#!/usr/bin/env python3
"""Grouped cascade attention: one shared-prefix pass PER GROUP, not one per batch.

vLLM's cascade (v1/attention/backends/flash_attn.py:1132) reads the shared prefix KV once
for the whole batch, but only when EVERY resident request shares it -- common blocks are
those whose ref_cnt equals the number of requests with allocated KV
(v1/core/kv_cache_manager.py:485). Our batch is G groups of 16 siblings, each group sharing
its own replay prefix, so that count collapses to the chat template and cascade never fires.

The generalization needs no new kernel. vLLM's single-group call is a degenerate use of a
kernel that is already variable-length batched:

    cu_prefix_query_lens = [0, num_tokens]     # one "sequence" spanning the whole batch
    prefix_kv_lens       = [common_prefix_len] # one prefix
    block_table[:1]                            # one block-table row

Give it G+1 cumulative query offsets, G prefix lengths, and G block-table rows and the same
flash_attn_varlen_func reads each group's prefix once. Two things then have to be handled
that the single-group case gets for free:

  1. QUERY ORDER. cu_prefix_query_lens partitions the token axis into contiguous spans, so a
     group's query tokens must be adjacent. Rather than reorder vLLM's input batch -- which
     would ripple into sampling, logprobs and cudagraph capture -- we permute the query
     tensor into group order, run one call, and scatter the result back. During decode
     that is one row per resident sequence against tens of thousands of prefix tokens per
     group, so the copy is negligible against the KV traffic it saves.

  2. SUFFIX BLOCK TABLES. The single-group path slices block_table[:, num_common_kv_blocks:]
     because every request skips the same number of common blocks. With per-group prefixes
     each request skips its own, so the uniform slice becomes a gather.

Correctness is not eyeballed: selftest() builds a paged KV cache, runs this against a
per-request reference (plain attention over the request's whole context) and compares. A
wrong attention corrupts training silently rather than crashing, so the equivalence test is
the point of this file, not an extra.
"""
from __future__ import annotations

import os
import torch

try:
    from vllm.vllm_flash_attn import flash_attn_varlen_func, get_scheduler_metadata
except ImportError:  # layout differs across vLLM builds
    from vllm.attention.utils.fa_utils import flash_attn_varlen_func  # type: ignore
    get_scheduler_metadata = None

# In this vLLM the helper lives at v1/attention/backends/fa_utils.py -- the first path tried
# here was wrong for this build, the except silently pinned FA2, splits collapsed to 1, and
# the "split" benchmark reproduced the unsplit number to 0.04%. Fail LOUD on total absence
# instead: a wrong version here doesn't crash anything, it just quietly measures nothing.
try:
    from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
except ImportError:
    from vllm.attention.utils.fa_utils import get_flash_attn_version  # older layouts
_FA_VERSION = get_flash_attn_version() or 2
print(f"[grouped-cascade] flash-attn version resolved: {_FA_VERSION}", flush=True)
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states


def plan_groups(group_ids: torch.Tensor, query_start_loc: torch.Tensor,
                prefix_lens: torch.Tensor, num_computed_tokens: torch.Tensor,
                block_size: int):
    """Build the permutation and per-group metadata for the prefix pass.

    Args:
        group_ids:       [num_reqs] int, which prefix-sharing group each request belongs to.
                         Requests in a singleton group (nothing to share) must be given a
                         prefix_len of 0 and are handled entirely by the suffix pass.
        query_start_loc: [num_reqs+1] cumulative query-token offsets (vLLM's own layout).
        prefix_lens:     [num_reqs] shared-prefix length for each request's group, a
                         multiple of block_size.
    Returns dict with the token permutation, cu_prefix_query_lens, per-group prefix kv
    lens, and the representative request index for each group's block-table row.
    """
    device = group_ids.device
    num_reqs = int(group_ids.numel())
    q_lens = (query_start_loc[1:] - query_start_loc[:-1]).to(torch.int64)

    # CAP THE PREFIX BY min(num_computed_tokens) WITHIN EACH GROUP. The prefix pass is
    # bi-directional, so if any member's own query tokens fall inside the shared region it
    # would attend to its own future tokens unmasked -- wrong output, no error. vLLM applies
    # this cap batch-wide (gpu_model_runner: common_prefix_len = min(common_prefix_len,
    # num_computed_tokens.min())); with per-group prefixes it has to hold per group. The cap
    # is NOT raised by +1 on purpose: every request must keep a non-empty suffix, because
    # the two-kernel merge has no path for a request handled entirely by the prefix pass.
    # SP_GC_NO_CAP=1 removes the cap. Only the selftest sets it, to prove the overlap cases
    # genuinely fail without it -- a test that passes either way proves nothing.
    if os.environ.get("SP_GC_NO_CAP") != "1":
        eff = prefix_lens.clone().to(torch.int64)
        for g in torch.unique(group_ids).tolist():
            m = group_ids == g
            cap = int(torch.minimum(prefix_lens[m].min(), num_computed_tokens[m].min()))
            eff[m] = (cap // block_size) * block_size
        prefix_lens = eff.to(prefix_lens.dtype)

    # Only groups that still have a real shared prefix take part in the prefix pass.
    active = prefix_lens > 0
    uniq, inverse = torch.unique(group_ids[active], return_inverse=True)
    num_groups = int(uniq.numel())

    order = []            # request indices, grouped
    rep = []              # one representative request per group (its block-table row)
    tokens_per_group = []
    active_idx = torch.nonzero(active, as_tuple=True)[0]
    for g in range(num_groups):
        members = active_idx[inverse == g]
        order.append(members)
        rep.append(int(members[0]))
        tokens_per_group.append(int(q_lens[members].sum()))
    order_reqs = torch.cat(order) if order else torch.empty(0, dtype=torch.int64, device=device)

    # Expand request order -> token order (a request contributes q_lens[i] consecutive rows).
    tok = [torch.arange(int(query_start_loc[i]), int(query_start_loc[i + 1]),
                        device=device, dtype=torch.int64) for i in order_reqs.tolist()]
    perm = torch.cat(tok) if tok else torch.empty(0, dtype=torch.int64, device=device)

    cu = torch.zeros(num_groups + 1, dtype=torch.int32, device=device)
    if num_groups:
        cu[1:] = torch.tensor(tokens_per_group, dtype=torch.int32,
                              device=device).cumsum(0)
    rep_t = torch.tensor(rep, dtype=torch.int64, device=device) if rep else \
        torch.empty(0, dtype=torch.int64, device=device)
    group_prefix = prefix_lens[rep_t].to(torch.int32) if num_groups else \
        torch.empty(0, dtype=torch.int32, device=device)

    order_list = order_reqs.tolist()
    return {"perm": perm, "cu_prefix_query_lens": cu, "prefix_kv_lens": group_prefix,
            "rep": rep_t, "num_groups": num_groups,
            # identity: every request is grouped and already in batch order -> the gather
            # and the scatter-back are no-ops and can be skipped entirely (the common decode
            # case: one group, siblings adjacent). Computed here because this function
            # already pays its one CPU sync per step; nothing new stalls.
            "identity": bool(active.all()) and order_list == list(range(num_reqs))
                        and bool((q_lens == 1).all()),
            "max_prefix": int(group_prefix.max()) if num_groups else 0,
            # capped per-request prefix -- the suffix pass MUST use this, not the caller's
            # requested prefix_lens, or the two passes disagree about where the split is.
            "req_prefix_lens": prefix_lens,
            "max_group_tokens": max(tokens_per_group) if tokens_per_group else 0,
            "covers_all": bool(active.all()) and num_reqs > 0,
            # UNIFORM: every group has the same member count and the same prefix length. That
            # is what the fused kernel trades generality for -- it indexes group g's queries as
            # a dense [S, D] tile at g*S, with no cu_seqlens, so a ragged batch would silently
            # read the wrong rows. Checked here rather than assumed from the workload.
            "uniform": (num_groups > 0
                        and len({len(o) for o in order}) == 1
                        and len(set(prefix_lens[rep_t].tolist())) == 1),
            "siblings": (len(order[0]) if order else 0),
            # CROSS-STEP cache. The caller reuses this plan for every step on which the
            # grouping holds, taking a fresh shallow copy each time (so "_rt" is per-step).
            # "_xrt" is a mutable dict, so all those copies share ONE object -- anything
            # derived from the grouping or the prefix alone gets computed on the first step
            # and read thereafter, instead of being rebuilt 5000 times.
            "_xrt": {}}


_LSE_TOKEN_AXIS: int | None = None


def lse_token_axis(lse: torch.Tensor, num_tokens: int, num_heads: int) -> int:
    """Which axis of flash-attn's softmax_lse indexes TOKENS.

    Inferring this from shape fails silently when num_heads == num_tokens -- exactly the
    decode case with 32 resident sequences on a 32-query-head model, where [32, 32] is
    ambiguous and guessing wrong permutes heads instead of tokens. That produced a 0.119
    logprob error end-to-end while every standalone case (8 heads, 32-64 tokens) passed.
    So resolve it ONCE from a non-square observation and cache it.
    """
    global _LSE_TOKEN_AXIS
    if _LSE_TOKEN_AXIS is None and num_tokens != num_heads:
        _LSE_TOKEN_AXIS = 0 if lse.shape[0] == num_tokens else 1
    if _LSE_TOKEN_AXIS is not None:
        return _LSE_TOKEN_AXIS
    raise RuntimeError(
        "grouped cascade: softmax_lse layout is ambiguous (num_heads == num_tokens) and no "
        "non-square batch has been seen yet. Call probe_lse_layout() at install time.")


def probe_lse_layout(device="cuda", dtype=torch.bfloat16) -> int:
    """Resolve the LSE token axis with a deliberately non-square dummy call."""
    global _LSE_TOKEN_AXIS
    if _LSE_TOKEN_AXIS is not None:
        return _LSE_TOKEN_AXIS
    nh, nkv, d, bs = 2, 1, 64, 16          # 2 heads
    ntok = 5                               # 5 tokens -- never equal to nh
    kc = torch.zeros(2, bs, nkv, d, device=device, dtype=dtype)
    vc = torch.zeros_like(kc)
    q = torch.zeros(ntok, nh, d, device=device, dtype=dtype)
    cu = torch.tensor([0, ntok], dtype=torch.int32, device=device)
    bt = torch.zeros(1, 2, dtype=torch.int32, device=device)
    _, lse = flash_attn_varlen_func(
        q=q, k=kc, v=vc, cu_seqlens_q=cu,
        seqused_k=torch.tensor([bs], dtype=torch.int32, device=device),
        max_seqlen_q=ntok, max_seqlen_k=bs, softmax_scale=1.0, causal=False,
        window_size=[-1, -1], block_table=bt, softcap=0.0, return_softmax_lse=True)
    _LSE_TOKEN_AXIS = 0 if lse.shape[0] == ntok else 1
    return _LSE_TOKEN_AXIS


# FAST PATH: call torch.ops._vllm_fa3_C.fwd directly instead of vLLM's python wrapper.
#
# Profiler on the n=32 cascade cell: the device is idle 29.8% of the step, and the gaps are
# attributable -- 2.03 ms/step of it sits in ONE place, the 56.4 us that follows every layer's
# reshape_and_cache_flash. That is the graph break: the KV write is the last graphed op before
# attention, so the gap is exactly the python between it and our first FA launch, paid 36x a
# step. flash_attn_varlen_func is a 130-line python prologue over a 37-positional-arg custom
# op: three asserts, a softmax_scale default, window_size normalisation, maybe_contiguous on
# q/k/v, and a torch.empty_like(cu_seqlens_q) allocated on EVERY call for a "dummy cu_seqlens_k"
# that the FA3 branch does not even use.
#
# All of that is loop-invariant here. The arg list is built once per step and only q/k/v move
# between layers, so the per-layer cost becomes three list writes plus the dispatch.
#
# The index map below is the FA3 branch of flash_attn_varlen_func verbatim. Getting an index
# wrong is a silent-wrong-answer bug, not a crash -- so this path is only taken when the probe
# below confirms the op exists, and selftest_grouped_cascade.py runs the whole thing against a
# per-request reference. A passing selftest is what licenses this, not the reading of the map.
_FA3_FWD = None
if _FA_VERSION >= 3 and os.environ.get("SP_GC_FASTCALL", "1") == "1":
    try:
        _FA3_FWD = torch.ops._vllm_fa3_C.fwd
    except Exception as _e:                       # op not registered in this build
        print(f"[grouped-cascade] fastcall unavailable ({_e}); using the python wrapper",
              flush=True)
        _FA3_FWD = None

# positional slots that change between layers; everything else is filled once per step
_FA_Q, _FA_K, _FA_V, _FA_OUT = 0, 1, 2, 6


def _fa3_args(q, k, v, out, cu_seqlens_q, seqused_k, max_seqlen_q, max_seqlen_k,
              block_table, softmax_scale, causal, softcap, scheduler_metadata, num_splits):
    """The 37 positional args of torch.ops._vllm_fa3_C.fwd, in order."""
    return [
        q, k, v,
        None, None,          # k_new, v_new
        None,                # q_v
        out,
        cu_seqlens_q,
        None,                # cu_seqlens_k (seqused_k is used instead)
        None,                # cu_seqlens_k_new
        None,                # seqused_q
        seqused_k,
        max_seqlen_q, max_seqlen_k,
        block_table,
        None,                # kv_batch_idx
        None,                # leftpad_k
        None, None, None,    # rotary_cos, rotary_sin, seqlens_rotary
        None, None, None,    # q_descale, k_descale, v_descale
        softmax_scale,
        causal,
        -1, -1,              # window_size left/right
        softcap,
        True,                # rotary_interleaved
        scheduler_metadata,
        num_splits,
        None,                # pack_gqa
        0,                   # sm_margin
        None,                # s_aux
        1, 0, None,          # cp_world_size, cp_rank, cp_tot_seqused_k
    ]


_PREFIX_SPLITS_ENV = os.environ.get("SP_GC_PREFIX_SPLITS", "auto")
# SP_GC_TIME=1: component-level timing, sampled 1-in-64 calls. Whole-call totals cannot
# answer where the time goes; this splits every sampled layer call into suffix kernel /
# prefix kernel / everything else (gather, scatter, merge). Printed every 200 samples.
_CT_ON = os.environ.get("SP_GC_TIME") == "1"
# SP_GC_OVERLAP (default on): run the prefix pass on a side stream, concurrent with the
# suffix pass. The two are independent until the merge; serialized they pay two wave tails
# and the smaller pass's full latency, which the baseline's single fused kernel never pays.
_OVERLAP = os.environ.get("SP_GC_OVERLAP", "1") == "1"
# SP_GC_FUSED=1 routes the identity/uniform decode case through fused_grouped_attn. Off by
# default: it is a different kernel, and it only earns the default after it is timed against
# the 81.95 s reference at n=32/prefix 50k/rollout 5k.
_FUSED_MODE = int(os.environ.get("SP_GC_FUSED", "0"))
_FUSED_ON = _FUSED_MODE == 1
# The fused path wants FEWER splits than FA3's auto-32. FA3 needs 32 to fill the machine
# because its combine is a separate kernel launch; here kernel B supplies its own parallelism
# (one CTA per token x KV head), so kernel A only has to cover the prefix phase -- and every
# extra split multiplies the fp32 partial workspace that kernel B must read back:
#   32 splits x 192 tokens x 8 heads x 128 dim x 4 B = 25.2 MB/layer, ~1.8 GB/step
# At 8 splits the grid is still (8, 48) = 384 CTAs, ~2.9 waves on 132 SMs, and the workspace
# traffic drops 4x. 0 = defer to FA3's choice.
_FUSED_SPLITS = int(os.environ.get("SP_GC_FUSED_SPLITS", "32"))
_HYB_PSPLIT = int(os.environ.get("SP_GC_HYBRID_PSPLIT", "8"))
_HYB_SEEN = False
_HYB_GATE_SEEN = set()      # distinct gate-failure signatures already reported           # one-shot "the hybrid actually dispatched" marker
_WALK_SEEN = False          # one-shot marker for SP_GC_FUSED=3
_OVL_SEEN = False           # one-shot marker for SP_GC_FUSED=4
_SIDE_STREAM = None


_EV_IN = None
_EV_SIDE = None


# SP_GC_CPUPROF=1: hand-rolled CPU timing of OUR per-layer path, because both profilers have
# now produced inflated numbers here -- CUPTI put +25% on the wall and charged it to idle, and
# with_stack put 3-8 us of tax on every python frame and made torch.cuda.current_stream look
# like the bubble (caching it changed nothing end to end). perf_counter pairs at four points
# cost ~70 ns each and cannot lie about where the boundary sits: everything between t0 and t3
# is ours; everything outside is vLLM machinery plus piecewise replays.
_CPUPROF = os.environ.get("SP_GC_CPUPROF", "0") == "1"
_DYN_SUF = os.environ.get("SP_GC_DYN", "0") == "1"
_CP = {"n": 0, "pref": 0.0, "walk": 0.0, "merge": 0.0, "body": 0.0}


def _cp_report():
    n = _CP["n"]
    if not n:
        return
    print(f"[gc-cpu] layers={n}  body {1e6*_CP['body']/n:7.1f} us/layer"
          f"  (prefix {1e6*_CP['pref']/n:6.1f}  walk {1e6*_CP['walk']/n:6.1f}"
          f"  merge {1e6*_CP['merge']/n:6.1f})"
          f"  -> {36e3*_CP['body']/n:6.3f} ms/step ours", flush=True)
    for k in _CP:
        _CP[k] = 0 if k == "n" else 0.0


_MAIN_STREAM = {}        # device -> the issuing stream object; see _main_stream()


def _main_stream(device):
    """torch.cuda.current_stream(), cached, because the uncached call is 30+ us.

    THIS IS THE BUBBLE. Dumping the CPU timeline inside the per-layer gap -- 294.8 us between
    reshape_and_cache_flash and our suffix kernel -- shows where it goes:

        record()            35.5 us   -> current_stream() -> _get_device_index()
                                      -> _get_available_device_type() -> torch.cuda.is_available()
                                      -> _nvml_based_avail() -> os.getenv()
        torch.cuda.stream() 18.9 us   (its __init__ calls current_stream() again)
        current_stream()     7.3 us

    Every Event.record() and every stream context re-resolves the device through an NVML
    availability probe and an environment lookup, ~60 us per layer, 36 layers deep, 2.3 ms/step
    -- against a Triton launch that measures 18.7 us. That is why the launcher was never the
    problem, why five attacks on it all measured negative, and why the isolated bench never
    showed it: there 30 iterations are queued back to back and the CPU runs ahead of a full
    GPU queue, while here each layer only has ~0.14 ms of device work to hide behind.

    The stream object itself is immutable and per-device, so it is cached once. Events are
    recorded against it explicitly, which skips the whole resolution chain.
    """
    st = _MAIN_STREAM.get(device)
    if st is None:
        st = torch.cuda.current_stream(device)
        _MAIN_STREAM[device] = st
    return st


def _side_stream(device):
    """Side stream plus the two reusable events used with it.

    Both used to be constructed inside the per-layer call: two cudaEventCreate/Destroy pairs,
    36 times a step, 5000 steps. Events are re-recordable, and cudaStreamWaitEvent captures
    the event's recorded work at CALL time, so a later re-record cannot retroactively weaken
    a wait that was already enqueued -- which is what makes reuse safe here: within a layer
    the record and the wait are enqueued back to back, and the next record happens only after
    the previous wait is already in.

    The MAIN stream is deliberately NOT cached: it is queried per call. vLLM is free to run
    the eager attention break on a stream of its choosing, and a stale handle would enqueue
    the pre-merge wait_event on the wrong stream -- the merge would then read the prefix
    buffer before the prefix kernel retired, silently, on some steps. That is the same
    silent-corruption class as the prefix-cap and LSE-axis bugs. A current_stream() query is
    ~1us; the two events it sits next to were the expensive part.
    """
    global _SIDE_STREAM, _EV_IN, _EV_SIDE
    if _SIDE_STREAM is None:
        _SIDE_STREAM = torch.cuda.Stream(device)
        _EV_IN = torch.cuda.Event()
        _EV_SIDE = torch.cuda.Event()
    return _SIDE_STREAM
_CT = {"n": 0, "suffix": [], "prefix": [], "other": []}


def _ct_print():
    k = len(_CT["suffix"])
    if k == 0 or k % 200:
        return
    s_ = sum(_CT["suffix"][-200:]) / 200
    p_ = sum(_CT["prefix"][-200:]) / 200
    o_ = sum(_CT["other"][-200:]) / 200
    print(f"[gc-comp] samples={k} per-layer ms: suffix={s_:.3f} prefix={p_:.3f} "
          f"other={o_:.3f} total={s_ + p_ + o_:.3f} pid={os.getpid()}", flush=True)
_NUM_SMS: int | None = None


def _num_sms(device) -> int:
    global _NUM_SMS
    if _NUM_SMS is None:
        _NUM_SMS = torch.cuda.get_device_properties(device).multi_processor_count
    return _NUM_SMS


_STATIC_MODE = os.environ.get("SP_GC_STATIC", "0") == "1"


def _prefix_bt(block_table, plan, xrt):
    if not plan["num_groups"]:
        return None
    if not _STATIC_MODE:
        return block_table[plan["rep"]]
    # STATIC: the captured graph reads this buffer's ADDRESS, so the per-step refresh must
    # write in place. A fresh fancy-index copy would leave the graph reading the capture-time
    # dummy pages forever -- silently wrong attention, the exact failure class the greedy A/B
    # gate downstream exists to catch.
    key = (int(plan["num_groups"]), int(block_table.shape[1]))
    if xrt.get("pbt_key") != key:
        xrt["pbt_key"] = key
        xrt["pbt_buf"] = torch.empty(key, dtype=block_table.dtype, device=block_table.device)
    torch.index_select(block_table, 0, plan["rep"], out=xrt["pbt_buf"])
    return xrt["pbt_buf"]


def _suffix_block_table(block_table: torch.Tensor, prefix_lens: torch.Tensor,
                        block_size: int, skip_min: int | None = None,
                        cols: torch.Tensor | None = None) -> torch.Tensor:
    """block_table[i] shifted left by each request's OWN common-block count.

    The single-group path can write block_table[:, k:] because k is shared. Here k varies by
    group, so gather row-wise. Out-of-range indices are clamped; the kernel never reads past
    seqused_k, so clamped tails are inert.

    skip_min/cols are the cross-step cache. The shift is a function of prefix_lens alone,
    which does not change while the grouping holds, so `int(skip.min())` -- a GPU->CPU sync
    on every step -- and the index tensor it sizes are both computed once per grouping
    instead of once per step.
    """
    if cols is not None:
        return torch.gather(block_table, 1, cols)
    num_reqs, width = block_table.shape
    skip = (prefix_lens // block_size).to(torch.int64)          # [num_reqs]
    if skip_min is None:
        skip_min = int(skip.min())
    keep = width - skip_min
    cols = torch.arange(keep, device=block_table.device).unsqueeze(0) + skip.unsqueeze(1)
    return torch.gather(block_table, 1, cols.clamp_(max=width - 1))


def grouped_cascade_attention(
    output: torch.Tensor, query: torch.Tensor,
    key_cache: torch.Tensor, value_cache: torch.Tensor,
    query_start_loc: torch.Tensor, seq_lens: torch.Tensor,
    group_ids: torch.Tensor, prefix_lens: torch.Tensor,
    block_table: torch.Tensor, softmax_scale: float,
    max_query_len: int, num_computed_tokens: torch.Tensor | None = None,
    logits_soft_cap: float = 0.0, plan: dict | None = None,
) -> torch.Tensor:
    """One shared-prefix pass per group + a per-request causal suffix pass, LSE-merged."""
    block_size = key_cache.shape[-3]
    if num_computed_tokens is None:                 # decode default: all but this step's query
        num_computed_tokens = seq_lens - (query_start_loc[1:] - query_start_loc[:-1])
    # The plan is a function of the BATCH, not the layer -- and this function is called once
    # per layer, 36x per decode step. Computing it here cost ~36 GPU->CPU syncs per step
    # (torch.unique + .tolist() + per-request arange), which single-handedly erased the
    # bandwidth saving: split-KV landed and the micro number did not move (158.8 vs 164.4).
    # Callers that can (the vLLM patch) pass a per-step plan; standalone callers pay once.
    if plan is None:
        plan = plan_groups(group_ids, query_start_loc, prefix_lens, num_computed_tokens,
                           block_size)
    prefix_lens = plan["req_prefix_lens"]           # capped; see plan_groups

    # Per-STEP runtime cache, filled by whichever layer runs first and reused by the other
    # 35. Everything here either syncs (int(tensor.max())) or launches small kernels (the
    # suffix block-table gather); at 36 layers/step these were ~220us/layer of the measured
    # 340us -- more than the prefix kernel itself (83us). The plan dict is shared across the
    # step, so the cache rides along with it.
    rt = plan.get("_rt")
    if rt is not None and rt.get("out_shape") != tuple(query.shape):
        # decode shapes are constant while the resident set is stable; any change
        # (a sequence finished, chunked prefill mixed in) just rebuilds the cache.
        rt = None
    if rt is None:
        xrt = plan.setdefault("_xrt", {})
        # CAPTURE-SAFE TENSOR LIFECYCLE (stage 1 of the FULL-cudagraph plan, useful on its
        # own). A captured graph holds POINTERS: every tensor a captured kernel reads must be
        # the SAME storage every step, refreshed in place by the builder outside the graph.
        # Until now suffix_kv and suffix_bt were freshly allocated per step -- harmless under
        # PIECEWISE, fatal under FULL (the graph would replay against the buffers from the
        # capture step while the fresh ones drift away). Persistent out= buffers change no
        # numerics and delete two per-step allocations.
        _nr = int(seq_lens.shape[0])
        if xrt.get("skv_key") != _nr:
            xrt["skv_key"] = _nr
            xrt["skv_buf"] = torch.empty(_nr, dtype=torch.int32, device=seq_lens.device)
        suffix_kv = torch.sub(seq_lens, prefix_lens, out=xrt["skv_buf"])
        # suffix_max from the host when the caller could supply it (seq_lens has a CPU
        # mirror in vLLM's metadata and the prefix lens are constant). int(tensor.max()) is
        # a GPU->CPU sync, and a sync here drains the queue mid-step, so the python for the
        # remaining layers stops overlapping the kernels already in flight.
        _smax = plan.get("suffix_max_cpu")
        if _smax is None:
            _smax = int(suffix_kv.max())
        if "sbt_cols" not in xrt:
            # the column index is a function of prefix_lens and the block-table width, both
            # constant while the grouping holds; building it costs an arange + add + clamp
            # and, worse, the int(skip.min()) sync that sizes it.
            _skip = (prefix_lens // key_cache.shape[-3]).to(torch.int64)
            _keep = int(block_table.shape[1]) - int(_skip.min())
            xrt["sbt_cols"] = (torch.arange(_keep, device=block_table.device).unsqueeze(0)
                               + _skip.unsqueeze(1)).clamp_(max=block_table.shape[1] - 1)
        _sbt_shape = (int(block_table.shape[0]), int(xrt["sbt_cols"].shape[1]))
        if xrt.get("sbt_buf_key") != _sbt_shape:
            xrt["sbt_buf_key"] = _sbt_shape
            xrt["sbt_buf"] = torch.empty(_sbt_shape, dtype=block_table.dtype,
                                         device=block_table.device)
            # device-side suffix_max, updated in place: the walk kernel can take its loop
            # bound from a device scalar (dynamic trip count), which removes the last host
            # value a captured launch would bake in. amax with out= keeps it allocation-free.
            xrt["smax_dev"] = torch.empty(1, dtype=torch.int32, device=block_table.device)
        torch.gather(block_table, 1, xrt["sbt_cols"], out=xrt["sbt_buf"])
        torch.amax(suffix_kv, dim=0, keepdim=True, out=xrt["smax_dev"])
        rt = {
            "suffix_kv": suffix_kv,
            "suffix_max": _smax,
            "suffix_max_dev": xrt["smax_dev"],
            "suffix_bt": xrt["sbt_buf"],
            "prefix_bt": _prefix_bt(block_table, plan, xrt),
        }
        if plan["num_groups"]:
            # The old auto formula (2*SMs // base_ctas) silently resolved to splits=1 in
            # production decode: a dozen groups x 32 heads already exceeds 2x the SM count,
            # so no CTA ever split its KV walk -- and with prefixes spanning 10k-80k in one
            # batch, the 80k group's serial walk set the tail while the 10k groups idled.
            # Measured: prefix 0.269 ms/layer (~1200 GB/s) vs 2460 GB/s in isolation. Same
            # cure as the suffix pass: FA3 AOT schedule with a shared split budget.
            _pq = int(plan["max_group_tokens"]) * int(plan["num_groups"])
            _pcap = 32 if _pq <= 256 else (8 if _pq <= 1024 else 1)
            # CAP BY THE PREFIX ITSELF. Everything above sizes the budget from QUERY tokens
            # and never asks how much KV there is to split, so a 16-token shared prefix got
            # the same 32 splits as a 50k one. Splitting a one-block walk 32 ways computes
            # nothing extra but still allocates FA3's fp32 accumulator of
            # [splits x tokens x heads x dim] -- 32 x 192 x 32 x 128 x 4 B = 100 MB per layer
            # -- and the combine kernel reads all of it back. Measured on the prefix=0 grid
            # control: cascade 80.27 s against stock 50.22 s, a 0.63x REGRESSION, ~167 us per
            # layer of pure accumulator traffic for zero dedup.
            # 256 tokens per split is the floor for a split being worth its share of the
            # combine. Deliberately chosen so this is a no-op at every prefix the grid
            # actually sweeps (10k -> 39 > 32, so the cap never binds); only degenerate
            # prefixes move, which is the bug and nothing else.
            _plen_cap = max(1, int(plan["max_prefix"]) // 256)
            if _PREFIX_SPLITS_ENV == "auto":
                prefix_splits = min(32, _pcap, _plen_cap)
            else:
                prefix_splits = min(_pcap, _plen_cap, max(1, int(_PREFIX_SPLITS_ENV)))
            if _FA_VERSION < 3:
                prefix_splits = 1        # FA2 rejects num_splits > 1
            rt["splits"] = prefix_splits
            # The prefix schedule is built from the group count, the per-group query token
            # count and the per-group prefix lengths -- none of which move while the
            # grouping holds. Build it once, not 5000 times.
            if "prefix_sched" in xrt:
                rt["prefix_sched"] = xrt["prefix_sched"]
            else:
                rt["prefix_sched"] = None
                if _FA_VERSION >= 3 and get_scheduler_metadata is not None and prefix_splits > 1:
                    try:
                        rt["prefix_sched"] = get_scheduler_metadata(
                            batch_size=int(plan["num_groups"]),
                            max_seqlen_q=int(plan["max_group_tokens"]),
                            max_seqlen_k=int(plan["max_prefix"]),
                            num_heads_q=int(query.shape[1]),
                            num_heads_kv=int(key_cache.shape[-2]),
                            headdim=int(query.shape[2]),
                            cache_seqlens=plan["prefix_kv_lens"],
                            qkv_dtype=query.dtype,
                            cu_seqlens_q=plan["cu_prefix_query_lens"],
                            page_size=int(key_cache.shape[-3]), causal=False,
                            window_size=(-1, -1), num_splits=prefix_splits)
                    except Exception:
                        rt["prefix_sched"] = None
                xrt["prefix_sched"] = rt["prefix_sched"]
        # NOTE the default here is "4" while the resolver below defaults to "auto", so an
        # unset env lands on auto and this line is only a floor. It must not int() the env
        # directly: SP_GC_SUFFIX_SPLITS=auto is a legal value for the resolver and int("auto")
        # raised ValueError, taking the whole cell down (the splits sweep lost its control
        # arm to exactly this). Parse defensively; the resolver below is the real decision.
        _ss0 = os.environ.get("SP_GC_SUFFIX_SPLITS", "4")
        rt["suffix_splits"] = (int(_ss0) if _ss0.isdigit() else 4) if _FA_VERSION >= 3 else 0
        # AOT tile schedule for the suffix pass. Production own-lengths span 0-90k in one
        # batch; a fixed split count load-balances nothing, and this pass is the MAJORITY of
        # attention traffic there (96 rows x own context vs a dozen deduped prefixes).
        # Stock FlashDecoding always passes this -- it is where its last ~30% of effective
        # bandwidth comes from on heterogeneous batches.
        rt["suffix_sched"] = None
        if _FA_VERSION >= 3 and get_scheduler_metadata is not None:
            try:
                # The kernel derives the metadata size from ITS OWN num_splits argument,
                # so the schedule must be built with the SAME split budget that the kernel
                # call passes -- stock does exactly this (schedule(num_splits=max_num_splits)
                # + kernel(num_splits=max_num_splits)). Building with one value and calling
                # with another raises 'scheduler_metadata must have shape (metadata_size)'.
                # budget scales with suffix length: 32 splits of a 500-token suffix is
                # confetti CTAs plus combine overhead; long suffixes want the full budget.
                _env_ss = os.environ.get("SP_GC_SUFFIX_SPLITS", "auto")
                # FA3 split-KV allocates an fp32 accumulator of [splits x tokens x heads x
                # dim]. A 32 budget on a chunked-PREFILL batch (~16k query tokens) demands
                # ~8GB and OOM'd the real run at step 298 (4GB alloc, 3.45GB free). Splits
                # only pay in decode (one token per seq); cap the budget by token count as
                # stock's builder does.
                _nq = int(query.shape[0])
                _tok_cap = 32 if _nq <= 256 else (8 if _nq <= 1024 else 1)
                if _env_ss == "auto":
                    _sm = rt["suffix_max"]
                    rt["suffix_splits"] = min(_tok_cap,
                                              32 if _sm > 16384 else (8 if _sm > 2048 else 2))
                else:
                    rt["suffix_splits"] = min(_tok_cap, int(_env_ss))
                rt["suffix_sched"] = get_scheduler_metadata(
                    batch_size=int(seq_lens.numel()), max_seqlen_q=max_query_len,
                    max_seqlen_k=rt["suffix_max"],
                    num_heads_q=int(query.shape[1]), num_heads_kv=int(key_cache.shape[-2]),
                    headdim=int(query.shape[2]), cache_seqlens=suffix_kv,
                    qkv_dtype=query.dtype, cu_seqlens_q=query_start_loc,
                    page_size=int(key_cache.shape[-3]), causal=True,
                    window_size=(-1, -1), num_splits=rt["suffix_splits"])
            except Exception:
                rt["suffix_sched"] = None      # fall back to manual splits
        # general-path indices, built once per step instead of mask machinery per layer:
        # merge treats lse=-inf as weight zero, so masked rows need NO output zeroing --
        # pre-fill lse with -inf and index_copy the computed columns over it.
        if not plan.get("identity", False) and plan["num_groups"]:
            rt["perm_idx"] = plan["perm"]
        # Preallocated kernel outputs. Both passes otherwise allocate output + LSE fresh,
        # 4 tensors x 36 layers x every step of allocator traffic on the eager hot path.
        # Shapes are constant during steady decode (guarded above); out= reuses these.
        rt["out_shape"] = tuple(query.shape)
        # Kernel output buffers, cross-step: the decode shape is constant while the grouping
        # holds, so these are two allocations per grouping rather than two per step. Keyed by
        # shape AND dtype so a changed batch rebuilds instead of aliasing the wrong buffer.
        _key = (tuple(query.shape), query.dtype)
        if xrt.get("buf_key") != _key:
            xrt["buf_key"] = _key
            xrt["suffix_out"] = torch.empty_like(query)
            xrt["prefix_out"] = torch.empty_like(query)
        rt["suffix_out"] = xrt["suffix_out"]
        rt["prefix_out"] = xrt["prefix_out"]
        # EQUAL SUFFIX LENGTH is the last precondition and the only per-step one: seq_lens grow
        # together only while every sibling emits the same number of tokens, which is exactly
        # this workload's contract. Checked on the CPU mirror so it costs no sync.
        if _FUSED_MODE:
            # suffix_uniform is decided by the caller from its CPU mirrors (see the patch);
            # absent it -- e.g. the standalone selftest path -- the fused route stays off
            # rather than guessing, because a ragged batch would read the wrong rows silently.
            #
            # THIS GUARD WAS `if _FUSED_ON:`, i.e. mode 1 ONLY, which left rt["fused_ok"]
            # unset under SP_GC_FUSED=2 and made the hybrid gate below dead code: the mode-2
            # branch tests rt.get("fused_ok") and got None every step, so every "hybrid"
            # end-to-end cell silently ran the three-kernel path. It read as "the fused kernel
            # does not help end to end" when the fused kernel was never dispatched -- n=48
            # hybrid 77.37 s vs 3-kernel 77.29 s was the same code timed twice.
            #
            # DECODE-ONLY IS A HARD PRECONDITION, and it was missing. Both fused kernels use
            # program_id(0) to index the query rows AND the suffix block-table rows with the
            # same t, which only lines up when every request contributes exactly one query
            # token. On a prefill or chunked-prefill step num_tokens >> num_reqs, t runs off
            # the end of suffix_bt, and the kernel takes an illegal memory access -- which is
            # exactly how the first genuine hybrid dispatch died, ~90 s in, on the first
            # prefill. Neither test could catch it: the selftest and bench_fused both build
            # q with one row per sequence, so num_tokens == num_reqs by construction there.
            # suffix_uniform is required by the PURE fused path (mode 1), whose kernel B still
            # takes one scalar length. The hybrid's epilogue now masks per row, so demanding it
            # there only kept the gate permanently shut: measured live, suffix_uniform is False
            # at every resident count because vLLM admits requests across several prefill steps
            # and the resulting offset never washes out.
            rt["fused_ok"] = bool(plan.get("uniform")
                                  and (plan.get("suffix_uniform") or _FUSED_MODE in (2, 3, 4))
                                  and plan["num_groups"] > 0
                                  and int(query.shape[0]) == int(block_table.shape[0]))
            # The workspace is the PURE fused path's fp32 partial store. The hybrid takes FA3's
            # already-combined (out, lse) and folds the epilogue in, so it needs no workspace --
            # keep the allocation on mode 1 rather than paying 25 MB/layer for mode 2.
            if _FUSED_ON and rt["fused_ok"] and xrt.get("fused_ws_key") != (_key, rt["splits"]):
                _S = int(plan["siblings"]); _H = int(query.shape[1]); _D = int(query.shape[2])
                _SIB = max(16, 1 << (_S - 1).bit_length())
                xrt["fused_ws_key"] = (_key, rt["splits"])
                xrt["fused_ws"] = (
                    torch.empty((plan["num_groups"], _H, rt["splits"], _SIB, _D),
                                dtype=torch.float32, device=query.device),
                    torch.empty((plan["num_groups"], _H, rt["splits"], _SIB),
                                dtype=torch.float32, device=query.device),
                    torch.empty((plan["num_groups"], _H, rt["splits"], _SIB),
                                dtype=torch.float32, device=query.device))
            rt["fused_ws"] = xrt.get("fused_ws")
        # Prebuilt positional arg lists for the fastcall path. Everything except q/k/v is
        # fixed for the whole step, so the per-layer cost collapses to three writes and the
        # dispatch. Only the identity decode case gets a prebuilt prefix list: the general
        # path passes out=None and gathers query[perm], so its args are not loop-invariant.
        if _FA3_FWD is not None:
            rt["suffix_args"] = _fa3_args(
                query, key_cache, value_cache, rt["suffix_out"], query_start_loc,
                rt["suffix_kv"], max_query_len, rt["suffix_max"], rt["suffix_bt"],
                softmax_scale, True, logits_soft_cap, rt.get("suffix_sched"),
                rt.get("suffix_splits", 0))
            if plan["num_groups"] and plan.get("identity", False):
                rt["prefix_args"] = _fa3_args(
                    query, key_cache, value_cache, rt["prefix_out"],
                    plan["cu_prefix_query_lens"], plan["prefix_kv_lens"],
                    plan["max_group_tokens"], plan["max_prefix"], rt["prefix_bt"],
                    softmax_scale, False, logits_soft_cap, rt.get("prefix_sched"),
                    rt["splits"])
        plan["_rt"] = rt

    # Mode 4 owns its own concurrency and must NOT have the prefix issued for it here, or the
    # prefix pass runs twice. See the mode-4 branch for why it schedules the halves the other
    # way round from this block.
    _ov = _OVERLAP and plan["num_groups"] > 0 and _FUSED_MODE != 4
    if _ov:
        # order: side stream must see Q/KV writes from the main stream, and the main
        # stream must see the prefix outputs before the merge. Two events, no syncs.
        _sstream = _side_stream(query.device)
        _EV_IN.record(_main_stream(query.device))
        _sstream.wait_event(_EV_IN)
    _ct = _CT_ON and (_CT.__setitem__("n", _CT["n"] + 1) or _CT["n"] % 64 == 0)
    if _ct:
        _e0 = torch.cuda.Event(enable_timing=True); _e1 = torch.cuda.Event(enable_timing=True)
        _e2 = torch.cuda.Event(enable_timing=True); _e3 = torch.cuda.Event(enable_timing=True)
        _e0.record()

    perm = plan["perm"] if plan["num_groups"] else None
    identity = plan.get("identity", False)

    # DEFINED HERE, ABOVE THE FUSED BRANCHES, NOT BELOW THEM. The hybrid branch calls
    # _prefix_call() to get FA3's prefix (out, lse) before running its own epilogue. While that
    # branch was unreachable the forward reference was invisible to every test; the first real
    # dispatch died in all four ranks with "cannot access local variable '_prefix_call'".
    _pa = rt.get("prefix_args")

    def _prefix_call():
        if _pa is not None:                       # identity decode: prebuilt, loop-invariant
            _pa[_FA_Q] = query
            _pa[_FA_K] = key_cache
            _pa[_FA_V] = value_cache
            o, lse, _, _ = _FA3_FWD(*_pa)
            return o, lse
        return flash_attn_varlen_func(
            q=query if identity else query[perm],
            k=key_cache, v=value_cache,
            cu_seqlens_q=plan["cu_prefix_query_lens"], seqused_k=plan["prefix_kv_lens"],
            max_seqlen_q=plan["max_group_tokens"], max_seqlen_k=plan["max_prefix"],
            softmax_scale=softmax_scale, causal=False, window_size=[-1, -1],
            block_table=rt["prefix_bt"], softcap=logits_soft_cap, return_softmax_lse=True,
            num_splits=rt["splits"], fa_version=_FA_VERSION,
            scheduler_metadata=rt.get("prefix_sched"),
            out=rt["prefix_out"] if identity else None,
        )

    # ---- FUSED PATH (SP_GC_FUSED=1) --------------------------------------------------------
    # One prefix split-walk + one suffix/combine/merge epilogue, replacing prefix FA3 +
    # FlashAttnFwdCombine + suffix FA3 + merge_attn_states. Requires exactly what the workload
    # guarantees and the general path cannot assume: identity order, uniform group sizes and
    # prefix lengths, and an EQUAL suffix length across sequences. All three are verified, not
    # assumed -- a ragged batch here would read the wrong rows silently.
    # SP_GC_FUSED=2: HYBRID. Isolated per-layer timing says each side wins a different half --
    # FA3's prefix 0.1250 ms vs my 0.1972, my suffix+combine+merge 0.1591 vs FA3's 0.1663 -- so
    # the best measured configuration is FA3's prefix walk plus my epilogue, which folds the
    # split-combine and merge_attn_states into the suffix pass and drops the 25 MB/layer fp32
    # workspace entirely. Measured 0.2768 ms/layer, 80% of roofline, 5.19x vs stock against
    # FA3's 4.91x, verified rel=0.00621 against the dense reference.
    # WHY THE GATE IS INSTRUMENTED. "HYBRID path live" fires on the FIRST dispatch, which is not
    # the same as firing on every step, and that difference is invisible from outside: the run
    # emits three-kernel numbers under a hybrid label. It happened twice. The traced hybrid arm
    # matched the three-kernel arm to two decimals in every bucket AND to the exact device-event
    # count (14437), and the one dispatch that did occur reported T=64, SUF=5013 -- the tail of
    # the run, where only one group of survivors is left. So the gate holds for a 64-request
    # remainder and fails for the 192-request body. Print WHICH term is false, once per distinct
    # combination, instead of guessing a fifth time.
    if _FUSED_MODE == 2 and plan["num_groups"] and not (
            identity and plan.get("uniform") and rt.get("fused_ok")):
        _sig = (bool(identity), bool(plan.get("uniform")), bool(plan.get("suffix_uniform")),
                int(query.shape[0]), int(block_table.shape[0]), int(plan["num_groups"]))
        global _HYB_GATE_SEEN
        if _sig not in _HYB_GATE_SEEN and len(_HYB_GATE_SEEN) < 8:
            _HYB_GATE_SEEN.add(_sig)
            print(f"[grouped-cascade] HYBRID gate FALSE: identity={_sig[0]} uniform={_sig[1]} "
                  f"suffix_uniform={_sig[2]} num_tokens={_sig[3]} bt_rows={_sig[4]} "
                  f"groups={_sig[5]}", flush=True)
    # SP_GC_FUSED=4: the schedule that measures 6.8-7.1x in isolation, wired in unchanged.
    #
    # WHY THE ORDER IS REVERSED from the _ov block above. That block issues the PREFIX on the
    # side stream and then the suffix on the main one -- short kernel first, long kernel second.
    # The prefix is only 80-120 us of GPU work, and between the two enqueues sits the whole
    # python cost of setting up the suffix launch, so the prefix has already retired by the time
    # the suffix is submitted and there is nothing left to overlap with. That is exactly what the
    # engine traces show: sum/union = 1.02, no concurrency, while the same two kernels reach 50%
    # hiding in the bench.
    #
    # overlapped_grouped_attention enqueues the LONG half first -- the suffix walk, ~0.15 ms, on
    # the side stream -- and only then calls prefix_fn() on the main stream. The long kernel is
    # already resident, so CPU latency on the second launch costs nothing and the short prefix
    # slots in beside it. Isolated: serial 0.27 -> overlapped 0.21 ms/layer, past the 6.57x
    # SERIAL floor, which no one-after-the-other schedule can do.
    if _FUSED_MODE == 4 and identity and rt.get("fused_ok"):
        from .fused_grouped_attn import overlapped_grouped_attention
        global _OVL_SEEN
        if not _OVL_SEEN:
            _OVL_SEEN = True
            print(f"[grouped-cascade] OVERLAP path live: groups={plan['num_groups']} "
                  f"siblings={plan['siblings']} splits={rt['splits']}", flush=True)
        _xr = plan.setdefault("_xrt", {})
        _T, _H, _D = int(query.shape[0]), int(query.shape[1]), int(query.shape[2])
        if _xr.get("ovl_ws_key") != (_T, _H, _D):
            _xr["ovl_ws_key"] = (_T, _H, _D)
            _xr["ovl_ws"] = (torch.empty((_T, _H, _D), dtype=torch.float32, device=query.device),
                             torch.empty((_T, _H), dtype=torch.float32, device=query.device),
                             torch.empty((_T, _H), dtype=torch.float32, device=query.device))
        # _sstream is bound inside the `if _ov:` block, which is disabled for this mode, so
        # taking it from there would be an UnboundLocalError on the first dispatch -- the same
        # forward-reference shape as the _prefix_call and xrt bugs earlier today. Ask the
        # helper directly; it caches per device.
        overlapped_grouped_attention(
            query, key_cache, value_cache, _prefix_call, rt["suffix_bt"],
            int(rt["suffix_max"]), softmax_scale, out=output, ws=_xr["ovl_ws"],
            stream=_side_stream(query.device),
            suffix_lens=(None if plan.get("suffix_uniform") else rt["suffix_kv"]))
        return output

    if _FUSED_MODE == 2 and identity and plan.get("uniform") and rt.get("fused_ok"):
        from .fused_grouped_attn import hybrid_grouped_attention
        # Say so ONCE, on the first dispatch. A dead gate here is invisible from the outside --
        # the run just quietly produces three-kernel numbers under a "hybrid" label, which is
        # exactly what happened while rt["fused_ok"] was mode-1-only. A single line makes the
        # difference between "the kernel did not help" and "the kernel did not run".
        global _HYB_SEEN
        if not _HYB_SEEN:
            _HYB_SEEN = True
            print(f"[grouped-cascade] HYBRID path live: groups={plan['num_groups']} "
                  f"siblings={plan['siblings']} psplit={_HYB_PSPLIT}", flush=True)
        # FA3's prefix wants FEWER splits in the hybrid than the three-kernel path does. There
        # its combine is a separate launch that has to fill the machine on its own; here the
        # epilogue supplies the parallelism, so the split budget only has to cover the prefix
        # walk -- and every extra split adds partial traffic FA3 must combine internally.
        # Swept: 8 -> 0.2685 ms/layer, 16 -> 0.2731, 32 -> 0.2730, 64 -> 0.2727.
        #
        # SLOT 29 MUST MOVE WITH SLOT 30. Slot 30 is num_splits, but slot 29 is the AOT
        # scheduler_metadata, and FA3 builds that schedule FOR a specific split count: it tells
        # each CTA which tile of which split to take. Overriding the split count while leaving
        # the schedule built for rt["splits"] makes FA3 index a split accumulator it was not
        # given, which is an illegal memory access -- and because the fault is asynchronous it
        # surfaced at the NEXT launch, my Triton kernel, sending three debugging passes after
        # index arithmetic that was never wrong. Passing None makes FA3 derive the schedule for
        # the split count actually in effect; the bench put psplit 8/16/32/64 within 1% of each
        # other, so the knob is not worth a second cached schedule.
        if _HYB_PSPLIT and rt.get("prefix_args") is not None:
            _saved = (rt["prefix_args"][29], rt["prefix_args"][30])
            rt["prefix_args"][29] = None
            rt["prefix_args"][30] = _HYB_PSPLIT
            po, pl = _prefix_call()
            rt["prefix_args"][29], rt["prefix_args"][30] = _saved
        else:
            po, pl = _prefix_call()
        hybrid_grouped_attention(
            query, key_cache, value_cache, po, pl, rt["suffix_bt"],
            plan["siblings"], int(rt["suffix_max"]), softmax_scale, out=output,
            lse_token_axis=(1 if pl.shape[-1] == query.shape[0] else 0),
            # per-row suffix lengths: the loop bound stays suffix_max, only the mask varies
            suffix_lens=(None if plan.get("suffix_uniform") else rt["suffix_kv"]))
        return output

    if _FUSED_ON and identity and plan.get("uniform") and rt.get("fused_ok"):
        from .fused_grouped_attn import fused_grouped_attention
        fused_grouped_attention(
            query, key_cache, value_cache,
            rt["prefix_bt"], rt["suffix_bt"],
            plan["num_groups"], plan["siblings"],
            int(plan["max_prefix"]), int(rt["suffix_max"]),
            softmax_scale, num_splits=_FUSED_SPLITS or rt["splits"], out=output,
            workspace=rt.get("fused_ws"))
        return output

    # OVERLAP: enqueue the prefix pass on the side stream BEFORE the suffix pass, so the two
    # independent reads run concurrently -- the deduped prefix hides inside the suffix walk,
    # recovering the wave packing the baseline's single fused kernel gets for free. The side
    # stream waited on _ev_in (Q/KV ready); the main stream waits on _ev_side before merge.
    # NOTE for gc-comp under overlap: the prefix region reads ~0 (already done when the main
    # stream gets there); suffix+other then equals true attention wall, which is the number
    # that matters.
    if _ov:
        _main = _main_stream(query.device)   # cached; the uncached call is 30+ us
        # set_stream instead of the `with torch.cuda.stream(...)` context. StreamContext.__init__
        # calls current_stream() with NO device, which re-runs the whole device-resolution chain
        # (_get_device_index -> _get_available_device_type -> torch.cuda.is_available ->
        # _nvml_based_avail -> os.getenv) at 18.9 us per layer, 36 layers deep. We already hold
        # the stream we are restoring to, so the save half of the context is redundant.
        # try/finally because leaving the side stream current would put every later kernel in
        # this step -- including other layers -- on it, which is a correctness bug, not a
        # slowdown.
        torch.cuda.set_stream(_sstream)
        try:
            prefix_output_p, prefix_lse_p = _prefix_call()
            # allocated on the side stream, consumed on the main stream: tell the caching
            # allocator, or these can be reused while the merge still reads them.
            prefix_lse_p.record_stream(_main)
            if not identity:
                prefix_output_p.record_stream(_main)
            _EV_SIDE.record(_sstream)
        finally:
            torch.cuda.set_stream(_main)

    # SP_GC_FUSED=3: my suffix walk in place of FA3's, and NOTHING else changes -- the prefix
    # still runs on the side stream and merge_attn_states still merges. Mode 2 fused the merge
    # into the walk and lost the overlap doing it; traced at n=192 that trade was 16% less
    # kernel work for 4% more wall (sum/union 1.01 vs 1.22). Here the walk needs nothing from
    # the prefix, so the halves stay concurrent and only the part that paid survives: the walk
    # is 9.5 ms/step against FA3's ~11.1, and FA3's suffix split-combine disappears with it.
    _walk = (_FUSED_MODE == 3 and identity and rt.get("fused_ok")
             and plan["num_groups"] > 0)
    # Instrumented copy of the branch below, taken only under SP_GC_CPUPROF=1 AND with the
    # side-stream machinery off: under _ov the prefix outputs live in the _ov block's locals
    # and are merged by the shared tail, so duplicating that here would merge None. The
    # profiling config is OVERLAP=0 (the shipping best) anyway.
    if _walk and _CPUPROF and not _ov:
        import time as _t
        _cp0 = _t.perf_counter()
        prefix_output_p, prefix_lse_p = _prefix_call()
        _cp1 = _t.perf_counter()
        from .fused_grouped_attn import suffix_walk_attention as _swa_p
        _xrp = plan.setdefault("_xrt", {})
        _Hp = int(query.shape[1])
        _lkp = (int(query.shape[0]), _Hp)
        if _xrp.get("walk_lse_key") != _lkp:
            _xrp["walk_lse_key"] = _lkp
            _xrp["walk_lse"] = torch.empty(_Hp, int(query.shape[0]),
                                           device=query.device, dtype=torch.float32)
        suffix_output, suffix_lse = _swa_p(
            query, key_cache, value_cache, rt["suffix_bt"],
            int(rt["suffix_max"]), softmax_scale,
            out=rt["suffix_out"], lse=_xrp["walk_lse"],
            suffix_lens=(None if plan.get("suffix_uniform") else rt["suffix_kv"]))
        _cp2 = _t.perf_counter()
        merge_attn_states(output, prefix_output_p, prefix_lse_p, suffix_output, suffix_lse)
        _cp3 = _t.perf_counter()
        _CP["pref"] += _cp1 - _cp0
        _CP["walk"] += _cp2 - _cp1
        _CP["merge"] += _cp3 - _cp2
        _CP["body"] += _cp3 - _cp0
        _CP["n"] += 1
        if _CP["n"] >= 36 * 300:
            _cp_report()
        return output
    if _walk:
        from .fused_grouped_attn import suffix_walk_attention
        global _WALK_SEEN
        if not _WALK_SEEN:
            _WALK_SEEN = True
            # SPREAD IS THE WHOLE QUESTION for this kernel. It gives every row the SAME loop
            # bound, suffix_max, and masks the tail per row, so a batch whose suffixes differ
            # by d tokens wastes d/suffix_max of every short row's work. FA3's varlen kernel
            # schedules per sequence and pays no such tax, which is a candidate explanation for
            # mode 3 measuring 72.55 s against the three-kernel path's 71.01. One sync, once,
            # on the first dispatch only -- and it is the number that decides whether forcing
            # the group into lockstep is worth a scheduler change or worth nothing.
            _sk = rt["suffix_kv"]
            _mn, _mx = int(_sk.min()), int(_sk.max())
            print(f"[grouped-cascade] WALK path live: groups={plan['num_groups']} "
                  f"siblings={plan['siblings']} suffix min={_mn} max={_mx} "
                  f"spread={_mx - _mn} waste={(_mx - _mn) / max(_mx, 1):.1%}", flush=True)
        _H = int(query.shape[1])
        # xrt is bound only on the step that REBUILDS the plan; on every cached step it is not
        # in scope at all, so referencing it here raised UnboundLocalError in all four ranks.
        # The cross-step dict lives on the plan -- take it from there rather than from a name
        # that happens to exist earlier in the same function on some paths.
        _xr = plan.setdefault("_xrt", {})
        _lk = (int(query.shape[0]), _H)
        if _xr.get("walk_lse_key") != _lk:
            _xr["walk_lse_key"] = _lk
            # [heads, tokens]: merge_attn_states consumes FA3's layout, so match it exactly
            _xr["walk_lse"] = torch.empty(_H, int(query.shape[0]),
                                          device=query.device, dtype=torch.float32)
        suffix_output, suffix_lse = suffix_walk_attention(
            query, key_cache, value_cache, rt["suffix_bt"],
            int(rt["suffix_max"]), softmax_scale,
            out=rt["suffix_out"], lse=_xr["walk_lse"],
            suffix_lens=(None if plan.get("suffix_uniform") else rt["suffix_kv"]),
            suffix_max_dev=(rt.get("suffix_max_dev") if _DYN_SUF else None))
    elif (_sa := rt.get("suffix_args")) is not None:
        _sa[_FA_Q] = query
        _sa[_FA_K] = key_cache
        _sa[_FA_V] = value_cache
        suffix_output, suffix_lse, _, _ = _FA3_FWD(*_sa)
    else:
        suffix_output, suffix_lse = flash_attn_varlen_func(
            q=query, k=key_cache, v=value_cache,
            cu_seqlens_q=query_start_loc, seqused_k=rt["suffix_kv"],
            max_seqlen_q=max_query_len, max_seqlen_k=rt["suffix_max"],
            softmax_scale=softmax_scale, causal=True, window_size=[-1, -1],
            block_table=rt["suffix_bt"], softcap=logits_soft_cap, return_softmax_lse=True,
            num_splits=rt.get("suffix_splits", 0), fa_version=_FA_VERSION,
            scheduler_metadata=rt.get("suffix_sched"),
            out=rt["suffix_out"],
        )
    if _ct:
        _e1.record()
    if plan["num_groups"] == 0:
        output.copy_(suffix_output)
        return output

    if _ov:
        _main_stream(query.device).wait_event(_EV_SIDE)
    else:
        prefix_output_p, prefix_lse_p = _prefix_call()

    if _ct:
        _e2.record()
    if identity:
        # every token is grouped and already in batch order: no scatter, no masking --
        # merge the kernel outputs directly.
        merge_attn_states(output, prefix_output_p, prefix_lse_p, suffix_output, suffix_lse)
        if _ct:
            _e3.record(); _e3.synchronize()
            _CT["suffix"].append(_e0.elapsed_time(_e1)); _CT["prefix"].append(_e1.elapsed_time(_e2))
            _CT["other"].append(_e2.elapsed_time(_e3)); _ct_print()
        return output

    # General path: scatter back to vLLM's token order via index_copy over a -inf
    # pre-fill. merge weights by exp(lse) -- but 0 x NaN = NaN, so ungrouped rows DO need
    # defined values: zeros_like, not empty_like (the selftest's mixed case went NaN on
    # uninitialized memory). Still one fill instead of the old per-layer mask machinery.
    prefix_output = torch.zeros_like(suffix_output)
    prefix_lse = torch.full_like(suffix_lse, float("-inf"))
    prefix_output.index_copy_(0, perm, prefix_output_p)
    axis = lse_token_axis(suffix_lse, int(query.shape[0]), int(query.shape[1]))
    if axis == 0:
        prefix_lse.index_copy_(0, perm, prefix_lse_p)
    else:
        prefix_lse.index_copy_(1, perm, prefix_lse_p)

    merge_attn_states(output, prefix_output, prefix_lse, suffix_output, suffix_lse)
    if _ct:
        _e3.record(); _e3.synchronize()
        _CT["suffix"].append(_e0.elapsed_time(_e1)); _CT["prefix"].append(_e1.elapsed_time(_e2))
        _CT["other"].append(_e2.elapsed_time(_e3)); _ct_print()
    return output
