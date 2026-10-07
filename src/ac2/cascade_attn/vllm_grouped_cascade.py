#!/usr/bin/env python3
"""Install grouped cascade attention into a running vLLM 0.23.0 engine.

Stock cascade reads a shared prefix once per batch, but only when EVERY resident request
shares it (v1/core/kv_cache_manager.py:485), so G groups of 16 siblings each with their own
replay prefix never trigger it. This replaces three functions so the same kernel runs one
shared-prefix pass PER GROUP. The kernel maths lives in grouped_cascade.py and is verified
standalone against plain attention; this file is only the plumbing.

  build()                 derive groups from the batch's block tables, emit per-group
                          cascade tensors instead of the single-prefix ones
  use_cascade_attention() decide on total KV traffic saved rather than one whole-batch prefix
  cascade_attention()     dispatch to the grouped implementation

Deliberately NOT patched: the scheduler and SchedulerOutput. Groups are recovered inside the
builder by comparing leading block ids, which the block tables already encode -- requests
sharing a prefix share physical blocks. That keeps the change out of the scheduler's
contract entirely.

Everything is fail-safe: if grouping finds nothing worth doing, or anything about the batch
looks unexpected, the stock code path runs unchanged. Enable with SP_GROUPED_CASCADE=1.
"""
from __future__ import annotations

import collections
import os
import torch
from time import perf_counter as _pc

_VERSION_OK = ("0.23.",)
_state = {"installed": False, "plan": None, "stamp": None, "hits": 0, "miss": 0,
          "cascade_calls": 0, "t_events": [], "fallback_calls": 0,
          "t_ours": 0.0, "t_theirs": 0.0, "t_step": 0.0, "t_last_exit": None,
          "fp_tick": 0,
}
# hoisted: os.environ.get in the per-layer hot path costs ~5-10us x 36 layers x every step
_CPUTIME_ON = os.environ.get("SP_GC_CPUTIME", "0") == "1"
_FP_EVERY = int(os.environ.get("SP_GC_FP_EVERY", "1"))
_TIME_ON = os.environ.get("SP_GC_TIME") == "1"
_VERIFY_ON = os.environ.get("SP_GC_VERIFY") == "1"
# SP_GC_TRACE_FILE: dump the exact batch shape the kernel is handed, so an offline benchmark
# can replay the real request distribution instead of a uniform synthetic one. The isolated
# benchmark builds G groups of exactly S members, one prefix length, one suffix length and a
# resident count that never changes, none of which the engine ever produces. Three integer
# vectors of length num_reqs fully determine the shape of a decode call, so the trace is tiny.
_TRACE_PATH = os.environ.get("SP_GC_TRACE_FILE", "")
# Sampled on plan REBUILDS, about once per decode step rather than once per layer. A rebuild
# has already paid one device to host copy for req_prefix_lens_cpu, so the two extra .tolist()
# calls ride along with a sync the code was going to do anyway. In the per layer path instead
# this would cost ~36 syncs per step, which the comments below record as having erased the
# bandwidth saving outright.
_TRACE_EVERY = int(os.environ.get("SP_GC_TRACE_EVERY", "50"))


def _cfg(name, default):
    return int(os.environ.get(name, default))


MIN_PREFIX_TOKENS = _cfg("SP_GC_MIN_PREFIX", 1024)   # below this the traffic saved is noise
MIN_GROUP_SIZE = _cfg("SP_GC_MIN_GROUP", 4)          # a prefix read by <4 queries saves little
MIN_GROUPS = _cfg("SP_GC_MIN_GROUPS", 1)

# Why a batch's coverage is what it is. `grouped_reqs/reqs` alone cannot distinguish "the
# siblings are not co-resident" from "their shared prefix is too short", and those two have
# completely different fixes -- the first is routing (free), the second is SP_REPLAY_CUT_GRAIN
# (a science knob). An earlier run measured coverage 0.45 in production and could not tell
# which limiter dominated, so the packing ceiling stayed a guess. This records, per batch, the
# candidate bucket sizes and what each rejection cost, and costs one dict update per bucket
# in a loop that is already Python.
_GDIAG: dict = {}


def derive_groups(block_table: torch.Tensor, num_computed_tokens: torch.Tensor,
                  block_size: int, diag: bool = False):
    """Recover prefix-sharing groups from the block tables.

    Requests that share a prefix share physical blocks, so two requests are in the same group
    iff their block tables agree on a leading run. Bucket by the first block id, then extend
    each bucket while every member still agrees. Returns (group_ids, prefix_lens) or None
    when there is nothing worth sharing.
    """
    num_reqs, width = block_table.shape
    if num_reqs < 2:
        return None
    # Only blocks the request has actually computed may be treated as shared; a block that is
    # allocated but not yet filled holds garbage. This mirrors the cap in grouped_cascade and
    # is applied again there per group.
    max_blocks = (num_computed_tokens // block_size).clamp(min=0)

    first = block_table[:, 0]
    group_ids = torch.full((num_reqs,), -1, dtype=torch.int64, device=block_table.device)
    prefix_lens = torch.zeros(num_reqs, dtype=torch.int32, device=block_table.device)
    gid = 0
    _GDIAG.clear()
    _sizes: list = []
    _rej_small = _rej_small_reqs = _rej_nolimit = _rej_shortpfx = _rej_shortpfx_reqs = 0
    _shortpfx_max = _recoverable = 0
    for key in torch.unique(first).tolist():
        members = torch.nonzero(first == key, as_tuple=True)[0]
        sz = int(members.numel())
        if diag:
            _sizes.append(sz)
        too_small = sz < MIN_GROUP_SIZE
        # Fast path skips the agreement scan for buckets the size gate already rejected. Under
        # `diag` we scan them anyway: the size gate fires FIRST, so without this a bucket that
        # is both too small AND short-prefixed is booked as "too small", and the packing lane
        # then claims requests that the prefix gate would reject regardless. That is the exact
        # confusion this diagnostic exists to remove, so it must not reproduce it.
        if too_small and not diag:
            continue
        limit = int(max_blocks[members].min())
        if limit <= 0:
            if diag:
                _rej_nolimit += 1
            continue
        head = block_table[members, :limit]
        agree = (head == head[0]).all(dim=0)                 # [limit] bool
        nz = torch.nonzero(~agree, as_tuple=True)[0]
        nblocks = int(nz[0]) if nz.numel() else limit
        short = nblocks * block_size < MIN_PREFIX_TOKENS
        if diag:
            if short:
                _rej_shortpfx += 1
                _rej_shortpfx_reqs += sz
                _shortpfx_max = max(_shortpfx_max, nblocks * block_size)
            elif too_small:
                # prefix is long enough; ONLY the size gate stands in the way, so co-residency
                # (routing) is what would recover these -- for free, no science knob
                _rej_small += 1
                _rej_small_reqs += sz
                if sz >= 2:
                    _recoverable += sz
        if too_small or short:
            continue
        group_ids[members] = gid
        prefix_lens[members] = nblocks * block_size
        gid += 1
    if diag:
        _GDIAG.update(
            buckets=len(_sizes),
            size_hist=collections.Counter(_sizes),
            rej_small=_rej_small, rej_small_reqs=_rej_small_reqs,
            rej_nolimit=_rej_nolimit,
            rej_shortpfx=_rej_shortpfx, rej_shortpfx_reqs=_rej_shortpfx_reqs,
            shortpfx_max=_shortpfx_max,
            # requests in buckets of >=2 whose shared prefix ALREADY clears MIN_PREFIX and are
            # blocked only by MIN_GROUP_SIZE: the free lane
            recoverable_by_min_group_2=_recoverable,
        )
    if gid < MIN_GROUPS:
        return None
    # Requests in no group get their own singleton id and a zero prefix: the grouped kernel
    # routes them entirely through the suffix pass, which is plain causal attention.
    solo = torch.nonzero(group_ids < 0, as_tuple=True)[0]
    if solo.numel():
        group_ids[solo] = torch.arange(gid, gid + solo.numel(), device=group_ids.device)
    return group_ids, prefix_lens



def _write_stats() -> None:
    """Publish counters where the parent process can see them.

    The engine runs out-of-process (and one worker per TP rank), so stats() in the driver is
    structurally blind to whether grouping engaged -- it reads 0 whether the plugin is working
    perfectly or not installed at all. Anything that gates on it, including tests, gets a
    false negative. One file per pid, so TP ranks never clobber each other.
    """
    path = os.environ.get("SP_GC_STATS_FILE")
    if not path or _state["hits"] % 50 not in (0, 1):
        return
    try:
        import json
        with open(f"{path}.{os.getpid()}", "w") as f:
            json.dump({"pid": os.getpid(), "batches_grouped": _state["hits"],
                       "batches_plain": _state["miss"]}, f)
    except Exception:
        pass          # observability must never break the engine


def _trace_batch(gc_plan, group_ids, ncomp, num_reqs) -> None:
    # Append one line describing the batch this decode call was handed. Fields are all of
    # length num_reqs unless noted:
    #   rebuild       plan rebuild counter. It orders the samples through a step and lets a
    #                 replay reproduce the resident count falling from full to nearly empty.
    #   num_reqs      how many requests were resident, a scalar
    #   group_ids     which group each request belongs to, negative when it belongs to none
    #   prefix_lens   the block aligned shared prefix each request gets, after capping
    #   num_computed  how many key and value tokens each request already holds, meaning its
    #                 prefix plus everything it has generated so far
    # One file per process id, matching the stats writer above: the engine runs one worker per
    # tensor parallel rank and a single shared path would interleave their lines.
    try:
        import json
        rec = {"rebuild": int(_state.get("cache_rebuilds", 0)),
               "num_reqs": int(num_reqs),
               "group_ids": group_ids[:num_reqs].tolist(),
               "prefix_lens": gc_plan["req_prefix_lens_cpu"][:num_reqs].tolist(),
               "num_computed": ncomp[:num_reqs].tolist()}
        with open(f"{_TRACE_PATH}.{os.getpid()}", "a") as fh:
            fh.write(json.dumps(rec) + chr(10))
    except Exception:
        pass          # observability must never break the engine


def install_attn_timer() -> None:
    """SP_ATTN_TIME=1: time the STOCK attention path per layer, sampled 1-in-64.

    The grouped path reports its own per-layer cost (gc-time), but nothing measures the
    baseline's on the same workload -- without it, 'attention share of the step' for the
    control arm is a model, not a measurement. Same event technique, same print cadence,
    active regardless of SP_GROUPED_CASCADE so a plain control arm can be instrumented.
    """
    if os.environ.get("SP_ATTN_TIME") != "1" or _state.get("attn_timer"):
        return
    from vllm.v1.attention.backends import flash_attn as fa
    impl = fa.FlashAttentionImpl
    stock_fwd = impl.forward
    st = {"n": 0, "t": []}
    _state["attn_timer"] = True

    def timed_forward(self, *a, **kw):
        st["n"] += 1
        if st["n"] % 64:
            return stock_fwd(self, *a, **kw)
        s_ev = torch.cuda.Event(enable_timing=True); e_ev = torch.cuda.Event(enable_timing=True)
        s_ev.record()
        r = stock_fwd(self, *a, **kw)
        e_ev.record(); e_ev.synchronize()
        st["t"].append(s_ev.elapsed_time(e_ev))
        if len(st["t"]) % 200 == 0:
            ts = st["t"][-200:]
            print(f"[attn-time] calls={st['n']} stock_ms_avg={sum(ts)/len(ts):.3f} "
                  f"pid={os.getpid()}", flush=True)
        return r

    impl.forward = timed_forward
    print("[attn-time] stock attention timer installed", flush=True)


def install() -> bool:
    """Patch the FlashAttention backend in place. Idempotent; returns False if unavailable."""
    if _state["installed"]:
        return True
    import vllm
    if not any(vllm.__version__.startswith(v) for v in _VERSION_OK):
        raise RuntimeError(
            f"grouped cascade is pinned to vLLM {_VERSION_OK}; found {vllm.__version__}. "
            f"The patch reaches into build()/cascade_attention() internals -- re-verify "
            f"against the new source before widening this check.")
    from vllm.v1.attention.backends import flash_attn as fa
    from .grouped_cascade import grouped_cascade_attention, plan_groups, probe_lse_layout
    # NOTE: probe_lse_layout() is NOT called here. install() runs at plugin load, which in
    # vLLM sleep-mode processes (the colocated judge) happens BEFORE the CuMemAllocator
    # pool exists -- a CUDA allocation here initializes the default allocator pool first,
    # and ~10GB of judge weights then land outside the sleep-managed pool, permanently
    # resident through level-1 sleep (the 10.22GB/GPU residual that OOM'd a20/a21's val
    # gen and a18's wave at redline). The probe runs lazily at the first scoped build()
    # instead: policy engines only, long after cumem is established.

    stock_build = fa.FlashAttentionMetadataBuilder.build
    stock_cascade = fa.cascade_attention
    stock_use_cascade = fa.FlashAttentionMetadataBuilder.use_cascade_attention

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        # SCOPE GATE. SP_GROUPED_CASCADE=1 travels in the shared ray runtime_env, so this
        # plugin loads inside EVERY vLLM engine in the allocation -- including the colocated
        # judge, whose memory budget (0.80 of each GPU) has no slack for a patched
        # attention path. Only the policy engine may run grouped cascade; everything else
        # gets the stock builder untouched.
        scope = _state.get("scope")
        if scope is None:
            model = str(getattr(getattr(self, "model_config", None), "model", "") or "")
            want = os.environ.get("SP_GC_MODEL_FILTER", "qwen").lower()
            scope = want in model.lower()
            _state["scope"] = scope
            print(f"[grouped-cascade] scope: model={model!r} filter={want!r} -> "
                  f"{'ACTIVE' if scope else 'DISABLED (not the policy engine)'} "
                  f"pid={os.getpid()}", flush=True)
            if scope:
                # First scoped batch: safe point for the one-time CUDA probe (idempotent;
                # cumem long established by now). See the install() note for why this
                # must never run at plugin-load time.
                probe_lse_layout()
        if not scope:
            _state["plan"] = None
            return stock_build(self, common_prefix_len, common_attn_metadata, fast_build)
        # DECODE-ONLY GATE. Cascade's win is decode bandwidth (the rollout benchmark is
        # decode-bound); on prefill/chunked batches the plugin only adds cost -- and its
        # per-layer gather/out workspaces on a 103k-token prefill spike ~1-2GB in-process,
        # which is exactly the inductor-OOM signature seen during rollout and validation
        # generation at high memory utilization. Prefill batches take the stock path untouched.
        if int(common_attn_metadata.max_query_len) > int(os.environ.get("SP_GC_MAX_QUERY_LEN", "8")):
            _state["plan"] = None
            _state["miss"] = _state.get("miss", 0) + 1
            return stock_build(self, common_prefix_len, common_attn_metadata, fast_build)
        # NOT-GROUPABLE FAST PATH. Measured on the prefix=0 grid control: cascade 219.95 s
        # against stock 158.53 s at rollout 10k -- 0.72x -- on a batch where grouping never
        # engaged at all (no [grouped-cascade] batch# line in the whole log). Nothing was
        # deduped and it still cost 39%, because both of the following were paid every step
        # for a batch that could not be grouped:
        #   1. stock_build was called with common_prefix_len=0, which is right when WE own the
        #      cascade but throws away vLLM's own common-prefix handling when we do not.
        #   2. the failure was not cached -- the old code set step_cache = None on the miss
        #      path -- so derive_groups re-ran every step, python loop and GPU->CPU syncs and
        #      all, to reach the same "no" 10000 times.
        # Caching the negative under the same fingerprint fixes both: an ungroupable batch
        # takes the untouched stock path from the second step onward.
        cam = common_attn_metadata
        _state["plan"] = None
        # The negative EXPIRES every 64 steps. derive_groups caps each group's prefix by the
        # members' num_computed_tokens, which grows, so a batch that is ungroupable now can
        # become groupable later -- during prefill most obviously. A permanent negative keyed
        # on the resident set alone would freeze that out for the rest of the run. Retrying
        # 1 step in 64 bounds the wasted derive at ~1.6% while still noticing the transition.
        _ng = _state.get("no_group_cache")
        if _ng is not None and _ng["n"] == int(cam.num_reqs) and _ng["age"] < 64:
            _fp_now = cam.block_table_tensor[: cam.num_reqs, 0]
            if _ng["fp"].shape == _fp_now.shape and not bool((_ng["fp"] != _fp_now).any()):
                _ng["age"] += 1
                _state["miss"] = _state.get("miss", 0) + 1
                return stock_build(self, common_prefix_len, cam, fast_build)
        md = stock_build(self, 0, common_attn_metadata, fast_build)
        try:
            bt = cam.block_table_tensor
            ncomp = getattr(cam, "num_computed_tokens_cpu", None)
            if ncomp is None:
                ncomp = cam._num_computed_tokens_cpu
            ncomp = torch.as_tensor(ncomp, device=bt.device)[: cam.num_reqs]
            # Cross-STEP grouping cache. During steady decode the resident set -- and
            # therefore the grouping -- does not change; only seq_lens grows. derive_groups
            # + plan_groups cost a python loop over group keys with several GPU->CPU syncs,
            # per step. Fingerprint = each request's first block id (grouping is a function
            # of shared leading blocks) + the request count; one small equality sync per
            # step buys back the whole planning cost. Any change rebuilds from scratch.
            fp = bt[: cam.num_reqs, 0]
            cached = _state.get("step_cache")
            qsl_now = cam.query_start_loc[: cam.num_reqs + 1]
            # q_lens must match too: same residents with different query splits (a chunked
            # prefill re-entry) would reuse cu_prefix_query_lens built for one-token decode.
            #
            # BOTH checks used to be torch.equal on DEVICE tensors, i.e. two GPU->CPU syncs
            # every step. A sync mid-step is not just its own latency: it drains the queue,
            # so the ~5ms of python per step stops overlapping the ~9ms of kernels and the
            # two add instead of hiding. Measured gap at n=32: step 14.32ms against 9.22ms
            # of instrumented kernel time.
            #   - query_start_loc has a CPU mirror in the metadata; compare that, free.
            #   - the block-id fingerprint has no CPU mirror, so it stays on device but is
            #     reduced to ONE bool: (a != b).any() instead of torch.equal.
            qsl_cpu = getattr(cam, "query_start_loc_cpu", None)
            if qsl_cpu is not None:
                qsl_cpu = qsl_cpu[: cam.num_reqs + 1]
            ok = (cached is not None and cached["n"] == int(cam.num_reqs)
                  and cached["fp"].shape == fp.shape)
            if ok and qsl_cpu is not None and cached.get("qsl_cpu") is not None:
                ok = (cached["qsl_cpu"].shape == qsl_cpu.shape
                      and torch.equal(cached["qsl_cpu"], qsl_cpu))     # CPU: no sync
                # THE LAST PER-STEP SYNC, now on a dial. The CPU-time buckets sum to 9.3
                # ms/step while the gen-5000 wall is 13.95: 4.6 ms is the CPU WAITING, and the
                # only forced wait left is this fingerprint sync -- it drains the queue every
                # step, so the ~9.3 ms of python and the ~9.35 ms of kernels add instead of
                # hiding. The CPU-side guards above (request count + query_start_loc mirror)
                # already catch every path that changes the batch: preemption and swap remove
                # the request from the running set first, and prefix-cache eviction cannot
                # touch blocks referenced by running requests. The device fingerprint is a
                # belt-and-braces check on top, so it can run every N steps instead of every
                # step. SP_GC_FP_EVERY: 1 = current behaviour, N = check every N steps,
                # 0 = trust the CPU guards entirely.
                if ok and _FP_EVERY != 0:
                    _state["fp_tick"] += 1
                    if _FP_EVERY == 1 or _state["fp_tick"] % _FP_EVERY == 0:
                        ok = not bool((cached["fp"] != fp).any())       # the sync
            elif ok:                                    # no CPU mirror: original two-sync path
                ok = (torch.equal(cached["fp"], fp)
                      and torch.equal(cached["qsl"], qsl_now))
            if ok:
                g = cached["g"]
                gc_plan_cached = cached["plan"]
                _state["cache_hits"] = _state.get("cache_hits", 0) + 1
            else:
                # Only ask for the diagnostic on the batches that will actually print it
                # (hits is incremented below, so predict the post-increment value). The extra
                # agreement scan over size-1..3 buckets is then paid on 1 batch in 500.
                _h = _state["hits"] + 1
                g = derive_groups(bt[: cam.num_reqs], ncomp, self.kv_cache_spec.block_size,
                                  diag=(_h == 1 or _h % 500 == 0))
                gc_plan_cached = None
                _state["cache_rebuilds"] = _state.get("cache_rebuilds", 0) + 1
            if g is not None:
                group_ids, prefix_lens = g
                qsl = qsl_now
                # ONE plan per step, shared by every layer. Computed inside the attention
                # call this was ~36 GPU->CPU syncs per decode step (torch.unique + .tolist()
                # + per-request arange, once per layer) and erased the bandwidth saving.
                if gc_plan_cached is not None:
                    gc_plan = dict(gc_plan_cached)   # fresh dict -> fresh per-step _rt
                    gc_plan.pop("_rt", None)
                    if _SP_STATIC:
                        # The captured graph reads the capture plan's storages; the per-step
                        # rebuild must therefore refresh THOSE, not a parallel set. One shared
                        # cross-step dict makes capture and runtime write the same memory.
                        gc_plan["_xrt"] = _state.setdefault("cap_xrt", {})
                        cpp = _state.get("cap_plan")
                        if cpp is not None:
                            cpp["plan"]["prefix_kv_lens"].copy_(
                                gc_plan["prefix_kv_lens"], non_blocking=True)
                            cpp["plan"]["rep"].copy_(gc_plan["rep"], non_blocking=True)
                else:
                    gc_plan = plan_groups(group_ids, qsl, prefix_lens, ncomp,
                                          self.kv_cache_spec.block_size)
                    # CPU mirror of the CAPPED prefix lens, taken on the one step that
                    # already paid to build the plan. It makes suffix_max computable on the
                    # host below, which is the last per-step sync in the hot path.
                    gc_plan["req_prefix_lens_cpu"] = gc_plan["req_prefix_lens"].cpu()
                    if _TRACE_PATH and _state["cache_rebuilds"] % _TRACE_EVERY == 0:
                        _trace_batch(gc_plan, group_ids, ncomp, int(cam.num_reqs))
                    _state["step_cache"] = {
                        "n": int(cam.num_reqs), "fp": fp.clone(),
                        "qsl": qsl_now.clone(),
                        "qsl_cpu": qsl_cpu.clone() if qsl_cpu is not None else None,
                        "g": g, "plan": gc_plan}
                # suffix_max = max(seq_len - prefix_len). seq_lens has a CPU mirror and the
                # prefix lens are constant while the grouping holds, so the kernel never has
                # to ask the device how long its longest suffix is.
                sl_cpu = getattr(cam, "seq_lens_cpu", None)
                pl_cpu = gc_plan.get("req_prefix_lens_cpu")
                if sl_cpu is not None and pl_cpu is not None:
                    _suf = sl_cpu[: cam.num_reqs] - pl_cpu
                    gc_plan["suffix_max_cpu"] = int(_suf.max())
                    # EQUAL suffix length is the fused kernel's last precondition, and the only
                    # per-step one: seq_lens only grow together while every sibling emits the
                    # same token count. Decided here because this is where the CPU mirrors
                    # already are -- asking the device would cost the sync this path removed.
                    gc_plan["suffix_uniform"] = bool((_suf == _suf[0]).all())
                _state["plan"] = {
                    "gc_plan": gc_plan,
                    "group_ids": group_ids, "prefix_lens": prefix_lens,
                    "num_computed": ncomp,
                    "query_start_loc": qsl,
                    "seq_lens": cam.seq_lens[: cam.num_reqs],
                    "block_table": bt[: cam.num_reqs],
                    "max_query_len": cam.max_query_len,
                }
                _state["stamp"] = (int(cam.num_actual_tokens), int(cam.num_reqs))
                _state["no_group_cache"] = None   # this set groups; drop any stale negative
                md.use_cascade = True
                _state["hits"] += 1
                _write_stats()
                if _state["hits"] == 1 or _state["hits"] % 500 == 0:
                    # The worker is a separate process, so parent-side stats() cannot see it.
                    # This line in the engine log is how you confirm grouping is live in a
                    # real run, and how you tell a no-op install from a working one.
                    ng = int((prefix_lens > 0).sum())
                    print(f"[grouped-cascade] batch#{_state['hits']} pid={os.getpid()} "
                          f"reqs={cam.num_reqs} grouped_reqs={ng} "
                          f"groups={len(torch.unique(group_ids[prefix_lens > 0]))} "
                          f"prefix_max={int(prefix_lens.max())} "
                          # the two rejection lanes, so a low coverage is self-diagnosing:
                          # rej_small_reqs is what better PACKING would recover (free),
                          # rej_shortpfx_reqs is what only a longer prefix would (science).
                          f"| buckets={_GDIAG.get('buckets')} "
                          f"sizes={dict(sorted(_GDIAG.get('size_hist', {}).items()))} "
                          f"rej_small={_GDIAG.get('rej_small')}({_GDIAG.get('rej_small_reqs')}reqs) "
                          f"gain_if_min_group_2={_GDIAG.get('recoverable_by_min_group_2')}reqs "
                          f"rej_shortpfx={_GDIAG.get('rej_shortpfx')}"
                          f"({_GDIAG.get('rej_shortpfx_reqs')}reqs,max={_GDIAG.get('shortpfx_max')})",
                          flush=True)
            else:
                _state["miss"] += 1
                _state["step_cache"] = None
                # Remember that THIS resident set cannot be grouped, so the next step takes
                # the early-out above instead of re-deriving the same "no". Keyed on the same
                # first-block fingerprint the positive cache uses, so any change to the
                # resident set re-derives exactly once.
                _state["no_group_cache"] = {"n": int(cam.num_reqs),
                                            "fp": bt[: cam.num_reqs, 0].clone(), "age": 0}
                # vLLM's own common-prefix path was disabled by the build above; give it back.
                if common_prefix_len:
                    md = stock_build(self, common_prefix_len, cam, fast_build)
        except Exception as e:                      # never take the engine down for this
            if os.environ.get("SP_GC_STRICT") == "1":
                raise
            print(f"[grouped-cascade] disabled for this batch: {type(e).__name__}: {e}",
                  flush=True)
            _state["plan"] = None
            md.use_cascade = False
        return md

    def cascade(output, query, key_cache, value_cache, cu_query_lens, max_query_len,
                cu_prefix_query_lens, prefix_kv_lens, suffix_kv_lens, max_kv_len,
                softmax_scale, alibi_slopes, sliding_window, logits_soft_cap, block_table,
                common_prefix_len, max_num_splits, fa_version, **kw):
        p = _state["plan"]
        _state["cascade_calls"] += 1
        # SP_GC_CPUTIME=1: pure perf_counter accumulation, no GPU syncs, ~100ns per call.
        # The question it answers: of the ~14 ms/step of CPU that walls the step (GPU busy is
        # 9.35 ms and BOTH suffix implementations measure ~14 ms of CPU), how much is OURS?
        # The torch profiler cannot answer it -- its per-frame instrumentation inflated
        # Event.record() to 35 us and sent a whole evening chasing torch.cuda plumbing that a
        # cached-stream fix then measurably did not fix (69.76 -> 69.76). Wall-clock deltas
        # around our own boundary are distortion-free: everything inside is ours, everything
        # between calls is vLLM + torch.compile machinery.
        if _CPUTIME_ON:
            _t_in = _pc()
        # SP_GC_TIME=1: cuda-event timing of the grouped call, and the build-vs-cascade call
        # ratio. With 36 layers, cascade_calls should be ~36x hits; a much smaller ratio
        # means CUDA-graph replay is bypassing this python wrapper entirely -- which is not
        # just slow-path news, it means replays attend with STALE captured plan tensors.
        if _TIME_ON and _state["cascade_calls"] % 1800 == 0:
            r = _state["cascade_calls"] / max(_state["hits"], 1)
            ts = _state["t_events"]
            avg = (sum(ts) / len(ts)) if ts else float("nan")
            print(f"[gc-time] cascade_calls={_state['cascade_calls']} hits={_state['hits']} "
                  f"calls/hit={r:.1f} (36=eager every layer) grouped_ms_avg={avg:.3f} "
                  f"fallbacks={_state['fallback_calls']} "
                  f"plan_cache={_state.get('cache_hits', 0)}h/"
                  f"{_state.get('cache_rebuilds', 0)}r", flush=True)
            _state["t_events"] = ts[-200:]
        # The stamp guards against a stale plan: if this call's shapes do not match what
        # build() last saw, we are not in the batch the plan describes.
        if _SP_STATIC and torch.cuda.is_current_stream_capturing():
            # CAPTURE: force the grouped path so OUR kernels are what the graph records.
            # DYN is a hard requirement here: without the device-scalar bound the walk kernel
            # bakes the dummy host suffix length (1) into the graph and every replay silently
            # attends over zero suffix tokens -- wrong outputs, no error.
            assert _DYN_SUF_V, "SP_GC_STATIC=1 requires SP_GC_DYN=1"
            nr = int(suffix_kv_lens.shape[0]) if suffix_kv_lens is not None else int(query.shape[0])
            cp = _capture_plan(query, block_table, nr)
            return grouped_cascade_attention(
                output, query, key_cache, value_cache,
                cu_query_lens, suffix_kv_lens if suffix_kv_lens is not None else prefix_kv_lens,
                cp["perm"], cp["req_prefix_lens"], block_table, softmax_scale,
                int(max_query_len), num_computed_tokens=None,
                logits_soft_cap=logits_soft_cap, plan=cp)
        if p is None or _state["stamp"] != (int(query.shape[0]), int(p["seq_lens"].numel())):
            _state["fallback_calls"] += 1
            if _CPUTIME_ON:
                _state["t_ours"] += _pc() - _t_in
            return stock_cascade(
                output, query, key_cache, value_cache, cu_query_lens, max_query_len,
                cu_prefix_query_lens, prefix_kv_lens, suffix_kv_lens, max_kv_len,
                softmax_scale, alibi_slopes, sliding_window, logits_soft_cap, block_table,
                common_prefix_len, max_num_splits, fa_version, **kw)
        assert alibi_slopes is None and tuple(sliding_window) == (-1, -1), \
            "grouped cascade does not support alibi or sliding window"
        if _VERIFY_ON:
            # Localise: run plain causal attention over each request's WHOLE context on these
            # exact tensors -- what the non-cascade path computes -- and compare. A large
            # delta here means the kernel composition is wrong under real conditions; a small
            # one means the attention is fine and the end-to-end difference is elsewhere.
            from .grouped_cascade import flash_attn_varlen_func as _fa
            ref, _ = _fa(q=query, k=key_cache, v=value_cache,
                         cu_seqlens_q=p["query_start_loc"], seqused_k=p["seq_lens"],
                         max_seqlen_q=p["max_query_len"], max_seqlen_k=int(p["seq_lens"].max()),
                         softmax_scale=softmax_scale, causal=True, window_size=[-1, -1],
                         block_table=p["block_table"], softcap=logits_soft_cap,
                         return_softmax_lse=True)
            got = torch.empty_like(ref)
            grouped_cascade_attention(
                got, query, key_cache, value_cache, p["query_start_loc"], p["seq_lens"],
                p["group_ids"], p["prefix_lens"], p["block_table"], softmax_scale,
                p["max_query_len"], num_computed_tokens=p["num_computed"],
                logits_soft_cap=logits_soft_cap, plan=p["gc_plan"])
            d = (got.float() - ref.float()).abs()
            ng = int((p["prefix_lens"] > 0).sum())
            # Express the worst element in bf16 ULPs at its own magnitude. A structural bug
            # scales with the value; rounding is pinned at 1-2 ULP no matter the batch.
            mag = ref.float().abs().max().clamp(min=1e-6)
            ulp = mag * 2 ** -8
            frac = (d > 0).float().mean()
            print(f"[gc-verify] tokens={query.shape[0]} heads={query.shape[1]} reqs="
                  f"{p['seq_lens'].numel()} grouped_reqs={ng} max|d|={d.max():.6f} "
                  f"mean|d|={d.mean():.6f} max|ref|={mag:.3f} ulps={d.max()/ulp:.2f} "
                  f"elems_differing={100*frac:.3f}%", flush=True)
            output.copy_(got)
            return output
        # sample 1-in-64: a per-call synchronize would serialize the whole pipeline and
        # measure its own stall instead of the kernel.
        if _TIME_ON and _state["cascade_calls"] % 64 == 0:
            s_ev = torch.cuda.Event(enable_timing=True); e_ev = torch.cuda.Event(enable_timing=True)
            s_ev.record()
            r = grouped_cascade_attention(
                output, query, key_cache, value_cache,
                p["query_start_loc"], p["seq_lens"], p["group_ids"], p["prefix_lens"],
                p["block_table"], softmax_scale, p["max_query_len"],
                num_computed_tokens=p["num_computed"], logits_soft_cap=logits_soft_cap,
                plan=p["gc_plan"])
            e_ev.record(); e_ev.synchronize()
            _state["t_events"].append(s_ev.elapsed_time(e_ev))
            return r
        if not _CPUTIME_ON:
            return grouped_cascade_attention(
                output, query, key_cache, value_cache,
                p["query_start_loc"], p["seq_lens"], p["group_ids"], p["prefix_lens"],
                p["block_table"], softmax_scale, p["max_query_len"],
                num_computed_tokens=p["num_computed"], logits_soft_cap=logits_soft_cap,
                plan=p["gc_plan"])
        r = grouped_cascade_attention(
            output, query, key_cache, value_cache,
            p["query_start_loc"], p["seq_lens"], p["group_ids"], p["prefix_lens"],
            p["block_table"], softmax_scale, p["max_query_len"],
            num_computed_tokens=p["num_computed"], logits_soft_cap=logits_soft_cap,
            plan=p["gc_plan"])
        _now = _pc()
        _state["t_ours"] += _now - _t_in
        # time BETWEEN our calls = vLLM + torch.compile machinery. First call of a step
        # follows a >1 ms gap (sampler etc.); folding those in would double-count the
        # step-level work, so gaps over 1 ms go to a separate bucket.
        _prev = _state["t_last_exit"]
        if _prev is not None:
            _d = _t_in - _prev
            if _d < 0.001:
                _state["t_theirs"] += _d
            else:
                _state["t_step"] += _d
        _state["t_last_exit"] = _now
        if _state["cascade_calls"] % 7200 == 0:      # every ~200 steps at 36 layers
            n = _state["cascade_calls"]
            print(f"[gc-cpu] calls={n} ours={1e3*_state['t_ours']/(n/36):.3f} "
                  f"theirs(inter-layer)={1e3*_state['t_theirs']/(n/36):.3f} "
                  f"step-level={1e3*_state['t_step']/(n/36):.3f} ms/step", flush=True)
        return r

    def use_cascade_attention(self, *a, **kw):
        # Stock gates on ONE whole-batch prefix, which is always ~0 for us. The grouped
        # decision was already made in build(); honour it. Scoped-out engines (the judge)
        # keep the stock decision function byte-for-byte.
        if not _state.get("scope", True):
            return stock_use_cascade(self, *a, **kw)
        return _state["plan"] is not None

    fa.FlashAttentionMetadataBuilder.build = build
    fa.FlashAttentionMetadataBuilder.use_cascade_attention = use_cascade_attention
    fa.cascade_attention = cascade

    # WAKE-PATH RELIEF. The run's sleep/wake cycle OOM'd at cumem re-map
    # (cumem_allocator.cpp:139, attempts 33-36): while the engine sleeps, the torch caching
    # allocator still holds freed-but-cached blocks (compile artifacts, our workspaces), so
    # physical memory is short exactly when cumem tries to re-map its reservation. Release
    # the cache before every wake. Runs in the worker process (where this plugin lives);
    # a server-side empty_cache would free nothing here.
    if os.environ.get("SP_GC_WAKE_EMPTY_CACHE", "0") == "1":
        try:
            from vllm.v1.worker.gpu_worker import Worker as _W
            if not getattr(_W, "_sp_wake_wrapped", False):
                _stock_wake = _W.wake_up

                def _wake(self, tags=None):
                    # scope is resolved by the first build(); None (no batch yet) or True
                    # -> policy engine may benefit; False (judge) stays stock.
                    if _state.get("scope") is not False:
                        torch.cuda.empty_cache()
                    return _stock_wake(self, tags)

                _W.wake_up = _wake
                _W._sp_wake_wrapped = True
                print("[grouped-cascade] wake-path empty_cache installed", flush=True)
        except Exception as e:
            print(f"[grouped-cascade] wake wrap unavailable: {e}", flush=True)
    _state["installed"] = True
    print(f"[grouped-cascade] installed (min_prefix={MIN_PREFIX_TOKENS}, "
          f"min_group={MIN_GROUP_SIZE})", flush=True)
    return True


def stats():
    return {"batches_grouped": _state["hits"], "batches_plain": _state["miss"]}


def maybe_install() -> bool:
    return install() if os.environ.get("SP_GROUPED_CASCADE") == "1" else False


_SP_STEPPROF = os.environ.get("SP_GC_STEPPROF", "0") == "1"
_SP_STATIC = os.environ.get("SP_GC_STATIC", "0") == "1"
_DYN_SUF_V = os.environ.get("SP_GC_DYN", "0") == "1"


def _capture_plan(query, block_table, num_reqs):
    """Synthetic G=1 plan for cudagraph CAPTURE, tensors pinned in module state.

    WHY FULL capture recorded stock: capture runs the forward on a dummy batch, the dummy
    carries no grouping, our gate falls through, and the STOCK kernels are what the graph
    records -- replay then runs stock forever (measured: 48.43 s at gen-800, i.e. stock).
    The graph holds pointers, so the fix is a plan whose every tensor is allocated ONCE here
    and refreshed in place by the per-step builder afterwards. Contents at capture are
    dummy-safe, not real: prefix length one block, page ids as the dummy table provides --
    kernels read valid memory and produce garbage nobody consumes; the capture's outputs are
    thrown away. What the graph keeps is addresses, shapes and the baked host scalars, and
    those are chosen to match the production decode shape exactly.
    """
    st = _state.get("cap_plan")
    if st is not None and st["nreqs"] == num_reqs:
        return st["plan"]
    dev = query.device
    bs = 16
    cu = torch.tensor([0, num_reqs], dtype=torch.int32, device=dev)
    pkv = torch.full((1,), bs, dtype=torch.int32, device=dev)      # one block: dummy-safe
    rep = torch.zeros(1, dtype=torch.int64, device=dev)
    prefix_lens = torch.full((num_reqs,), bs, dtype=torch.int32, device=dev)
    plan = {
        "perm": torch.arange(num_reqs, dtype=torch.int64, device=dev),
        "cu_prefix_query_lens": cu, "prefix_kv_lens": pkv, "rep": rep,
        "num_groups": 1, "identity": True, "uniform": True, "suffix_uniform": False,
        "siblings": num_reqs, "max_group_tokens": num_reqs,
        # BAKED HOST SCALAR: FA3's max_seqlen_k for the prefix pass. seqused_k (pkv, device)
        # carries the true per-step length, so this only sizes the schedule; it is pinned to
        # the production prefix cap so the baked schedule fits the real workload.
        "max_prefix": int(os.environ.get("SP_GC_STATIC_PREFIX", "50176")),
        "req_prefix_lens": prefix_lens,
        "suffix_max_cpu": 1,   # host int is INERT under DYN; None would int(max()) = a sync, illegal in capture
        "_xrt": _state.setdefault("cap_xrt", {}),
    }
    _state["cap_plan"] = {"nreqs": num_reqs, "plan": plan}
    return plan



def _install_stepprof():
    """perf_counter brackets on the worker's step boundary, printed every 200 steps.

    The 5.33 ms/step of serial step-level CPU decomposes into two very different problems:
    time INSIDE execute_model (sampler, output pythonization, prepare_inputs -- worker-side
    work) and time BETWEEN execute_model calls (ZMQ round trip to EngineCore, scheduling,
    input deserialization -- pipeline latency). async_scheduling targets only the second, and
    it measured ~0 end to end, which is either "the second bucket is small" or "the flag never
    engaged" -- logs cannot tell because INFO is suppressed. This measures the split directly,
    in the worker, with no profiler distortion.
    """
    if not _SP_STEPPROF:
        return
    try:
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    except ImportError:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner  # older layout
    st = {"n": 0, "inside": 0.0, "between": 0.0, "prep": 0.0, "last_exit": None}
    _em = GPUModelRunner.execute_model
    _pi = getattr(GPUModelRunner, "prepare_inputs", None)

    def em(self, *a, **kw):
        t0 = _pc()
        if st["last_exit"] is not None:
            st["between"] += t0 - st["last_exit"]
        r = _em(self, *a, **kw)
        t1 = _pc()
        st["inside"] += t1 - t0
        st["last_exit"] = t1
        st["n"] += 1
        if st["n"] % 200 == 0:
            n = st["n"]
            print(f"[gc-step] n={n} inside_execute={1e3*st['inside']/n:.3f} "
                  f"between_execute={1e3*st['between']/n:.3f} "
                  f"prepare={1e3*st['prep']/n:.3f} ms/step", flush=True)
        return r

    GPUModelRunner.execute_model = em
    if _pi is not None:
        def pi(self, *a, **kw):
            t0 = _pc(); r = _pi(self, *a, **kw); st["prep"] += _pc() - t0
            return r
        GPUModelRunner.prepare_inputs = pi
    print("[gc-step] worker step profiler installed", flush=True)


def register() -> None:
    """vLLM general-plugin entry point.

    vLLM calls load_general_plugins() inside EngineCore and inside every worker
    (v1/engine/core.py, v1/worker/worker_base.py), which is the only place a patch can reach
    the process that actually runs attention -- V1 runs the engine in a separate process, so
    a monkeypatch applied by the training driver does not exist there. vLLM's own docstring
    warns plugins may load several times per process tree; install() is idempotent.

    Off unless SP_GROUPED_CASCADE=1, so merely installing the package changes nothing.
    """
    install_attn_timer()
    if os.environ.get("SP_GROUPED_CASCADE") != "1":
        return
    try:
        install()
    except Exception as e:
        # A plugin that raises takes the worker down with it. Grouped cascade is an
        # optimisation: log and leave the engine on the stock path.
        print(f"[grouped-cascade] plugin registration failed, using stock attention: "
              f"{type(e).__name__}: {e}", flush=True)


if _SP_STEPPROF:
    try:
        _install_stepprof()
    except Exception as _e:  # never let profiling kill the engine
        print(f'[gc-step] install failed: {type(_e).__name__}: {_e}', flush=True)
