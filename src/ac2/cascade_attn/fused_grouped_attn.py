#!/usr/bin/env python3
"""Grouped-prefix + per-sequence-suffix attention: prefix split-walk, then a fused
suffix+combine+merge epilogue. Two kernels instead of four.

WHAT IT REPLACES, per layer:

    FA3 prefix pass  ->  fp32 partials in HBM  ->  FlashAttnFwdCombine   1.25 ms/step
    FA3 suffix pass  ->  suffix out + lse
    merge_attn_states                                                    0.09 ms/step

Measured at n=32 / prefix 50k / rollout 5k / TP=4 the combine alone is 1.25 ms/step against
stock's 0.18. That is structural, not tuning: reading each 50k prefix once PER GROUP is the
whole win, but it collapses 192 independent sequences into 6, so at 8 query heads/rank the
prefix pass offers 48 CTAs -- a third of one wave on 132 SMs -- and FA3 must split the KV walk
32 ways purely to fill the machine, with combine cost linear in the split count. A sweep
confirms it cannot be tuned away: prefix splits 32/16/8 land within 0.5% (81.63/81.18/81.61 s)
because a cheaper combine costs exactly as much occupancy; 4 splits collapses it (89.97 s).

TWO PROPERTIES THIS WORKLOAD GUARANTEES and the general cascade kernel cannot assume:
  1. every sibling in a group emits the SAME token count -- all shapes static and identical
     across groups, so no cu_seqlens, no ragged tails, and no per-step scheduler metadata
     (today rebuilt every step for a shape that never changes);
  2. the prefix is shared by exactly S siblings; each suffix is private.

THE SPLIT, and why each phase gets its own grid:

  KERNEL A  grid (H, G*NSPLIT).  CTA (h, g, s) walks slice s of group g's prefix with all S
            sibling queries resident, so the prefix bytes are read once per group per split --
            never once per sibling -- and writes (acc, m, l) partials. This phase is
            parallelism-starved by construction (6 groups), which is exactly what splitting is
            for.
  KERNEL B  grid (H, T), T = G*S = 192.  CTA (h, t) reduces the NSPLIT prefix partials for its
            own token, then walks that token's OWN suffix, merges both states online, and
            writes the final output. This phase has 192 sequences -- it never needed splitting,
            and giving it a token-shaped grid saturates the machine.

So the separate combine AND merge_attn_states both vanish into kernel B's epilogue, and the
suffix output tensor is never materialised. A kernel boundary is a true global barrier, so
there is no cross-CTA visibility hazard -- the single-launch version of this needed a
release/acquire pair that tl.debug_barrier() does NOT provide (it orders threads within a
block), and it failed its equivalence test at rel=0.61 for exactly that reason.

L2 IS A SCHEDULING DECISION, not a hope. blockIdx.x is the fastest-varying dimension, so HEADS
go there: the query heads sharing a KV head are co-scheduled and read one K/V tile once from
HBM, hitting cache for the rest (8 query heads over 2 KV heads at TP=4). The opposite order
puts a different group's 51.2 MB prefix in each adjacent block and evicts everything -- one
group's prefix already exceeds the 50 MB L2, so the cache only pays when the reuse is
CONCURRENT rather than sequential. Stock gets ~2x of this accidentally, from siblings being
adjacent inside one kernel; here it is arranged deliberately.

A wrong attention corrupts training silently rather than crashing, so selftest() gates
everything: it compares against a dense per-request reference, and also runs num_splits=1 to
separate "attention math wrong" from "reduction wrong".
"""
from __future__ import annotations

import torch
from os import environ as _os_env

# LAUNCH COST IS THE BOTTLENECK, not the kernel. Traced at n=192: the walk path does 16% LESS
# GPU work than the three-kernel path (attention sum 13.13 vs 15.61 ms/step) and is still
# slower, because GPU idle goes 10.8% (three-kernel, one Triton launch per layer) -> 12.2%
# (mode 2, one) -> 27.3% (mode 3, two). Roughly 67 us of wall per extra Triton launch, 36
# layers deep, 5000 steps.
# FA3 avoids this with a prebuilt 37-slot positional list called through a fastcall; every
# Triton launch here was instead re-reading three environment variables and calling int() on
# them, per launch. Hoisted to import time -- the knobs are swept between processes, never
# within one.
_ENV_BN = int(_os_env.get("SP_FUSED_BN", "64"))
_ENV_STAGES = int(_os_env.get("SP_FUSED_STAGES", "4"))
_ENV_WARPS_A = int(_os_env.get("SP_FUSED_WARPS_A", "8"))
_ENV_WARPS_B = int(_os_env.get("SP_FUSED_WARPS_B", "4"))
_ENV_SIBT = int(_os_env.get("SP_FUSED_SIBT", "32"))

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except ImportError:                                  # CPU-only checkout
    _HAVE_TRITON = False


if _HAVE_TRITON:

    @triton.jit
    def _prefix_partial_kernel(
        Q, K, V, PBT, Wacc, Wm, Wl,
        sm_scale, P, S, NSPLIT,
        stride_qt, stride_qh,
        stride_kb, stride_ks, stride_kh,
        stride_pbt_g,
        stride_wa_g, stride_wa_h, stride_wa_s, stride_wa_i,
        stride_wm_g, stride_wm_h, stride_wm_s,
        BLOCK_SIZE: tl.constexpr, HKV_RATIO: tl.constexpr,
        D: tl.constexpr, BN: tl.constexpr, SIB: tl.constexpr,
        NB: tl.constexpr, NPG: tl.constexpr, ROWS: tl.constexpr,
        SIBT: tl.constexpr, NBLOCKS: tl.constexpr,
    ):
        # THE SHARED PREFIX IS THE WHOLE POINT, so one K/V tile must serve every query row that
        # shares it. Two levels of sharing exist here and FA3's varlen path can only use one:
        #   * the group's S siblings share the prefix  -- FA3 does exploit this
        #   * HKV_RATIO query heads share a KV head    -- FA3 issues these as separate work
        # Together one tile serves S x HKV_RATIO = 128 query rows. The previous grid was
        # (H, G*NSPLIT), i.e. one query head per CTA, so the same prefix bytes were fetched 4x
        # (L2-absorbed, but 4x the requests) and each dot was only [32, D] x [D, BN]. Measured:
        # 0.2817 ms/layer, 33% of roofline, against FA3's 0.1250.
        # Now a CTA owns (group, KV head, split) and carries a [ROWS, D] query tile with
        # ROWS = S * HKV_RATIO, so the tile is read ONCE and feeds a [128,D] x [D,BN] dot --
        # 4x the arithmetic intensity per byte, which is the advantage the sharing structure
        # actually confers. Row r is (head hh = r // S, sibling i = r % S).
        # SIBLING TILING. ROWS = S * HKV_RATIO = 128 made acc[128,D] fp32 = 64 KB, and qk and
        # p_ another 64 KB each at BN=128 -- far past the 232 KB SMEM budget, so the kernel spilled
        # and ran at 46% of roofline. Splitting the group's S siblings into tiles of SIBT keeps
        # the sharing that matters (one K/V tile still serves SIBT * HKV_RATIO rows, well above
        # FA3's single head) while cutting every live tile proportionally AND multiplying the CTA
        # count, which this parallelism-starved phase wants anyway.
        kvh = tl.program_id(0)
        pid = tl.program_id(1)
        ntile = (S + SIBT - 1) // SIBT
        g = pid // (NSPLIT * ntile)
        rem = pid % (NSPLIT * ntile)
        s = rem // ntile
        stile = rem % ntile

        offs_d = tl.arange(0, D)
        offs_r = tl.arange(0, ROWS)
        hh = offs_r // SIBT
        sib = stile * SIBT + (offs_r % SIBT)
        r_ok = (hh < HKV_RATIO) & (sib < S)
        q_tok = g * S + sib
        q_head = kvh * HKV_RATIO + hh
        q = tl.load(Q + q_tok[:, None] * stride_qt + q_head[:, None] * stride_qh
                    + offs_d[None, :], mask=r_ok[:, None], other=0.0)

        acc = tl.zeros([ROWS, D], dtype=tl.float32)
        m_i = tl.full([ROWS], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([ROWS], dtype=tl.float32)

        p_lo = (P * s) // NSPLIT
        p_hi = (P * (s + 1)) // NSPLIT
        for start in range(p_lo, p_hi, BN):
            offs_n = start + tl.arange(0, BN)
            n_ok = offs_n < p_hi
            pg = tl.load(PBT + g * stride_pbt_g + (start // BLOCK_SIZE) + tl.arange(0, NB),
                         mask=(start // BLOCK_SIZE) + tl.arange(0, NB) < NPG, other=0)
            blk = tl.reshape(tl.broadcast_to(pg[:, None], (NB, BLOCK_SIZE)), (BN,))
            # CLAMP THE PAGE ID. The synthetic tests build the suffix table themselves so every
            # entry is a live page, but vLLM's block_table is preallocated [max_reqs, max_blks]
            # and is not cleared per step, while _suffix_block_table's gather clamps its column
            # index to the last column -- so columns past a request's own allocation can hold
            # stale or never-written ids. Those lanes are masked out of the MATH by n_ok, but
            # Triton still forms the address, and an id past the end of the cache is the
            # illegal memory access that killed the production dispatches. One op on an address
            # that was going to be discarded anyway.
            blk = tl.minimum(tl.maximum(blk, 0), NBLOCKS - 1)
            koff = blk * stride_kb + (offs_n % BLOCK_SIZE) * stride_ks + kvh * stride_kh
            k = tl.load(K + koff[:, None] + offs_d[None, :], mask=n_ok[:, None], other=0.0)
            v = tl.load(V + koff[:, None] + offs_d[None, :], mask=n_ok[:, None], other=0.0)
            qk = tl.dot(q, tl.trans(k)) * sm_scale
            qk = tl.where(r_ok[:, None] & n_ok[None, :], qk, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.exp(m_i - m_new)
            p_ = tl.exp(qk - m_new[:, None])
            acc = acc * alpha[:, None] + tl.dot(p_.to(v.dtype), v).to(tl.float32)
            l_i = l_i * alpha + tl.sum(p_, 1)
            m_i = m_new

        base_a = Wacc + g * stride_wa_g + q_head[:, None] * stride_wa_h + s * stride_wa_s
        tl.store(base_a + sib[:, None] * stride_wa_i + offs_d[None, :], acc, mask=r_ok[:, None])
        base_m = g * stride_wm_g + q_head * stride_wm_h + s * stride_wm_s + sib
        tl.store(Wm + base_m, m_i, mask=r_ok)
        tl.store(Wl + base_m, l_i, mask=r_ok)

    @triton.jit
    def _suffix_merge_kernel(
        Q, K, V, SBT, Out, Wacc, Wm, Wl,
        sm_scale, SUF, S, NSPLIT,
        stride_qt, stride_qh,
        stride_kb, stride_ks, stride_kh,
        stride_sbt_t, stride_ot, stride_oh,
        stride_wa_g, stride_wa_h, stride_wa_s, stride_wa_i,
        stride_wm_g, stride_wm_h, stride_wm_s,
        BLOCK_SIZE: tl.constexpr, HKV_RATIO: tl.constexpr,
        D: tl.constexpr, BN: tl.constexpr, HP: tl.constexpr,
        NB: tl.constexpr, NPGS: tl.constexpr, NBLOCKS: tl.constexpr,
    ):
        # ONE CTA PER (token, KV head), carrying that KV head's HKV_RATIO query heads as a
        # tile. The first version used one CTA per (token, query head), which left a single
        # query row and therefore no tile to feed tl.dot -- the suffix walk fell back to
        # tl.sum(q * k), a manual FMA reduction with the tensor cores idle. The suffix is the
        # LARGER half of the attention (10.24 vs 5.30 ms/step in the FA3 breakdown), so that
        # cost about as much as the fused epilogue saves, and the first A/B came out 0.8%
        # SLOWER than the three-kernel path. GQA makes the fix free: 4 query heads already
        # share one K/V tile, so grouping them gives a [HP, D] x [D, BN] dot AND reads the tile
        # once for all four.
        t = tl.program_id(0)
        kvh = tl.program_id(1)
        g = t // S
        i = t % S
        offs_d = tl.arange(0, D)
        offs_h = tl.arange(0, HP)
        h_ok = offs_h < HKV_RATIO
        hq = kvh * HKV_RATIO + offs_h                    # this KV head's query heads

        # --- reduce the NSPLIT prefix partials for each of those query heads ----------------
        m_i = tl.full([HP], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([HP], dtype=tl.float32)
        acc = tl.zeros([HP, D], dtype=tl.float32)
        for j in range(0, NSPLIT):
            off_m = g * stride_wm_g + hq * stride_wm_h + j * stride_wm_s + i
            mj = tl.load(Wm + off_m, mask=h_ok, other=float("-inf"))
            lj = tl.load(Wl + off_m, mask=h_ok, other=0.0)
            aj = tl.load(Wacc + g * stride_wa_g + hq[:, None] * stride_wa_h
                         + j * stride_wa_s + i * stride_wa_i + offs_d[None, :],
                         mask=h_ok[:, None], other=0.0)
            m_new = tl.maximum(m_i, mj)
            a = tl.exp(m_i - m_new)
            b = tl.exp(mj - m_new)
            acc = acc * a[:, None] + aj * b[:, None]
            l_i = l_i * a + lj * b
            m_i = m_new

        # --- walk this token's own suffix, on tensor cores ---------------------------------
        q = tl.load(Q + t * stride_qt + hq[:, None] * stride_qh + offs_d[None, :],
                    mask=h_ok[:, None], other=0.0)          # bf16: see kernel A
        for start in range(0, SUF, BN):
            offs_n = start + tl.arange(0, BN)
            n_ok = offs_n < SUF
            pg = tl.load(SBT + t * stride_sbt_t + (start // BLOCK_SIZE) + tl.arange(0, NB),
                         mask=(start // BLOCK_SIZE) + tl.arange(0, NB) < NPGS, other=0)
            blk = tl.reshape(tl.broadcast_to(pg[:, None], (NB, BLOCK_SIZE)), (BN,))
            # CLAMP THE PAGE ID. The synthetic tests build the suffix table themselves so every
            # entry is a live page, but vLLM's block_table is preallocated [max_reqs, max_blks]
            # and is not cleared per step, while _suffix_block_table's gather clamps its column
            # index to the last column -- so columns past a request's own allocation can hold
            # stale or never-written ids. Those lanes are masked out of the MATH by n_ok, but
            # Triton still forms the address, and an id past the end of the cache is the
            # illegal memory access that killed the production dispatches. One op on an address
            # that was going to be discarded anyway.
            blk = tl.minimum(tl.maximum(blk, 0), NBLOCKS - 1)
            koff = blk * stride_kb + (offs_n % BLOCK_SIZE) * stride_ks + kvh * stride_kh
            k = tl.load(K + koff[:, None] + offs_d[None, :], mask=n_ok[:, None], other=0.0)
            v = tl.load(V + koff[:, None] + offs_d[None, :], mask=n_ok[:, None], other=0.0)
            qk = tl.dot(q, tl.trans(k)) * sm_scale
            qk = tl.where(h_ok[:, None] & n_ok[None, :], qk, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            a = tl.exp(m_i - m_new)
            p_ = tl.exp(qk - m_new[:, None])
            acc = acc * a[:, None] + tl.dot(p_.to(v.dtype), v).to(tl.float32)
            l_i = l_i * a + tl.sum(p_, 1)
            m_i = m_new

        out = acc / tl.maximum(l_i, 1e-20)[:, None]
        tl.store(Out + t * stride_ot + hq[:, None] * stride_oh + offs_d[None, :],
                 out.to(Out.dtype.element_ty), mask=h_ok[:, None])


def fused_grouped_attention(query, key_cache, value_cache, prefix_block_table,
                            suffix_block_table, num_groups, siblings, prefix_len,
                            suffix_len, softmax_scale, num_splits=32, out=None,
                            workspace=None):
    """query [T,H,D] group-major, T = num_groups*siblings. Equal prefix_len across groups and
    equal suffix_len across sequences -- that is the generality this trades away."""
    if not _HAVE_TRITON:
        raise RuntimeError("triton unavailable")
    T, H, D = query.shape
    assert T == num_groups * siblings, (T, num_groups, siblings)
    if out is None:
        out = torch.empty_like(query)
    SIB = max(16, triton.next_power_of_2(siblings))
    dev = query.device
    if workspace is None:
        wacc = torch.empty((num_groups, H, num_splits, SIB, D), dtype=torch.float32, device=dev)
        wm = torch.empty((num_groups, H, num_splits, SIB), dtype=torch.float32, device=dev)
        wl = torch.empty_like(wm)
    else:
        wacc, wm, wl = workspace
    BLOCK_SIZE = key_cache.shape[1]
    HKV = key_cache.shape[2]
    # KV tile and pipeline depth. 64 was a first guess; a wider tile means fewer loop trips,
    # fewer block-table loads and more work in flight per stage, which is what hides HBM
    # latency on a pure streaming walk like this. Must stay a multiple of the page size so the
    # page-broadcast above is exact. Overridable so a sweep can find the real optimum.
    import os as _os
    BN = _ENV_BN
    NSTAGES = _ENV_STAGES
    # warp count is the knob never swept: it sets how the [ROWS, BN] tiles are partitioned across
    # the CTA, and with a 64 KB accumulator the split between parallelism and per-thread state
    # is exactly what decides whether this spills.
    NWARPS_A = _ENV_WARPS_A
    NWARPS_B = _ENV_WARPS_B
    assert BN % BLOCK_SIZE == 0, (BN, BLOCK_SIZE)

    # SIBT: siblings per CTA tile. 16 keeps ROWS at 64 (16 x 4 GQA heads) -- still 2x the query
    # rows FA3 gets per K/V tile, at a quarter of the SMEM the full 128-row tile needed.
    # Measured: SIBT=16 (ROWS 64) is WORSE than the full 32 (ROWS 128) -- 0.4091 vs 0.3306
    # ms/layer -- so the sharing outweighs the SMEM pressure it causes, and the right direction
    # is MORE rows per K/V tile, not fewer. 128 is the ceiling here: rows come from
    # S * HKV_RATIO, and different groups do not share a prefix, so there is nothing else to add.
    SIBT = _ENV_SIBT
    SIBT = min(SIBT, siblings)
    NTILE = (siblings + SIBT - 1) // SIBT
    ROWS = max(16, triton.next_power_of_2(SIBT * (H // HKV)))
    _prefix_partial_kernel[(HKV, num_groups * num_splits * NTILE)](
        query, key_cache, value_cache, prefix_block_table, wacc, wm, wl,
        softmax_scale, prefix_len, siblings, num_splits,
        query.stride(0), query.stride(1),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        prefix_block_table.stride(0),
        wacc.stride(0), wacc.stride(1), wacc.stride(2), wacc.stride(3),
        wm.stride(0), wm.stride(1), wm.stride(2),
        BLOCK_SIZE=BLOCK_SIZE, HKV_RATIO=H // HKV, D=D, BN=BN, SIB=SIB,
        NB=BN // BLOCK_SIZE, NPG=prefix_block_table.shape[1], ROWS=ROWS, SIBT=SIBT,
        NBLOCKS=key_cache.shape[0],
        num_warps=NWARPS_A, num_stages=NSTAGES,
    )
    HP = max(16, triton.next_power_of_2(H // HKV))
    _suffix_merge_kernel[(T, HKV)](
        query, key_cache, value_cache, suffix_block_table, out, wacc, wm, wl,
        softmax_scale, suffix_len, siblings, num_splits,
        query.stride(0), query.stride(1),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        suffix_block_table.stride(0), out.stride(0), out.stride(1),
        wacc.stride(0), wacc.stride(1), wacc.stride(2), wacc.stride(3),
        wm.stride(0), wm.stride(1), wm.stride(2),
        BLOCK_SIZE=BLOCK_SIZE, HKV_RATIO=H // HKV, D=D, BN=BN, HP=HP,
        NB=BN // BLOCK_SIZE, NPGS=suffix_block_table.shape[1],
        NBLOCKS=key_cache.shape[0],
        num_warps=NWARPS_B, num_stages=NSTAGES,
    )
    return out


def _reference(query, key_cache, value_cache, pbt, sbt, G, S, P, SUF, scale):
    """Dense per-request reference: one softmax over [group prefix ++ own suffix]."""
    T, H, D = query.shape
    BS, HKV = key_cache.shape[1], key_cache.shape[2]
    out = torch.empty_like(query)

    def gather(bt_row, n):
        ks, vs = [], []
        for x in range(n):
            b = int(bt_row[x // BS]); o = x % BS
            ks.append(key_cache[b, o]); vs.append(value_cache[b, o])
        return torch.stack(ks), torch.stack(vs)

    for t in range(T):
        kp, vp = gather(pbt[t // S], P)
        ks_, vs_ = gather(sbt[t], SUF)
        k = torch.cat([kp, ks_], 0).float()
        v = torch.cat([vp, vs_], 0).float()
        for h in range(H):
            kvh = h // (H // HKV)
            sc = (query[t, h].float() * scale) @ k[:, kvh].T
            out[t, h] = (torch.softmax(sc, -1) @ v[:, kvh]).to(out.dtype)
    return out



if _HAVE_TRITON:

    # NO do_not_specialize HERE, and no direct-launch bypass either. Both were tried and
    # measured on the production cell (n=192, k=192, gen 5000):
    #
    #     mode 3 baseline                                    72.55 s
    #     + do_not_specialize, direct launch OFF             94.98 s   (+31%)
    #     + do_not_specialize, direct launch ON              94.15 s
    #
    # So bypassing JITFunction.run is worth 0.9%, not the ~15% the launch-overhead reasoning
    # predicted -- the JIT bind was never the cost. And do_not_specialize is catastrophic on
    # its own: it gives up the divisibility-by-16 hints, and this kernel is a memory-bound
    # paged gather whose loads then stop vectorising. Pinning the signature was the
    # precondition for the bypass, so the bypass cannot be had at a price worth paying.
    @triton.jit
    def _suffix_walk_kernel(
        Q, K, V, SBT, Out, OutLSE, SUFLEN, SUFMAX,
        sm_scale, SUF,
        stride_qt, stride_qh,
        stride_kb, stride_ks, stride_kh,
        stride_sbt_t, stride_ot, stride_oh,
        stride_lh, stride_lt,
        BLOCK_SIZE: tl.constexpr, HKV_RATIO: tl.constexpr,
        D: tl.constexpr, BN: tl.constexpr, HP: tl.constexpr,
        NB: tl.constexpr, NPGS: tl.constexpr, NBLOCKS: tl.constexpr,
        RAGGED: tl.constexpr, DYN: tl.constexpr,
    ):
        """Suffix walk ONLY -- writes (out, lse), merges nothing.

        WHY THIS EXISTS when _suffix_merge_prefixout_kernel already walks the suffix AND folds
        in the prefix: fusing the merge costs the OVERLAP, and the overlap is worth more than
        the merge. Traced at n=192, both arms, identical shape:

            attention SUM   hybrid 13.13 ms/step   3-kernel 15.61   hybrid does 16% LESS work
            sum/union       hybrid  1.01x          3-kernel  1.22x  hybrid lost all of it
            wall/step       hybrid 18.120 ms       3-kernel 17.488  and is 4% SLOWER

        The three-kernel path issues FA3's prefix on a side stream and FA3's suffix on the main
        stream concurrently, then merges. The fused epilogue cannot: it seeds its accumulator
        from the prefix (out, lse) on its FIRST instruction, so it has to wait for the prefix
        pass. Deleting merge_attn_states (0.09 ms/step) bought a serialisation costing ~2.5.

        So keep the merge as its own launch and keep the concurrency: side stream runs FA3's
        prefix, main stream runs THIS, then merge_attn_states. The suffix walk needs nothing
        from the prefix, so the dependency -- and the stall -- disappears, while the part that
        actually paid survives: this walk is 9.5 ms/step against FA3's suffix ~11.1, and FA3's
        suffix split-combine goes with it.
        """
        t = tl.program_id(0)
        kvh = tl.program_id(1)
        offs_d = tl.arange(0, D)
        offs_h = tl.arange(0, HP)
        h_ok = offs_h < HKV_RATIO
        hq = kvh * HKV_RATIO + offs_h

        q = tl.load(Q + t * stride_qt + hq[:, None] * stride_qh + offs_d[None, :],
                    mask=h_ok[:, None], other=0.0)
        acc = tl.zeros([HP, D], dtype=tl.float32)
        m_i = tl.full([HP], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([HP], dtype=tl.float32)

        # DYN: the loop bound comes from a DEVICE scalar refreshed in place by the builder,
        # not from a host int baked into the launch. Triton supports dynamic trip counts, and
        # this is the last host value a FULL-cudagraph capture of this kernel would freeze:
        # with it, every tensor and every bound the kernel touches lives in persistent
        # storage the per-step builder rewrites, which is the contract graph replay needs.
        suf_hi = tl.load(SUFMAX).to(tl.int32) if DYN else SUF
        suf_t = tl.load(SUFLEN + t) if RAGGED else suf_hi
        for start in range(0, suf_hi, BN):
            offs_n = start + tl.arange(0, BN)
            n_ok = offs_n < suf_t
            pg = tl.load(SBT + t * stride_sbt_t + (start // BLOCK_SIZE) + tl.arange(0, NB),
                         mask=(start // BLOCK_SIZE) + tl.arange(0, NB) < NPGS, other=0)
            blk = tl.reshape(tl.broadcast_to(pg[:, None], (NB, BLOCK_SIZE)), (BN,))
            blk = tl.minimum(tl.maximum(blk, 0), NBLOCKS - 1)
            koff = blk * stride_kb + (offs_n % BLOCK_SIZE) * stride_ks + kvh * stride_kh
            k = tl.load(K + koff[:, None] + offs_d[None, :], mask=n_ok[:, None], other=0.0)
            v = tl.load(V + koff[:, None] + offs_d[None, :], mask=n_ok[:, None], other=0.0)
            qk = tl.dot(q, tl.trans(k)) * sm_scale
            qk = tl.where(h_ok[:, None] & n_ok[None, :], qk, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            a = tl.exp(m_i - m_new)
            p_ = tl.exp(qk - m_new[:, None])
            acc = acc * a[:, None] + tl.dot(p_.to(v.dtype), v).to(tl.float32)
            l_i = l_i * a + tl.sum(p_, 1)
            m_i = m_new

        out = acc / tl.maximum(l_i, 1e-20)[:, None]
        tl.store(Out + t * stride_ot + hq[:, None] * stride_oh + offs_d[None, :],
                 out.to(Out.dtype.element_ty), mask=h_ok[:, None])
        # merge_attn_states consumes FA3's convention: lse = m + log(l), laid out [heads, tokens].
        # A row with no suffix would give log(0) = -inf against m = -inf and produce NaN, so pin
        # it to -inf explicitly -- the merge then ignores that row instead of poisoning the
        # output. The prefix cap guarantees a non-empty suffix, but a silent NaN here would
        # corrupt training rather than crash, which is the failure mode this file exists to stop.
        lse = tl.where(l_i > 0, m_i + tl.log(tl.maximum(l_i, 1e-20)), float("-inf"))
        tl.store(OutLSE + hq * stride_lh + t * stride_lt, lse, mask=h_ok)


_WALK_ARGS = {}          # shape -> (grid, strides tuple, constexpr kwargs); see the note below
_WALK_FAST = {}          # shape -> CompiledKernel, or False once the fast path is ruled out
_WALK_PATH_SEEN = False
# HARD OFF, not a knob. The direct launch only works with do_not_specialize
# pinning the signature, and that costs 31% on the GPU side while the bypass
# itself returns 0.9%. Enabling this without the pin is an IndexError; enabling it
# with the pin is a large regression. Kept as documentation of a closed avenue.
_FAST_LAUNCH = False


def _harvest_compiled(jit_fn):
    """Pull the already-compiled kernel out of Triton's JIT cache, or None.

    THE LAUNCH IS THE BOTTLENECK AND CUDAGRAPH IS NOT THE ONLY WAY OUT. Traced: the fused path
    does 16% less GPU work and still loses, because ~67 us of CPU goes into every Triton launch
    while the prefix pass it should overlap with is only ~78 us of GPU -- the CPU never gets
    far enough ahead for the two to run together. Most of that 67 us is JITFunction.run:
    building a specialization key from the arg types, hashing it, looking it up, binding. None
    of that changes once the shape is fixed, which it is for the whole decode phase.

    CompiledKernel.__getitem__ skips all of it and goes more or less straight to the CUDA
    launch. The layout of Triton's cache has moved between versions (JITFunction.cache in 2.x,
    device_caches in 3.x), so probe rather than assume, and fall back to the ordinary launch on
    anything unexpected -- a slower kernel is a bad day, a wrong one corrupts training.
    """
    for attr in ("device_caches", "cache"):
        c = getattr(jit_fn, attr, None)
        if not c:
            continue
        try:
            for entry in c.values():
                d = entry[0] if isinstance(entry, (tuple, list)) else entry
                if isinstance(d, dict) and len(d) == 1:
                    k = next(iter(d.values()))
                    if hasattr(k, "__getitem__") and hasattr(k, "run"):
                        return k
        except Exception:
            return None
    return None


def suffix_walk_attention(query, key_cache, value_cache, suffix_block_table,
                          suffix_len, softmax_scale, out, lse, suffix_lens=None,
                          suffix_max_dev=None):
    """FA3-compatible (out, lse) for the suffix half, so the existing merge path is unchanged.

    EVERYTHING THAT DOES NOT CHANGE BETWEEN STEPS IS CACHED. Strides, the grid, and the
    constexpr set are functions of the SHAPE, which is fixed for the whole decode phase, but
    they were being recomputed on every launch: ten Tensor.stride() calls, three shape lookups,
    a next_power_of_2, and a dict build, 36 layers deep and 5000 steps long. That is the same
    reason FA3 is called through a prebuilt positional list here rather than by keyword.
    """
    T, H, D = query.shape
    HKV = key_cache.shape[2]
    key = (T, H, D, HKV, key_cache.shape[1], suffix_block_table.shape[1],
           key_cache.shape[0], suffix_lens is not None, suffix_max_dev is not None)
    ent = _WALK_ARGS.get(key)
    if ent is None:
        assert suffix_block_table.shape[0] == T, (
            f"suffix_walk_attention is decode-only: {T} query rows vs "
            f"{suffix_block_table.shape[0]} suffix block-table rows")
        BN = _ENV_BN
        ent = (
            (T, HKV),
            (query.stride(0), query.stride(1),
             key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
             suffix_block_table.stride(0), out.stride(0), out.stride(1),
             lse.stride(0), lse.stride(1)),
            dict(BLOCK_SIZE=key_cache.shape[1], HKV_RATIO=H // HKV, D=D, BN=BN,
                 HP=max(16, triton.next_power_of_2(H // HKV)),
                 NB=BN // key_cache.shape[1], NPGS=suffix_block_table.shape[1],
                 NBLOCKS=key_cache.shape[0], RAGGED=suffix_lens is not None,
                 DYN=suffix_max_dev is not None,
                 num_warps=_ENV_WARPS_B, num_stages=_ENV_STAGES),
            # The direct launcher takes EVERY declared parameter positionally, constexprs
            # included, in declaration order -- probed, not guessed: a 6-param kernel wants
            # 13 + 6 = 19 launcher args, and this 28-param one wants 13 + 28 = 41, which is
            # exactly the count the failed attempt reported. num_warps/num_stages are not
            # parameters, they are compilation options, so they are NOT in this tuple.
            (key_cache.shape[1], H // HKV, D, BN,
             max(16, triton.next_power_of_2(H // HKV)),
             BN // key_cache.shape[1], suffix_block_table.shape[1],
             key_cache.shape[0], suffix_lens is not None, suffix_max_dev is not None),
        )
        _WALK_ARGS[key] = ent
    grid, st, cx, cxv = ent
    sl = suffix_lens if suffix_lens is not None else query
    smx = suffix_max_dev if suffix_max_dev is not None else query   # unused when DYN=False
    # DEFAULT OFF. Bypassing JITFunction.run is real (the probe confirms CompiledKernel's
    # __getitem__ goes almost straight to the launch), but Triton SPECIALIZES on argument
    # values -- strides divisible by 16, values equal to 1 -- and packed_metadata describes the
    # arg list AFTER specialization, so passing the full list is a count mismatch that surfaces
    # as IndexError inside the launcher. Getting it right means reimplementing Triton's
    # specialization rules against a private API that has already moved once between versions.
    # Left in, behind a flag, because the measurement that motivates it stands: ~67 us of CPU
    # per launch against a 78 us prefix pass is what destroys the overlap. Not worth shipping
    # on by default -- a working path is worth more than an unproven 15%.
    fast = _WALK_FAST.get(key) if _FAST_LAUNCH else False
    if fast:
        # Non-constexpr arguments only, in declaration order; the constexprs are baked in.
        # CompiledKernel.__getitem__ indexes grid[0..2] unconditionally, so it needs the full
        # 3-tuple -- JITFunction accepts a short one and pads, CompiledKernel does not, which
        # is an IndexError on the first replay rather than at bind time.
        fast[(grid[0], grid[1], 1)](
            query, key_cache, value_cache, suffix_block_table, out, lse, sl, smx,
            softmax_scale, suffix_len, *st, *cxv)
        return out, lse
    _suffix_walk_kernel[grid](
        query, key_cache, value_cache, suffix_block_table, out, lse, sl, smx,
        softmax_scale, suffix_len, *st, **cx)
    if fast is None:
        global _WALK_PATH_SEEN
        got = _harvest_compiled(_suffix_walk_kernel)
        _WALK_FAST[key] = got or False
        if not _WALK_PATH_SEEN:
            _WALK_PATH_SEEN = True
            print(f"[fused] walk launch path: "
                  f"{'FAST (compiled kernel, JIT bind skipped)' if got else 'SLOW (JIT bind per launch)'}",
                  flush=True)
    return out, lse

def selftest(device="cuda", dtype=torch.bfloat16, verbose=True) -> bool:
    torch.manual_seed(0)
    G, S, H, HKV, D, BS = 3, 8, 8, 2, 128, 16
    P, SUF = 128, 48
    T = G * S
    kc = torch.randn(512, BS, HKV, D, device=device, dtype=dtype) * 0.1
    vc = torch.randn(512, BS, HKV, D, device=device, dtype=dtype) * 0.1
    q = torch.randn(T, H, D, device=device, dtype=dtype) * 0.3
    pbt = torch.arange(G * (P // BS), device=device, dtype=torch.int32).view(G, P // BS)
    base = G * (P // BS)
    sbt = (base + torch.arange(T * (SUF // BS), device=device,
                               dtype=torch.int32)).view(T, SUF // BS)
    scale = D ** -0.5
    ref = _reference(q, kc, vc, pbt, sbt, G, S, P, SUF, scale)

    ok = True
    # num_splits=1 first: it bypasses the multi-partial reduction entirely, so a failure here
    # is the attention math and a failure ONLY at >1 is the reduction. The single-launch
    # version failed at rel=0.61 and this split is what tells the two apart.
    for ns in (1, 4, 8):
        got = fused_grouped_attention(q, kc, vc, pbt, sbt, G, S, P, SUF, scale, num_splits=ns)
        d = (got.float() - ref.float()).abs()
        rel = (d.max() / ref.float().abs().max()).item()
        good = rel < 0.02
        ok &= good
        if verbose:
            print(f"[fused] splits={ns:>2}  max|d|={d.max().item():.5f}  rel={rel:.5f}  "
                  f"-> {'PASS' if good else 'FAIL'}")
    if verbose:
        print(f"[fused] {'ALL PASS' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)


if _HAVE_TRITON:

    @triton.jit
    def _suffix_merge_prefixout_kernel(
        Q, K, V, SBT, Out, PO, PL, SUFLEN,
        sm_scale, SUF, S,
        stride_qt, stride_qh,
        stride_kb, stride_ks, stride_kh,
        stride_sbt_t, stride_ot, stride_oh,
        stride_pot, stride_poh, stride_plh, stride_plt,
        BLOCK_SIZE: tl.constexpr, HKV_RATIO: tl.constexpr,
        D: tl.constexpr, BN: tl.constexpr, HP: tl.constexpr,
        NB: tl.constexpr, NPGS: tl.constexpr, NBLOCKS: tl.constexpr, LSE_T_AXIS: tl.constexpr,
        RAGGED: tl.constexpr,
    ):
        """Suffix walk + merge against an ALREADY-COMBINED prefix (out, lse).

        HYBRID. Per-kernel timing says each side wins a different half: FA3's prefix pass is
        0.1250 ms/layer against my 0.1972, while my suffix+combine+merge is 0.1591 against FA3's
        0.1663. FA3's prefix is better because it is a hand-tuned SM90 kernel; my suffix half is
        better because it folds the split-combine and merge_attn_states into one epilogue.
        Taking the better of each -- FA3 prefix, this epilogue -- beats both pure paths, and it
        drops the 32-way partial reduction entirely since FA3 hands back a single combined
        (out, lse) pair. That also removes kernel A's 25 MB/layer fp32 workspace round trip.
        """
        t = tl.program_id(0)
        kvh = tl.program_id(1)
        offs_d = tl.arange(0, D)
        offs_h = tl.arange(0, HP)
        h_ok = offs_h < HKV_RATIO
        hq = kvh * HKV_RATIO + offs_h

        # prefix state: one combined output and lse per (token, head) -- no reduction loop
        acc = tl.load(PO + t * stride_pot + hq[:, None] * stride_poh + offs_d[None, :],
                      mask=h_ok[:, None], other=0.0).to(tl.float32)
        if LSE_T_AXIS == 0:
            lse = tl.load(PL + t * stride_plt + hq * stride_plh, mask=h_ok, other=float("-inf"))
        else:
            lse = tl.load(PL + hq * stride_plh + t * stride_plt, mask=h_ok, other=float("-inf"))
        # FA3 returns a NORMALISED output and its lse. Recover the unnormalised accumulator the
        # online merge needs: acc_raw = out * exp(lse) is unsafe (overflows), so carry m = lse
        # and l = 1, which is the same state up to the invariant acc/l with m folded in.
        m_i = lse
        l_i = tl.where(h_ok, 1.0, 0.0)

        q = tl.load(Q + t * stride_qt + hq[:, None] * stride_qh + offs_d[None, :],
                    mask=h_ok[:, None], other=0.0)
        # RAGGED SUFFIXES ARE THE NORM, NOT THE EXCEPTION. This kernel took ONE scalar suffix
        # length for the whole batch and the caller gated on suffix_uniform to honour it. That
        # gate is false in essentially every real step: vLLM admits requests over several
        # prefill steps, so request 0 begins generating before request 191 and the offset
        # persists. Measured live -- identity=True uniform=True suffix_uniform=False at 71,
        # 127, 144 and 192 resident requests -- so the hybrid never dispatched and the run
        # quietly emitted three-kernel numbers under a hybrid label.
        # The precondition was never needed. Only the LOOP BOUND must be shared; the per-row
        # length moves into the mask. Rows that end early mask their tail, and with a spread of
        # tens of tokens against a 5k suffix the wasted lanes are under 1%.
        suf_t = tl.load(SUFLEN + t) if RAGGED else SUF
        for start in range(0, SUF, BN):
            offs_n = start + tl.arange(0, BN)
            n_ok = offs_n < suf_t
            pg = tl.load(SBT + t * stride_sbt_t + (start // BLOCK_SIZE) + tl.arange(0, NB),
                         mask=(start // BLOCK_SIZE) + tl.arange(0, NB) < NPGS, other=0)
            blk = tl.reshape(tl.broadcast_to(pg[:, None], (NB, BLOCK_SIZE)), (BN,))
            # CLAMP THE PAGE ID. The synthetic tests build the suffix table themselves so every
            # entry is a live page, but vLLM's block_table is preallocated [max_reqs, max_blks]
            # and is not cleared per step, while _suffix_block_table's gather clamps its column
            # index to the last column -- so columns past a request's own allocation can hold
            # stale or never-written ids. Those lanes are masked out of the MATH by n_ok, but
            # Triton still forms the address, and an id past the end of the cache is the
            # illegal memory access that killed the production dispatches. One op on an address
            # that was going to be discarded anyway.
            blk = tl.minimum(tl.maximum(blk, 0), NBLOCKS - 1)
            koff = blk * stride_kb + (offs_n % BLOCK_SIZE) * stride_ks + kvh * stride_kh
            k = tl.load(K + koff[:, None] + offs_d[None, :], mask=n_ok[:, None], other=0.0)
            v = tl.load(V + koff[:, None] + offs_d[None, :], mask=n_ok[:, None], other=0.0)
            qk = tl.dot(q, tl.trans(k)) * sm_scale
            qk = tl.where(h_ok[:, None] & n_ok[None, :], qk, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            a = tl.exp(m_i - m_new)
            p_ = tl.exp(qk - m_new[:, None])
            acc = acc * a[:, None] + tl.dot(p_.to(v.dtype), v).to(tl.float32)
            l_i = l_i * a + tl.sum(p_, 1)
            m_i = m_new

        out = acc / tl.maximum(l_i, 1e-20)[:, None]
        tl.store(Out + t * stride_ot + hq[:, None] * stride_oh + offs_d[None, :],
                 out.to(Out.dtype.element_ty), mask=h_ok[:, None])


_HYB_SHAPES_SEEN = False    # one-shot ground-truth dump; see the note at the launch


def hybrid_grouped_attention(query, key_cache, value_cache, prefix_out, prefix_lse,
                             suffix_block_table, siblings, suffix_len, softmax_scale,
                             out=None, lse_token_axis=1, suffix_lens=None):
    """FA3 supplies the prefix (out, lse); this fuses the suffix walk and the merge."""
    import os as _os
    T, H, D = query.shape
    HKV = key_cache.shape[2]
    if out is None:
        out = torch.empty_like(query)
    BN = _ENV_BN
    HP = max(16, triton.next_power_of_2(H // HKV))
    # program_id(0) indexes BOTH the query rows and the suffix block-table rows, so this is a
    # decode-only kernel: one query token per request. On a prefill step num_tokens >> num_reqs
    # and t runs past the end of SBT, which surfaces as an illegal memory access rather than as
    # anything readable. Cheap shape check, no device sync -- and it fails where the cause is,
    # not somewhere inside Triton 36 layers later.
    assert suffix_block_table.shape[0] == T, (
        f"hybrid_grouped_attention is decode-only: {T} query rows vs "
        f"{suffix_block_table.shape[0]} suffix block-table rows")
    # EVERY load and store in this kernel is masked, and Triton predicates masked lanes so they
    # cannot fault. An illegal access therefore cannot come from an index this code bounds --
    # it has to come from a SHAPE assumption that is wrong, i.e. the kernel is reading the
    # cache with the wrong strides entirely. Four guesses at the index arithmetic have now been
    # refuted, so dump the ground truth once instead of guessing a fifth time.
    global _HYB_SHAPES_SEEN
    if not _HYB_SHAPES_SEEN:
        _HYB_SHAPES_SEEN = True
        _mx = int(suffix_block_table.max())
        print(f"[fused] HYBRID shapes: q={tuple(query.shape)}/{query.stride()} "
              f"kc={tuple(key_cache.shape)}/{key_cache.stride()} "
              f"vc={tuple(value_cache.shape)}/{value_cache.stride()} "
              f"sbt={tuple(suffix_block_table.shape)}/{suffix_block_table.stride()} "
              f"sbt_max={_mx} nblocks={key_cache.shape[0]} "
              f"po={tuple(prefix_out.shape)}/{prefix_out.stride()} "
              f"pl={tuple(prefix_lse.shape)}/{prefix_lse.stride()} lse_axis={lse_token_axis} "
              f"SUF={suffix_len} S={siblings} H={H} HKV={HKV} D={D} "
              f"BLOCK_SIZE={key_cache.shape[1]} NB={BN // key_cache.shape[1]}", flush=True)
    _suffix_merge_prefixout_kernel[(T, HKV)](
        query, key_cache, value_cache, suffix_block_table, out, prefix_out, prefix_lse,
        suffix_lens if suffix_lens is not None else query,   # unused when RAGGED=False
        softmax_scale, suffix_len, siblings,
        query.stride(0), query.stride(1),
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        suffix_block_table.stride(0), out.stride(0), out.stride(1),
        prefix_out.stride(0), prefix_out.stride(1),
        prefix_lse.stride(0 if lse_token_axis == 1 else 1),
        prefix_lse.stride(1 if lse_token_axis == 1 else 0),
        BLOCK_SIZE=key_cache.shape[1], HKV_RATIO=H // HKV, D=D, BN=BN, HP=HP,
        NB=BN // key_cache.shape[1], NPGS=suffix_block_table.shape[1],
        NBLOCKS=key_cache.shape[0],
        LSE_T_AXIS=lse_token_axis, RAGGED=suffix_lens is not None,
        num_warps=_ENV_WARPS_B,
        num_stages=_ENV_STAGES,
    )
    return out


if _HAVE_TRITON:

    @triton.jit
    def _suffix_only_kernel(
        Q, K, V, SBT, SAcc, Sm, Sl, SUFLEN,
        sm_scale, SUF,
        stride_qt, stride_qh,
        stride_kb, stride_ks, stride_kh,
        stride_sbt_t, stride_sa_t, stride_sa_h, stride_sm_t, stride_sm_h,
        BLOCK_SIZE: tl.constexpr, HKV_RATIO: tl.constexpr,
        D: tl.constexpr, BN: tl.constexpr, HP: tl.constexpr,
        NB: tl.constexpr, NPGS: tl.constexpr, NBLOCKS: tl.constexpr,
        RAGGED: tl.constexpr,
    ):
        """Suffix walk ONLY -- no prefix dependency, so it can run CONCURRENTLY with the prefix.

        The two halves are sequential today purely because the epilogue seeds its accumulator
        from the prefix result. But the suffix walk needs nothing from the prefix; only the final
        merge does. Splitting it lets both kernels be in flight at once on separate streams.

        Total bytes do not change, so this cannot beat the roofline -- but neither kernel
        saturates it alone (prefix 76%, epilogue 87%), and their idle SMs are idle for different
        reasons (the prefix is starved to 6 groups and pays tail effects; the suffix is 192 short
        independent walks). Overlapping lets one fill the other's gaps, which raises AGGREGATE
        utilisation. 82% -> 92% would be exactly the 6x target.
        """
        t = tl.program_id(0)
        kvh = tl.program_id(1)
        offs_d = tl.arange(0, D)
        offs_h = tl.arange(0, HP)
        h_ok = offs_h < HKV_RATIO
        hq = kvh * HKV_RATIO + offs_h

        acc = tl.zeros([HP, D], dtype=tl.float32)
        m_i = tl.full([HP], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([HP], dtype=tl.float32)
        q = tl.load(Q + t * stride_qt + hq[:, None] * stride_qh + offs_d[None, :],
                    mask=h_ok[:, None], other=0.0)
        # Per-row suffix length, exactly as _suffix_merge_prefixout_kernel needed: vLLM admits
        # requests over several prefill steps, so suffix_uniform is false in essentially every
        # real step and a shared scalar bound would keep this path gated off forever.
        suf_t = tl.load(SUFLEN + t) if RAGGED else SUF
        for start in range(0, SUF, BN):
            offs_n = start + tl.arange(0, BN)
            n_ok = offs_n < suf_t
            pg = tl.load(SBT + t * stride_sbt_t + (start // BLOCK_SIZE) + tl.arange(0, NB),
                         mask=(start // BLOCK_SIZE) + tl.arange(0, NB) < NPGS, other=0)
            blk = tl.reshape(tl.broadcast_to(pg[:, None], (NB, BLOCK_SIZE)), (BN,))
            # CLAMP THE PAGE ID. The synthetic tests build the suffix table themselves so every
            # entry is a live page, but vLLM's block_table is preallocated [max_reqs, max_blks]
            # and is not cleared per step, while _suffix_block_table's gather clamps its column
            # index to the last column -- so columns past a request's own allocation can hold
            # stale or never-written ids. Those lanes are masked out of the MATH by n_ok, but
            # Triton still forms the address, and an id past the end of the cache is the
            # illegal memory access that killed the production dispatches. One op on an address
            # that was going to be discarded anyway.
            blk = tl.minimum(tl.maximum(blk, 0), NBLOCKS - 1)
            koff = blk * stride_kb + (offs_n % BLOCK_SIZE) * stride_ks + kvh * stride_kh
            k = tl.load(K + koff[:, None] + offs_d[None, :], mask=n_ok[:, None], other=0.0)
            v = tl.load(V + koff[:, None] + offs_d[None, :], mask=n_ok[:, None], other=0.0)
            qk = tl.dot(q, tl.trans(k)) * sm_scale
            qk = tl.where(h_ok[:, None] & n_ok[None, :], qk, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            a = tl.exp(m_i - m_new)
            p_ = tl.exp(qk - m_new[:, None])
            acc = acc * a[:, None] + tl.dot(p_.to(v.dtype), v).to(tl.float32)
            l_i = l_i * a + tl.sum(p_, 1)
            m_i = m_new
        tl.store(SAcc + t * stride_sa_t + hq[:, None] * stride_sa_h + offs_d[None, :], acc,
                 mask=h_ok[:, None])
        tl.store(Sm + t * stride_sm_t + hq * stride_sm_h, m_i, mask=h_ok)
        tl.store(Sl + t * stride_sm_t + hq * stride_sm_h, l_i, mask=h_ok)

    @triton.jit
    def _merge2_kernel(
        Out, PO, PL, SAcc, Sm, Sl,
        stride_ot, stride_oh, stride_pot, stride_poh, stride_plh, stride_plt,
        stride_sa_t, stride_sa_h, stride_sm_t, stride_sm_h,
        H: tl.constexpr, D: tl.constexpr,
    ):
        """Merge FA3's (out, lse) with the suffix state. Tiny: reads ~1 MB/layer total."""
        t = tl.program_id(0)
        h = tl.program_id(1)
        offs_d = tl.arange(0, D)
        po = tl.load(PO + t * stride_pot + h * stride_poh + offs_d).to(tl.float32)
        pl = tl.load(PL + h * stride_plh + t * stride_plt)
        sa = tl.load(SAcc + t * stride_sa_t + h * stride_sa_h + offs_d)
        sm = tl.load(Sm + t * stride_sm_t + h * stride_sm_h)
        sl = tl.load(Sl + t * stride_sm_t + h * stride_sm_h)
        m = tl.maximum(pl, sm)
        wp = tl.exp(pl - m)                 # FA3's out is normalised: weight is exp(lse - m)
        ws = tl.exp(sm - m)
        out = (po * wp + sa * ws) / tl.maximum(wp + sl * ws, 1e-20)
        tl.store(Out + t * stride_ot + h * stride_oh + offs_d, out.to(Out.dtype.element_ty))


def overlapped_grouped_attention(query, key_cache, value_cache, prefix_fn,
                                 suffix_block_table, suffix_len, softmax_scale,
                                 out=None, ws=None, stream=None, suffix_lens=None):
    """Run the FA3 prefix and the suffix walk CONCURRENTLY, then merge.

    prefix_fn() must launch FA3's prefix pass and return (out, lse); it is issued on the
    caller's stream while the suffix kernel runs on `stream`.
    """
    import os as _os
    T, H, D = query.shape
    HKV = key_cache.shape[2]
    if out is None:
        out = torch.empty_like(query)
    if ws is None:
        sacc = torch.empty((T, H, D), dtype=torch.float32, device=query.device)
        sm = torch.empty((T, H), dtype=torch.float32, device=query.device)
        sl = torch.empty_like(sm)
    else:
        sacc, sm, sl = ws
    BN = _ENV_BN
    HP = max(16, triton.next_power_of_2(H // HKV))
    main = torch.cuda.current_stream(query.device)
    side = stream or torch.cuda.Stream(query.device)
    ev = torch.cuda.Event()
    ev.record(main)
    side.wait_event(ev)
    with torch.cuda.stream(side):
        _suffix_only_kernel[(T, HKV)](
            query, key_cache, value_cache, suffix_block_table, sacc, sm, sl,
            suffix_lens if suffix_lens is not None else query,
            softmax_scale, suffix_len,
            query.stride(0), query.stride(1),
            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
            suffix_block_table.stride(0), sacc.stride(0), sacc.stride(1),
            sm.stride(0), sm.stride(1),
            BLOCK_SIZE=key_cache.shape[1], HKV_RATIO=H // HKV, D=D, BN=BN, HP=HP,
            NB=BN // key_cache.shape[1], NPGS=suffix_block_table.shape[1],
        NBLOCKS=key_cache.shape[0], RAGGED=suffix_lens is not None,
            num_warps=_ENV_WARPS_B, num_stages=4)
        for tsr in (sacc, sm, sl):
            tsr.record_stream(main)
        ev2 = torch.cuda.Event()
        ev2.record(side)
    po, pl = prefix_fn()                      # FA3 prefix, concurrent on the main stream
    main.wait_event(ev2)
    lse_t_axis = 1 if pl.shape[-1] == T else 0
    _merge2_kernel[(T, H)](
        out, po, pl, sacc, sm, sl,
        out.stride(0), out.stride(1), po.stride(0), po.stride(1),
        pl.stride(0 if lse_t_axis == 1 else 1), pl.stride(1 if lse_t_axis == 1 else 0),
        sacc.stride(0), sacc.stride(1), sm.stride(0), sm.stride(1),
        H=H, D=D, num_warps=4)
    return out
