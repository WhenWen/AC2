# Copyright 2026 Tilde Research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Nitrobrew: fused, constant-memory KL divergence from hidden states (FSDP path).

Computes KL(teacher || student) or KL(student || teacher) without materialising
the full [N, V] teacher logit tensor. Teacher logits are reconstructed on-the-fly
as z @ W.T in vocabulary chunks of size C using single-pass online-softmax.

Two entry families:
- compute_nitrobrew_kl / _NitrobrewKL[Clipped]: logit-processor path — the engine already
  materialized the packed (N, V) student logits (use_fused_kernels=False); only the TEACHER
  side is chunk-reconstructed.
- compute_fused_nitrobrew_aux / _NitrobrewKLTwoSided: fused-kernels path — runs INSIDE the
  monkeypatched model forward; BOTH logit streams are chunk-reconstructed from hidden states,
  so the (N, V) student logits NEVER materialize (the OPSD memory contract).

Peak extra memory: O(N * C) per chunk, instead of O(N * V).
"""

from dataclasses import dataclass
from typing import Optional

import torch

from verl.utils.ulysses import get_ulysses_sequence_parallel_world_size, slice_input_tensor
from verl.workers.config import DistillationConfig

_CHUNK_V: int = 1024


# ---------------------------------------------------------------------------
# Forward KL: KL(p_T || p_S)
# ---------------------------------------------------------------------------


@torch.compile
def _fwd_chunk_update(
    z_f: torch.Tensor,       # (N, D_t) float32
    W_chunk: torch.Tensor,   # (C, D_t) float32
    zs_chunk: torch.Tensor,  # (N, C)   float32
    mt: torch.Tensor,        # (N,) running teacher max
    st: torch.Tensor,        # (N,) sum exp(zt - mt)
    tt: torch.Tensor,        # (N,) sum exp(zt - mt) * zt
    ut: torch.Tensor,        # (N,) sum exp(zt - mt) * zs_clamped
    s_min: torch.Tensor,     # (N, 1) floor for student logits
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Online-softmax update for one teacher vocab chunk."""
    zt = z_f @ W_chunk.T
    tile_mt = zt.max(dim=1).values
    new_mt = torch.maximum(mt, tile_mt)
    alpha = (mt - new_mt).exp()
    pt = (zt - new_mt.unsqueeze(1)).exp()
    st = st * alpha + pt.sum(dim=1)
    tt = tt * alpha + (pt * zt).sum(dim=1)
    zs_clamped = torch.maximum(zs_chunk, s_min)
    ut = ut * alpha + (pt * zs_clamped).sum(dim=1)
    return new_mt, st, tt, ut


def _s_chunk(student_logits: torch.Tensor, v0: int, v1: int, inv_T: float) -> torch.Tensor:
    """One fp32 vocab chunk of the student logits (never materializes a full fp32 copy)."""
    c = student_logits[:, v0:v1].float()
    return c * inv_T if inv_T != 1.0 else c


def _student_lse(student_logits: torch.Tensor, chunk_V: int, inv_T: float) -> torch.Tensor:
    """Chunked online logsumexp over the vocab axis. Returns (N,) float32."""
    N, V = student_logits.shape
    m = torch.full((N,), float("-inf"), dtype=torch.float32, device=student_logits.device)
    acc = torch.zeros(N, dtype=torch.float32, device=student_logits.device)
    for v in range(0, V, chunk_V):
        c = _s_chunk(student_logits, v, v + chunk_V, inv_T)
        new_m = torch.maximum(m, c.max(dim=1).values)
        acc = acc * (m - new_m).exp() + (c - new_m.unsqueeze(1)).exp().sum(dim=1)
        m = new_m
    return m + acc.log()


def _chunked_kl_forward(
    z: torch.Tensor,              # (N, D_t)
    W: torch.Tensor,              # (V, D_t)
    student_logits: torch.Tensor, # (N, V)
    chunk_V: int = _CHUNK_V,
    temperature: float = 1.0,
    log_prob_min_clamp: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Single-pass chunked forward. Returns (kl, t_lse) both (N,) float32.

    Student logits are cast to fp32 per vocab chunk — a full (N, V) fp32 copy is never
    materialized (the bf16 logits from the engine forward are the only full-width tensor)."""
    N = z.shape[0]
    V = W.shape[0]
    device = z.device

    z_f = z.float()
    W_f = W.to(device=device, dtype=torch.float32)
    inv_T = 1.0 / temperature if temperature != 1.0 else 1.0
    if temperature != 1.0:
        z_f = z_f * inv_T

    s_lse = _student_lse(student_logits, chunk_V, inv_T)

    if log_prob_min_clamp is not None:
        s_min = (s_lse + log_prob_min_clamp).unsqueeze(1)
    else:
        s_min = torch.full((1, 1), float("-inf"), dtype=torch.float32, device=device)

    mt = torch.full((N,), float("-inf"), dtype=torch.float32, device=device)
    st = torch.zeros(N, dtype=torch.float32, device=device)
    tt = torch.zeros(N, dtype=torch.float32, device=device)
    ut = torch.zeros(N, dtype=torch.float32, device=device)

    for v in range(0, V, chunk_V):
        mt, st, tt, ut = _fwd_chunk_update(
            z_f, W_f[v : v + chunk_V], _s_chunk(student_logits, v, v + chunk_V, inv_T),
            mt, st, tt, ut, s_min,
        )

    t_lse = mt + st.log()
    kl = tt / st - t_lse - ut / st + s_lse
    return kl, t_lse


# ---------------------------------------------------------------------------
# Pointwise-clipped forward KL (OPSD, arXiv:2601.18734 D_clip):
#   kl_n = sum_v min( p_T(v) * [log p_T(v) - log p_S(v)], tau )
# Needs t_lse BEFORE contributions can be formed, so it is two-pass (vs the
# single-pass algebraic aggregation above, which cannot express the clip).
# ---------------------------------------------------------------------------


@torch.compile
def _clip_fwd_chunk(
    z_f: torch.Tensor,       # (N, D_t) float32
    W_chunk: torch.Tensor,   # (C, D_t) float32
    zs_chunk: torch.Tensor,  # (N, C)   float32
    t_lse: torch.Tensor,     # (N,)
    s_lse: torch.Tensor,     # (N,)
    s_min: torch.Tensor,     # (N, 1) floor for student logits
    token_clip: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clipped-KL contribution and clipped-entry count for one vocab chunk."""
    log_q = z_f @ W_chunk.T - t_lse.unsqueeze(1)
    q = log_q.exp()
    log_p = torch.maximum(zs_chunk, s_min) - s_lse.unsqueeze(1)
    contrib = q * (log_q - log_p)
    clipped = contrib > token_clip
    return contrib.clamp_max(token_clip).sum(dim=1), clipped.sum(dim=1).float()


@torch.compile
def _clip_qsum_chunk(
    z_f: torch.Tensor,
    W_chunk: torch.Tensor,
    zs_chunk: torch.Tensor,
    t_lse: torch.Tensor,
    s_lse: torch.Tensor,
    s_min: torch.Tensor,
    token_clip: float,
) -> torch.Tensor:
    """sum_v q(v) * 1[contrib_v <= tau] for one vocab chunk (backward pass 1)."""
    log_q = z_f @ W_chunk.T - t_lse.unsqueeze(1)
    q = log_q.exp()
    log_p = torch.maximum(zs_chunk, s_min) - s_lse.unsqueeze(1)
    keep = (q * (log_q - log_p)) <= token_clip
    return (q * keep).sum(dim=1)


@torch.compile
def _clip_bwd_chunk(
    z_f: torch.Tensor,
    W_chunk: torch.Tensor,
    zs_chunk: torch.Tensor,
    t_lse: torch.Tensor,
    s_lse: torch.Tensor,
    s_min: torch.Tensor,
    token_clip: float,
    qsum_keep: torch.Tensor,  # (N,) sum_v q_v * 1[unclipped]
    grad: torch.Tensor,       # (N,)
) -> torch.Tensor:
    """d(clipped KL)/d(student_logits) for one vocab chunk.

    d/d(zs_u) sum_v min(q_v (log q_v - log p_v), tau)
      = -q_u * 1[unclipped_u] + p_S(u) * sum_v q_v 1[unclipped_v]
    (log_prob_min_clamp treated as identity, matching the unclipped backward).
    """
    log_q = z_f @ W_chunk.T - t_lse.unsqueeze(1)
    q = log_q.exp()
    log_p = torch.maximum(zs_chunk, s_min) - s_lse.unsqueeze(1)
    keep = ((q * (log_q - log_p)) <= token_clip).to(q.dtype)
    p_S = (zs_chunk - s_lse.unsqueeze(1)).exp()
    return (p_S * qsum_keep.unsqueeze(1) - q * keep) * grad.unsqueeze(1)


def _t_lse_only(z_f, W_f, chunk_V):
    """Teacher log-partition (N,) via online logsumexp over vocab chunks."""
    N = z_f.shape[0]
    mt = torch.full((N,), float("-inf"), dtype=torch.float32, device=z_f.device)
    st = torch.zeros(N, dtype=torch.float32, device=z_f.device)
    for v in range(0, W_f.shape[0], chunk_V):
        zt = z_f @ W_f[v : v + chunk_V].T
        new_mt = torch.maximum(mt, zt.max(dim=1).values)
        st = st * (mt - new_mt).exp() + (zt - new_mt.unsqueeze(1)).exp().sum(dim=1)
        mt = new_mt
    return mt + st.log()


class _NitrobrewKLClipped(torch.autograd.Function):
    """Pointwise-clipped KL(p_T || p_S): per-(position,vocab) contributions clamped to <= tau
    before the vocab sum. Returns (kl, clipped_frac); clipped_frac is non-differentiable."""

    @staticmethod
    def forward(ctx, z, W, student_logits, chunk_V, temperature=1.0,
                log_prob_min_clamp=None, token_clip=10.0):
        N = z.shape[0]
        V = W.shape[0]
        device = z.device
        z_f = z.float()
        W_f = W.to(device=device, dtype=torch.float32)
        inv_T = 1.0 / temperature if temperature != 1.0 else 1.0
        if temperature != 1.0:
            z_f = z_f * inv_T
        s_lse = _student_lse(student_logits, chunk_V, inv_T)
        if log_prob_min_clamp is not None:
            s_min = (s_lse + log_prob_min_clamp).unsqueeze(1)
        else:
            s_min = torch.full((1, 1), float("-inf"), dtype=torch.float32, device=device)

        t_lse = _t_lse_only(z_f, W_f, chunk_V)  # pass A
        kl = torch.zeros(N, dtype=torch.float32, device=device)
        n_clipped = torch.zeros(N, dtype=torch.float32, device=device)
        for v in range(0, V, chunk_V):  # pass B
            k_c, c_c = _clip_fwd_chunk(
                z_f, W_f[v : v + chunk_V], _s_chunk(student_logits, v, v + chunk_V, inv_T),
                t_lse, s_lse, s_min, token_clip,
            )
            kl = kl + k_c
            n_clipped = n_clipped + c_c

        ctx.save_for_backward(z, W, student_logits, t_lse, s_lse)
        ctx.chunk_V = chunk_V
        ctx.temperature = temperature
        ctx.log_prob_min_clamp = log_prob_min_clamp
        ctx.token_clip = token_clip
        clipped_frac = n_clipped / float(V)
        ctx.mark_non_differentiable(clipped_frac)
        return kl, clipped_frac

    @staticmethod
    def backward(ctx, grad_output, _grad_frac):
        z, W, student_logits, t_lse, s_lse = ctx.saved_tensors
        chunk_V = ctx.chunk_V
        temperature = ctx.temperature
        token_clip = ctx.token_clip
        N, V = student_logits.shape
        device = z.device

        z_f = z.float()
        W_f = W.to(device=device, dtype=torch.float32)
        inv_T = 1.0 / temperature if temperature != 1.0 else 1.0
        if temperature != 1.0:
            z_f = z_f * inv_T
        if ctx.log_prob_min_clamp is not None:
            s_min = (s_lse + ctx.log_prob_min_clamp).unsqueeze(1)
        else:
            s_min = torch.full((1, 1), float("-inf"), dtype=torch.float32, device=device)
        grad = grad_output.float()

        qsum_keep = torch.zeros(N, dtype=torch.float32, device=device)
        for v in range(0, V, chunk_V):  # backward pass 1: masked teacher mass
            qsum_keep = qsum_keep + _clip_qsum_chunk(
                z_f, W_f[v : v + chunk_V], _s_chunk(student_logits, v, v + chunk_V, inv_T),
                t_lse, s_lse, s_min, token_clip,
            )
        # grad in the student dtype, computed fp32 per chunk — no full (N, V) fp32 tensors
        grad_s = torch.empty(N, V, dtype=student_logits.dtype, device=device)
        for v in range(0, V, chunk_V):  # backward pass 2: per-chunk gradient
            g = _clip_bwd_chunk(
                z_f, W_f[v : v + chunk_V], _s_chunk(student_logits, v, v + chunk_V, inv_T),
                t_lse, s_lse, s_min, token_clip, qsum_keep, grad,
            )
            if temperature != 1.0:
                g = g * inv_T
            grad_s[:, v : v + chunk_V] = g.to(grad_s.dtype)
        return None, None, grad_s, None, None, None, None


# ---------------------------------------------------------------------------
# TWO-SIDED fused KL (OPSD fused-kernels path): BOTH logit streams are
# reconstructed chunk-wise — student from (hidden @ lm_head.T), teacher from
# (teacher_hidden @ teacher_unembed.T) — so the (N, V) student logits NEVER
# materialize. Gradients flow into the student hidden states AND lm_head.
# Runs INSIDE the monkeypatched model forward (dense_common), where FSDP has
# the lm_head parameters gathered. token_clip=None gives the exact forward KL.
# ---------------------------------------------------------------------------


def _two_sided_lse(h_f, W, chunk_V, inv_T):
    """Online logsumexp of (h_f @ W.T) * inv_T over vocab chunks. W cast to fp32 per chunk."""
    N = h_f.shape[0]
    m = torch.full((N,), float("-inf"), dtype=torch.float32, device=h_f.device)
    acc = torch.zeros(N, dtype=torch.float32, device=h_f.device)
    for v in range(0, W.shape[0], chunk_V):
        c = h_f @ W[v : v + chunk_V].float().T
        if inv_T != 1.0:
            c = c * inv_T
        new_m = torch.maximum(m, c.max(dim=1).values)
        acc = acc * (m - new_m).exp() + (c - new_m.unsqueeze(1)).exp().sum(dim=1)
        m = new_m
    return m + acc.log()


class _NitrobrewKLTwoSided(torch.autograd.Function):
    """Pointwise-clipped forward KL(p_T || p_S) with chunked logit reconstruction on BOTH sides.

    forward(h, W_s, z, W_t): h=(N, D_s) student hidden (grad), W_s=(V, D_s) student lm_head
    (grad), z=(N, D_t) frozen teacher hidden, W_t=(V, D_t) frozen teacher unembed.
    Returns (kl, clipped_frac) both (N,) float32; clipped_frac is non-differentiable and all-zero
    when token_clip is None (exact unclipped KL)."""

    @staticmethod
    def forward(ctx, h, W_s, z, W_t, chunk_V, temperature=1.0,
                log_prob_min_clamp=None, token_clip=None):
        N = h.shape[0]
        V = W_s.shape[0]
        device = h.device
        inv_T = 1.0 / temperature if temperature != 1.0 else 1.0
        h_f = h.float()
        z_f = z.float()

        s_lse = _two_sided_lse(h_f, W_s, chunk_V, inv_T)
        t_lse = _two_sided_lse(z_f, W_t, chunk_V, inv_T)
        if log_prob_min_clamp is not None:
            s_min = (s_lse + log_prob_min_clamp).unsqueeze(1)
        else:
            s_min = torch.full((1, 1), float("-inf"), dtype=torch.float32, device=device)

        kl = torch.zeros(N, dtype=torch.float32, device=device)
        n_clipped = torch.zeros(N, dtype=torch.float32, device=device)
        for v in range(0, V, chunk_V):
            zt = z_f @ W_t[v : v + chunk_V].float().T
            zs = h_f @ W_s[v : v + chunk_V].float().T
            if inv_T != 1.0:
                zt = zt * inv_T
                zs = zs * inv_T
            log_q = zt - t_lse.unsqueeze(1)
            q = log_q.exp()
            log_p = torch.maximum(zs, s_min) - s_lse.unsqueeze(1)
            contrib = q * (log_q - log_p)
            if token_clip is not None:
                n_clipped = n_clipped + (contrib > token_clip).sum(dim=1).float()
                contrib = contrib.clamp_max(token_clip)
            kl = kl + contrib.sum(dim=1)

        ctx.save_for_backward(h, W_s, z, W_t, t_lse, s_lse)
        ctx.chunk_V = chunk_V
        ctx.temperature = temperature
        ctx.log_prob_min_clamp = log_prob_min_clamp
        ctx.token_clip = token_clip
        clipped_frac = n_clipped / float(V)
        ctx.mark_non_differentiable(clipped_frac)
        return kl, clipped_frac

    @staticmethod
    def backward(ctx, grad_output, _grad_frac):
        h, W_s, z, W_t, t_lse, s_lse = ctx.saved_tensors
        chunk_V = ctx.chunk_V
        temperature = ctx.temperature
        token_clip = ctx.token_clip
        N = h.shape[0]
        V = W_s.shape[0]
        device = h.device
        inv_T = 1.0 / temperature if temperature != 1.0 else 1.0
        h_f = h.float()
        z_f = z.float()
        if ctx.log_prob_min_clamp is not None:
            s_min = (s_lse + ctx.log_prob_min_clamp).unsqueeze(1)
        else:
            s_min = torch.full((1, 1), float("-inf"), dtype=torch.float32, device=device)
        grad = grad_output.float()

        if token_clip is not None:
            qsum_keep = torch.zeros(N, dtype=torch.float32, device=device)
            for v in range(0, V, chunk_V):  # pass 1: teacher mass on unclipped entries
                zt = z_f @ W_t[v : v + chunk_V].float().T
                zs = h_f @ W_s[v : v + chunk_V].float().T
                if inv_T != 1.0:
                    zt = zt * inv_T
                    zs = zs * inv_T
                log_q = zt - t_lse.unsqueeze(1)
                q = log_q.exp()
                log_p = torch.maximum(zs, s_min) - s_lse.unsqueeze(1)
                keep = (q * (log_q - log_p)) <= token_clip
                qsum_keep = qsum_keep + (q * keep).sum(dim=1)
        else:
            qsum_keep = torch.ones(N, dtype=torch.float32, device=device)

        grad_h = torch.zeros(N, h.shape[1], dtype=torch.float32, device=device)
        grad_Ws = torch.zeros(V, W_s.shape[1], dtype=torch.float32, device=device)
        for v in range(0, V, chunk_V):  # pass 2: G = (p_S*qsum_keep - q*keep) * grad, contracted
            Ws_chunk = W_s[v : v + chunk_V].float()
            zt = z_f @ W_t[v : v + chunk_V].float().T
            zs = h_f @ Ws_chunk.T
            if inv_T != 1.0:
                zt = zt * inv_T
                zs = zs * inv_T
            q = (zt - t_lse.unsqueeze(1)).exp()
            p_S = (zs - s_lse.unsqueeze(1)).exp()
            if token_clip is not None:
                log_q = zt - t_lse.unsqueeze(1)
                log_p = torch.maximum(zs, s_min) - s_lse.unsqueeze(1)
                keep = ((q * (log_q - log_p)) <= token_clip).to(q.dtype)
            else:
                keep = 1.0
            G = (p_S * qsum_keep.unsqueeze(1) - q * keep) * grad.unsqueeze(1)
            if inv_T != 1.0:
                G = G * inv_T  # chain rule through logits = (h @ W.T) / T
            grad_h = grad_h + G @ Ws_chunk
            grad_Ws[v : v + chunk_V] = G.T @ h_f

        return (
            grad_h.to(h.dtype),
            grad_Ws.to(W_s.dtype),
            None, None, None, None, None, None,
        )


@dataclass
class FusedLinearAux:
    """Per-token distillation outputs produced inside the fused model forward.

    Field names match what transformer_impl.prepare_model_outputs extracts; fields left None
    are skipped by the (None-tolerant) consumer loop."""

    distillation_losses: Optional[torch.Tensor] = None      # (1, N) float32
    student_mass: Optional[torch.Tensor] = None
    teacher_mass: Optional[torch.Tensor] = None
    distillation_clipped_frac: Optional[torch.Tensor] = None  # (1, N) float32


def compute_fused_nitrobrew_aux(
    hidden_states: torch.Tensor,     # (1, N, D_s) packed student hidden (grad)
    lm_head_weight: torch.Tensor,    # (V, D_s) student unembed (grad)
    teacher_hidden: torch.Tensor,    # (1, N, D_t) frozen teacher hidden, position-aligned
    teacher_unembed: torch.Tensor,   # (V, D_t) frozen teacher unembed
    response_index: Optional[torch.Tensor] = None,  # (M,) positions to compute; None = all
    token_clip: Optional[float] = None,
    temperature: float = 1.0,
    log_prob_min_clamp: Optional[float] = None,
    chunk_V: int = _CHUNK_V,
) -> FusedLinearAux:
    """Fused-forward OPSD KL: both logit streams chunk-reconstructed, (N, V) never materialized.

    Called from the monkeypatched CausalLM forward (dense_common), where FSDP has the lm_head
    parameters gathered. When response_index is given, only those positions are computed
    (the ones no_padding_2_padding later reads); the rest are exact zeros."""
    h_all = hidden_states.view(-1, hidden_states.shape[-1])
    z_all = teacher_hidden.view(-1, teacher_hidden.shape[-1])
    assert h_all.shape[0] == z_all.shape[0], (
        f"student hidden {h_all.shape[0]} vs teacher hidden {z_all.shape[0]} position mismatch"
    )
    N = h_all.shape[0]

    if response_index is not None:
        if response_index.numel() == 0:
            zeros = torch.zeros(1, N, dtype=torch.float32, device=h_all.device)
            return FusedLinearAux(
                distillation_losses=zeros,
                distillation_clipped_frac=zeros.clone() if token_clip is not None else None,
            )
        h_run = h_all.index_select(0, response_index)
        z_run = z_all.index_select(0, response_index)
    else:
        h_run, z_run = h_all, z_all

    kl, clipped_frac = _NitrobrewKLTwoSided.apply(
        h_run, lm_head_weight, z_run, teacher_unembed, chunk_V,
        temperature, log_prob_min_clamp, token_clip,
    )
    if response_index is not None:
        kl = torch.zeros(N, dtype=kl.dtype, device=kl.device).index_copy(0, response_index, kl)
        clipped_frac = torch.zeros(N, dtype=clipped_frac.dtype, device=clipped_frac.device).index_copy(
            0, response_index, clipped_frac
        )
    return FusedLinearAux(
        distillation_losses=kl.view(1, N),
        distillation_clipped_frac=clipped_frac.view(1, N) if token_clip is not None else None,
    )


# ---------------------------------------------------------------------------
# Backward: standard d(KL)/d(zs_v) = p_S(v) - p_T(v)
# ---------------------------------------------------------------------------


@torch.compile
def _bwd_chunk(
    z_f: torch.Tensor,       # (N, D_t) float32
    W_chunk: torch.Tensor,   # (C, D_t) float32
    zs_chunk: torch.Tensor,  # (N, C)   float32
    t_lse: torch.Tensor,     # (N,)
    s_lse: torch.Tensor,     # (N,)
    grad: torch.Tensor,      # (N,)
) -> torch.Tensor:
    """d(KL)/d(student_logits) for one vocab chunk. Returns (N, C)."""
    zt = z_f @ W_chunk.T
    p_T = (zt - t_lse.unsqueeze(1)).exp()
    p_S = (zs_chunk - s_lse.unsqueeze(1)).exp()
    return (p_S - p_T) * grad.unsqueeze(1)


# ---------------------------------------------------------------------------
# autograd.Function -- wires forward & backward
# ---------------------------------------------------------------------------


class _NitrobrewKL(torch.autograd.Function):
    """KL(p_T || p_S) with chunked forward and backward over vocab axis."""

    @staticmethod
    def forward(ctx, z, W, student_logits, chunk_V, temperature=1.0,
                log_prob_min_clamp=None):
        kl, t_lse = _chunked_kl_forward(
            z, W, student_logits, chunk_V, temperature, log_prob_min_clamp,
        )
        inv_T = 1.0 / temperature if temperature != 1.0 else 1.0
        s_lse = _student_lse(student_logits, chunk_V, inv_T)
        ctx.save_for_backward(z, W, student_logits, t_lse, s_lse)
        ctx.chunk_V = chunk_V
        ctx.temperature = temperature
        return kl

    @staticmethod
    def backward(ctx, grad_output):
        z, W, student_logits, t_lse, s_lse = ctx.saved_tensors
        chunk_V = ctx.chunk_V
        temperature = ctx.temperature
        N, V = student_logits.shape
        device = z.device

        z_f = z.float()
        W_f = W.to(device=device, dtype=torch.float32)
        inv_T = 1.0 / temperature if temperature != 1.0 else 1.0
        if temperature != 1.0:
            z_f = z_f * inv_T
        grad = grad_output.float()

        # grad in the student dtype, computed fp32 per chunk — no full (N, V) fp32 tensors
        grad_s = torch.empty(N, V, dtype=student_logits.dtype, device=device)
        for v in range(0, V, chunk_V):
            g = _bwd_chunk(
                z_f, W_f[v : v + chunk_V], _s_chunk(student_logits, v, v + chunk_V, inv_T),
                t_lse, s_lse, grad,
            )
            if temperature != 1.0:
                g = g * inv_T
            grad_s[:, v : v + chunk_V] = g.to(grad_s.dtype)

        return None, None, grad_s, None, None, None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _response_position_index(data, T_sp: int, device) -> torch.Tensor | None:
    """Global rmpad indices of the positions that PREDICT response tokens —
    `[cu[i]+prompt_len_i-1 : cu[i]+prompt_len_i-1+resp_len_i)` per sample. These are the ONLY
    positions `no_padding_2_padding` later slices back out ([seq_offset-resp_len-1 : seq_offset-1]),
    so restricting the KL to them changes nothing downstream while skipping prompt positions and
    (via an all-zero response_mask row) OPSD-skipped samples entirely.

    Returns None when the layout doesn't allow it (missing keys, non-nested input_ids, Ulysses
    SP>1 where positions are sequence-sharded, or an offsets/T_sp mismatch)."""
    if data is None:
        return None
    try:
        if get_ulysses_sequence_parallel_world_size() > 1:
            return None
    except Exception:
        return None
    input_ids = data.get("input_ids", None)
    prompts = data.get("prompts", None)
    attention_mask = data.get("attention_mask", None)
    if input_ids is None or prompts is None or attention_mask is None or not input_ids.is_nested:
        return None
    cu = input_ids.offsets()
    if int(cu[-1]) != T_sp:
        return None
    prompt_max_len = prompts.shape[1]
    prompt_lens = attention_mask[:, :prompt_max_len].sum(dim=1)
    resp_lens = attention_mask[:, prompt_max_len:].sum(dim=1)
    response_mask = data.get("response_mask", None)
    spans = []
    for i in range(prompt_lens.shape[0]):
        if response_mask is not None and not bool(response_mask[i].any()):
            continue  # fully masked sample (e.g. OPSD skipped group): its loss is discarded anyway
        start = int(cu[i]) + int(prompt_lens[i]) - 1
        spans.append(torch.arange(start, start + int(resp_lens[i]), device=device))
    if not spans:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.cat(spans)


def compute_nitrobrew_kl(
    student_logits: torch.Tensor,
    teacher_hidden_states: torch.Tensor,
    teacher_unembed: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
    data=None,
) -> dict[str, torch.Tensor]:
    """Compute Nitrobrew forward KL loss KL(p_T || p_S) for the FSDP path.

    If distillation_loss.token_clip is set, uses the OPSD pointwise-clipped variant
    (per-(position,vocab) contribution clamped to <= tau before the vocab sum) and
    additionally returns per-position "distillation_clipped_frac".

    When `data` allows it, the kernel runs ONLY on the response-predicting positions (the ones
    `no_padding_2_padding` later reads); prompt positions and fully-masked samples get exact
    zeros without touching the vocab axis."""
    z, T_sp = _unpack_hidden_states(teacher_hidden_states, student_logits)
    z_flat = z.view(T_sp, -1)
    s_flat = student_logits.view(T_sp, -1)

    idx = _response_position_index(data, T_sp, s_flat.device)
    if idx is not None:
        z_run = z_flat[idx]
        s_run = s_flat.index_select(0, idx)
    else:
        z_run, s_run = z_flat, s_flat

    loss_config = config.distillation_loss
    token_clip = getattr(loss_config, "token_clip", None)
    clipped_frac = None
    if idx is not None and idx.numel() == 0:
        per_token_kl = s_flat.sum(dim=-1, dtype=torch.float32) * 0.0  # zeros, keeps grad graph
        if token_clip is not None:
            clipped_frac = torch.zeros(T_sp, dtype=torch.float32, device=s_flat.device)
    elif token_clip is not None:
        per_token_kl, clipped_frac = _NitrobrewKLClipped.apply(
            z_run, teacher_unembed, s_run, _CHUNK_V,
            loss_config.kd_temperature, loss_config.log_prob_min_clamp, token_clip,
        )
    else:
        per_token_kl = _NitrobrewKL.apply(
            z_run, teacher_unembed, s_run, _CHUNK_V,
            loss_config.kd_temperature, loss_config.log_prob_min_clamp,
        )

    if idx is not None and idx.numel() > 0:
        per_token_kl = torch.zeros(
            T_sp, dtype=per_token_kl.dtype, device=per_token_kl.device
        ).index_copy(0, idx, per_token_kl)
        if clipped_frac is not None:
            clipped_frac = torch.zeros(
                T_sp, dtype=clipped_frac.dtype, device=clipped_frac.device
            ).index_copy(0, idx, clipped_frac)

    out = {"distillation_losses": per_token_kl.view(1, T_sp)}
    if clipped_frac is not None:
        out["distillation_clipped_frac"] = clipped_frac.view(1, T_sp)
    return out


# ---------------------------------------------------------------------------
# Reverse KL: KL(p_S || p_T)
# ---------------------------------------------------------------------------


@torch.compile
def _rev_fwd_chunk(
    z_f: torch.Tensor,       # (N, D_t)
    W_chunk: torch.Tensor,   # (C, D_t)
    zs_chunk: torch.Tensor,  # (N, C)
    s_lse: torch.Tensor,     # (N,) pre-computed student log-partition
    mt: torch.Tensor,        # (N,) running teacher max
    st: torch.Tensor,        # (N,) running sum exp(z_t - mt)
    ut: torch.Tensor,        # (N,) running sum p_S * z_t
    et: torch.Tensor,        # (N,) running sum p_S * z_s
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Online-softmax update for one teacher vocab chunk of reverse KL."""
    zt = z_f @ W_chunk.T

    tile_mt = zt.max(dim=1).values
    new_mt = torch.maximum(mt, tile_mt)
    alpha = (mt - new_mt).exp()
    st = st * alpha + (zt - new_mt.unsqueeze(1)).exp().sum(dim=1)

    ps_chunk = (zs_chunk - s_lse.unsqueeze(1)).exp()
    ut = ut + (ps_chunk * zt).sum(dim=1)
    et = et + (ps_chunk * zs_chunk).sum(dim=1)

    return new_mt, st, ut, et


def _chunked_reverse_kl_forward(
    z: torch.Tensor,              # (N, D_t)
    W: torch.Tensor,              # (V, D_t)
    student_logits: torch.Tensor, # (N, V)
    chunk_V: int = _CHUNK_V,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Chunked reverse KL forward. Returns (kl, t_lse, s_lse) all (N,) float32."""
    N = z.shape[0]
    V = W.shape[0]
    device = z.device

    z_f = z.float()
    W_f = W.to(device=device, dtype=torch.float32)
    s_f = student_logits.float()

    if temperature != 1.0:
        inv_T = 1.0 / temperature
        z_f = z_f * inv_T
        s_f = s_f * inv_T

    s_lse = s_f.logsumexp(dim=-1)

    mt = torch.full((N,), float("-inf"), dtype=torch.float32, device=device)
    st = torch.zeros(N, dtype=torch.float32, device=device)
    ut = torch.zeros(N, dtype=torch.float32, device=device)
    et = torch.zeros(N, dtype=torch.float32, device=device)

    for v in range(0, V, chunk_V):
        mt, st, ut, et = _rev_fwd_chunk(
            z_f, W_f[v : v + chunk_V], s_f[:, v : v + chunk_V],
            s_lse, mt, st, ut, et,
        )

    t_lse = mt + st.log()
    kl = et - ut - s_lse + t_lse
    return kl, t_lse, s_lse


@torch.compile
def _rev_bwd_chunk(
    z_f: torch.Tensor,       # (N, D_t)
    W_chunk: torch.Tensor,   # (C, D_t)
    zs_chunk: torch.Tensor,  # (N, C)
    t_lse: torch.Tensor,     # (N,)
    s_lse: torch.Tensor,     # (N,)
    kl: torch.Tensor,        # (N,)
    grad: torch.Tensor,      # (N,)
) -> torch.Tensor:
    """dKL(p_S||p_T)/dz_s(v) = p_S(v) * [log(p_S(v)/p_T(v)) - KL]. Returns (N, C)."""
    zt = z_f @ W_chunk.T
    log_ps = zs_chunk - s_lse.unsqueeze(1)
    log_pt = zt - t_lse.unsqueeze(1)
    ps = log_ps.exp()
    return ps * (log_ps - log_pt - kl.unsqueeze(1)) * grad.unsqueeze(1)


class _NitrobrewReverseKL(torch.autograd.Function):
    """KL(p_S || p_T) with chunked forward and backward over vocab axis."""

    @staticmethod
    def forward(ctx, z, W, student_logits, chunk_V, temperature=1.0):
        kl, t_lse, s_lse = _chunked_reverse_kl_forward(z, W, student_logits, chunk_V, temperature)
        ctx.save_for_backward(z, W, student_logits, t_lse, s_lse, kl)
        ctx.chunk_V = chunk_V
        ctx.temperature = temperature
        return kl

    @staticmethod
    def backward(ctx, grad_output):
        z, W, student_logits, t_lse, s_lse, kl = ctx.saved_tensors
        chunk_V = ctx.chunk_V
        temperature = ctx.temperature
        N, V = student_logits.shape

        z_f = z.float()
        W_f = W.to(device=z.device, dtype=torch.float32)
        s_f = student_logits.float()
        if temperature != 1.0:
            inv_T = 1.0 / temperature
            z_f = z_f * inv_T
            s_f = s_f * inv_T
        grad = grad_output.float()

        grad_s = torch.empty(N, V, dtype=torch.float32, device=z.device)
        for v in range(0, V, chunk_V):
            grad_s[:, v : v + chunk_V] = _rev_bwd_chunk(
                z_f, W_f[v : v + chunk_V], s_f[:, v : v + chunk_V],
                t_lse, s_lse, kl, grad,
            )

        if temperature != 1.0:
            grad_s = grad_s / temperature

        return None, None, grad_s.to(student_logits.dtype), None, None


def compute_nitrobrew_reverse_kl(
    student_logits: torch.Tensor,
    teacher_hidden_states: torch.Tensor,
    teacher_unembed: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
    data=None,  # accepted for signature parity with compute_nitrobrew_kl; slicing not implemented
) -> dict[str, torch.Tensor]:
    """Compute Nitrobrew reverse KL loss KL(p_S || p_T) for the FSDP path."""
    z, T_sp = _unpack_hidden_states(teacher_hidden_states, student_logits)
    z_flat = z.view(T_sp, -1)
    s_flat = student_logits.view(T_sp, -1)

    loss_config = config.distillation_loss
    per_token_kl = _NitrobrewReverseKL.apply(
        z_flat, teacher_unembed, s_flat, _CHUNK_V, loss_config.kd_temperature,
    )
    return {"distillation_losses": per_token_kl.view(1, T_sp)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unpack_hidden_states(
    teacher_hidden_states: torch.Tensor,
    student_logits: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Convert nested teacher hidden states to flat packed tensor. Returns (z, T_sp)."""
    if teacher_hidden_states.is_nested:
        z = teacher_hidden_states.values().unsqueeze(0)
    else:
        z = teacher_hidden_states.flatten(0, 1).unsqueeze(0)

    if get_ulysses_sequence_parallel_world_size() > 1:
        z = slice_input_tensor(z, dim=1)

    assert z.shape[:2] == student_logits.shape[:2], (
        f"Shape mismatch: z {z.shape[:2]} vs student_logits {student_logits.shape[:2]}"
    )
    return z, z.shape[1]
