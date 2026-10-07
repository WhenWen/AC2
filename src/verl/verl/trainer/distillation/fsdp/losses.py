# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
# Modified by the AC2 authors (2026) to implement AC2; see src/verl/README.md for the list of changed files.


import torch
import torch.nn.functional as F

from verl.utils.ulysses import (
    get_ulysses_sequence_parallel_world_size,
    slice_input_tensor,
)
from verl.workers.config import DistillationConfig, DistillationLossConfig


def kl_divergence(log_q: torch.Tensor, log_p: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence between two distributions given their log probabilities."""
    log_p = log_p.float()
    log_q = log_q.float()
    p = log_p.exp()
    kld = p * (log_p - log_q)
    return kld.sum(dim=-1)


def compute_forward_kl_topk(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute forward KL distillation loss using top-k log probabilities.

    Args:
        student_logits: (bsz, seqlen/sp_size, vocab_size).
        teacher_topk_log_probs: (bsz, seqlen, topk).
        teacher_topk_ids: (bsz, seqlen, topk).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - distillation_losses: (bsz, seqlen/sp_size)
    - student_mass: (bsz, seqlen/sp_size)
    - teacher_mass: (bsz, seqlen/sp_size)
    """
    assert teacher_topk_log_probs.is_nested and teacher_topk_ids.is_nested
    teacher_topk_log_probs = teacher_topk_log_probs.values().unsqueeze(0)  # (1, total_nnz, topk)
    teacher_topk_ids = teacher_topk_ids.values().unsqueeze(0)  # (1, total_nnz, topk)

    # 1. split across sp groups (bsz, seqlen, topk) => (bsz, seqlen/sp_size, topk)
    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1)
        teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1)
    assert teacher_topk_log_probs.shape[:2] == teacher_topk_ids.shape[:2] == student_logits.shape[:2]

    # 2. compute token-wise KL divergence across sp groups
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    student_topk_ids = torch.topk(student_log_probs, k=teacher_topk_ids.shape[-1], dim=-1).indices
    student_topk_log_probs = torch.gather(student_log_probs, dim=-1, index=teacher_topk_ids)
    student_mass = student_topk_log_probs.exp().sum(dim=-1)
    teacher_mass = teacher_topk_log_probs.exp().sum(dim=-1)
    loss_config: DistillationLossConfig = config.distillation_loss
    if loss_config.log_prob_min_clamp is not None:
        student_topk_log_probs = student_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
        teacher_topk_log_probs = teacher_topk_log_probs.clamp_min(loss_config.log_prob_min_clamp)
    distillation_losses = kl_divergence(log_q=student_topk_log_probs, log_p=teacher_topk_log_probs)

    # Diagnostics for tracking teacher/student top-k overlap in OPD, following
    # "Rethinking On-Policy Distillation of Large Language Models" (arXiv:2604.13016).
    overlap_mask = (teacher_topk_ids.unsqueeze(-1) == student_topk_ids.unsqueeze(-2)).any(dim=-1)
    overlap_count = overlap_mask.sum(dim=-1)
    token_kl = teacher_topk_log_probs.exp() * (teacher_topk_log_probs - student_topk_log_probs)
    overlap_token_advantage_sum = (-token_kl * overlap_mask).sum(dim=-1)
    overlap_token_advantage = overlap_token_advantage_sum / overlap_count.clamp_min(1)
    overlap_token_advantage = torch.where(
        overlap_count > 0, overlap_token_advantage, torch.zeros_like(overlap_token_advantage)
    )

    return {
        "distillation_losses": distillation_losses,
        "student_mass": student_mass,
        "teacher_mass": teacher_mass,
        "overlap_count": overlap_count,
        "overlap_token_advantage": overlap_token_advantage,
    }


def _slice_nested_topk(tensor: torch.Tensor) -> torch.Tensor:
    """(bsz, seqlen, K) nested -> (1, total_nnz[/sp], K) dense, sliced for Ulysses SP."""
    assert tensor.is_nested
    tensor = tensor.values().unsqueeze(0)  # (1, total_nnz, K)
    if get_ulysses_sequence_parallel_world_size() > 1:
        tensor = slice_input_tensor(tensor, dim=1)
    return tensor


def _lookup_scorer_log_probs(
    candidate_ids: torch.Tensor,
    scorer_topk_ids: torch.Tensor,
    scorer_topk_log_probs: torch.Tensor,
    floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Look up each candidate id in a scorer's rank-ordered top-K list.

    Args:
        candidate_ids: (1, S, k) student-support token ids.
        scorer_topk_ids: (1, S, K) scorer top-K token ids (unique per position).
        scorer_topk_log_probs: (1, S, K) matching log probabilities.
        floor: log-prob assigned to candidates MISSING from the scorer's list; found values are
            also clamped to >= floor so present-vs-barely-present stays continuous.

    Returns:
    - log_q: (1, S, k) float32 scorer log-probs on the candidate support.
    - found: (1, S, k) bool, True where the candidate was in the scorer's list.
    """
    match = candidate_ids.unsqueeze(-1) == scorer_topk_ids.unsqueeze(-2)  # (1, S, k, K)
    found = match.any(dim=-1)
    matched = (match.to(scorer_topk_log_probs.dtype) * scorer_topk_log_probs.unsqueeze(-2)).sum(dim=-1)
    log_q = torch.where(found, matched, torch.full_like(matched, floor)).clamp_min(floor)
    return log_q.float(), found


def compute_direct_opd(
    student_logits: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    teacher_ref_topk_log_probs: torch.Tensor,
    teacher_ref_topk_ids: torch.Tensor,
    config: DistillationConfig,
    data_format: str,
) -> dict[str, torch.Tensor]:
    """Direct On-Policy Distillation loss (arXiv:2607.05394, Eq. 6/10-12).

    Per position t on the student rollout, with S_t = the student's own top-k_student ids:
        r_t(v)   = log q_T(v|s_t) - log q_ref(v|s_t)          (scorer log-ratio reward)
        pbar_t   = softmax over S_t of the student logits      (renormalized support, DETACHED)
        A_t(v)   = stop_gradient( pbar_t(v) * r_t(v) )
        loss_t   = - sum_{v in S_t} A_t(v) * log pi_theta(v|s_t)
    Scorer log-probs for ids missing from that scorer's top-K list are floored at
    log_prob_min_clamp. Gradient flows ONLY through log pi_theta (the full-vocab log_softmax).

    Args:
        student_logits: (bsz, seqlen/sp_size, vocab_size).
        teacher_topk_log_probs / teacher_topk_ids: (bsz, seqlen, K) nested, teacher scorer.
        teacher_ref_topk_log_probs / teacher_ref_topk_ids: (bsz, seqlen, K) nested, reference scorer.
        data_format: "thd" or "bshd" (same handling as compute_forward_kl_topk).

    Returns dict of (bsz, seqlen/sp_size) tensors:
    - distillation_losses: per-position loss (carries the gradient).
    - dopd_rbar: per-position mean over candidates of pbar*r (detached; batch mean -> dopd/rbar).
    - dopd_coverage_teacher / dopd_coverage_ref: fraction of S_t found in each scorer's top-K.
    - dopd_overlap_teacher: |S_t ∩ teacher top-k_student| / k_student (scorer lists are rank-ordered).
    """
    loss_config: DistillationLossConfig = config.distillation_loss
    assert loss_config.log_prob_min_clamp is not None, "direct_opd requires log_prob_min_clamp"
    floor = float(loss_config.log_prob_min_clamp)
    k_student = int(loss_config.direct_opd_student_topk)

    # 1. nested (bsz, seqlen, K) -> dense (1, total_nnz/sp, K), sliced across sp groups
    teacher_topk_log_probs = _slice_nested_topk(teacher_topk_log_probs)
    teacher_topk_ids = _slice_nested_topk(teacher_topk_ids)
    teacher_ref_topk_log_probs = _slice_nested_topk(teacher_ref_topk_log_probs)
    teacher_ref_topk_ids = _slice_nested_topk(teacher_ref_topk_ids)
    assert (
        teacher_topk_log_probs.shape[:2]
        == teacher_topk_ids.shape[:2]
        == teacher_ref_topk_log_probs.shape[:2]
        == teacher_ref_topk_ids.shape[:2]
        == student_logits.shape[:2]
    )

    # 2. student support: own top-k_student ids + renormalized (detached) weights
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    student_topk_log_probs, student_topk_ids = torch.topk(student_log_probs, k=k_student, dim=-1)
    student_topk_log_probs = student_topk_log_probs.float()  # keeps grad; fp32 for the loss
    # softmax of the selected log-probs == pi(v) / sum_{u in S_t} pi(u)
    pbar = torch.softmax(student_topk_log_probs.detach(), dim=-1)

    # 3. scorer log-ratio reward on the student support (missing ids floored)
    log_q_teacher, found_teacher = _lookup_scorer_log_probs(
        student_topk_ids, teacher_topk_ids, teacher_topk_log_probs, floor
    )
    log_q_ref, found_ref = _lookup_scorer_log_probs(
        student_topk_ids, teacher_ref_topk_ids, teacher_ref_topk_log_probs, floor
    )
    log_ratio = log_q_teacher - log_q_ref

    # 4. detached per-candidate advantage; loss carries gradient only via log pi_theta
    advantage = (pbar * log_ratio).detach()
    distillation_losses = -(advantage * student_topk_log_probs).sum(dim=-1)

    # 5. diagnostics (all per-position, detached)
    dopd_rbar = advantage.mean(dim=-1)  # mean over candidates; masked batch mean = paper's rbar_m
    dopd_coverage_teacher = found_teacher.float().mean(dim=-1)
    dopd_coverage_ref = found_ref.float().mean(dim=-1)
    # Scorer lists are rank-ordered (extract_prompt_logprobs), so [:k_student] is the teacher's own top-k.
    teacher_head_ids = teacher_topk_ids[..., : min(k_student, teacher_topk_ids.shape[-1])]
    overlap = (student_topk_ids.unsqueeze(-1) == teacher_head_ids.unsqueeze(-2)).any(dim=-1)
    dopd_overlap_teacher = overlap.float().mean(dim=-1)

    return {
        "distillation_losses": distillation_losses,
        "dopd_rbar": dopd_rbar,
        "dopd_coverage_teacher": dopd_coverage_teacher,
        "dopd_coverage_ref": dopd_coverage_ref,
        "dopd_overlap_teacher": dopd_overlap_teacher,
    }
