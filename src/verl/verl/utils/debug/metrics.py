# Copyright 2025 Individual Contributor: TomQunChaoA
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

import logging

import torch

from verl.protocol import DataProto

logger = logging.getLogger(__file__)


def calculate_token_list_diff(tensor1: torch.Tensor, tensor2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # verify inputs
    if tensor1.numel() == 0 or tensor2.numel() == 0:
        return torch.zeros(tensor1.shape[0], dtype=torch.long, device=tensor1.device)
    if tensor1.shape != tensor2.shape or mask.shape != tensor1.shape or mask.shape != tensor2.shape:
        print(
            f"<WARN> dim of tensor1, tensor2, mask is not equal, {(tensor1.shape)=},{(tensor2.shape)=}, {(mask.shape)=}"
        )
        return torch.ones_like(tensor1)
    # transfer to same device
    if tensor2.device != tensor1.device:
        tensor2 = tensor2.to(tensor1.device)
    if mask.device != tensor1.device:
        mask = mask.to(tensor1.device)

    # calculate diff
    diff_mask = tensor1 != tensor2

    valid_diff_mask = diff_mask & (mask == 1)

    diff_counts = valid_diff_mask.sum(dim=1)

    return diff_counts


def pearson_correlation_coefficient(tensor1: torch.Tensor, tensor2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # implemention of https://arxiv.org/pdf/2506.13585
    if tensor1.shape != tensor2.shape or mask.shape != tensor1.shape or mask.shape != tensor2.shape:
        return 0
    mt1 = torch.masked_select(tensor1, mask)
    mt2 = torch.masked_select(tensor2, mask)
    result = torch.corrcoef(torch.stack([mt1, mt2], dim=0))
    return result[0][1].detach().item()


def calculate_log_prob_diff(log_probs1: torch.Tensor, log_probs2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    full_diff = torch.abs(log_probs1 - log_probs2)
    return torch.masked_select(full_diff, mask)


def calculate_debug_metrics(data: DataProto) -> dict:
    """
    calculate rollout vs actor logprobs diff, for debugging purpose

    Args:
        data: DataProto
            the data batch to calculate
            rollout_log_probs: log_probs record when rollout forward tokens
            old_log_probs(actor log probs): log_probs record when actor forward tokens
            loss_mask or attention_mask: to mask unrelated token
            responses: the response tokens, for calculating size
    Returns:
        dict: metrics
            "training/rollout_probs_diff_valid": 1->input is valid, 0->input is invalid
            "training/rollout_probs_diff_max": max value of logprob diff of rollout vs. actor
            "training/rollout_probs_diff_mean": mean value of logprob diff of rollout vs. actor
            "training/rollout_probs_diff_std": std value of logprob diff of rollout vs. actor
            "training/rollout_actor_probs_pearson_corr": logprob's pearson corrcoef of rollout vs. actor, reference to https://arxiv.org/pdf/2506.13585
    """

    rollout_old_log_probs = data.batch["rollout_log_probs"]
    actor_old_log_probs = data.batch["old_log_probs"]
    if "response_mask" in data.batch:
        logger.debug("response mask found, use it to mask log probs")
        log_prob_mask = data.batch["response_mask"]
    elif "attention_mask" in data.batch:
        log_prob_mask = data.batch["attention_mask"]
    else:
        logger.warning(f"no mask info found, use all log probs, {(data.batch.keys())=}")
        log_prob_mask = torch.ones_like(rollout_old_log_probs)
    responses = data.batch["responses"]
    response_length = responses.size(1)

    response_mask = log_prob_mask[:, -response_length:]
    # calculate pearson corrcoef
    actor_probs = torch.exp(actor_old_log_probs)
    rollout_probs = torch.exp(rollout_old_log_probs)
    response_mask_bool = response_mask.bool()

    # check if there are any valid tokens before computing metrics
    if not response_mask_bool.any():
        logger.warning("response_mask is all False, returning default metrics")
        return {
            "training/rollout_probs_diff_valid": 0,
            "training/rollout_probs_diff_max": float("nan"),
            "training/rollout_probs_diff_mean": float("nan"),
            "training/rollout_probs_diff_std": float("nan"),
            "training/rollout_actor_probs_pearson_corr": float("nan"),
        }

    pearson_corrcoef = pearson_correlation_coefficient(actor_probs, rollout_probs, response_mask_bool)
    rollout_probs_diff = calculate_log_prob_diff(actor_probs, rollout_probs, response_mask_bool)
    return {
        "training/rollout_probs_diff_valid": 1,
        "training/rollout_probs_diff_max": torch.max(rollout_probs_diff).detach().item(),
        "training/rollout_probs_diff_mean": torch.mean(rollout_probs_diff).detach().item(),
        "training/rollout_probs_diff_std": torch.std(rollout_probs_diff).detach().item(),
        "training/rollout_actor_probs_pearson_corr": pearson_corrcoef,
    }


def calculate_train_inference_mismatch_metrics(data: DataProto) -> dict:
    """Train/inference logprob mismatch diagnostics (self-play addition).

    Extends :func:`calculate_debug_metrics` with the tail- and distribution-level
    signals that matter for judging whether the GRPO importance ratio is well
    behaved (cf. maxtext's ``docs/train_inference_mismatch.md``). It compares the
    *same* generated tokens under two distributions:

    - **trainer** = ``old_log_probs`` — the actor's recomputed logprobs. verl scales
      the actor logits by the rollout sampling temperature, so this is the
      temperature-matched ("sampled") comparison to the rollout.
    - **inference** = ``rollout_log_probs`` — the vLLM per-token logprobs captured at
      generation (requires ``actor_rollout_ref.rollout.calculate_log_probs=True``).

    All reductions are masked over completion tokens (``response_mask``). Emitted
    keys (logged to W&B as-is by the trainer):

    - ``train_inference/old_available`` — 1 if real rollout logprobs were present and
      any token was unmasked, else 0. A "0 mismatch" with ``old_available=0`` is
      vacuous, so always check this first.
    - ``train_inference/logprob_diff_abs_{mean,max}`` — ``|trainer - inference|`` in
      logprob space.
    - ``train_inference/prob_diff_abs_{mean,p50,p95,p99,max}`` — ``|exp(trainer) -
      exp(inference)|`` in probability space; the tail (p95/p99/max) is where sparse
      outlier tokens show up even when the mean agrees.
    - ``train_inference/{trainer,rollout}_next_token_{loss,perplexity}`` — masked
      ``mean(-logprob)`` and its ``exp`` for each distribution over the same tokens.

    Returns ``{"train_inference/old_available": 0}`` (only) when inference logprobs
    are missing/empty or every token is masked.
    """
    if "rollout_log_probs" not in data.batch:
        return {"train_inference/old_available": 0}

    rollout_log_probs = data.batch["rollout_log_probs"]
    actor_log_probs = data.batch["old_log_probs"]
    if rollout_log_probs.numel() == 0 or actor_log_probs.numel() == 0:
        return {"train_inference/old_available": 0}

    if "response_mask" in data.batch:
        log_prob_mask = data.batch["response_mask"]
    elif "attention_mask" in data.batch:
        log_prob_mask = data.batch["attention_mask"]
    else:
        logger.warning(f"no mask info found, use all log probs, {(data.batch.keys())=}")
        log_prob_mask = torch.ones_like(rollout_log_probs)

    response_length = data.batch["responses"].size(1)
    response_mask = log_prob_mask[:, -response_length:].bool()
    if not response_mask.any():
        return {"train_inference/old_available": 0}

    if rollout_log_probs.device != actor_log_probs.device:
        rollout_log_probs = rollout_log_probs.to(actor_log_probs.device)
    if response_mask.device != actor_log_probs.device:
        response_mask = response_mask.to(actor_log_probs.device)

    # logprob-space diff over the completion tokens
    logprob_diff = torch.masked_select(torch.abs(actor_log_probs - rollout_log_probs), response_mask)
    # probability-space diff; bounded in [0, 1], tail is the interesting part
    prob_diff = torch.masked_select(torch.abs(torch.exp(actor_log_probs) - torch.exp(rollout_log_probs)), response_mask)

    # per-distribution NLL / perplexity of the same tokens
    trainer_nll = torch.masked_select(-actor_log_probs, response_mask).mean()
    rollout_nll = torch.masked_select(-rollout_log_probs, response_mask).mean()

    # torch.quantile() raises "input tensor is too large" above 2**24 (~16.7M) elements. At the
    # faithful scale (batch * n * up-to-50k tokens) prob_diff far exceeds that, so the call would
    # crash right after the reward step. Subsample (random, with replacement) under the cap for the
    # quantile estimate; mean/max above use the full tensor (no size limit).
    _qsrc = prob_diff.float()
    _QCAP = 16_000_000
    if _qsrc.numel() > _QCAP:
        _idx = torch.randint(0, _qsrc.numel(), (_QCAP,), device=_qsrc.device)
        _qsrc = _qsrc[_idx]
    quantiles = torch.quantile(_qsrc, torch.tensor([0.5, 0.95, 0.99], device=_qsrc.device))

    return {
        "train_inference/old_available": 1,
        "train_inference/logprob_diff_abs_mean": logprob_diff.mean().detach().item(),
        "train_inference/logprob_diff_abs_max": logprob_diff.max().detach().item(),
        "train_inference/prob_diff_abs_mean": prob_diff.mean().detach().item(),
        "train_inference/prob_diff_abs_p50": quantiles[0].detach().item(),
        "train_inference/prob_diff_abs_p95": quantiles[1].detach().item(),
        "train_inference/prob_diff_abs_p99": quantiles[2].detach().item(),
        "train_inference/prob_diff_abs_max": prob_diff.max().detach().item(),
        "train_inference/trainer_next_token_loss": trainer_nll.detach().item(),
        "train_inference/trainer_next_token_perplexity": torch.exp(trainer_nll).detach().item(),
        "train_inference/rollout_next_token_loss": rollout_nll.detach().item(),
        "train_inference/rollout_next_token_perplexity": torch.exp(rollout_nll).detach().item(),
    }
