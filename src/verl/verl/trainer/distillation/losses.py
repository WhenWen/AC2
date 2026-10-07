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

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
from tensordict import TensorDict

from verl.base_config import BaseConfig
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import ActorConfig, DistillationConfig, DistillationLossConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import no_padding_2_padding

DistillationLossFn = Callable[
    [
        ActorConfig,  # actor_config
        DistillationConfig,  # distillation_config
        dict,  # model_output
        TensorDict,  # micro batch input
    ],
    tuple[torch.Tensor, dict[str, Any]],
]


def is_distillation_enabled(config: Optional[DistillationConfig]) -> bool:
    """Check if distillation is enabled based on the provided configuration."""
    if config is None:
        return False
    return config.enabled


@dataclass
class DistillationLossSettings(BaseConfig):
    """
    Settings for a distillation loss function to be registered.

    Args:
        names (str | list[str]): Name(s) to register the distillation loss function under.
        use_topk (bool): Whether the loss function uses top-k log probabilities.
        use_estimator (bool): Whether the loss function uses single-sample KL estimators.
    """

    names: Any = field(default_factory=list)
    use_topk: bool = False
    use_estimator: bool = False
    use_hidden_states: bool = False

    _mutable_fields = {"names"}

    def __post_init__(self):
        self.names = [self.names] if isinstance(self.names, str) else self.names
        if sum([self.use_topk, self.use_estimator, self.use_hidden_states]) != 1:
            raise ValueError(
                f"Expected exactly one of use_estimator, use_topk, use_hidden_states, "
                f"but got {self.use_estimator=}, {self.use_topk=}, {self.use_hidden_states=}."
            )


DISTILLATION_LOSS_REGISTRY: dict[str, DistillationLossFn] = {}
DISTILLATION_SETTINGS_REGISTRY: dict[str, DistillationLossSettings] = {}


def register_distillation_loss(
    loss_settings: DistillationLossSettings,
) -> Callable[[DistillationLossFn], DistillationLossFn]:
    """Register a distillation loss function with the given name."""

    def decorator(func: DistillationLossFn) -> DistillationLossFn:
        for name in loss_settings.names:
            if name in DISTILLATION_LOSS_REGISTRY:
                raise ValueError(f"Distillation loss function with name '{name}' is already registered.")
            DISTILLATION_LOSS_REGISTRY[name] = func
            DISTILLATION_SETTINGS_REGISTRY[name] = loss_settings
        return func

    return decorator


def get_distillation_loss_fn(loss_name: str) -> DistillationLossFn:
    """Get the distillation loss function with a given name."""
    if loss_name not in DISTILLATION_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_LOSS_REGISTRY.keys())}"
        )
    return DISTILLATION_LOSS_REGISTRY[loss_name]


def get_distillation_loss_settings(loss_name: str) -> DistillationLossSettings:
    """Get the distillation loss settings with a given name."""
    if loss_name not in DISTILLATION_SETTINGS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_SETTINGS_REGISTRY.keys())}"
        )
    return DISTILLATION_SETTINGS_REGISTRY[loss_name]


def compute_distillation_loss_range(
    distillation_losses: torch.Tensor, response_mask: torch.Tensor
) -> dict[str, Metric]:
    """Compute min and max distillation loss over valid response tokens."""
    if response_mask.is_nested:
        distillation_losses_response = distillation_losses[response_mask.bool().to_padded_tensor(False)]
    else:
        distillation_losses_response = distillation_losses[response_mask.bool()]
    if distillation_losses_response.numel() == 0:
        # A mini-batch can hold zero valid response tokens (e.g. every sample in it is an
        # OPSD-skipped group after shuffling); .min()/.max() on empty tensors would raise.
        return {}
    return {
        "distillation/loss_min": Metric(AggregationType.MIN, distillation_losses_response.min()),
        "distillation/loss_max": Metric(AggregationType.MAX, distillation_losses_response.max()),
    }


def compute_topk_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    data: TensorDict,
    student_logits: torch.Tensor,
    data_format: str,
) -> torch.Tensor:
    """Compute the topk loss in logit processor.

    Returns:
    - distillation_losses: (bsz, seqlen/cp_size)
    - student_mass: (bsz, seqlen/cp_size)
    - teacher_mass: (bsz, seqlen/cp_size)
    """
    loss_settings = distillation_config.distillation_loss.loss_settings

    if getattr(loss_settings, "use_hidden_states", False):
        # Nitrobrew: full-vocab forward KL from teacher HIDDEN states (constant memory).
        loss_mode = distillation_config.distillation_loss.loss_mode
        use_reverse = loss_mode == "nitrobrew_reverse_kl"
        match config.strategy:
            case "fsdp" | "veomni":
                import verl.trainer.distillation.fsdp.nitrobrew_loss as fsdp_nb

                distillation_loss_fn = fsdp_nb.compute_nitrobrew_reverse_kl if use_reverse else fsdp_nb.compute_nitrobrew_kl
            case "megatron":
                import verl.trainer.distillation.megatron.nitrobrew_loss as megatron_nb

                distillation_loss_fn = megatron_nb.compute_nitrobrew_kl
                if use_reverse:
                    raise NotImplementedError("Nitrobrew reverse KL not yet implemented for Megatron")
            case _:
                raise NotImplementedError(f"Nitrobrew not implemented for strategy: {config.strategy=}")

        teacher_unembed = data["teacher_unembed"]
        if hasattr(teacher_unembed, "data"):
            teacher_unembed = teacher_unembed.data
        outputs = distillation_loss_fn(
            student_logits=student_logits,
            teacher_hidden_states=data["teacher_hidden_states"],
            teacher_unembed=teacher_unembed,
            config=distillation_config,
            data_format=data_format,
            data=data,  # lets the FSDP path restrict the KL to response-predicting positions
        )
    elif distillation_config.distillation_loss.loss_mode == "direct_opd":
        # Direct-OPD (arXiv:2607.05394): two-scorer log-ratio reward on the student's own top-k
        # support. Consumes the teacher_* pair PLUS the teacher_ref_* pair emitted by the
        # Direct-OPD agent loop (distillation.direct_opd_ref_key).
        match config.strategy:
            case "fsdp" | "veomni":
                import verl.trainer.distillation.fsdp.losses as fsdp_losses

                outputs = fsdp_losses.compute_direct_opd(
                    student_logits=student_logits,
                    teacher_topk_log_probs=data["teacher_logprobs"],
                    teacher_topk_ids=data["teacher_ids"],
                    teacher_ref_topk_log_probs=data["teacher_ref_logprobs"],
                    teacher_ref_topk_ids=data["teacher_ref_ids"],
                    config=distillation_config,
                    data_format=data_format,
                )
            case _:
                raise NotImplementedError(f"direct_opd not implemented for strategy: {config.strategy=}")
    else:
        match config.strategy:
            # VeOmni uses FSDP2 internally, so its loss computation is identical to FSDP.
            case "fsdp" | "veomni":
                import verl.trainer.distillation.fsdp.losses as fsdp_losses

                distillation_loss_fn = fsdp_losses.compute_forward_kl_topk
            case "megatron":
                import verl.trainer.distillation.megatron.losses as megatron_losses

                distillation_loss_fn = megatron_losses.compute_forward_kl_topk
            case _:
                raise NotImplementedError(f"Unsupported strategy: {config.strategy=}")

        outputs = distillation_loss_fn(
            student_logits=student_logits,
            teacher_topk_log_probs=data["teacher_logprobs"],
            teacher_topk_ids=data["teacher_ids"],
            config=distillation_config,
            data_format=data_format,
        )

    expected_shape = student_logits.shape[:2]
    for k, v in outputs.items():
        assert v.shape == expected_shape, f"Expected shape {expected_shape}, but got {v.shape} for {k=}."

    return outputs


def distillation_ppo_loss(
    config: ActorConfig,
    distillation_config: Optional[DistillationConfig],
    model_output: dict = None,
    data: TensorDict = None,
    dp_group=None,
    student_logits: torch.Tensor = None,
    data_format: str = "thd",
):
    """Loss function used both for logit processor and final policy loss.
    - student_logits is not None, compute the topk loss in logit processor.
    - student_logits is None, compute final policy loss.

    [split sequence across sp/cp groups]
                   |
    [model forward and output logits: (bsz, seqlen/cp_size, vocab_size/tp_size)]
                   |
    [logits processor compute topk loss: (bsz, seqlen/cp_size)]
                   |
    [all gather topk loss across sp/cp groups: (bsz, seqlen)]
                   |
    [combine topk loss with policy loss]

    Args:
        config: Actor configuration.
        distillation_config: Distillation configuration.
        model_output: Model output, including log_probs, entropy.
        data: Micro input batch, contains
          - teacher_logprobs: (bsz, seqlen, topk)
          - teacher_ids: (bsz, seqlen, topk)
        student_logits: (bsz, seqlen/cp_size, vocab_size/tp_size).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - student_logits is not None, return the topk loss tensor (bsz, seqlen/cp_size).
    - student_logits is None, return the final policy loss scalar and metrics.
    """

    # Called as logits processor
    if student_logits is not None:
        return compute_topk_loss(config, distillation_config, data, student_logits, data_format)

    # Called as final policy loss
    distillation_loss_config = distillation_config.distillation_loss
    distill_loss, distill_metrics = distillation_loss(config, distillation_config, model_output, data)
    policy_loss, policy_metrics = ppo_loss(config, model_output, data, dp_group)
    if not distillation_loss_config.use_task_rewards:
        policy_loss = 0.0
        if config.use_kl_loss:
            # Pure distillation discards the whole ppo_loss result (PPO surrogate + entropy),
            # but Direct-OPD (arXiv:2607.05394 Eq. 12) needs the KL(pi||pi_ref) anchor to
            # survive — recompute just that term here. Existing pure-distillation runs (OPSD)
            # set actor.use_kl_loss=False and are unchanged.
            policy_loss = _kl_anchor_loss(config, model_output, data, policy_metrics)

    # Combine distillation with policy loss
    policy_metrics.update(distill_metrics)
    distillation_loss_coef = (
        distillation_loss_config.distillation_loss_coef if distillation_loss_config.use_task_rewards else 1.0
    )
    policy_loss += distill_loss * distillation_loss_coef
    policy_metrics["distillation/loss"] = Metric(value=distill_loss, aggregation=AggregationType.SUM)

    return policy_loss, policy_metrics


def _kl_anchor_loss(
    config: ActorConfig,
    model_output: dict,
    data: TensorDict,
    metrics: dict[str, Any],
) -> torch.Tensor:
    """Standalone KL(pi||pi_ref) anchor for pure-distillation runs (use_task_rewards=False).

    Mirrors the use_kl_loss block of workers/utils/losses.py:ppo_loss, whose result
    distillation_ppo_loss zeroes out. The coefficient is the driver-broadcast non-tensor
    ``dopd_alpha`` when present (Direct-OPD adaptive controller, one-step lag), else the
    static actor.kl_loss_coef. global_batch_info is already populated by the preceding
    distillation_loss/ppo_loss calls.
    """
    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    try:
        _alpha = data["dopd_alpha"]
    except Exception:
        _alpha = None
    kl_coef = float(_alpha) if _alpha is not None else config.kl_loss_coef
    kl_data = data.select("response_mask", "ref_log_prob").to_padded_tensor()
    response_mask = kl_data["response_mask"].to(bool)
    kld = kl_penalty(logprob=log_prob, ref_logprob=kl_data["ref_log_prob"], kl_penalty=config.kl_loss_type)
    kl_loss = agg_loss(
        loss_mat=kld, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode, **config.global_batch_info
    )
    metrics["actor/kl_anchor_loss"] = Metric(value=kl_loss, aggregation=AggregationType.SUM)
    metrics["actor/kl_anchor_coef"] = kl_coef
    return kl_loss * kl_coef


def distillation_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics.

    Returns:
    - distillation_loss: Aggregated distillation loss scalar.
    - distillation_metrics: Dictionary of metrics.
    """
    assert distillation_config is not None
    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    # Populate global-batch normalization from `data` BEFORE aggregating. ppo_loss also sets
    # these, but it runs AFTER distillation_loss inside distillation_ppo_loss — without this the
    # first mini-batch aggregates with EMPTY info (local-only normalization, wrong under dp>1)
    # and every later one with the previous mini-batch's stale values.
    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor
    distillation_loss_fn = get_distillation_loss_fn(loss_config.loss_mode)
    distillation_losses, distillation_metrics = distillation_loss_fn(
        config=config,
        distillation_config=distillation_config,
        model_output=model_output,
        data=data,
    )
    response_mask = data["response_mask"]
    loss_agg_mode = config.loss_agg_mode

    # OPD-rewrite (gated on SP_OPD_REWRITE=1 in the worker env so runs without it stay
    # byte-identical): the min/max range diagnostics must cover the positions the loss actually
    # aggregates (think tokens of non-dropped rows), not the full response — otherwise proof and
    # driver-dropped positions, where zero teacher hidden yields a large uniform-teacher KL that
    # the mask zeroes out of the loss, dominate distillation/loss_max and hide real behavior.
    # Gate for the diff_c IS-correction + distill-masked range diagnostics below. SP_OPD_REWRITE
    # implies it; SP_OPD_DIFF_CORRECT=1 opts a PLAIN-OPD run in. A run that sets neither keeps
    # the uncorrected-OPD behavior byte-identical.
    _opd_rewrite = (
        os.environ.get("SP_OPD_REWRITE") in ("1", "true", "True")
        or os.environ.get("SP_OPD_DIFF_CORRECT") in ("1", "true", "True")
    )
    range_mask = response_mask
    if _opd_rewrite and "opd_think_mask" in data.keys():
        _rm = response_mask.to_padded_tensor(False) if response_mask.is_nested else response_mask
        _tm = data["opd_think_mask"]
        _tm = _tm.to_padded_tensor(False) if _tm.is_nested else _tm
        range_mask = _rm * _tm.to(_rm.dtype)
    distillation_metrics.update(
        compute_distillation_loss_range(distillation_losses=distillation_losses, response_mask=range_mask)
    )
    if loss_config.loss_max_clamp is not None:
        # clamping min is for k1 loss which can be negative
        distillation_losses = distillation_losses.clamp(min=-loss_config.loss_max_clamp, max=loss_config.loss_max_clamp)

    if loss_config.use_policy_gradient:
        # Use negative distillation loss as reward, as done by https://thinkingmachines.ai/blog/on-policy-distillation/.
        policy_loss_fn = get_policy_loss_fn(loss_config.policy_loss_mode)
        for k, v in config.global_batch_info.items():
            loss_config.global_batch_info[k] = v
        log_prob = no_padding_2_padding(model_output["log_probs"], data)
        old_log_prob = data["old_log_probs"]
        if old_log_prob.is_nested:
            old_log_prob = data["old_log_probs"].to_padded_tensor(0.0)
        if response_mask.is_nested:
            response_mask = response_mask.to_padded_tensor(False)
        rollout_is_weights = data.get("rollout_is_weights", None)
        distillation_loss, pg_metrics = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=-distillation_losses.detach(),
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=loss_config,
            rollout_is_weights=rollout_is_weights,
        )
        pg_metrics = {f"distillation/{k[len('actor/') :]}": v for k, v in pg_metrics.items()}
        distillation_metrics.update(pg_metrics)
    else:
        # Directly backpropagate distillation loss as a supervised loss, as in https://arxiv.org/abs/2306.13649.
        if response_mask.is_nested:
            response_mask = response_mask.to_padded_tensor(False)
        # OPD: an OPTIONAL distill-ONLY mask (batch["opd_think_mask"]) restricts the distillation
        # loss to the thinking region strictly BEFORE </think> — without touching response_mask,
        # which ppo_loss (the GRPO task term) shares and must keep over the full response. Same
        # [bsz, resp_len] layout as response_mask; multiplied in here only for the distill agg.
        distill_mask = response_mask
        if "opd_think_mask" in data.keys():
            think_mask = data["opd_think_mask"]
            if think_mask.is_nested:
                think_mask = think_mask.to_padded_tensor(False)
            distill_mask = response_mask * think_mask.to(response_mask.dtype)
        # Difficulty-sampling IS correction (OPD-rewrite, gated on SP_OPD_REWRITE=1 to keep runs
        # without it byte-identical): the PPO surrogate, entropy bonus and KL penalty
        # are all corrected back to the uniform-sampling objective by the per-row diff_c weight
        # (workers/utils/losses.py); the supervised distill term is part of the same objective and
        # needs the identical correction — without it GRPO optimizes the corrected uniform
        # objective while OPD optimizes the biased difficulty-sampled one. diff_c is (bsz, 1),
        # broadcasting over the response length, and is already rescaled so the c-weighted global
        # token count matches batch_num_tokens (normalize_c_for_loss).
        if _opd_rewrite:
            diff_c = data.get("diff_c", None)
            if diff_c is not None:
                assert not diff_c.is_nested and diff_c.dim() == 2 and diff_c.shape[0] == distillation_losses.shape[0], (
                    f"diff_c shape {tuple(diff_c.shape)} misaligned with distillation_losses "
                    f"{tuple(distillation_losses.shape)}"
                )
                distillation_losses = distillation_losses * diff_c.to(distillation_losses.dtype)
        distillation_loss = agg_loss(
            loss_mat=distillation_losses,
            loss_mask=distill_mask,
            loss_agg_mode=loss_agg_mode,
            **config.global_batch_info,
        )

    return distillation_loss, distillation_metrics


@register_distillation_loss(DistillationLossSettings(names=["forward_kl_topk"], use_topk=True))  # type: ignore[arg-type]
def compute_forward_kl_topk(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute forward KL distillation loss and related metrics using top-k log probabilities.

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    # topk loss has been computed in logits processor
    distillation_losses = no_padding_2_padding(model_output["distillation_losses"], data)
    student_mass = no_padding_2_padding(model_output["student_mass"], data)
    teacher_mass = no_padding_2_padding(model_output["teacher_mass"], data)
    overlap_count = model_output.get("overlap_count")
    overlap_token_advantage = model_output.get("overlap_token_advantage")
    if overlap_count is not None and overlap_token_advantage is not None:
        overlap_count = no_padding_2_padding(overlap_count, data)
        overlap_token_advantage = no_padding_2_padding(overlap_token_advantage, data)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert distillation_losses.shape == student_mass.shape == teacher_mass.shape == response_mask_bool.shape

    overlap_metrics = {}
    if overlap_count is not None and overlap_token_advantage is not None:
        assert overlap_count.shape == overlap_token_advantage.shape == response_mask_bool.shape
        valid_overlap_count = overlap_count[response_mask_bool]
        k = distillation_config.distillation_loss.topk
        assert k is not None
        # Diagnostics for tracking teacher/student top-k overlap in OPD, following
        # "Rethinking On-Policy Distillation of Large Language Models" (arXiv:2604.13016):
        # overlap ratio and average teacher-token KL contribution on overlapped tokens.
        overlap_metrics["distillation/overlap_ratio"] = (valid_overlap_count.float().mean() / k).item()
        overlap_position_mask = response_mask_bool & (overlap_count > 0)
        if overlap_position_mask.any():
            overlap_metrics["distillation/overlap_token_advantage"] = (
                overlap_token_advantage[overlap_position_mask].mean().item()
            )
        else:
            overlap_metrics["distillation/overlap_token_advantage"] = 0.0

    # Log amount of mass in the top-k log probabilities for both student and teacher.
    student_mass = student_mass[response_mask_bool]
    teacher_mass = teacher_mass[response_mask_bool]
    distillation_metrics = {
        "distillation/student_mass": student_mass.mean().item(),
        "distillation/student_mass_min": Metric(AggregationType.MIN, student_mass.min()),
        "distillation/student_mass_max": Metric(AggregationType.MAX, student_mass.max()),
        "distillation/teacher_mass": teacher_mass.mean().item(),
        "distillation/teacher_mass_min": Metric(AggregationType.MIN, teacher_mass.min()),
        "distillation/teacher_mass_max": Metric(AggregationType.MAX, teacher_mass.max()),
        **overlap_metrics,
    }

    # Due to use of top-k, student and teacher distributions don't sum to 1 -> divergences can be negative.
    distillation_losses = distillation_losses.clamp_min(0.0)

    return distillation_losses, distillation_metrics


@register_distillation_loss(DistillationLossSettings(names=["direct_opd"], use_topk=True))  # type: ignore[arg-type]
def compute_direct_opd_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Aggregate the Direct-OPD per-position loss (arXiv:2607.05394, computed in the logits
    processor — see fsdp/losses.py:compute_direct_opd) and its diagnostics.

    Returns:
    - distillation_losses: (bsz, resp_len). NOT clamped to >= 0: the per-position value
      -sum_v A(v) log pi(v) is signed by construction (A is a signed advantage).
    - distillation_metrics: dopd/rbar (drives the adaptive-alpha controller), scorer coverage
      of the student support, and student/teacher top-k overlap (paper §5.1 diagnostic).
    """
    distillation_losses = no_padding_2_padding(model_output["distillation_losses"], data)
    rbar = no_padding_2_padding(model_output["dopd_rbar"], data)
    coverage_teacher = no_padding_2_padding(model_output["dopd_coverage_teacher"], data)
    coverage_ref = no_padding_2_padding(model_output["dopd_coverage_ref"], data)
    overlap_teacher = no_padding_2_padding(model_output["dopd_overlap_teacher"], data)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert (
        distillation_losses.shape == rbar.shape == coverage_teacher.shape == response_mask_bool.shape
    )

    if not response_mask_bool.any():
        # A mini-batch can hold zero valid response tokens after shuffling; keep metric keys
        # absent rather than emitting NaNs (same guard as compute_distillation_loss_range).
        return distillation_losses, {}

    distillation_metrics = {
        "dopd/rbar": Metric(AggregationType.MEAN, rbar[response_mask_bool].mean()),
        "dopd/coverage_teacher": Metric(AggregationType.MEAN, coverage_teacher[response_mask_bool].mean()),
        "dopd/coverage_ref": Metric(AggregationType.MEAN, coverage_ref[response_mask_bool].mean()),
        "dopd/overlap_teacher": Metric(AggregationType.MEAN, overlap_teacher[response_mask_bool].mean()),
    }
    return distillation_losses, distillation_metrics


@register_distillation_loss(
    DistillationLossSettings(names=["nitrobrew", "nitrobrew_reverse_kl"], use_hidden_states=True)
)  # type: ignore[arg-type]
def compute_nitrobrew_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Aggregate Nitrobrew per-token KL (computed in the logits processor)."""
    distillation_losses = no_padding_2_padding(model_output["distillation_losses"], data)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert distillation_losses.shape == response_mask_bool.shape
    distillation_losses = distillation_losses.clamp_min(0.0)
    metrics: dict[str, Any] = {}
    if "distillation_clipped_frac" in model_output:
        # OPSD pointwise clip tau: fraction of (position, vocab) contributions that hit the clip.
        clipped_frac = no_padding_2_padding(model_output["distillation_clipped_frac"], data)
        denom = response_mask_bool.sum().clamp_min(1)
        metrics["opsd/kl_clipped_frac"] = Metric(
            AggregationType.MEAN, (clipped_frac * response_mask_bool).sum() / denom
        )
    return distillation_losses, metrics


@register_distillation_loss(
    DistillationLossSettings(names=["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"], use_estimator=True)
)  # type: ignore[arg-type]
def compute_distillation_loss_reverse_kl_estimator(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics using single-sample KL estimators.

    Uses the kl_penalty function from core_algos which supports various KL divergence
    estimators: "kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3".

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert teacher_log_probs.shape == student_log_probs.shape == response_mask_bool.shape

    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    distillation_losses = kl_penalty(
        logprob=student_log_probs, ref_logprob=teacher_log_probs, kl_penalty=loss_config.loss_mode
    )
    # Since k1 can be negative, log the mean absolute loss.
    metrics = {
        "distillation/abs_loss": Metric(AggregationType.MEAN, distillation_losses[response_mask_bool].abs().mean()),
    }
    return distillation_losses, metrics
