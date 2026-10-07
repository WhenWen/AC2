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
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import agg_loss, compute_value_loss, get_policy_loss_fn, kl_penalty
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.metric import AggregationType, Metric
from verl.utils.torch_functional import masked_mean, masked_sum
from verl.workers.config import ActorConfig, CriticConfig
from verl.workers.utils.padding import no_padding_2_padding


def sft_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]

    log_prob = model_output["log_probs"]

    if pad_mode == DatasetPadMode.NO_PADDING:
        # log_prob and loss mask are nested tensors of shape [bsz, j1]
        # for each sample, loss mask shape is [1, prompt_length + response_length]
        loss_mask = data["loss_mask"]

        log_prob_flatten = log_prob.values()
        loss_mask_flatten = loss_mask.values()

        # left-shift the loss mask by one token to align with log_prob
        loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)

        # NOTE: loss is averaged over all tokens in the batch across all data parallel groups,
        # For FSDP backend, the loss is directly used for backward; while for Megatron backend,
        # the loss should be scaled by `num_microbatches` for pp schedule.
        loss = -masked_sum(log_prob_flatten, loss_mask_flatten) / batch_num_tokens * dp_size
    else:
        response_mask = data["response_mask"].to(bool)
        loss = -masked_sum(log_prob, response_mask) / batch_num_tokens * dp_size

    return loss, {}


def sp_q_ce_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    """Generative-Q teacher-forced CE.

    The driver builds each row as (statement prompt | Q context riding response-side)
    with the forced ``Q value: `` prefill, teacher-forces the grid value + <|im_end|>,
    and ships:
      * ``sp_q_tok_w``  [bsz, resp_len] float: (row_weight / 4) on the TARGET token
        positions (grid value + end-of-turn), 0 elsewhere. row_weight = 1 for an
        uncovered record's no-reference row, 1/2 for each variant row of a covered
        record -- so L_Q(i) = CE/4 (uncovered) or the mean of the two variants.
      * ``sp_q_is_ref`` [bsz] int: 1 = with-reference variant row (metrics split).
      * non-tensor ``sp_q_denom``: the GLOBAL number of sampled base records N; the
        logical batch loss is mean over base records.

    Loss (FSDP-reduction-compensated, the sft_loss convention): the local microbatch
    contributes ``-(log_prob * tok_w).sum() / N * dp_size``; gradient averaging over dp
    ranks then yields exactly grad[ (1/N) sum_i L_Q(i) ]. Metrics are LOCAL sums (plain
    floats); the sp_q worker method all-reduces them itself.
    """
    log_prob = no_padding_2_padding(model_output["log_probs"], data)  # [bsz, resp_len]
    tok_w = data["sp_q_tok_w"]
    denom = float(data["sp_q_denom"])
    dp_size = float(data["dp_size"])
    assert tok_w.shape == log_prob.shape, (tok_w.shape, log_prob.shape)

    weighted = -(log_prob * tok_w)                    # [bsz, resp_len]
    loss = weighted.sum() / denom * dp_size

    is_ref = data["sp_q_is_ref"].to(bool)             # [bsz]
    per_row = weighted.sum(dim=-1)                    # w_row * CE_row / 4
    metrics = {
        "sp_q_ce4_sum_ref": per_row[is_ref].sum().detach().item(),
        "sp_q_ce4_sum_noref": per_row[~is_ref].sum().detach().item(),
        "sp_q_rows_ref": int(is_ref.sum().item()),
        "sp_q_rows_noref": int((~is_ref).sum().item()),
        "sp_q_target_tokens": int((tok_w > 0).sum().item()),
    }
    return loss, metrics


def ppo_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    """Computes ppo loss from model output (log_prob, entropy, values, etc. ) and old_log_probs from data."""
    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    entropy = model_output.get("entropy", None)
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)

    # global batch info for loss aggregation
    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor
    # AEC: adopt the driver-computed clip constant (broadcast identically to every worker via
    # assign_non_tensor(aec_k=...)); compute_policy_loss_vanilla reads it through aec.current_k().
    try:
        _aeck = data["aec_k"]
    except Exception:
        _aeck = None
    if _aeck is not None:
        from verl.trainer.ppo import aec as _aec
        _aec.set_k(float(_aeck))

    # Direct-OPD adaptive KL anchor (arXiv:2607.05394 Eq. 13): the driver broadcasts the current
    # alpha as non-tensor "dopd_alpha" (same mechanism as aec_k); it overrides the static
    # kl_loss_coef. Absent key -> plain PPO, untouched. Read BEFORE the select below drops it.
    try:
        _dopd_alpha = data["dopd_alpha"]
    except Exception:
        _dopd_alpha = None

    # Difficulty sampling: per-problem IS correction.
    # The driver attaches `diff_c` as a (bsz, 1) tensor, already rescaled so the c-weighted global
    # token count equals batch_num_tokens (normalize_c_for_loss). Scaling advantages by c>0 scales
    # every token's pg loss EXACTLY by c -- all vanilla clip terms (-A*r, -A*clip(r), -A*clip_c) are
    # linear in A, and the max/min/where selections are invariant under positive scaling -- so the
    # aggregate becomes the c-weighted token-mean with an UNCHANGED denominator.
    # Absent key -> plain PPO (non-difficulty runs are untouched). Grabbed before the select below.
    diff_c = data.get("diff_c", None)

    # assumes that if any of the global batch info is set, the policy_loss_fn will
    # normalize using dp_size/global_bsz/global_token; in this case, metric aggregation should be SUM
    # to reflect the mean loss over the global batch
    if (
        data["dp_size"] > 1
        or data["batch_num_tokens"] is not None
        or data["global_batch_size"] is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    metrics = {}

    # select fields and convert to padded tensor
    fields = ["response_mask", "old_log_probs", "advantages"]
    if "rollout_is_weights" in data:
        fields.append("rollout_is_weights")
    if "ref_log_prob" in data:
        fields.append("ref_log_prob")
    data = data.select(*fields).to_padded_tensor()

    response_mask = data["response_mask"].to(bool)
    # compute policy loss
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]
    rollout_is_weights = data.get("rollout_is_weights", None)

    if diff_c is not None:
        # (bsz, 1) broadcasts over response length; kept separate from rollout_is_weights (the two
        # compose multiplicatively if both are ever active).
        assert not diff_c.is_nested, "diff_c must stay a regular (bsz, 1) tensor through dispatch"
        assert diff_c.dim() == 2 and diff_c.shape[0] == advantages.shape[0], (
            f"diff_c shape {tuple(diff_c.shape)} misaligned with advantages {tuple(advantages.shape)}"
        )
        advantages = advantages * diff_c.to(advantages.dtype)

    loss_agg_mode = config.loss_agg_mode

    loss_mode = config.policy_loss.get("loss_mode", "vanilla")

    policy_loss_fn = get_policy_loss_fn(loss_mode)
    pg_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
    )

    # AggregationType.MEAN for pg metrics: assumes policy_loss_fn normalizes by local_bsz/local_tokens
    # Ex: in compute_policy_loss_vanilla, pg_metrics are pg_clipfrac, ppo_kl, pg_clipfrac_lower
    pg_metrics = Metric.from_dict(pg_metrics, aggregation=AggregationType.MEAN)

    metrics.update(pg_metrics)
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=metric_aggregation)
    policy_loss = pg_loss

    # add entropy loss
    # Difficulty sampling: the entropy bonus is part of the objective and contributes to the
    # gradient, so it is IS-corrected exactly like the PPO surrogate and the KL term. Weighting each
    # token's entropy by its row's diff_c gives the c-weighted token-mean, whose expectation is the
    # uniform-sampling entropy regularizer E_p[H] (E_q[c·∇H] = E_p[∇H]). Without this the entropy
    # term would regularize under the difficulty-sampled distribution, over-weighting hard problems.
    if entropy is not None:
        entropy_mat = entropy if diff_c is None else entropy * diff_c.to(entropy.dtype)
        entropy_loss = agg_loss(
            loss_mat=entropy_mat, loss_mask=response_mask, loss_agg_mode=loss_agg_mode,
            **config.global_batch_info
        )
        entropy_coeff = config.entropy_coeff
        policy_loss -= entropy_coeff * entropy_loss
        metrics["actor/entropy_loss"] = Metric(value=entropy_loss, aggregation=metric_aggregation)

    # add kl loss
    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]
        # compute kl loss
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        if diff_c is not None:
            # same c-weighted token-mean: the KL penalty is a per-problem expectation too
            kld = kld * diff_c.to(kld.dtype)
        kl_loss = agg_loss(
            loss_mat=kld, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode, **config.global_batch_info
        )

        kl_loss_coef = float(_dopd_alpha) if _dopd_alpha is not None else config.kl_loss_coef
        policy_loss += kl_loss * kl_loss_coef
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=metric_aggregation)
        metrics["kl_coef"] = kl_loss_coef

    return policy_loss, metrics


def value_loss(config: CriticConfig, model_output, data: TensorDict, dp_group=None):
    """value loss

    Args:
        config: CriticConfig
        model_output: model output from the model
        data: the input to the model
        dp_group: data paralle group

    Returns:
        value loss
    """
    vpreds = no_padding_2_padding(model_output["values"], data)  # (bsz, response_length)

    # select fields and convert to padded tensor
    data = data.select("values", "returns", "response_mask").to_padded_tensor()
    values = data["values"]
    returns = data["returns"]
    response_mask = data["response_mask"].to(bool)

    vf_loss, vf_clipfrac = compute_value_loss(
        vpreds=vpreds,
        values=values,
        returns=returns,
        response_mask=response_mask,
        cliprange_value=config.cliprange_value,
        loss_agg_mode=config.loss_agg_mode,
    )

    metrics = {}

    metrics.update(
        {
            "critic/vf_loss": vf_loss.detach().item(),
            "critic/vf_clipfrac": vf_clipfrac.detach().item(),
            "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
        }
    )

    return vf_loss, metrics
