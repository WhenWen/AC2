# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import math
import os
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pprint import pprint
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.distillation.losses import is_distillation_enabled
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    _compute_response_info,
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.utils import (
    Role,
    WorkerType,
    need_critic,
    need_reference_policy,
    need_reward_model,
    need_teacher_policy,
)
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import deprecated, load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.config import DistillationConfig, EngineConfig
from verl.workers.rollout.llm_server import LLMServerManager
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_spec_decode_metrics(
    spec_drafts,
    spec_accepts,
    spec_verifies,
    non_padding_mask=None,
) -> dict:
    """Aggregate per-request speculative decoding stats.

    Ratios are computed per request and then averaged, so long and short
    responses have equal metric weight.

    The three inputs come from the rollout engine (vLLM request spec-decode
    stats or sglang ``meta_info["spec_*"]`` keys). Either all three are ``None``
    (caller didn't fetch them, e.g. spec rollout disabled) and the function
    is a no-op, or all three are populated; mixed state is a programmer error.

    ``non_padding_mask`` is a numpy bool array used by sync PPO to drop padded
    placeholder samples; pass ``None`` for async PPO.
    """
    if spec_drafts is None and spec_accepts is None and spec_verifies is None:
        return {}
    assert spec_drafts is not None and spec_accepts is not None and spec_verifies is not None, (
        "spec_decode metrics require all three of spec_num_draft_tokens / "
        "spec_num_accepted_tokens / spec_num_verify_steps; got partial inputs"
    )

    drafts = spec_drafts.tolist() if hasattr(spec_drafts, "tolist") else list(spec_drafts)
    accepts = spec_accepts.tolist() if hasattr(spec_accepts, "tolist") else list(spec_accepts)
    verifies = spec_verifies.tolist() if hasattr(spec_verifies, "tolist") else list(spec_verifies)

    if non_padding_mask is not None:
        drafts = [d for d, keep in zip(drafts, non_padding_mask, strict=True) if keep]
        accepts = [a for a, keep in zip(accepts, non_padding_mask, strict=True) if keep]
        verifies = [v for v, keep in zip(verifies, non_padding_mask, strict=True) if keep]

    if len(drafts) == 0:
        return {}

    # Treat zero-denominator samples as 0.0 and keep them in the mean.
    per_sample_accept_rate = [(a / d) if d > 0 else 0.0 for a, d in zip(accepts, drafts, strict=True)]
    per_sample_accept_length = [(1.0 + a / v) if v > 0 else 0.0 for a, v in zip(accepts, verifies, strict=True)]

    n = len(drafts)
    return {
        "rollout/spec_accept_rate": float(sum(per_sample_accept_rate) / n),
        "rollout/spec_accept_length": float(sum(per_sample_accept_length) / n),
    }


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]
        # sp_segment (core_algos.compute_sp_segment_advantage): needs extra_info for the stamped
        # interior Q values, and its per-step statistics come back on the module-level
        # SP_SEGMENT_LAST_STATS -- copied onto meta_info here so fit() can merge them into
        # metrics without the registry contract growing a third return value.
        if adv_estimator in ("sp_segment",):
            adv_kwargs["non_tensor_batch"] = data.non_tensor_batch
            _algo = config or {}
            _sm = _algo.get("sp_q_seg_min_valid", None) if hasattr(_algo, "get") else None
            _se = _algo.get("sp_q_seg_collapse_eps", None) if hasattr(_algo, "get") else None
            if _sm is not None:
                adv_kwargs["seg_min_valid"] = int(_sm)
            if _se is not None:
                adv_kwargs["seg_collapse_eps"] = float(_se)
        # GDPO: pass raw data for per-dimension reward extraction
        if adv_estimator in (AdvantageEstimator.GDPO, "gdpo"):
            adv_kwargs["non_tensor_batch"] = data.non_tensor_batch
            adv_kwargs["batch"] = data.batch
        # Add sum_pi_squared for Optimal Token Baseline
        if adv_estimator in (AdvantageEstimator.OPTIMAL_TOKEN_BASELINE, AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE):
            # Check if sum_pi_squared is available
            assert "sum_pi_squared" in data.batch, (
                "Step-dependent optimal baseline requires sum_pi_squared from actor. "
                "Please set actor.calculate_sum_pi_squared=True in config."
            )
            adv_kwargs["sum_pi_squared"] = data.batch["sum_pi_squared"]
            # old_log_probs needed for path-variance proxy: w_t = 1 - 2*exp(old_log_probs) + sum_pi_squared
            adv_kwargs["old_log_probs"] = data.batch["old_log_probs"]
            # Get pre-computed rollout IS weights if available
            rollout_is_weights = data.batch.get("rollout_is_weights", None)
            adv_kwargs["rollout_is_weights"] = rollout_is_weights

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if adv_estimator in ("sp_segment",):
            data.meta_info["sp_seg_metrics"] = dict(core_algos.SP_SEGMENT_LAST_STATS)
    return data


def _sp_td_copy_ranges(counts):
    """(src_row, first_copy_pos, n_copies) for an interleaved repeat: both
    DataProto.repeat(interleave=True) and sample_level_repeat lay a source row's copies
    out contiguously, so the copy index within a slot is position arithmetic."""
    out, pos = [], 0
    for src, c in enumerate(counts):
        out.append((src, pos, int(c)))
        pos += int(c)
    return out


def sp_td_rewrite_gen_copies(pre_ntb, out_proto, counts) -> int:
    """No-group TD (sp_q_td_enable), GEN side: give each copy of a TD slot its OWN prefix
    and short-lane cap BEFORE dispatch. The dataset stamps the slot's full trajectory
    (sp_td_traj_ids) plus per-sibling cut lengths (sp_td_cuts) and caps (sp_td_caps) on
    the single dataset row; the repeat clones that row, and this rewrite makes copy j the
    (traj[:cuts[j]], caps[j]) request. Only per-row VALUES the prefix agent already reads
    change -- no new field crosses the Ray-actor/vLLM boundary.

    pre_ntb: the PRE-repeat gen batch non-tensors (source rows); out_proto: the repeated
    gen DataProto. Returns the number of TD source rows rewritten.
    """
    cuts_col = pre_ntb.get("sp_td_cuts")
    if cuts_col is None:
        return 0
    traj_col = pre_ntb["sp_td_traj_ids"]
    caps_col = pre_ntb["sp_td_caps"]
    out_prefix = out_proto.non_tensor_batch["sp_prefix_token_ids"]
    # sp_q_max_new_tokens is an int per row: depending on the collate path it lands in
    # .batch (tensor) or non_tensor -- handle both, like _sp_per_row_repeat_counts does.
    out_cap_nt = out_proto.non_tensor_batch.get("sp_q_max_new_tokens")
    out_cap_t = out_proto.batch.get("sp_q_max_new_tokens") if out_proto.batch is not None else None
    n_td = 0
    for src, base, c in _sp_td_copy_ranges(counts):
        cuts = cuts_col[src]
        if cuts is None or len(cuts) == 0:
            continue
        assert len(cuts) == c, (
            f"sp_q_td: slot stamped {len(cuts)} cuts but repeats {c} copies -- "
            "sp_q_td_siblings must equal rollout.n"
        )
        traj = traj_col[src]
        caps = caps_col[src]
        for j in range(c):
            k = int(cuts[j])
            out_prefix[base + j] = [int(x) for x in traj[:k]]
            if out_cap_nt is not None:
                out_cap_nt[base + j] = int(caps[j])
            if out_cap_t is not None:
                out_cap_t[base + j] = int(caps[j])
        n_td += 1
    return n_td


def sp_td_rewrite_driver_copies(ntb, counts) -> int:
    """No-group TD, DRIVER side (after the post-generation repeat, before union): each TD
    copy becomes its own singleton uid group with per-copy bookkeeping. extra_info dicts
    are SHARED across copies after the repeat (repeat duplicates dict references), so
    each TD copy gets its own shallow dict -- the wave does the same at its own entry --
    before its sp_prefix_len / sp_cut_fraction are corrected to its own cut and it is
    marked sp_q_td_row=1 (the stamp the probe->seg relay and the reward-site drop key
    on). Returns the number of TD source rows rewritten."""
    extra = ntb["extra_info"]
    uids = ntb["uid"]
    n_td = 0
    for src, base, c in _sp_td_copy_ranges(counts):
        ei0 = extra[base]
        if not (isinstance(ei0, dict) and int(ei0.get("sp_q_td_group", 0))):
            continue
        cuts = ei0.get("sp_td_cuts") or []
        assert len(cuts) == c, (
            f"sp_q_td: slot stamped {len(cuts)} cuts but repeats {c} copies -- "
            "sp_q_td_siblings must equal rollout.n"
        )
        t = int(ei0.get("sp_td_len", 0))
        for j in range(c):
            ei = dict(ei0)
            k = int(cuts[j])
            ei["sp_prefix_len"] = k
            ei["sp_cut_fraction"] = (k / t) if t else 0.0
            ei["sp_q_td_row"] = 1
            extra[base + j] = ei
            if j > 0:
                uids[base + j] = str(uuid.uuid4())
        n_td += 1
    return n_td


def sp_td_mirror_driver_keys(batch_ntb, gen_ntb) -> list:
    """No-group TD, DRIVER side, step 2: make the gen output agree with the rewritten driver
    batch BEFORE `batch.union(gen_batch_output)`. _get_gen_batch copies the reward keys
    (uid, extra_info, ...) INTO the gen batch, and the agent loop returns them, so after the
    Ray round-trip the gen output holds PICKLED COPIES of the pre-rewrite uid/extra_info.
    union_numpy_dict asserts deep equality on every shared key -- with per-copy uids and
    per-copy extra_info on the driver side that assertion fails ("`extra_info` in
    tensor_dict1 and tensor_dict2 are not the same object"). Row r of
    both batches describes the same request (same interleaved repeat counts), so the
    driver's arrays are simply installed on the gen side. Returns the keys mirrored."""
    mirrored = []
    for key in ("uid", "extra_info"):
        if key in gen_ntb and key in batch_ntb:
            assert len(gen_ntb[key]) == len(batch_ntb[key]), (
                f"sp_q_td: {key} length mismatch gen={len(gen_ntb[key])} driver={len(batch_ntb[key])}"
            )
            gen_ntb[key] = batch_ntb[key]
            mirrored.append(key)
    return mirrored


@deprecated(
    "main_ppo.py is deprecated, and wil be replaced by main_ppo_sync.py in v0.8.0, please use main_ppo_sync.py instead."
)
class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)
        self.use_teacher_policy = need_teacher_policy(self.config)
        # ref_log_prob is only consumed by the KL terms. Colocated OPSD forces the ref model into
        # existence (it is the privileged teacher for compute_ref_hidden) with both KL terms off —
        # don't burn a full unprivileged ref forward per step producing a tensor nothing reads.
        self.use_ref_log_prob = bool(
            self.config.algorithm.get("use_kl_in_reward", False)
            or self.config.actor_rollout_ref.actor.use_kl_loss
        )

        self.use_rm = need_reward_model(self.config)

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        self.checkpoint_manager = None
        self._init_dump_executor()

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    @staticmethod
    def _write_generations(inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path, global_steps):
        """Write generation samples as JSONL (runs in background thread)."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

        print(f"Dumped generations to {filename}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL asynchronously."""
        global_steps = self.global_steps
        future = self._dump_executor.submit(
            self._write_generations,
            inputs,
            outputs,
            gts,
            scores,
            reward_extra_infos_dict,
            dump_path,
            global_steps,
        )
        self._dump_futures.append(future)
        # Clean up completed futures and surface any exceptions early
        still_pending = []
        for f in self._dump_futures:
            if f.done():
                f.result()  # re-raises if the write failed
            else:
                still_pending.append(f)
        self._dump_futures = still_pending

    def _init_dump_executor(self):
        """Create or recreate the dump executor and futures list."""
        self._dump_executor = ThreadPoolExecutor(max_workers=1)
        self._dump_futures = []

    def _shutdown_dump_executor(self):
        """Drain pending dump futures and shut down the executor."""
        for f in self._dump_futures:
            f.result()
        self._dump_futures.clear()
        self._dump_executor.shutdown(wait=True)

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = {
                k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in reward_extra_infos_dict.items()
            }
            # Per-sample token lengths so downstream plotting can compute response-length
            # distributions/percentiles (mean/min/max scalars in metrics.jsonl can't). Tokens,
            # not chars -- mixing the two silently mislabels response-length panels.
            length_info = _compute_response_info(batch)
            prompt_lengths = length_info["prompt_length"].detach().cpu().long().tolist()
            response_lengths = length_info["response_length"].detach().cpu().long().tolist()
            reward_extra_infos_to_dump["prompt_length"] = prompt_lengths
            reward_extra_infos_to_dump["response_length"] = response_lengths
            reward_extra_infos_to_dump["total_tokens"] = [p + r for p, r in zip(prompt_lengths, response_lengths)]
            # Persist the RAW token IDs alongside the decoded input/output text so the durable
            # dump is self-sufficient for token-level work (exact re-tokenization, offline replay,
            # length audits) without the transient step_cache .pt. Purely additive + backward
            # compatible: new "prompt_token_ids"/"response_token_ids" keys; readers that don't know
            # them ignore them, and older dumps lacking them still parse. Reconstructed from
            # input_ids under attention_mask (real tokens are prompt-then-response in order), so we
            # make NO left/right padding assumption and write only the un-padded IDs -- never the
            # full 50k-wide rows. Gated by SP_DUMP_TOKEN_IDS (default on); set =0 to save disk on
            # long-generation runs where these lists ~double the jsonl size.
            if os.environ.get("SP_DUMP_TOKEN_IDS", "1") != "0":
                _ids = batch.batch["input_ids"]
                _attn = batch.batch["attention_mask"].bool()
                _prompt_tok, _response_tok = [], []
                for _i, _pl in enumerate(prompt_lengths):
                    _real = _ids[_i][_attn[_i]].cpu().tolist()  # [real prompt ids..., real response ids...]
                    _prompt_tok.append(_real[:_pl])
                    _response_tok.append(_real[_pl:])
                reward_extra_infos_to_dump["prompt_token_ids"] = _prompt_tok
                reward_extra_infos_to_dump["response_token_ids"] = _response_tok
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_to_dump.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )
            # Explicit uid unlocks prompt-group viz panels (pass-count distribution,
            # per-group judge counts); data_source + extra_info split/index unlock the
            # paired-theorem view. All are no-ops when absent from the batch.
            if "uid" in batch.non_tensor_batch:
                reward_extra_infos_to_dump.setdefault("uid", batch.non_tensor_batch["uid"].tolist())
            # Difficulty sampling: per-row draw-time correction + stable problem id, so the viz
            # parser can SNIS-weight its dump-derived train aggregations (correctness/pass-count
            # panels) back to the uniform distribution. No-ops when absent.
            if "diff_c_raw" in batch.non_tensor_batch:
                reward_extra_infos_to_dump.setdefault(
                    "diff_c_raw", [float(x) for x in batch.non_tensor_batch["diff_c_raw"]]
                )
            if "qid" in batch.non_tensor_batch:
                reward_extra_infos_to_dump.setdefault("qid", batch.non_tensor_batch["qid"].tolist())
            if "data_source" in batch.non_tensor_batch:
                reward_extra_infos_to_dump.setdefault(
                    "data_source", batch.non_tensor_batch["data_source"].tolist()
                )
            if "extra_info" in batch.non_tensor_batch:
                _ei = batch.non_tensor_batch["extra_info"]
                reward_extra_infos_to_dump.setdefault(
                    "extra_split", [e.get("split") if isinstance(e, dict) else None for e in _ei]
                )
                reward_extra_infos_to_dump.setdefault(
                    "extra_index", [e.get("index") if isinstance(e, dict) else None for e in _ei]
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        # qid/diff_c_raw: difficulty-sampling identity + draw-time correction -- must survive in
        # `batch` (like uid) through generation so the reward/update stages can key on them.
        reward_keys = set(
            {"data_source", "reward_model", "extra_info", "uid", "qid", "diff_c_raw"}
        ) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _compute_reward_colocate(self, batch: DataProto) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor:
        """
        compute reward use colocate reward model
        """
        assert self.reward_loop_manager is not None, "RewardLoopManager is None"
        batch_reward = self.reward_loop_manager.compute_rm_score(batch)
        return batch_reward

    def _validate(self, merged: bool = False):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            if self.use_rm and "rm_scores" not in test_output_gen_batch_padded.batch.keys():
                # for colocate reward models, we need to sleep rollout model
                # to spare GPU memory for reward model
                self.checkpoint_manager.sleep_replicas()
                batch_reward = self._compute_reward_colocate(test_output_gen_batch_padded)
                test_output_gen_batch_padded = test_output_gen_batch_padded.union(batch_reward)
                # wake up rollout model
                # replace with wake_up method once supported
                self.checkpoint_manager.update_weights(self.global_steps)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # Store original inputs
            input_ids = test_batch.batch["prompts"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            # evaluate using reward_function
            reward_tensor, reward_extra_info = extract_reward(test_batch)

            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            # Per-sample token lengths for downstream val response-length distributions
            # (mirrors the train dump in _log_rollout_data; tokens, not chars).
            val_length_info = _compute_response_info(test_batch)
            val_prompt_lengths = val_length_info["prompt_length"].detach().cpu().long().tolist()
            val_response_lengths = val_length_info["response_length"].detach().cpu().long().tolist()
            reward_extra_infos_dict["prompt_length"].extend(val_prompt_lengths)
            reward_extra_infos_dict["response_length"].extend(val_response_lengths)
            reward_extra_infos_dict["total_tokens"].extend(
                [p + r for p, r in zip(val_prompt_lengths, val_response_lengths)]
            )

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        if merged:
            print("_merge_validation_results validate result will be merged")
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }
        data_sources = np.concatenate(data_source_lst, axis=0)
        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _merge_validation_results(self, result_a, result_b):
        if result_a is None and result_b is None:
            return {}
        if result_a is None:
            result_a = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}
        if result_b is None:
            result_b = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}

        if not result_a.get("data_sources") and not result_b.get("data_sources"):
            return {}

        data_sources = np.concatenate(result_a["data_sources"] + result_b["data_sources"], axis=0)
        sample_uids = result_a["sample_uids"] + result_b["sample_uids"]
        sample_turns = result_a["sample_turns"] + result_b["sample_turns"]

        reward_extra_infos_dict = {}
        all_keys = set(result_a["reward_extra_infos_dict"].keys()) | set(result_b["reward_extra_infos_dict"].keys())
        for key in all_keys:
            list_a = result_a["reward_extra_infos_dict"].get(key, [])
            list_b = result_b["reward_extra_infos_dict"].get(key, [])
            reward_extra_infos_dict[key] = list_a + list_b

        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                distillation_config=self.config.get("distillation"),
                role=str(actor_role),
            )
            self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            # convert critic_cfg into TrainingWorkerConfig for the unified model engine worker
            from verl.workers.engine_workers import TrainingWorkerConfig

            orig_critic_cfg = critic_cfg
            engine_config: EngineConfig = orig_critic_cfg.engine
            engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
            engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu

            critic_cfg = TrainingWorkerConfig(
                model_type="value_model",
                model_config=orig_critic_cfg.model,
                engine_config=engine_config,
                optimizer_config=orig_critic_cfg.optim,
                checkpoint_config=orig_critic_cfg.checkpoint,
                extra_context=getattr(self, "_critic_extra_context", {}),
            )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/verl-project/verl/blob/master/examples/tutorial/ray/tutorial.ipynb
        # for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.reset()
            # assign critic loss
            from functools import partial

            from verl.workers.utils.losses import value_loss

            value_loss_ = partial(value_loss, config=orig_critic_cfg)
            self.critic_wg.set_loss_fn(value_loss_)

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # OPSD colocated-judge memory fix: OPSD adds a frozen-teacher ref that plain GRPO lacks. With
        # no LoRA the ref is NOT a separate pool — it is a second engine (`self.ref`) INSIDE the
        # ActorRolloutRef hybrid worker (so `ref_policy_wg is actor_rollout_wg`). The colocated reward
        # model (gpt-oss-120b judge) is constructed next and runs vLLM `profile_run` to size its KV
        # cache; the extra resident ref tips that profile into a cublasCreate/NVLink OOM (actor+judge
        # alone fit — the no-ref baseline). Offload the ref AND actor engines to CPU + free the reserve
        # so the judge profiles in a no-ref run's free memory; both auto-onload on next forward. Guarded
        # to a colocated RM with a reference policy, so plain-GRPO / standalone-judge are untouched.
        colocated_rm = self.use_rm and not self.config.reward.reward_model.enable_resource_pool
        if colocated_rm and getattr(self, "use_reference_policy", False):
            try:
                self.actor_rollout_wg.offload_colocated_for_rm()
                print("[opsd] offloaded ref+actor to CPU before colocated judge init", flush=True)
            except Exception as _e:  # never let a memory-hint break init
                print(f"[opsd] offload-before-judge skipped: {_e}", flush=True)

        # create reward loop manager
        from verl.experimental.reward_loop import RewardLoopManager

        # initalize reward loop manager
        # reward model (colocate or standalone): get resource_pool
        # no reward model: resource_pool = None
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel) if self.use_rm else None
        self.reward_loop_manager = RewardLoopManager(
            config=self.config,
            rm_resource_pool=resource_pool,
        )

        # create async rollout manager and request scheduler
        # Note: mode is always "async" since sync mode is deprecated
        self.async_rollout_mode = True

        # Distillation loss config is needed by _update_actor whenever distillation is enabled —
        # including colocated OPSD (distillation.opsd_colocated=True), where use_teacher_policy is
        # False because the teacher is the colocated ref model and there is no teacher pool/manager.
        if is_distillation_enabled(self.config.get("distillation")):
            self.distillation_config: DistillationConfig = omega_conf_to_dataclass(self.config.distillation)
        else:
            self.distillation_config = None

        # initialize teacher loop manager (separate teacher pool only)
        if self.use_teacher_policy:
            use_hidden_states = self.distillation_config.distillation_loss.loss_settings.use_hidden_states

            if use_hidden_states:
                from verl.experimental.teacher_loop.nitrobrew_teacher import NitrobrewTeacherModelManager

                teacher_resource_pool = self.resource_pool_manager.get_resource_pool(Role.TeacherModel)
                gpus_per_replica = self.distillation_config.teacher_models[
                    next(iter(self.distillation_config.teacher_models))
                ].per_replica_world_size

                self.teacher_model_manager = NitrobrewTeacherModelManager(
                    teacher_model_configs=self.distillation_config.teacher_models,
                    gpus_per_replica=gpus_per_replica,
                )
                self.actor_rollout_wg.set_teacher_unembed(self.teacher_model_manager.w_up)
            else:
                from verl.experimental.teacher_loop import MultiTeacherModelManager

                teacher_resource_pool = self.resource_pool_manager.get_resource_pool(Role.TeacherModel)
                self.teacher_model_manager = MultiTeacherModelManager(
                    config=self.config,
                    resource_pool=teacher_resource_pool,
                )
        else:
            self.teacher_model_manager = None

        # Support custom AgentLoopManager via config
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        # infrastructure overview: https://verl.readthedocs.io/en/latest/advance/reward_loop.html#architecture-design
        # agent_reward_loop: streaming reward computation with actor rollout
        # two conditions satisfied: (1) no reward model, or (2) reward model with extra resource pool
        enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool

        self.llm_server_manager = LLMServerManager.create(
            config=self.config, worker_group=self.actor_rollout_wg, rollout_resource_pool=actor_rollout_resource_pool
        )

        # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
        # to stream reward computation with actor rollout
        # To stream teacher computation with actor rollout, we instead pass the full manager so that the
        # teacher loop workers can sleep/wake together with rollout workers
        reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None
        self.async_rollout_manager = AgentLoopManager.create(
            config=self.config,
            llm_client=self.llm_server_manager.get_client(),
            teacher_client=self.teacher_model_manager.get_client() if self.use_teacher_policy else None,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )

        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        # Support custom CheckpointEngineManager via config
        checkpoint_manager_class_fqn = self.config.actor_rollout_ref.rollout.get("checkpoint_manager_class")
        if checkpoint_manager_class_fqn:
            CheckpointEngineManager = load_class_from_fqn(checkpoint_manager_class_fqn, "CheckpointEngineManager")
        else:
            from verl.checkpoint_engine import CheckpointEngineManager
        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            trainer=self.actor_rollout_wg,
            replicas=self.llm_server_manager.get_replicas(),
        )

        # sleep all replicas to load checkpoint
        self.checkpoint_manager.sleep_replicas()

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        # Retention "keep last N + best-training-reward" (SP_KEEP_BEST_CKPT): DEFAULT-ON -- runs unless
        # explicitly disabled with SP_KEEP_BEST_CKPT=0 (prevents runs from silently hoarding a
        # checkpoint per step, e.g. 102 steps x 47GB = 3.2TB). The DRIVER becomes the
        # sole pruner (see _retain_recent_and_best, called from fit() once the corrected reward is
        # known), so disable the workers' own last-N rotation -- otherwise it would blindly drop the
        # best-reward ckpt the moment it ages past N. None => ensure_checkpoint_capacity no-ops.
        if os.environ.get("SP_KEEP_BEST_CKPT", "1") not in ("0", "false", "False", "no"):
            max_actor_ckpt_to_keep = None
            max_critic_ckpt_to_keep = None
            self._last_saved_global_step = self.global_steps

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # AEC: persist the driver's integral clip constant k next to the checkpoint so it survives
        # resume / job restarts (else k resets to 0 on each restart and re-climbs from scratch).
        from verl.trainer.ppo import aec as _aec
        _aec.save_k(local_global_step_folder)
        # Difficulty sampling: persist per-qid pass-rate stats the same way (difficulty_state.json).
        from verl.trainer.ppo import difficulty as _difficulty
        _difficulty.save_state(local_global_step_folder)
        # Replay-prefix training (SP_REPLAY_ENABLE): persist the success-EMA table + the local
        # dataset-step cursor the same way (sp_replay_state.json). The step's delta was already
        # appended at the reward site, so "delta k durable before checkpoint k" holds here.
        from verl.trainer.ppo import sp_replay as _sp_replay
        _sp_replay.save_state(local_global_step_folder)
        # Generative-Q readiness (SP_Q_ENABLE): persist q_state.json
        # (cursor/counters/readiness/MAE window) + the O_Q per-rank optimizer shards. The
        # step's q delta was appended after the update (before this save), so the same
        # "delta k durable before checkpoint k" contract holds.
        # Both saves FAIL CLOSED: a post-branch checkpoint without q_state.json or
        # without O_Q shards is fatal on resume, so it must never be published
        # as `latest`. Raising here happens before the tracker file is written below, which
        # leaves auto-resume pointing at the previous complete checkpoint.
        from verl.trainer.ppo import sp_q_readiness as _sp_q
        if _sp_q.enabled():
            _sp_q.save_state(local_global_step_folder)
            self._sp_q_save_optim_verified(local_global_step_folder)
            if self._sp_q_separate():
                self._sp_q_save_q_model_verified(local_global_step_folder)
                # Periodically freeze a copy OUTSIDE the auto-pruned tree so a
                # recovery point can never be lost to retention. Backgrounded + exception-guarded: never blocks/breaks the
                # run. SP_Q_FREEZE_EVERY=0 disables.
                try:
                    _fe = int(os.environ.get("SP_Q_FREEZE_EVERY", "0"))
                    if _fe > 0 and int(self.global_steps) % _fe == 0:
                        import subprocess
                        _root = os.path.dirname(local_global_step_folder)
                        _prot = os.path.join(os.path.dirname(_root), "protected_checkpoints")
                        os.makedirs(_prot, exist_ok=True)
                        _dst = os.path.join(_prot, f"global_step_{self.global_steps}_frozen")
                        # complete iff the .frozen_complete marker exists; a bare dir (no
                        # marker) is a torn/aborted freeze and is re-done.
                        if not os.path.exists(os.path.join(_dst, ".frozen_complete")):
                            # clean any stale partials, copy, atomic-rename, write the
                            # completeness marker, THEN chmod read-only; on ANY failure
                            # remove the partials so a torn copy is never mistaken for good.
                            _cmd = (
                                f"rm -rf '{_dst}.tmp' '{_dst}' && cp -r '{local_global_step_folder}' '{_dst}.tmp' && "
                                f"mv '{_dst}.tmp' '{_dst}' && date > '{_dst}/.frozen_complete' && "
                                f"chmod -R a-w '{_dst}' && echo \"[sp_q-sep] froze {_dst}\" "
                                f"|| {{ echo \"[sp_q-sep] freeze FAILED for {_dst}\"; rm -rf '{_dst}.tmp' '{_dst}'; }}"
                            )
                            subprocess.Popen(_cmd, shell=True)
                            print(f"[sp_q-sep] freezing protected copy -> {_dst} (background, marker-gated)", flush=True)
                except Exception as _e:  # noqa: BLE001
                    print(f"[sp_q-sep] periodic freeze skipped (non-fatal): {_e}", flush=True)

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _retain_recent_and_best(self, train_reward):
        """Checkpoint retention (SP_KEEP_BEST_CKPT, DEFAULT-ON): keep the last ``max_actor_ckpt_to_keep``
        steps (5 if unset), the single highest-training-reward step, AND every step at a multiple of
        ``SP_CKPT_KEEP_EVERY`` (DEFAULT 10) -- removing only the heavy actor/critic weight dirs of
        every other step. data.pt / aec_k.json / difficulty_state.json are left in place so the
        dashboard + difficulty warm-start still see every step. The best-reward pointer is persisted
        to ``checkpoints/best_reward.json`` so it survives resume.

        The keep-every stride exists because the rolling window alone makes mid-run steps
        unrecoverable: with keep_last=5 and save_freq=1, step 137 is gone by step 143, so no later
        analysis can re-evaluate or branch from it. A stride of 10 matches the usual SP_TEST_FREQ, so
        the retained steps are the ones that also have validation rows.

        THIS COSTS DISK, deliberately and unboundedly: ~47 GB per retained checkpoint at 4B/FSDP
        means a 500-step run holds ~50 of them, ~2.4 TB, on top of the rolling window. Set
        SP_CKPT_KEEP_EVERY=0 to disable the stride and keep only last-N + best.

        Driver-only (the workers' own last-N rotation is disabled in _save_checkpoint when enabled).
        Never raises -- a pruning error must not crash training."""
        import glob as _glob
        import shutil as _shutil

        try:
            ckpt_root = self.config.trainer.default_local_dir
            keep_last = self.config.trainer.get("max_actor_ckpt_to_keep", None)
            keep_last = int(keep_last) if keep_last else 5
            best_path = os.path.join(ckpt_root, "best_reward.json")
            best = {"step": None, "reward": None}
            if os.path.exists(best_path):
                try:
                    with open(best_path) as f:
                        best = json.load(f)
                except Exception:
                    best = {"step": None, "reward": None}
            # Update the best pointer with THIS (already-saved) step.
            if train_reward is not None:
                br = best.get("reward")
                if br is None or float(train_reward) > float(br):
                    best = {"step": int(self.global_steps), "reward": float(train_reward)}
                    try:
                        with open(best_path, "w") as f:
                            json.dump(best, f)
                    except Exception as e:
                        print(f"[ckpt-retain] could not persist best_reward.json: {e}", flush=True)
            # Keep-set: last `keep_last` step numbers + the best-reward step.
            steps = []
            for d in _glob.glob(os.path.join(ckpt_root, "global_step_*")):
                s = os.path.basename(d).rsplit("_", 1)[-1]
                if s.isdigit():
                    steps.append(int(s))
            steps.sort()
            keep = set(steps[-keep_last:]) if keep_last > 0 else set(steps)
            if best.get("step") is not None:
                keep.add(int(best["step"]))
            # Every multiple of the stride is retained permanently. Parsed defensively: a malformed
            # value must fall back to the default rather than silently disabling retention (a typo
            # here is unrecoverable -- the weights are already deleted by the time anyone notices).
            try:
                keep_every = int(os.environ.get("SP_CKPT_KEEP_EVERY", "10"))
            except ValueError:
                print("[ckpt-retain] SP_CKPT_KEEP_EVERY is not an integer; using the default 10",
                      flush=True)
                keep_every = 10
            if keep_every > 0:
                keep |= {s for s in steps if s % keep_every == 0}
            # Prune the weight dirs of every other step (leave the small state files intact).
            # sp_q_optim: the generative-Q O_Q shards (~8 bytes/param fp32 moments) are as
            # heavy as an optimizer state and must rotate with the weights they belong to.
            for s in steps:
                if s in keep:
                    continue
                # sp_q_optim = fused-Q O_Q shards; q_model = separate-Q theta_Q shards
                # — both are model-heavy and must rotate with the weights.
                for sub in ("actor", "critic", "sp_q_optim", "q_model"):
                    p = os.path.join(ckpt_root, f"global_step_{s}", sub)
                    if os.path.isdir(p):
                        _shutil.rmtree(p, ignore_errors=True)
                        print(f"[ckpt-retain] pruned weights global_step_{s}/{sub} "
                              f"(keep last {keep_last} + best step {best.get('step')}"
                              f"{f' + every {keep_every}' if keep_every > 0 else ''})", flush=True)
        except Exception as e:
            print(f"[ckpt-retain] retention pass failed (non-fatal): {e}", flush=True)

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                # Replay-prefix training: finalize with a fresh cursor (0) + truncate any stale
                # delta log so the first batch is built from the pristine frozen seed.
                from verl.trainer.ppo import sp_replay as _sp_replay
                _sp_replay.on_checkpoint_load(None)
                from verl.trainer.ppo import sp_q_readiness as _sp_q
                if _sp_q.enabled():
                    _sp_q.on_checkpoint_load(None)
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")
        # AEC: restore the driver's integral clip constant k persisted at save time (keyed to this
        # exact global_step dir). Seeds both aec._S["k"] and self._aec_k so step N+1 continues from it.
        from verl.trainer.ppo import aec as _aec
        _restored_aec_k = _aec.load_k(global_step_folder)
        if _restored_aec_k is not None:
            self._aec_k = _restored_aec_k
        # Difficulty sampling: restore per-qid pass-rate stats (load BEFORE any weight()/observe()
        # so a lazy _init cannot interleave -- same init-order rule as aec.set_k). Priority:
        #   1. difficulty_state.json in the resumed checkpoint (difficulty resume)
        #   2. warm-start p-hat by replaying the run's historical rollout dumps up to the resumed
        #      step (grafting onto a PRE-difficulty checkpoint -- don't start blind)
        #   3. fresh/empty -> all weights equal -> c == 1 (plain uniform PPO until observed)
        from verl.trainer.ppo import difficulty as _difficulty
        if _difficulty.load_state(global_step_folder) is None and _difficulty.enabled():
            _rollout_dir = os.path.abspath(os.path.join(global_step_folder, "..", "..", "rollouts"))
            _difficulty.warmstart_from_rollouts(_rollout_dir, max_step=self.global_steps)
        # Replay-prefix training: restore the EMA table + local dataset-step cursor from
        # sp_replay_state.json in this checkpoint (absent on the first attach after a graft ->
        # fresh cursor 0, EMA prior = bucket occupancy k/capacity), then apply the delta log up
        # to the cursor and physically truncate the invalidated tail (the double-apply guard).
        from verl.trainer.ppo import sp_replay as _sp_replay
        _sp_replay.on_checkpoint_load(global_step_folder)
        # Generative-Q readiness (SP_Q_ENABLE): restore q_state.json + rebuild the Q FIFO
        # (seed shards + truncated delta log), then load O_Q's per-rank shards. A checkpoint
        # WITH q_state.json but WITHOUT O_Q shards is fatal; the branch-point
        # first attach (parent checkpoint, no q_state.json) legitimately starts O_Q fresh.
        from verl.trainer.ppo import sp_q_readiness as _sp_q
        if _sp_q.enabled():
            _sp_q.on_checkpoint_load(global_step_folder)
            _q_state_present = os.path.exists(
                os.path.join(global_step_folder, _sp_q.Q_STATE_FILE_NAME)
            )
            _qmodel_present = os.path.isdir(os.path.join(global_step_folder, "q_model"))
            # Separate-Q branch-first attach: theta_Q + O_Q are initialized from the SFT
            # checkpoint (SP_Q_INIT_PATH; below, after the actor load), so do NOT force-load the fused O_Q from the
            # branch checkpoint. Post-branch separate resume (q_model present) DOES restore
            # its own O_Q. Fused-Q resume is unchanged.
            if (not self._sp_q_separate()) or _qmodel_present:
                if os.environ.get("SP_Q_OPTIM_FRESH", "0") == "1":
                    # MLP-head branch: the grafted checkpoint's O_Q was trained under the
                    # generative CE loss; the MLP surface re-inits it.
                    # The attach script passes this only while latest ckpt <= BRANCH, so
                    # the branch's own later checkpoints load normally.
                    print("[sp_q] SP_Q_OPTIM_FRESH=1: skipping O_Q load, fresh optimizer",
                          flush=True)
                else:
                    self.actor_rollout_wg.sp_q_load_optim(global_step_folder, strict=_q_state_present)

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # separate-Q: now that the module holds the POLICY, restore theta_Q from
        # the checkpoint (post-branch resume) or initialize it from the SFT endpoint
        # (SP_Q_INIT_PATH; branch-point first attach). Must run AFTER the actor load so the theta_pi
        # snapshot inside sp_q_init_separate is the real policy.
        if _sp_q.enabled() and self._sp_q_separate():
            _res = [r for r in (self.actor_rollout_wg.sp_q_load_q_model(global_step_folder) or []) if r is not None]
            _loaded_n = sum(1 for r in _res if r.get("loaded"))
            if _res and 0 < _loaded_n < len(_res):
                raise RuntimeError(
                    f"sp_q-sep: PARTIAL theta_Q resume ({_loaded_n}/{len(_res)} ranks loaded) under "
                    f"{global_step_folder} -- some ranks would reach the wave without theta_Q; "
                    "refusing. Resume an earlier complete checkpoint."
                )
            if _loaded_n == 0:
                # fail-CLOSED: only the BRANCH-POINT attach (resuming the branch
                # checkpoint, which has no q_model/) legitimately initializes theta_Q from
                # the SFT checkpoint. A post-branch separate-Q checkpoint (step > branch)
                # that lacks q_model/ is corrupt — refuse rather than silently reset Q to
                # the SFT init.
                try:
                    _step = int(os.path.basename(global_step_folder.rstrip("/")).rsplit("_", 1)[-1])
                except (ValueError, IndexError):
                    _step = -1
                _branch = int(os.environ.get("SP_Q_BRANCH_STEP", "55"))
                if _step != _branch:
                    raise RuntimeError(
                        f"sp_q-sep: resuming global_step {_step} (!= branch {_branch}) but q_model/ "
                        f"is missing/incomplete under {global_step_folder} -- init-from-SFT is "
                        f"legitimate ONLY at the exact branch step. A step<branch resume would pair "
                        f"theta_Q/readiness trained through step {_branch} with an EARLIER policy "
                        f"(future-data leakage); a step>branch resume means a corrupt "
                        "checkpoint. Refusing. Resume the branch or a complete post-branch "
                        "checkpoint."
                    )
                _sft = os.environ["SP_Q_INIT_PATH"]  # SFT init checkpoint step dir
                _hf = os.path.join(_sft, "huggingface")
                _hf = _hf if os.path.isdir(_hf) else _sft
                self.actor_rollout_wg.sp_q_init_separate(
                    _hf, _sft, float(os.environ.get("SP_Q_LR_BASE", "1e-5"))
                )
                print(f"[sp_q-sep] theta_Q + O_Q initialized from the SFT checkpoint {_sft} (branch step {_step})", flush=True)
                # readiness warm-start from the prequential sim — branch only
                _sp_q.harness().load_readiness_warmstart(os.environ.get("SP_Q_WARMSTART_PATH", ""))
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        # Difficulty sampling: DifficultyWeightedSampler draws i.i.d.-with-replacement, so there is no
        # epoch position worth restoring (and p̂ is restored separately via difficulty_state.json /
        # warm-start). Crucially, a checkpoint written by the stock RandomSampler carries a torchdata
        # sampler state that is INCOMPATIBLE with this sampler: restoring it asserts inside torchdata
        # (stateful_dataloader/sampler.py:load_state_dict) and crashes at first iteration. So skip the
        # dataloader-state restore entirely when difficulty sampling is on — a fresh sampler each
        # resume is correct for i.i.d. draws.
        from verl.trainer.ppo import difficulty as _difficulty
        from verl.trainer.ppo import sp_replay as _sp_replay
        if _difficulty.enabled():
            print(
                "[difficulty] skipping dataloader state restore (i.i.d.-with-replacement sampler; "
                "p-hat restored separately) — avoids the RandomSampler->DifficultyWeightedSampler "
                "torchdata state-schema mismatch.",
                flush=True,
            )
        elif _sp_replay.enabled():
            # The replay dataset is a deterministic (step,slot) stream: its position derives from
            # the sp_replay dataset-step cursor (restored above), NOT from torchdata state. A
            # grafted parent checkpoint carries the PARENT dataset's dataloader state, which is
            # incompatible with this dataset's length/sampler — never restore it.
            print(
                "[sp_replay] skipping dataloader state restore (deterministic (step,slot) dataset; "
                "position derives from the sp_replay dataset-step cursor).",
                flush=True,
            )
        elif os.path.exists(dataloader_local_path):
            steps_per_epoch = len(self.train_dataloader)
            at_epoch_boundary = steps_per_epoch > 0 and self.global_steps % steps_per_epoch == 0
            if at_epoch_boundary:
                print(
                    f"Skipping dataloader state restore: global_steps={self.global_steps} "
                    f"is at an epoch boundary (steps_per_epoch={steps_per_epoch}). "
                    f"The saved state marks the dataloader as exhausted. "
                    f"Next epoch will iterate from scratch."
                )
            else:
                dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
                self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()

    def _get_dp_size(self, worker_group, role: str) -> int:
        """Get data parallel size from worker group dispatch info.

        This method retrieves the data parallel size by querying the dispatch info
        for the specified role. The dispatch info is cached for subsequent calls.

        Args:
            worker_group: The worker group to query dispatch info from.
            role: The role name (e.g., "actor", "critic") to get DP size for.

        Returns:
            The data parallel size (number of DP ranks).
        """
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _sp_dp_pad_for_dispatch(self, batch: DataProto, worker_group, role: str):
        """sp_dp_pad (SP_DP_PAD=1): make `batch` chunkable by the role's dp size.

        Returns (batch_to_dispatch, keep) where keep is a bool mask over the padded rows
        (True = real) to apply to the call's per-row OUTPUT, or None when nothing was padded
        (flag off, or the length already divides -- the pre-existing path, untouched). The
        caller's own `batch` is never mutated. Pads are placed per `self._sp_dp_plan` (the
        positions the seqlen balancer gave them this step) when the plan matches this batch's
        length, else appended. See verl.trainer.ppo.sp_dp_pad.
        """
        from verl.trainer.ppo import sp_dp_pad as _sp_dp

        if not _sp_dp.enabled():
            return batch, None
        dp_size = self._get_dp_size(worker_group, role)
        n_pad = _sp_dp.pad_count(len(batch), dp_size)
        if n_pad == 0:
            return batch, None
        plan = getattr(self, "_sp_dp_plan", None)
        if not _sp_dp.plan_fits(plan, len(batch), dp_size):
            plan = None
        padded = _sp_dp.pad_batch(batch, dp_size, positions=plan)
        print(f"[sp_dp_pad] {role}: {len(batch)} rows + {n_pad} pad -> {len(padded)} "
              f"({len(padded) // dp_size}/rank over dp={dp_size}; "
              f"{'balancer positions' if plan is not None else 'appended'})", flush=True)
        return padded, _sp_dp.real_mask(padded)

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens.

        When use_prefix_grouper is enabled, uses group-level balancing to keep samples with
        the same uid together on the same rank for prefix sharing optimization.
        """
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        # Get dp_size from dispatch info to correctly balance across data parallel ranks
        # Note: world_size may include tensor/pipeline parallel dimensions, but we only want DP
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")

        # Use group-level balancing for PrefixGrouper to keep same-uid samples together
        if getattr(self, "use_prefix_grouper", False) and "uid" in batch.non_tensor_batch:
            from verl.utils.seqlen_balancing import get_group_balanced_partitions

            uid_list = list(batch.non_tensor_batch["uid"])
            seqlen_list = global_seqlen_lst.tolist()

            # Count number of uid groups
            num_groups = len(set(uid_list))

            if num_groups % dp_size != 0:
                raise ValueError(
                    f"PrefixGrouper with balance_batch requires num_uid_groups ({num_groups}) "
                    f"% dp_size ({dp_size}) == 0. "
                    f"This ensures each rank gets equal number of groups. "
                    f"Current batch_size={batch_size}, adjust batch_size to be a multiple of "
                    f"dp_size * rollout.n."
                )

            global_partition_lst = get_group_balanced_partitions(
                seqlen_list=seqlen_list,
                uid_list=uid_list,
                k_partitions=dp_size,
            )

        elif keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        # Skip reordering within partitions for PrefixGrouper to maintain uid grouping
        if not getattr(self, "use_prefix_grouper", False):
            for idx, partition in enumerate(global_partition_lst):
                partition.sort(key=lambda x: (workload_lst[x], x))
                ordered_partition = partition[::2] + partition[1::2][::-1]
                global_partition_lst[idx] = ordered_partition

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _compute_values(self, batch: DataProto) -> DataProto:
        # sp_dp_pad: dispatch padded, drop the pad rows' outputs (see _compute_old_log_prob)
        batch, _dp_keep = self._sp_dp_pad_for_dispatch(batch, self.critic_wg, "train")
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to nopadding
        batch_td = left_right_2_no_padding(batch_td)
        # step 3: add meta info
        tu.assign_non_tensor(batch_td, compute_loss=False)
        output = self.critic_wg.infer_batch(batch_td)
        output = output.get()
        values = tu.get(output, "values")
        values = no_padding_2_padding(values, batch_td)
        values = tu.get_tensordict({"values": values.float()})
        values = DataProto.from_tensordict(values)
        if _dp_keep is not None:
            values = values.select_idxs(_dp_keep)
        return values

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        # sp_dp_pad: dispatch padded, drop the pad rows' outputs (see _compute_old_log_prob)
        if self.ref_in_actor:
            batch, _dp_keep = self._sp_dp_pad_for_dispatch(batch, self.actor_rollout_wg, "actor")
        else:
            batch, _dp_keep = self._sp_dp_pad_for_dispatch(batch, self.ref_policy_wg, "ref")
        # step 1: convert dataproto to tensordict.
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to nopadding
        batch_td = left_right_2_no_padding(batch_td)
        # step 3: add meta info
        metadata = {"calculate_entropy": False, "compute_loss": False}
        if self.ref_in_actor:
            metadata["no_lora_adapter"] = True
        tu.assign_non_tensor(batch_td, **metadata)
        if self.ref_in_actor:
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
        else:
            output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
        # gather output
        log_probs = tu.get(output, "log_probs")
        # step 4. No padding to padding
        log_probs = no_padding_2_padding(log_probs, batch_td)
        # step 5: rebuild a tensordict and convert to dataproto
        ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
        ref_log_prob = DataProto.from_tensordict(ref_log_prob)
        if _dp_keep is not None:
            ref_log_prob = ref_log_prob.select_idxs(_dp_keep)

        return ref_log_prob

    def _compute_old_log_prob(self, batch: DataProto):
        # sp_dp_pad: a non-divisible batch is dispatched padded; the pad rows' outputs are
        # dropped below so the returned proto lines up with the caller's (unpadded) batch.
        batch, _dp_keep = self._sp_dp_pad_for_dispatch(batch, self.actor_rollout_wg, "actor")
        # TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free
        # step 1: convert dataproto to tensordict.
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to nopadding
        batch_td = left_right_2_no_padding(batch_td)
        # step 3: add meta info
        calculate_sum_pi_squared = self.config.actor_rollout_ref.actor.get("calculate_sum_pi_squared", False)
        tu.assign_non_tensor(
            batch_td,
            calculate_entropy=True,
            calculate_sum_pi_squared=calculate_sum_pi_squared,
            compute_loss=False,
        )
        output = self.actor_rollout_wg.compute_log_prob(batch_td)
        # gather output
        entropy = tu.get(output, "entropy")
        log_probs = tu.get(output, "log_probs")
        routed_experts = tu.get(output, "routed_experts")
        sum_pi_squared = tu.get(output, "sum_pi_squared") if calculate_sum_pi_squared else None

        old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
        # step 4. No padding to padding
        entropy = no_padding_2_padding(entropy, batch_td)
        log_probs = no_padding_2_padding(log_probs, batch_td)
        if sum_pi_squared is not None:
            sum_pi_squared = no_padding_2_padding(sum_pi_squared, batch_td)
        # step 5: rebuild a tensordict and convert to dataproto
        result = {"old_log_probs": log_probs.float(), "entropys": entropy.float()}
        if routed_experts is not None:
            result["routed_experts"] = routed_experts
        if sum_pi_squared is not None:
            result["sum_pi_squared"] = sum_pi_squared.float()
        old_log_prob = tu.get_tensordict(result)
        old_log_prob = DataProto.from_tensordict(old_log_prob)
        if _dp_keep is not None:
            old_log_prob = old_log_prob.select_idxs(_dp_keep)
        return old_log_prob, old_log_prob_mfu

    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        batch.meta_info["temperature"] = rollout_config.temperature
        # update actor (the DataProto -> no-padding TensorDict conversion happens below, after
        # the sp_dp_pad block has decided whether the dispatched batch carries pad rows)
        calculate_entropy = self.config.actor_rollout_ref.actor.calculate_entropy or (
            self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
        )
        if is_distillation_enabled(self.config.get("distillation")):
            _ls = self.distillation_config.distillation_loss.loss_settings
            distillation_use_topk = _ls.use_topk or _ls.use_hidden_states
        else:
            distillation_use_topk = False
        ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
        ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        from verl.trainer.ppo import sp_dyn_group as _sp_dyn
        if _sp_dyn.enabled():
            # Dynamic group size: the trained sequence count varies per step and the
            # actor's iterator asserts exact divisibility. Keep the fixed arm's structure
            # of exactly TWO PPO minibatches per step (the interleaved Q phase sits between
            # them) by sizing the minibatch to half the batch. The round loop pads the
            # trained total to a multiple of 2*world_size, so this divides by dp.
            _n_seq = len(batch)
            _dp = self._get_dp_size(self.actor_rollout_wg, "actor")
            assert _n_seq % (2 * _dp) == 0, (
                f"sp_dyn: {_n_seq} trained sequences is not divisible by 2*dp={2 * _dp}")
            ppo_mini_batch_size = _n_seq // 2
            print(f"[sp_dyn] update_actor: {_n_seq} seqs -> 2 minibatches of "
                  f"{ppo_mini_batch_size}", flush=True)
        # sp_dp_pad (SP_DP_PAD=1): a data-parallel size that divides neither the trained
        # sequence count nor the global minibatch (e.g. 3072 and 1536 over dp=56 on 7 nodes).
        # Pad the trained batch to a multiple of dp with zero-response-mask duplicates (exactly
        # zero contribution to the token-mean loss AND its all-reduced denominator) and let the
        # DRIVER assign minibatch membership so that every minibatch holds exactly
        # ppo_mini_batch_size REAL sequences and every rank runs every minibatch -- the
        # interleaved order PPO(M1) -> Q -> PPO(M2) is preserved with the same real minibatch sizes an
        # 8-node run trains. Divisible shapes (and sp_dyn, which pads its own rounds) take the
        # pre-existing positional path untouched. See verl.trainer.ppo.sp_dp_pad.
        from verl.trainer.ppo import sp_dp_pad as _sp_dp
        _mini_kwargs = {"mini_batch_size": ppo_mini_batch_size}
        _dp = self._get_dp_size(self.actor_rollout_wg, "actor")
        if _sp_dp.enabled() and not _sp_dyn.enabled() and (
            len(batch) % _dp != 0 or ppo_mini_batch_size % _dp != 0
        ):
            _n_real = len(batch)
            assert _n_real % ppo_mini_batch_size == 0, (
                f"sp_dp_pad: {_n_real} trained seqs do not form whole minibatches of "
                f"{ppo_mini_batch_size}"
            )
            _k = _n_real // ppo_mini_batch_size
            batch = _sp_dp.pad_batch(batch, _dp)  # new proto; the caller keeps the real rows
            _ids = _sp_dp.minibatch_ids(_sp_dp.pad_flags(batch).tolist(), _dp, _k)
            batch.batch[_sp_dp.MINIBATCH_ID_KEY] = torch.as_tensor(_ids, dtype=torch.long)
            _mini_kwargs = {"num_mini_batch": _k}
            print(f"[sp_dp_pad] update_actor: {_n_real} real seqs + {len(batch) - _n_real} pad -> "
                  f"{len(batch)} ({len(batch) // _dp}/rank over dp={_dp}); {_k} driver-assigned "
                  f"minibatches of {ppo_mini_batch_size} real seqs", flush=True)
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to no-padding
        batch_td = left_right_2_no_padding(batch_td)
        ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
        seed = self.config.actor_rollout_ref.actor.data_loader_seed
        shuffle = self.config.actor_rollout_ref.actor.shuffle
        tu.assign_non_tensor(
            batch_td,
            calculate_entropy=calculate_entropy,
            aec_k=float(getattr(self, "_aec_k", 0.0)),  # AEC clip constant: driver-computed, identical on all workers
            distillation_use_topk=distillation_use_topk,
            global_batch_size=ppo_mini_batch_size,
            **_mini_kwargs,
            epochs=ppo_epochs,
            seed=seed,
            dataloader_kwargs={"shuffle": shuffle},
            compute_loss=True,
        )
        actor_output = self.actor_rollout_wg.update_actor(batch_td)
        actor_output = tu.get(actor_output, "metrics")
        actor_output = rename_dict(actor_output, "actor/")
        # modify key name
        actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
        actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})

        return actor_output

    def _update_critic(self, batch: DataProto) -> DataProto:
        batch_td = batch.to_tensordict()
        # step 2: convert from padding to no-padding
        batch_td = left_right_2_no_padding(batch_td)
        ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
        ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        ppo_epochs = self.config.critic.ppo_epochs
        seed = self.config.critic.data_loader_seed
        shuffle = self.config.critic.shuffle
        tu.assign_non_tensor(
            batch_td,
            global_batch_size=ppo_mini_batch_size,
            mini_batch_size=ppo_mini_batch_size,
            epochs=ppo_epochs,
            seed=seed,
            dataloader_kwargs={"shuffle": shuffle},
        )

        output = self.critic_wg.train_mini_batch(batch_td)
        output = output.get()
        output = tu.get(output, "metrics")
        output = rename_dict(output, "critic/")
        # modify key name
        output["perf/mfu/critic"] = output.pop("critic/mfu")
        critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        return critic_output

    # ------------------------------------------------------------------
    # Generative-Q readiness driver helpers (the per-step algorithm sites
    # are marked in fit()). All are called only when
    # sp_q_readiness.enabled().
    # ------------------------------------------------------------------

    def _sp_q_save_optim_verified(self, ckpt_dir: str):
        """Save O_Q's per-rank shards and verify the set is COMPLETE before the caller
        publishes the checkpoint. Every expected rank must report a save and
        every shard file must exist non-empty; only then is the `_complete.json` marker
        written -- which sp_q_load_optim requires on a strict (post-branch) resume, so a
        torn or partially pruned shard set is caught at load time too."""
        results = [r for r in (self.actor_rollout_wg.sp_q_save_optim(ckpt_dir) or []) if r is not None]
        if not results:
            raise RuntimeError(f"sp_q: sp_q_save_optim returned no results for {ckpt_dir}")
        world_sizes = {int(r["world_size"]) for r in results}
        if len(world_sizes) != 1:
            raise RuntimeError(f"sp_q: workers disagree on world_size {sorted(world_sizes)}")
        ws = world_sizes.pop()
        if len(results) != ws:
            raise RuntimeError(
                f"sp_q: {len(results)} O_Q shard results for world_size {ws} ({ckpt_dir}) -- "
                "incomplete optimizer state, refusing to publish this checkpoint"
            )
        unsaved = sorted(int(r["rank"]) for r in results if not r.get("saved"))
        if unsaved:
            raise RuntimeError(f"sp_q: ranks {unsaved} did not save an O_Q shard ({ckpt_dir})")
        d = os.path.join(ckpt_dir, "sp_q_optim")
        bad = [
            i for i in range(ws)
            if not os.path.exists(os.path.join(d, f"rank_{i}.pt"))
            or os.path.getsize(os.path.join(d, f"rank_{i}.pt")) == 0
        ]
        if bad:
            raise RuntimeError(
                f"sp_q: O_Q shards missing/empty for ranks {bad} under {d} -- resuming from "
                "this checkpoint would be fatal, refusing to publish it"
            )
        marker = os.path.join(d, "_complete.json")
        tmp = marker + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"world_size": ws, "global_step": int(self.global_steps)}, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, marker)
        print(f"[sp_q] O_Q shard set verified complete ({ws} ranks) -> {d}", flush=True)

    # ---------------- separate-Q driver surface ----------------
    def _sp_q_separate(self) -> bool:
        return os.environ.get("SP_Q_SEPARATE", "0") in ("1", "true", "True")

    def _sp_q_interleave(self) -> bool:
        """Tied-Q interleaving: the update order PPO(M1) -> Q-only -> PPO(M2), with
        the Q step LEFT APPLIED and its size priced by the next step's LR (the ladder)
        instead of the per-step movement cap. Mutually exclusive with separate-Q
        (that surface has no shared weights to interleave into)."""
        on = os.environ.get("SP_Q_INTERLEAVE", "0") in ("1", "true", "True")
        if on and self._sp_q_separate():
            raise ValueError(
                "SP_Q_INTERLEAVE=1 and SP_Q_SEPARATE=1 are mutually exclusive: interleaving "
                "prices the Q step against the PPO minibatch displacements on the SHARED "
                "weights, which separate-Q does not have (theta_Q is its own trajectory)."
            )
        return on

    def _sp_inflow_only(self) -> bool:
        """Inflow-only scratch rows: scratch (statement-stream) rows are buffer INFLOW only —
        generated with per-row rollout_n=1 (dataset stamps `sp_rollout_n`), admitted at the
        reward site, then physically dropped before advantage/update so the PPO loss sees
        only replay rows. Off = scratch rows are kept and trained like any other row."""
        return os.environ.get("SP_SCRATCH_INFLOW_ONLY", "0") in ("1", "true", "True")

    def _sp_per_row_repeat_counts(self, batch):
        """Per-row rollout counts when inflow-only is on and the dataset stamped them;
        None -> use the uniform config rollout.n at the repeat sites."""
        if not self._sp_inflow_only():
            return None
        # AUTHORITATIVE source: extra_info (per-row dicts, empirically proven to survive
        # the dataloader->trainer path — the reward-site drop reads it). The top-level
        # sp_rollout_n key has been observed to VANISH from the live batch (cause
        # unresolved) — kept only as a fallback.
        counts = None
        extra = batch.non_tensor_batch.get("extra_info")
        if (extra is not None and len(extra) and isinstance(extra[0], dict)
                and "sp_rollout_n" in extra[0]):
            counts = [int(ei.get("sp_rollout_n", 0)) for ei in extra]
        if counts is None:
            counts = batch.non_tensor_batch.get("sp_rollout_n")
        if counts is None and batch.batch is not None and "sp_rollout_n" in batch.batch.keys():
            counts = batch.batch["sp_rollout_n"].tolist()
        if counts is None:
            # LOUD: silently falling back to uniform repeat floods the global FIFO
            # (e.g. 512 admits/step). Dump the visible keys so the failure
            # diagnoses itself in the log.
            _nk = sorted(batch.non_tensor_batch.keys())
            _ek = sorted(extra[0].keys()) if (extra is not None and len(extra)
                                              and isinstance(extra[0], dict)) else None
            print(f"[sp_inflow] sp_rollout_n ABSENT -> uniform fallback; "
                  f"non_tensor={_nk} extra_info[0]={_ek}", flush=True)
            return None
        assert self.config.algorithm.adv_estimator != AdvantageEstimator.REMAX, (
            "SP_SCRATCH_INFLOW_ONLY is not validated with REMAX (baseline slicing)"
        )
        n = int(self.config.actor_rollout_ref.rollout.n)
        # Burst inflow (the stale-replay-buffer ablation): the stamp is three-valued.
        # POSITIVE = that many rollouts (1 = the classic single-rollout inflow row; >1 = a
        # burst-step inflow row).
        # 0 = the uniform config n (replay / cold_scratch rows — unchanged semantics).
        # NEGATIVE = drop the row from the step entirely: repeat count 0 removes it from
        # BOTH sample_level_repeat sites (gen batch and driver batch share this list), so
        # the row is never generated, never judged, and never admitted — that is what a
        # zero-inflow step looks like. A dataset that never stamps a negative resolves
        # exactly as with the two-valued stamp.
        resolved = [n if int(c) == 0 else max(int(c), 0) for c in counts]
        assert any(resolved), (
            "sp_inflow: every row of this batch resolved to 0 rollouts — the step would be "
            "empty. A burst schedule must never stamp the trained lane."
        )
        n_drop = sum(1 for c in counts if int(c) < 0)
        n_train = sum(1 for c in counts if int(c) == 0)
        inflow_seqs = sum(int(c) for c in counts if int(c) > 0)
        print(f"[sp_inflow] per-row repeat engaged: "
              f"{len(counts) - n_drop - n_train} inflow rows ({inflow_seqs} seqs) + "
              f"{n_train} train x{n} + {n_drop} dropped = {sum(resolved)} sequences",
              flush=True)
        return resolved

    def _sp_q_wave_maybe_separate(self, batch, probes: bool = True):
        """Run the Q wave. For separate-Q, swap theta_Q into the module and sync it into
        vLLM (so the wave predicts with the SEPARATE Q weights), then restore the policy
        weights into the module afterward (vLLM gets the policy back at the end-of-step
        update_weights). For fused-Q this is just the wave, unchanged.

        probes=False: a later round of a dynamic-group-size step (sp_dyn_group); the
        per-uid diagnostics were issued by round 1 and the wave rows are APPENDED."""
        self._sp_q_wave_slept = False
        sep = self._sp_q_separate()
        if sep:
            # naive weight-sync contract (engine_workers.update_weights): the rollout must
            # be ASLEEP before update_weights (it resumes/wakes internally). Post-generation
            # the replicas are AWAKE, so sleep first, then swap theta_Q into the module and
            # sync it into the (asleep -> wake) vLLM engine for the wave.
            self.checkpoint_manager.sleep_replicas()
            self.actor_rollout_wg.sp_q_swap_in_q()
            self.checkpoint_manager.update_weights(self.global_steps)  # asleep -> load theta_Q -> wake
        self._sp_q_run_wave_and_stamp(batch, probes=probes)
        if sep:
            self.actor_rollout_wg.sp_q_swap_out_q()                    # module <- policy (vLLM re-synced at end-of-step)

    def _sp_q_save_q_model_verified(self, ckpt_dir: str):
        """Persist theta_Q shards + completeness marker. World size is derived from the
        WORKER results (this runs on the non-distributed Ray controller, where
        torch.distributed is unavailable)."""
        import glob as _glob
        results = [r for r in (self.actor_rollout_wg.sp_q_save_q_model(ckpt_dir) or []) if r is not None]
        if not results:
            raise RuntimeError(f"sp_q-sep: sp_q_save_q_model returned no results for {ckpt_dir}")
        world_sizes = {int(r["world_size"]) for r in results}
        if len(world_sizes) != 1:
            raise RuntimeError(f"sp_q-sep: workers disagree on world_size {sorted(world_sizes)}")
        ws = world_sizes.pop()
        d = os.path.join(ckpt_dir, "q_model")
        present = _glob.glob(os.path.join(d, "rank_*.pt"))
        if len(present) != ws or len(results) != ws:
            raise RuntimeError(
                f"sp_q-sep: theta_Q shards incomplete ({len(present)} files / {len(results)} results "
                f"for world_size {ws}) under {d} -- refusing to publish this checkpoint"
            )
        marker = os.path.join(d, "_complete.json")
        with open(marker + ".tmp", "w") as f:
            json.dump({"world_size": ws, "global_step": int(self.global_steps)}, f)
        os.replace(marker + ".tmp", marker)
        print(f"[sp_q-sep] theta_Q shard set verified complete ({ws}) -> {d}", flush=True)

    def _sp_q_build_q_batch(self, rows, n_records):
        """Build the teacher-forced Q DataProto (identical layout to the fused Q phase)."""
        from verl.trainer.ppo import sp_q_readiness as _sp_q
        h = _sp_q.harness()
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")
        pad_rows = (-len(rows)) % dp_size
        shortest = min(range(len(rows)), key=lambda k: len(rows[k]["ctx_ids"]))
        all_rows = rows + [dict(rows[shortest], weight=0.0) for _ in range(pad_rows)]
        plens, rlens = [], []
        for r in all_rows:
            plen = len(h.prompt_ids_for_qid(r["rec"]["qid"]))
            plens.append(plen)
            rlens.append(len(r["ctx_ids"]) - plen + len(r["target_ids"]))
        P, R = max(plens), max(rlens)
        N = len(all_rows)
        pad_id = self.tokenizer.pad_token_id or 0
        input_ids = torch.full((N, P + R), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((N, P + R), dtype=torch.long)
        response_mask = torch.zeros((N, R), dtype=torch.long)
        tok_w = torch.zeros((N, R), dtype=torch.float32)
        is_ref = torch.zeros((N,), dtype=torch.int8)
        for i, r in enumerate(all_rows):
            plen = plens[i]
            ctx, tgt = r["ctx_ids"], r["target_ids"]
            prompt_ids = ctx[:plen]
            resp_ids = list(ctx[plen:]) + list(tgt)
            L = len(resp_ids)
            input_ids[i, P - plen:P] = torch.as_tensor(prompt_ids, dtype=torch.long)
            attention_mask[i, P - plen:P] = 1
            input_ids[i, P:P + L] = torch.as_tensor(resp_ids, dtype=torch.long)
            attention_mask[i, P:P + L] = 1
            t0 = L - len(tgt)
            response_mask[i, t0:L] = 1
            tok_w[i, t0:L] = float(r["weight"]) / 4.0
            is_ref[i] = 1 if r["variant"] == "ref" else 0
        position_ids = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0)
        # per-row REAL flag (1 for a genuine record, 0 for a dp-divisibility pad row). Rides
        # the batch through balancing/scatter so the worker can compute an EXACT per-minibatch
        # real-record denominator regardless of where padding lands.
        sp_q_real = torch.tensor([1 if float(r["weight"]) > 0 else 0 for r in all_rows], dtype=torch.int8)
        proto = DataProto.from_single_dict({
            "input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids,
            "response_mask": response_mask, "prompts": input_ids[:, :P], "responses": input_ids[:, P:],
            "sp_q_tok_w": tok_w, "sp_q_is_ref": is_ref, "sp_q_real": sp_q_real,
        })
        self._balance_batch(proto, metrics={}, logging_prefix="sp_q_seqlen")
        td = proto.to_tensordict()
        return left_right_2_no_padding(td)

    def _sp_q_train_separate_phase(self, metrics: dict):
        """Separate-Q update: draw the NEW step-k records, run 3 x minibatch-64 O_Q steps
        on theta_Q (left applied). No theta_0 capture, no movement cap. Then advance the Q
        cursor + durably append the step's delta (admissions) — the same bookkeeping the
        fused path does in _sp_q_finish_and_apply, which separate mode bypasses."""
        from verl.trainer.ppo import sp_q_readiness as _sp_q
        h = _sp_q.harness()
        sample = h._train_sample or {}
        rows = sample.get("rows") or []
        n_records = len(sample.get("picked_seqs") or rows)
        if not rows:
            metrics["q/q_phase_skipped"] = 1
            disp = {"q/q_phase_skipped": 1, "q/delta_q_applied": 0.0, "q/delta_ppo_net": 0.0}
        else:
            td = self._sp_q_build_q_batch(rows, n_records)
            tu.assign_non_tensor(
                td,
                sp_q_denom=int(n_records),
                sp_q_n_real=int(n_records),   # worker uses this for per-minibatch loss normalization
                sp_q_lr=float(os.environ.get("SP_Q_LR_BASE", "1e-5")),
                sp_q_clip=float(os.environ.get("SP_Q_GRAD_CLIP", "1.0")),
                sp_q_grad_steps=int(os.environ.get("SP_Q_GRAD_STEPS", "3")),
                sp_q_mb=int(os.environ.get("SP_Q_MB", "64")),
                sp_q_max_token_len=int(os.environ.get("SP_Q_MAX_TOKEN_LEN", "53360")),
                temperature=1.0,
            )
            out = self.actor_rollout_wg.sp_q_train_separate(td)
            qm = reduce_metrics(tu.get(out, "metrics"))
            metrics.update({
                "q/loss": qm.get("sp_q/loss", float("nan")),
                "q/grad_norm": qm.get("sp_q/grad_norm", float("nan")),
                "q/grad_steps": qm.get("sp_q/grad_steps", 0),
                "q/q_phase_skipped": 0,
            })
            disp = {"q/loss": qm.get("sp_q/loss", float("nan")),
                    "q/grad_steps": qm.get("sp_q/grad_steps", 0),
                    "q/delta_q_applied": 0.0, "q/delta_ppo_net": 0.0, "q/q_phase_skipped": 0}
        # Advance next_dataset_step + fsync the step's delta (admissions) BEFORE the
        # checkpoint. Without this the Q cursor stays behind the replay cursor and step t+1
        # aborts at the reward-site cursor assertion; admissions never reach q_state_deltas.jsonl.
        h.append_delta(disp)

    def _sp_q_mlp_wave_score(self, live):
        """MLP-Q wave: score contexts with the actor's own FSDP module + head.

        The generative wave runs on the still-awake vLLM replicas; this one needs the
        FSDP training module instead, so the replicas are put to sleep FIRST -- an FSDP
        forward beside a fully awake vLLM reservation is the same OOM family the
        step-cache resume sleep in fit() guards against. fit()'s own post-wave sleep is
        skipped via the flag."""
        self.checkpoint_manager.sleep_replicas()
        self._sp_q_wave_slept = True
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")
        n = len(live)
        pad = (-n) % dp_size
        shortest = min(range(n), key=lambda k: len(live[k]["ctx_ids"]))
        rows = live + [live[shortest]] * pad
        L = max(len(r["ctx_ids"]) for r in rows)
        ids = torch.zeros((len(rows), L), dtype=torch.int32)
        lens = torch.zeros((len(rows),), dtype=torch.int64)
        for i, r in enumerate(rows):
            c = r["ctx_ids"]
            ids[i, : len(c)] = torch.as_tensor(c, dtype=torch.int32)
            lens[i] = len(c)
        proto = DataProto.from_single_dict({"sp_q_ctx_ids": ids, "sp_q_ctx_len": lens})
        td = proto.to_tensordict()
        tu.assign_non_tensor(td, sp_q_head_path=os.environ.get("SP_Q_MLP_HEAD_PATH", ""))
        out = self.actor_rollout_wg.sp_q_mlp_score(td)
        vals = out["sp_q_values"]
        for j, r in enumerate(live):
            v = float(vals[j])
            r["value"], r["gen_text"], r["fail_reason"] = v, "%.4f" % v, None

    def _sp_q_run_wave_and_stamp(self, batch: DataProto, probes: bool = True):
        """Post-generation Q wave: assemble consumed-Q calls, readiness
        probes and the audit-lane counterfactual calls, run them greedily on the
        still-awake rollout replicas at theta_0, parse, stash results, and stamp consumed
        rows' extra_info so the judge path skips them (the judge-cost win). Only consumed
        rows are stamped: audit rows keep their real judge reward by construction."""
        from verl.trainer.ppo import sp_q_readiness as _sp_q

        h = _sp_q.harness()
        ntb = batch.non_tensor_batch
        # repeat(interleave) duplicates dict REFERENCES: one extra_info object per
        # group. Per-row stamps (sp_q_value differs per rollout) need per-row dicts.
        extra = [dict(ei) for ei in ntb["extra_info"]]
        ntb["extra_info"] = np.array(extra, dtype=object)

        resp = batch.batch["responses"]
        resp_len = resp.shape[1]
        resp_attn = batch.batch["attention_mask"][:, -resp_len:]
        valid_lens = resp_attn.sum(dim=-1).tolist()
        rmask = batch.batch["response_mask"] if "response_mask" in batch.batch.keys() \
            else compute_response_mask(batch)
        gen_lens = rmask.sum(dim=-1).tolist()
        uids = ntb["uid"]
        step = int(extra[0]["sp_dataset_step"])

        _decode_cache: dict[int, str] = {}

        def resp_ids_of_row(i):
            return resp[i][: int(valid_lens[i])].tolist()

        def decoded_of_row(i):
            if i not in _decode_cache:
                _decode_cache[i] = self.tokenizer.decode(resp_ids_of_row(i), skip_special_tokens=True)
            return _decode_cache[i]

        rows = h.build_wave_rows(
            step=step,
            extra=extra,
            uids=uids,
            resp_ids_of_row=resp_ids_of_row,
            gen_len_of_row=lambda i: int(gen_lens[i]),
            decoded_of_row=decoded_of_row,
            probes=probes,
        )

        live = [r for r in rows if not r["overflow"]]
        for r in rows:
            if r["overflow"]:
                r["value"], r["gen_text"], r["fail_reason"] = None, "", "overflow"
        if live and os.environ.get("SP_Q_PARAM", "gen") == "mlp":
            self._sp_q_mlp_wave_score(live)
        elif live:
            n = len(live)
            ctx_arr = np.empty(n, dtype=object)
            rp_arr = np.empty(n, dtype=object)
            for j, r in enumerate(live):
                ctx_arr[j] = [int(t) for t in r["ctx_ids"]]
                rp_arr[j] = []  # postprocess requires a raw_prompt key; unused by sp_q_agent
            wave = DataProto.from_single_dict({"dummy_tensor": torch.zeros(n, 1)})
            wave.non_tensor_batch["sp_q_ctx_token_ids"] = ctx_arr
            wave.non_tensor_batch["raw_prompt"] = rp_arr
            wave.non_tensor_batch["agent_name"] = np.array(["sp_q_agent"] * n, dtype=object)
            wave.non_tensor_batch["sp_q_max_tokens"] = np.array([h.gen_reserve] * n, dtype=np.int64)
            wave.meta_info = {"global_steps": self.global_steps, "validate": False}
            size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
            wave_padded, pad_size = pad_dataproto_to_divisor(wave, size_divisor)
            wave_out = self.async_rollout_manager.generate_sequences(wave_padded)
            wave_out = unpad_dataproto(wave_out, pad_size=pad_size)
            out_resp = wave_out.batch["responses"]
            out_attn = wave_out.batch["attention_mask"][:, -out_resp.shape[1]:]
            out_lens = out_attn.sum(dim=-1).tolist()
            for j, r in enumerate(live):
                gen_ids = out_resp[j][: int(out_lens[j])].tolist()
                text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                v = _sp_q.parse_grid_value(text)
                r["value"], r["gen_text"] = v, text
                r["fail_reason"] = None if v is not None else "unparseable"

        h.record_wave_results(step, rows, append=not probes)

        # ---- SP_Q_DUMP_WAVE: persist every completion sent to Q for scoring ----------
        # These rows are otherwise TRANSIENT: record_wave_results holds exactly one step in
        # memory and the next step overwrites it, so the only durable trace of a Q scoring
        # call was the scalar error inside q_state_deltas.jsonl. Writing ctx_token_ids (the
        # exact context Q scored) alongside gen_text (what Q returned) makes the
        # Q-prediction-vs-realized-outcome join reproducible offline, instead of having to
        # re-derive prefixes by joining rollout dumps on uid.
        # Default OFF, so runs without it are byte-for-byte unaffected.
        # Failure here must never take down training -- this is instrumentation, so the
        # whole block is exception-isolated, and the file is written to a temp name and
        # atomically renamed so a crash mid-write cannot leave a truncated file that looks
        # complete to a downstream reader.
        if os.environ.get("SP_Q_DUMP_WAVE", "0") not in ("", "0", "false", "False"):
            try:
                _rd = self.config.trainer.get("rollout_data_dir", None)
                if _rd:
                    _qdir = os.path.join(os.path.dirname(os.path.normpath(str(_rd))), "q_wave")
                    os.makedirs(_qdir, exist_ok=True)
                    _final = os.path.join(_qdir, "%s.jsonl" % step)
                    # dynamic group size: rounds after the first APPEND to the step's file
                    # (the first round wrote it atomically via the tmp+rename below).
                    _tmp = _final if not probes else os.path.join(_qdir, ".%s.jsonl.tmp" % step)
                    _n = 0
                    with open(_tmp, "a" if not probes else "w", encoding="utf-8") as _f:
                        for _r in rows:
                            _f.write(json.dumps({
                                "step": int(step),
                                "kind": _r.get("kind"),
                                "uid": _r.get("uid"),
                                "qid": _r.get("qid"),
                                "variant": _r.get("variant"),
                                "overflow": bool(_r.get("overflow")),
                                "value": _r.get("value"),
                                "fail_reason": _r.get("fail_reason"),
                                "gen_text": _r.get("gen_text", ""),
                                "ctx_token_ids": [int(t) for t in (_r.get("ctx_ids") or [])],
                            }) + "\n")
                            _n += 1
                    if probes:
                        os.replace(_tmp, _final)
                    print("[sp_q] dumped %d wave rows for step %s -> %s%s" % (
                        _n, step, _qdir, "" if probes else " (appended)"), flush=True)
            except Exception as _e:
                print("[sp_q][WARN] wave dump failed for step %s: %s" % (step, _e), flush=True)

        # Stamp per-ROW dicts, never row indices: _balance_batch reorders the batch
        # between here and the reward site, so a stored index would point at a different
        # rollout there. Stamps ride with their row through the reorder.
        for r in rows:
            if r["kind"] == "probe":
                # ---- no-group TD: the probe IS the row's baseline Q(p_i). With td mode
                # on, every TD copy is its own uid group, so "one probe per uid" is one
                # probe per ROW, at that row's own prefix -- exactly V(p_i) at K=0. Relay
                # it into the sp_q_seg_* stamps at bound = prefix_len; the sp_segment
                # estimator's singleton-split path then yields A = r_i - Q(p_i) on the
                # generated span (a1 spans only the mask-0 prefix). Stamped EXPLICITLY
                # invalid/valid like the seg branch below -- the estimator treats a
                # missing key as "no split", so absence-means-valid would silently turn
                # the mode off. Non-TD rows: the probe stays readiness-only, unstamped.
                ei = extra[r["row"]]
                if int(ei.get("sp_q_td_row", 0)):
                    invalid = bool(r["overflow"] or r.get("value") is None)
                    ei["sp_q_seg_value"] = None if invalid else float(r["value"])
                    ei["sp_q_seg_bound"] = int(ei.get("sp_prefix_len", 0))
                    ei["sp_q_seg_invalid"] = int(invalid)
                    ei["sp_q_seg_lane"] = str(ei.get("sp_q_route", "short"))
                continue
            if r["kind"] == "audit_cut":
                # marks the one member of this audit group whose counterfactual Q was
                # measured, so the reward site can pair it with ITS terminal reward
                extra[r["row"]]["sp_q_audit_cut"] = 1
                continue
            if r["kind"] == "seg":
                # SEGMENTED-Q interior value. Stamped per ROW, like every
                # other wave result, because _balance_batch reorders the batch between here
                # and compute_advantage -- a stored row index would attach this q_i to a
                # different rollout there, which is the whole reason the audit pairing uses
                # stamps too. sp_q_seg_invalid is stamped EXPLICITLY (0/1) rather than left
                # absent on success: the estimator treats a missing key as "no seg", so
                # relying on absence to mean valid would silently disable the method.
                ei = extra[r["row"]]
                invalid = bool(r["overflow"] or r.get("value") is None)
                ei["sp_q_seg_value"] = None if invalid else float(r["value"])
                ei["sp_q_seg_bound"] = int(r["seg_bound"])
                ei["sp_q_seg_invalid"] = int(invalid)
                ei["sp_q_seg_lane"] = str(r.get("lane", ""))
                continue
            if r["kind"] != "consumed":
                continue
            ei = extra[r["row"]]
            invalid = bool(r["overflow"] or r.get("value") is None)
            ei["sp_q_route_taken"] = "q_consumed"
            ei["sp_q_value"] = None if invalid else float(r["value"])
            ei["sp_q_invalid"] = int(invalid)

    def _sp_q_stash_wave_rows(self, batch: DataProto):
        """Copy the harness's wave results into batch.meta_info for the postwave /
        postreward caches, WITHOUT the context token ids (up to ~53K ints per row --
        hundreds of MB the reward site never reads). Everything the reward-site
        rehydration needs (kind/uid/qid/row/variant/overflow/value/gen_text/
        fail_reason + audit-cut fields) is kept verbatim."""
        from verl.trainer.ppo import sp_q_readiness as _sp_q

        h = _sp_q.harness()
        wr = h._wave_result
        assert wr is not None, "sp_q stash called before the wave recorded results"
        slim = [{k: v for k, v in r.items() if k != "ctx_ids"} for r in wr["rows"]]
        # The tier-1 reference RESOLUTION has to survive the cache as well, not just the wave
        # rows. The wave resolves trajectory_ref_proof BEFORE replay rotation; on a postwave/
        # postreward resume the Q-admission lookup (observe_and_update) instead runs AFTER
        # sp_replay.observe_and_update has evicted 32 entries, so a fresh lookup cannot find
        # those source entries and silently falls back to the bank / no-reference variant. The
        # resumed step would then train on different Q contexts than the same step run
        # uninterrupted. Carry the resolved verdict (entry_id -> proof text, or None for "refused
        # / no proof") for every source entry in this batch; text only, no token ids.
        _eids = {str(ei.get("sp_entry_id", "") or "")
                 for ei in batch.non_tensor_batch["extra_info"]}
        _eids.discard("")
        _refs = {eid: h._ref_proof_cache[eid] for eid in _eids if eid in h._ref_proof_cache}
        batch.meta_info["sp_q_wave_rows"] = {
            "step": wr["step"], "rows": slim, "ref_proofs": _refs}

    def _sp_q_rehydrate_wave_rows(self, batch: DataProto):
        """postwave/postreward resume: restore the COMPLETED wave's results into the
        harness from the cache's meta stash -- the wave itself is never rerun."""
        from verl.trainer.ppo import sp_q_readiness as _sp_q

        stash = batch.meta_info.get("sp_q_wave_rows")
        assert stash is not None, (
            "sp_q: cache resume without sp_q_wave_rows in meta_info -- a postwave/"
            "postreward cache written by this code always carries it; a stale cache "
            "from an older code version must be deleted, not silently accepted"
        )
        h = _sp_q.harness()
        h.record_wave_results(int(stash["step"]), list(stash["rows"]))
        # Restore the PRE-ROTATION tier-1 verdicts (see _sp_q_stash_wave_rows). .get() rather
        # than an assert: caches written before this field existed must still resume, they just
        # fall back to the old post-rotation lookup. setdefault so anything already resolved in
        # this process wins.
        for _eid, _pr in (stash.get("ref_proofs") or {}).items():
            h._ref_proof_cache.setdefault(str(_eid), _pr)
        print(f"[sp_q] rehydrated {len(stash['rows'])} wave rows for step {stash['step']} "
              "from the step cache (wave not rerun)", flush=True)

    def _sp_dyn_generate_wave_reward(self, batch: DataProto, gen_batch: DataProto,
                                     timing_raw: dict, metrics: dict, td_counts) -> DataProto:
        """Dynamic group size (see verl/trainer/ppo/sp_dyn_group.py).

        Replaces generate -> wave -> sleep of the normal step with rounds:

            counts_r  = per-row repeat counts for round r (inflow rows round 1 only,
                        trained rows 8/8/16 while unsatisfied, 0 once satisfied)
            gen_r     = generate(gen_batch.sample_level_repeat(counts_r))      (awake)
            sub_r     = batch.sample_level_repeat(counts_r).union(gen_r)        (paired by construction)
            wave(sub_r, probes = r == 0); sleep replicas
            sub_r     = sub_r.union(judge(sub_r))                               (rm_scores)
            satisfied |= rows with any score >= SP_DYN_THRESH
            wake replicas if another round follows

        and returns DataProto.concat(sub_1..sub_R) with rm_scores present, so the reward
        stage below skips the judge and goes straight to the Q reward site. The wave
        stash is merged across rounds (record_wave_results append) and copied into
        meta_info for the postreward cache. Each round's TRAINED total is padded to a
        multiple of 2*world_size so the two-minibatch actor update divides.
        """
        from time import perf_counter
        from verl.trainer.ppo import sp_dyn_group as _dyn
        from verl.trainer.ppo import sp_q_readiness as _sp_q

        assert td_counts is None, "sp_dyn: no-group TD mode is not supported with dynamic group size"
        assert self.config.algorithm.adv_estimator != AdvantageEstimator.REMAX, \
            "sp_dyn: REMAX baseline generation is not supported with dynamic group size"
        assert _sp_q.enabled(), \
            "sp_dyn needs the Q wave (SP_Q_ENABLE=1): capped short rows are scored by Q, not the judge"
        assert self.use_rm, "sp_dyn: the stopping test reads rm_scores; a run without a reward model cannot use it"
        print(_dyn.describe(), flush=True)

        extra = batch.non_tensor_batch["extra_info"]
        stamps = [int(ei.get("sp_rollout_n", 0)) if isinstance(ei, dict) else 0 for ei in extra]
        ws = int(self.actor_rollout_wg.world_size)
        dp = self._get_dp_size(self.actor_rollout_wg, "actor")
        n_inflow = sum(st for st in stamps if st > 0)
        assert n_inflow % dp == 0, (
            f"sp_dyn: {n_inflow} inflow sequences is not divisible by dp={dp}; the pre-drop "
            f"_balance_batch would refuse the assembled batch")
        multiple = 2 * ws
        trained_rows = {i for i, st in enumerate(stamps) if st == 0}

        parts, per_round, satisfied = [], [], set()
        gen_timing: dict = {}
        t_gen = t_wave = t_judge = 0.0
        for rnd in range(_dyn.n_rounds()):
            counts = _dyn.round_counts(stamps, rnd, satisfied)
            if not any(c > 0 for c in counts):
                break
            counts = _dyn.pad_to_multiple(counts, multiple, stamps)
            per_round.append(counts)

            if rnd > 0:
                # The replicas were put to sleep for the previous round's judge. The proven
                # wake path is the weight sync (asleep -> load -> wake); the weights are
                # still theta_0 at this point, so the sync is a no-op in content.
                t0 = perf_counter()
                self.checkpoint_manager.update_weights(self.global_steps)
                timing_raw["sp_dyn_wake"] = timing_raw.get("sp_dyn_wake", 0.0) + (perf_counter() - t0)

            t0 = perf_counter()
            gen_r = self.async_rollout_manager.generate_sequences(gen_batch.sample_level_repeat(counts))
            t_gen += perf_counter() - t0
            for k, v in (gen_r.meta_info.pop("timing", None) or {}).items():
                try:
                    gen_timing[k] = gen_timing.get(k, 0.0) + float(v)
                except (TypeError, ValueError):
                    pass
            if "__do_sample__" in gen_r.non_tensor_batch:
                gen_r.pop(non_tensor_batch_keys=["__do_sample__"])
            sub = batch.sample_level_repeat(counts).union(gen_r)

            t0 = perf_counter()
            self._sp_q_wave_maybe_separate(sub, probes=(rnd == 0))
            if not getattr(self, "_sp_q_wave_slept", False):
                self.checkpoint_manager.sleep_replicas()
            t_wave += perf_counter() - t0

            t0 = perf_counter()
            if "rm_scores" not in sub.batch.keys():
                sub = sub.union(self._compute_reward_colocate(sub))
            t_judge += perf_counter() - t0
            parts.append(sub)

            scores = sub.batch["rm_scores"].sum(-1).tolist()
            satisfied |= _dyn.satisfied_rows(_dyn.expand_rows(counts), scores)
            n_sat = len(satisfied & trained_rows)
            print(f"[sp_dyn] step {self.global_steps} round {rnd + 1}/{_dyn.n_rounds()}: "
                  f"{len(sub)} seqs ({sum(c for i, c in enumerate(counts) if i in trained_rows)} trained), "
                  f"{n_sat}/{len(trained_rows)} groups satisfied so far", flush=True)

        assert parts, "sp_dyn: no round generated anything"
        out = DataProto.concat(parts)
        timing_raw.update(gen_timing)
        timing_raw["gen"] = t_gen
        timing_raw["sp_q_wave"] = t_wave
        timing_raw["sp_dyn_judge"] = t_judge
        m = _dyn.summarize(per_round, stamps)
        tot = _dyn.total_counts(per_round)
        m["sp_dyn/trained_seqs"] = sum(tot[i] for i in trained_rows)
        m["sp_dyn/total_seqs"] = len(out)
        metrics.update(m)
        print(f"[sp_dyn] step {self.global_steps}: {len(per_round)} rounds, {len(out)} seqs "
              f"({m['sp_dyn/trained_seqs']} trained, mean group {m['sp_dyn/mean_group_size']:.2f}), "
              f"gen {t_gen:.0f}s wave {t_wave:.0f}s judge {t_judge:.0f}s", flush=True)
        # The postreward cache (saved by the reward stage) must carry the merged wave rows,
        # exactly as the postwave save does on the normal path.
        self._sp_q_stash_wave_rows(out)
        return out

    def _sp_q_reward_site(self, batch: DataProto, reward_tensor: torch.Tensor) -> dict:
        """Reward-site Q update: valid-member flags, FIFO admission + readiness
        measurement (pending delta), neutralize invalid consumed-Q rollouts (survivor-
        mean fill keeps the GRPO mean baseline exactly as if they were absent; mask 0
        removes them from the PPO loss), and draw the step's Q-training sample."""
        from verl.trainer.ppo import sp_q_readiness as _sp_q

        h = _sp_q.harness()
        ntb = batch.non_tensor_batch
        extra = ntb["extra_info"]
        uids = ntb["uid"]
        n = len(extra)
        jhe = ntb.get("judge_http_error")
        jpf = ntb.get("judge_parse_failed")

        valid_flags, dropped = [], []
        td_dropped_no_baseline = 0
        for i in range(n):
            ei = extra[i]
            # no-group TD: a TD row whose baseline probe overflowed or was unparseable
            # has a reward but no Q(p_i) to subtract. Left in, the sp_segment fallback
            # would hand this singleton uid its RAW reward as the advantage (the GRPO
            # singleton convention baselines at 0.0) -- an uncentered, always-positive
            # push. Drop it like an invalid consumed row instead, whatever its reward
            # source (consumed Q or an early-finish judge score).
            td_bad = bool(int(ei.get("sp_q_td_row", 0))) and int(ei.get("sp_q_seg_invalid", 1)) != 0
            if td_bad:
                td_dropped_no_baseline += 1
            if ei.get("sp_q_route_taken") == "q_consumed":
                ok = not int(ei.get("sp_q_invalid", 1))
                valid_flags.append(ok and not td_bad)
                if not ok or td_bad:
                    dropped.append(i)
            else:
                he = int(float(jhe[i])) if jhe is not None else 0
                pf = int(float(jpf[i])) if jpf is not None else 0
                # http/parse failures are invalid MEMBERS for Q targets but the
                # rows stay in PPO (only invalid consumed-Q rollouts are dropped)
                valid_flags.append((he == 0 and pf == 0) and not td_bad)
                if td_bad:
                    dropped.append(i)

        resp = batch.batch["responses"]
        resp_len = resp.shape[1]
        valid_lens = batch.batch["attention_mask"][:, -resp_len:].sum(dim=-1).tolist()

        def resp_ids_of_row(i):
            return resp[i][: int(valid_lens[i])].tolist()

        seq_rewards = reward_tensor.sum(-1).tolist()
        m = h.observe_and_update(
            non_tensor_batch=ntb,
            seq_rewards=seq_rewards,
            valid_flags=valid_flags,
            resp_ids_of_row=resp_ids_of_row,
        )

        # ---- neutralize dropped rollouts: survivor-mean fill + mask 0 ----
        if dropped:
            group_rows: dict[str, list[int]] = {}
            for i in range(n):
                group_rows.setdefault(str(uids[i]), []).append(i)
            for i in dropped:
                g = group_rows[str(uids[i])]
                surv = [seq_rewards[j] for j in g if valid_flags[j]]
                fill = (sum(surv) / len(surv)) if surv else 0.0
                reward_tensor[i].zero_()
                last = max(int(valid_lens[i]) - 1, 0)
                reward_tensor[i, last] = fill
                batch.batch["response_mask"][i].zero_()
        m["q/dropped_rollouts"] = len(dropped)
        m["q/td_dropped_no_baseline"] = td_dropped_no_baseline

        # ---- Q-training sample for this step (consumed by the train phase) ----
        step = h._pending_delta["dataset_step"]
        if self._sp_q_separate():
            h.draw_new_records(step)   # separate-Q: NEW step-k records only
        else:
            h.draw_train_sample(step)
        m.update(h.train_sample_meta())
        return m

    def _sp_q_build_train_td(self, metrics: dict, *, lr: float):
        """Build the teacher-forced Q batch from this step's drawn sample.

        Shared by BOTH tied-weight Q surfaces (the capture-at-theta_0 phase and the
        interleaved phase) -- they differ only in when the O_Q step lands relative to
        the PPO minibatches, never in what the Q batch contains. Returns
        (tensordict, n_records), or None when the drawn sample is empty."""
        from verl.trainer.ppo import sp_q_readiness as _sp_q

        h = _sp_q.harness()
        sample = h._train_sample or {}
        rows = sample.get("rows") or []
        if not rows:
            metrics["q/q_phase_skipped"] = 1
            return None
        n_records = len(sample["picked_seqs"])

        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")
        pad_rows = (-len(rows)) % dp_size
        # dummy rows (weight 0) to make the batch divisible by dp; no gradient, and
        # driver-side counts come from n_records, so metrics are unaffected.
        shortest = min(range(len(rows)), key=lambda k: len(rows[k]["ctx_ids"]))
        all_rows = rows + [dict(rows[shortest], weight=0.0) for _ in range(pad_rows)]

        if os.environ.get("SP_Q_PARAM", "gen") == "mlp":
            # MLP-Q batch: ctx only (the model must NOT see the teacher-forced target),
            # plus per-row z/weight. No left_right_2_no_padding -- the worker forwards
            # each row at exact length, so the packed layout would only be undone.
            L = max(len(r["ctx_ids"]) for r in all_rows)
            N = len(all_rows)
            ids = torch.zeros((N, L), dtype=torch.int32)
            lens = torch.zeros((N,), dtype=torch.int64)
            zs = torch.zeros((N,), dtype=torch.float32)
            ws = torch.zeros((N,), dtype=torch.float32)
            mref = torch.zeros((N,), dtype=torch.int8)
            for i, r in enumerate(all_rows):
                c = r["ctx_ids"]
                ids[i, : len(c)] = torch.as_tensor(c, dtype=torch.int32)
                lens[i] = len(c)
                zs[i] = float(r["rec"]["z"])
                ws[i] = float(r["weight"])
                mref[i] = 1 if r["variant"] == "ref" else 0
            proto = DataProto.from_single_dict({
                "sp_q_ctx_ids": ids, "sp_q_ctx_len": lens, "sp_q_bce_z": zs,
                "sp_q_row_w": ws, "sp_q_is_ref": mref,
            })
            td = proto.to_tensordict()
            tu.assign_non_tensor(
                td,
                sp_q_denom=int(n_records),
                sp_q_lr=float(lr),
                sp_q_clip=float(os.environ.get("SP_Q_GRAD_CLIP", "0.2")),
                sp_q_param="mlp",
                sp_q_head_lr=float(os.environ.get("SP_Q_MLP_HEAD_LR", "1e-4")),
                sp_q_head_path=os.environ.get("SP_Q_MLP_HEAD_PATH", ""),
                sp_q_max_token_len=int(os.environ.get("SP_Q_MAX_TOKEN_LEN", "53360")),
                temperature=1.0,
            )
            return td, n_records

        plens, rlens = [], []
        for r in all_rows:
            plen = len(h.prompt_ids_for_qid(r["rec"]["qid"]))
            plens.append(plen)
            rlens.append(len(r["ctx_ids"]) - plen + len(r["target_ids"]))
        P, R = max(plens), max(rlens)
        N = len(all_rows)
        pad_id = self.tokenizer.pad_token_id or 0
        input_ids = torch.full((N, P + R), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((N, P + R), dtype=torch.long)
        response_mask = torch.zeros((N, R), dtype=torch.long)
        tok_w = torch.zeros((N, R), dtype=torch.float32)
        is_ref = torch.zeros((N,), dtype=torch.int8)
        for i, r in enumerate(all_rows):
            plen = plens[i]
            ctx, tgt = r["ctx_ids"], r["target_ids"]
            prompt_ids = ctx[:plen]
            resp_ids = list(ctx[plen:]) + list(tgt)
            L = len(resp_ids)
            input_ids[i, P - plen:P] = torch.as_tensor(prompt_ids, dtype=torch.long)
            attention_mask[i, P - plen:P] = 1
            input_ids[i, P:P + L] = torch.as_tensor(resp_ids, dtype=torch.long)
            attention_mask[i, P:P + L] = 1
            t0 = L - len(tgt)
            response_mask[i, t0:L] = 1
            tok_w[i, t0:L] = float(r["weight"]) / 4.0
            is_ref[i] = 1 if r["variant"] == "ref" else 0
        position_ids = torch.clip(torch.cumsum(attention_mask, dim=-1) - 1, min=0)

        proto = DataProto.from_single_dict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_mask": response_mask,
            # prompts/responses: REQUIRED by no_padding_2_padding (the loss's log_prob
            # re-padding derives prompt/response lens from them; without them the first Q
            # phase raises KeyError 'prompts'). Same P/R split
            # the rows were built with: prompt part left-padded, response right-padded.
            "prompts": input_ids[:, :P],
            "responses": input_ids[:, P:],
            "sp_q_tok_w": tok_w,
            "sp_q_is_ref": is_ref,
        })
        self._balance_batch(proto, metrics=metrics, logging_prefix="sp_q_seqlen")
        td = proto.to_tensordict()
        td = left_right_2_no_padding(td)
        tu.assign_non_tensor(
            td,
            sp_q_denom=int(n_records),
            sp_q_lr=float(lr),
            sp_q_clip=float(os.environ.get("SP_Q_GRAD_CLIP", "0.2")),
            sp_q_max_token_len=int(os.environ.get("SP_Q_MAX_TOKEN_LEN", "53360")),
            temperature=1.0,
        )
        return td, n_records

    def _sp_q_train_capture_phase(self, metrics: dict):
        """Q phase step 1: build the teacher-forced batch from the drawn sample
        and run sp_q_train_capture on the workers (Delta_Q captured at theta_0, not
        applied), then arm the per-PPO-step displacement tracker."""
        built = self._sp_q_build_train_td(
            metrics, lr=float(os.environ.get("SP_Q_LR_BASE", "2e-6"))
        )
        if built is None:
            return {"skipped": True}
        td, n_records = built
        out = self.actor_rollout_wg.sp_q_train_capture(td)
        qm = reduce_metrics(tu.get(out, "metrics"))
        denom = max(n_records, 1)
        worker_skipped = bool(qm.get("sp_q/skipped", 0))
        metrics.update({
            "q/loss": qm.get("sp_q/loss", float("nan")),
            "q/loss_noref": qm.get("sp_q/sp_q_ce4_sum_noref", float("nan")) / denom,
            "q/loss_ref": qm.get("sp_q/sp_q_ce4_sum_ref", float("nan")) / denom,
            "q/grad_norm": qm.get("sp_q/grad_norm", float("nan")),
            "q/target_tokens": qm.get("sp_q/sp_q_target_tokens", float("nan")),
            "q/q_phase_skipped": int(worker_skipped),
        })
        self.actor_rollout_wg.sp_q_arm_ppo_norms()
        return {
            "skipped": False,
            "worker_skipped": worker_skipped,
            "delta_q": float(qm.get("sp_q/delta_q_norm", 0.0)),
        }

    def _sp_q_interleave_arm_phase(self, metrics: dict):
        """Interleaved Q phase, part 1: build the Q batch at the LADDER's current LR and arm the
        workers so the Q-only optimizer step fires inside update_actor, after PPO
        minibatch `SP_Q_INTERLEAVE_AFTER` (default 1)."""
        from verl.trainer.ppo import sp_q_readiness as _sp_q

        h = _sp_q.harness()
        lr = h.q_lr.lr if h.q_lr is not None else float(os.environ.get("SP_Q_LR_BASE", "2e-6"))
        built = self._sp_q_build_train_td(metrics, lr=lr)
        if built is None:
            metrics["q/lr_current"] = lr
            return {"skipped": True, "lr": lr}
        td, n_records = built
        tu.assign_non_tensor(
            td, sp_q_after_minibatch=int(os.environ.get("SP_Q_INTERLEAVE_AFTER", "1"))
        )
        self.actor_rollout_wg.sp_q_arm_interleave(td)
        metrics["q/lr_current"] = lr
        return {"skipped": False, "lr": lr, "n_records": n_records}

    def _sp_q_interleave_finish_phase(self, ctx, metrics: dict):
        """Interleaved Q phase, part 2 (after PPO minibatch 2): collect the applied Delta_Q and
        the per-minibatch PPO displacements, advance the LR ladder on
        rho = ||Delta_Q|| / (||Delta_PPO1|| + ||Delta_PPO2||), and append the durable
        delta record. Unlike the movement-cap path there is NOTHING to apply here -- the Q step
        is already in the weights."""
        from verl.trainer.ppo import sp_q_readiness as _sp_q

        h = _sp_q.harness()
        lr_used = float((ctx or {}).get("lr", 0.0))
        if ctx is None or ctx.get("skipped"):
            res = {"delta_q": 0.0, "delta_ppo_steps": [], "delta_ppo_sum": 0.0,
                   "q_non_finite": False, "q_loss_non_finite": False,
                   "q_grad_non_finite": False, "late": False, "metrics": {}}
            skipped = True
        else:
            res = self.actor_rollout_wg.sp_q_finish_interleave()[0]
            skipped = False
            qm = res.get("metrics") or {}
            denom = max(int(ctx.get("n_records", 1)), 1)
            metrics.update({
                "q/loss": qm.get("sp_q/loss", float("nan")),
                "q/loss_noref": qm.get("sp_q/sp_q_ce4_sum_noref", float("nan")) / denom,
                "q/loss_ref": qm.get("sp_q/sp_q_ce4_sum_ref", float("nan")) / denom,
                "q/grad_norm": qm.get("sp_q/grad_norm", float("nan")),
                "q/target_tokens": qm.get("sp_q/sp_q_target_tokens", float("nan")),
            })
            if res.get("late"):
                # the actor ran fewer minibatches than the trigger index: the Q step still
                # happened (after PPO), but the pinned PPO(M1) -> Q -> PPO(M2) order did NOT
                # hold this step.
                print("[sp_q] WARNING: interleaved Q step fired LATE (after the full PPO "
                      "update) -- fewer PPO minibatches than SP_Q_INTERLEAVE_AFTER", flush=True)
            metrics["q/interleave_late"] = int(bool(res.get("late")))
            metrics["q/loss_non_finite"] = int(bool(res.get("q_loss_non_finite")))
            metrics["q/grad_non_finite"] = int(bool(res.get("q_grad_non_finite")))

        steps = [float(s) for s in (res.get("delta_ppo_steps") or [])]
        d_ppo1 = steps[0] if len(steps) > 0 else 0.0
        d_ppo2 = steps[1] if len(steps) > 1 else 0.0
        delta_q = float(res.get("delta_q", 0.0))
        non_finite = bool(res.get("q_non_finite", False))

        disp = {
            "q/delta_ppo1": d_ppo1 if len(steps) > 0 else float("nan"),
            "q/delta_ppo2": d_ppo2 if len(steps) > 1 else float("nan"),
            "q/delta_ppo_sum": float(res.get("delta_ppo_sum", math.fsum(steps))),
            # the interleaved surface has no theta_0 anchor, so "net" is the sum of the
            # applied PPO steps -- the quantity the ladder's rho divides by.
            "q/delta_ppo_net": float(res.get("delta_ppo_sum", math.fsum(steps))),
            "q/delta_q": delta_q,
            "q/delta_q_applied": 0.0 if non_finite else delta_q,
            "q/q_phase_skipped": int(skipped or non_finite),
            "q/lr_current": lr_used,
        }
        if h.q_lr is not None:
            section = h.q_lr.apply_step(
                delta_q=None if (skipped or non_finite) else delta_q,
                delta_ppo1=d_ppo1,
                delta_ppo2=d_ppo2,
                q_phase_skipped=bool(skipped),
                q_non_finite=non_finite,
            )
            disp["q_lr_ladder"] = section
            disp.update({
                "q/rho": section["rho"] if section["rho"] is not None else float("nan"),
                "q/lr_next": section["lr_next"],
                "q/lr_breach": int(section["breach"]),
                "q/lr_breach_streak": section["breach_streak_after"],
                "q/lr_reduced": int(section["reduced"]),
                "q/lr_floor_alert": int(section["floor_alert"]),
            })
            if section["reduced"]:
                print(f"[sp_q] Q-LR ladder REDUCED {section['lr_current']:.4g} -> "
                      f"{section['lr_next']:.4g} (rho={section['rho']}, "
                      f"non_finite={non_finite})", flush=True)
            if section["floor_alert"]:
                print(f"[sp_q] Q-LR ladder at the FLOOR {h.q_lr.floor:.4g} and still "
                      f"breaching (rho={section['rho']}) -- alert only, no further "
                      "reduction", flush=True)
        else:
            disp["q/rho"] = (delta_q / max(d_ppo1 + d_ppo2, 1e-12)) if not skipped else float("nan")
        metrics.update({k: v for k, v in disp.items() if k.startswith("q/")})
        h.append_delta(disp)
        metrics["q/delta_q_cumsum"] = h.cum_delta_q
        metrics["q/delta_ppo_net_cumsum"] = h.cum_delta_ppo_net
        if h.cum_delta_ppo_net > 0:
            metrics["q/cum_ratio"] = h.cum_delta_q / h.cum_delta_ppo_net

    def _sp_q_finish_and_apply(self, ctx, metrics: dict):
        """Q phase step 3: delta_PPO_net + per-step norms, the movement cap
        s_t = min(1, rho_cap/rho_t), apply s_t*Delta_Q, log the displacement norms, and
        append the step's durable delta record."""
        from verl.trainer.ppo import sp_q_readiness as _sp_q

        h = _sp_q.harness()
        rho_cap = float(os.environ.get("SP_Q_RHO_CAP", "0.5"))
        lr_base = float(os.environ.get("SP_Q_LR_BASE", "2e-6"))
        disp = {}
        if ctx is None or ctx.get("skipped"):
            disp = {"q/delta_q": 0.0, "q/delta_q_applied": 0.0, "q/q_phase_skipped": 1}
            metrics.setdefault("q/q_phase_skipped", 1)
        else:
            res = self.actor_rollout_wg.sp_q_finish_update()[0]
            net = float(res["delta_ppo_net"])
            steps = [float(s) for s in res["delta_ppo_steps"]]
            delta_q = float(ctx["delta_q"])
            rho = delta_q / max(net, 1e-12)
            s = 0.0 if ctx["worker_skipped"] else (min(1.0, rho_cap / rho) if rho > 0 else 1.0)
            applied = self.actor_rollout_wg.sp_q_apply(s)[0]
            disp = {
                "q/delta_ppo1": steps[0] if len(steps) > 0 else float("nan"),
                "q/delta_ppo2": steps[1] if len(steps) > 1 else float("nan"),
                "q/delta_ppo_net": net,
                "q/delta_ppo_sum": sum(steps),
                "q/delta_q": delta_q,
                "q/delta_q_applied": float(applied["applied_delta_q_norm"]),
                "q/rho": rho,
                "q/s": s,
                "q/lr_eff": s * lr_base,
                "q/q_phase_skipped": int(ctx["worker_skipped"]),
            }
        metrics.update({k: v for k, v in disp.items()})
        h.append_delta(disp)
        metrics["q/delta_q_cumsum"] = h.cum_delta_q
        metrics["q/delta_ppo_net_cumsum"] = h.cum_delta_ppo_net
        if h.cum_delta_ppo_net > 0:
            metrics["q/cum_ratio"] = h.cum_delta_q / h.cum_delta_ppo_net

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        if self._dump_executor._shutdown:
            self._init_dump_executor()

        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint and update weights before doing anything
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                self._shutdown_dump_executor()
                return

        if self.config.actor_rollout_ref.rollout.skip.get("enable", False):
            rollout_skip = RolloutSkip(self.config, self.async_rollout_manager)
            rollout_skip.wrap_generate_sequences()

        # Resuming AT/PAST the terminal step must be a no-op: the += 1 below would otherwise run
        # (and checkpoint) one extra step per relaunch — e.g. an auto-roll manager restarting a
        # finished run trains 151, 152, ... forever. Exit before touching the model.
        if self.global_steps >= self.total_training_steps:
            print(
                f"[fit] resumed at global_step {self.global_steps} >= total_training_steps "
                f"{self.total_training_steps} — training already complete, exiting without stepping",
                flush=True,
            )
            self._shutdown_dump_executor()
            return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                # Difficulty sampling: attach the STABLE problem id (qid = statement hash; uid is a
                # fresh uuid per draw and never matches across steps) and the draw-time correction c
                # recorded by the weighted sampler's FIFO. On mismatch (e.g. dataloader prefetch
                # replay after a resume) self-heal: drop the FIFO and recompute c from current
                # weights -- one step of second-order staleness at most.
                from verl.trainer.ppo import difficulty as _difficulty
                if _difficulty.enabled():
                    _qids = [
                        _difficulty.qid_from_messages(m) for m in batch.non_tensor_batch["raw_prompt"]
                    ]
                    _recs = _difficulty.pop_draws(len(_qids))
                    if len(_recs) == len(_qids) and all(r[1] == q for r, q in zip(_recs, _qids)):
                        _cs = [r[2] for r in _recs]
                    else:
                        print(
                            f"[difficulty] draw FIFO mismatch ({len(_recs)}/{len(_qids)} recs); "
                            "clearing and recomputing c from current weights",
                            flush=True,
                        )
                        _difficulty.clear_fifo()
                        _smp = getattr(self.train_dataloader, "sampler", None)
                        if hasattr(_smp, "correction"):
                            _cs = [_smp.correction(q) for q in _qids]
                        else:
                            _cs = [1.0] * len(_qids)
                    batch.non_tensor_batch["qid"] = np.array(_qids, dtype=object)
                    batch.non_tensor_batch["diff_c_raw"] = np.array(_cs, dtype=np.float64)

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                rollout_n = self.config.actor_rollout_ref.rollout.n
                _sp_row_counts = self._sp_per_row_repeat_counts(batch)
                if _sp_row_counts is not None:
                    gen_batch_output = gen_batch.sample_level_repeat(_sp_row_counts)
                else:
                    gen_batch_output = gen_batch.repeat(repeat_times=rollout_n, interleave=True)

                # ---- no-group TD (sp_q_td_enable): per-copy prefix rewrite BEFORE dispatch.
                # A ready slot's copies stop being siblings of one prefix and become
                # td_siblings single-rollout requests at distinct cuts of the same stored
                # trajectory. Key presence gates the whole path: the dataset stamps
                # sp_td_cuts only when the mode is on, so every other run skips this in one
                # dict lookup. The same counts drive the mirrored driver-batch rewrite
                # below -- the two MUST agree or stamps attach to the wrong prefix.
                _sp_td_counts = None
                if "sp_td_cuts" in gen_batch.non_tensor_batch:
                    _sp_td_counts = (list(_sp_row_counts) if _sp_row_counts is not None
                                     else [rollout_n] * len(gen_batch))
                    _n_td = sp_td_rewrite_gen_copies(
                        gen_batch.non_tensor_batch, gen_batch_output, _sp_td_counts)
                    if _n_td:
                        print(f"[sp_q_td] step {self.global_steps}: rewrote {_n_td} ready "
                              f"slot(s) into per-copy TD prefixes", flush=True)

                if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                    # NOTE: REMAX needs one sampled rollout plus one greedy baseline per prompt.
                    # Keep them in a single agent-loop/vLLM request to avoid sending a second
                    # rollout after replicas have been put to sleep, which can leave async vLLM
                    # engines in an invalid state for multi-turn agent workloads.
                    gen_batch_output.non_tensor_batch["__do_sample__"] = np.ones(len(gen_batch_output), dtype=bool)
                    gen_baseline_batch = gen_batch.slice(0, None)
                    gen_baseline_batch.non_tensor_batch["__do_sample__"] = np.zeros(len(gen_baseline_batch), dtype=bool)
                    combined_gen_batch = DataProto.concat([gen_batch_output, gen_baseline_batch])
                    num_sampled_prompts = len(gen_batch_output)
                else:
                    combined_gen_batch = gen_batch_output
                    num_sampled_prompts = len(gen_batch_output)

                is_last_step = self.global_steps >= self.total_training_steps
                # --- step-cache resume (env-gated via VERL_STEP_CACHE_DIR) ---
                # Three cache points under SP_Q:
                #   postgen    = batch after union, BEFORE the wave (no Q stamps).
                #   postwave   = batch after the wave, WITH the wave results stashed slim
                #                (no ctx ids) in meta_info["sp_q_wave_rows"] -- the wave
                #                results live in the driver harness, so the cache must
                #                carry them for the reward-site rehydration.
                #   postreward = as before; inherits the meta stash through the batch.
                # A COMPLETED wave is never rerun: only a postgen resume (crash inside
                # the wave itself, whose outputs never existed) executes the wave -- and
                # that path must WAKE the replicas first (the wave needs engines at
                # theta_0). postwave/postreward resumes rehydrate the harness and go
                # straight to sleep + judge/update. Replay safety: nothing Q-related is
                # durably persisted before append_delta (post-update), so re-running the
                # reward site off a cache cannot double-apply admissions.
                _sc_dir = os.environ.get("VERL_STEP_CACHE_DIR")
                from verl.trainer.ppo import sp_q_readiness as _sp_q
                from verl.trainer.ppo import sp_dyn_group as _sp_dyn
                _sc_pg = _sc_pw = _sc_pr = None
                _resume_batch = None
                _resume_kind = None
                if _sc_dir:
                    os.makedirs(_sc_dir, exist_ok=True)
                    _sc_pg = os.path.join(_sc_dir, f"step{self.global_steps}_postgen.pt")
                    _sc_pw = os.path.join(_sc_dir, f"step{self.global_steps}_postwave.pt")
                    _sc_pr = os.path.join(_sc_dir, f"step{self.global_steps}_postreward.pt")
                    # prune caches of older (completed) steps
                    for _f in os.listdir(_sc_dir):
                        if _f.startswith("step") and "_post" in _f:
                            try:
                                _n = int(_f.split("_")[0][4:])
                                if _n < self.global_steps:
                                    os.unlink(os.path.join(_sc_dir, _f))
                            except (ValueError, OSError):
                                pass
                    if os.path.exists(_sc_pr):
                        _resume_batch = DataProto.load_from_disk(_sc_pr)
                        _resume_kind = "postreward"
                        print(f"[step-cache] step {self.global_steps}: resuming from POST-REWARD cache "
                              f"(skips generation + wave + judge)", flush=True)
                    elif os.path.exists(_sc_pw):
                        _resume_batch = DataProto.load_from_disk(_sc_pw)
                        _resume_kind = "postwave"
                        print(f"[step-cache] step {self.global_steps}: resuming from POST-WAVE cache "
                              f"(skips generation + wave)", flush=True)
                    elif os.path.exists(_sc_pg):
                        _resume_batch = DataProto.load_from_disk(_sc_pg)
                        _resume_kind = "postgen"
                        print(f"[step-cache] step {self.global_steps}: resuming from POST-GEN cache "
                              f"(skips generation; the wave will run" +
                              (" on woken replicas)" if _sp_q.enabled() else ")"), flush=True)
                with marked_timer("step", timing_raw):
                    if _resume_batch is None and _sp_dyn.enabled():
                        # Dynamic group size (sp_dyn_group): rounds of generate -> Q wave ->
                        # judge, assembled into ONE batch that already carries rm_scores and
                        # the merged wave stash. No postgen/postwave cache exists for such a
                        # step (only postreward, saved below as usual); a crash mid-step
                        # regenerates the step.
                        batch = self._sp_dyn_generate_wave_reward(
                            batch, gen_batch, timing_raw, metrics, _sp_td_counts)
                        del combined_gen_batch
                    elif _resume_batch is None:
                        # generate a batch
                        with marked_timer("gen", timing_raw, color="red"):
                            if curr_step_profile:
                                self.llm_server_manager.start_profile()
                            combined_gen_output = self.async_rollout_manager.generate_sequences(combined_gen_batch)
                            # Generative-Q readiness: the Q wave (consumed-Q calls + readiness
                            # probes) must run on the STILL-AWAKE
                            # rollout replicas at theta_0 -- sleep is deferred to right after
                            # the wave (below, once the batch is assembled).
                            if not _sp_q.enabled():
                                self.checkpoint_manager.sleep_replicas()
                            if curr_step_profile:
                                self.llm_server_manager.stop_profile()

                            timing_raw.update(combined_gen_output.meta_info["timing"])
                            combined_gen_output.meta_info.pop("timing", None)

                        gen_batch_output = combined_gen_output.slice(0, num_sampled_prompts)
                        if "__do_sample__" in gen_batch_output.non_tensor_batch:
                            gen_batch_output.pop(non_tensor_batch_keys=["__do_sample__"])

                        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                            gen_baseline_output = combined_gen_output.slice(num_sampled_prompts, None)
                            if "__do_sample__" in gen_baseline_output.non_tensor_batch:
                                gen_baseline_output.pop(non_tensor_batch_keys=["__do_sample__"])

                            if self.use_rm and "rm_scores" not in gen_baseline_output.batch.keys():
                                baseline_reward = self._compute_reward_colocate(gen_baseline_output)
                                gen_baseline_output = gen_baseline_output.union(baseline_reward)

                            reward_baseline_tensor = gen_baseline_output.batch["rm_scores"].sum(dim=-1)
                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_output
                        del combined_gen_batch, combined_gen_output
                        # repeat to align with repeated responses in rollout
                        if _sp_row_counts is not None:
                            batch = batch.sample_level_repeat(_sp_row_counts)
                        else:
                            batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        # no-group TD: mirror of the gen-side rewrite, on the driver batch's
                        # bookkeeping (fresh uid per copy -> singleton groups; per-copy
                        # extra_info with the copy's own sp_prefix_len). Same counts as the
                        # gen rewrite by construction.
                        if _sp_td_counts is not None:
                            sp_td_rewrite_driver_copies(batch.non_tensor_batch, _sp_td_counts)
                            # the gen output carries pickled pre-rewrite uid/extra_info copies;
                            # the union below asserts deep equality on shared keys.
                            sp_td_mirror_driver_keys(batch.non_tensor_batch, gen_batch_output.non_tensor_batch)
                        batch = batch.union(gen_batch_output)
                        # postgen cache BEFORE the wave: a crash inside the wave resumes
                        # here and runs the (never-completed) wave instead of regenerating.
                        if _sc_pg:
                            try:
                                batch.save_to_disk(_sc_pg)
                                print(f"[step-cache] saved post-gen cache for step {self.global_steps}", flush=True)
                            except Exception as _e:
                                print(f"[step-cache] post-gen save failed: {_e}", flush=True)
                        if _sp_q.enabled():
                            # Q wave at theta_0 on the awake replicas, then sleep for the judge.
                            with marked_timer("sp_q_wave", timing_raw, color="magenta"):
                                self._sp_q_wave_maybe_separate(batch)
                            if not getattr(self, "_sp_q_wave_slept", False):
                                self.checkpoint_manager.sleep_replicas()
                            if _sc_pw:
                                try:
                                    self._sp_q_stash_wave_rows(batch)
                                    batch.save_to_disk(_sc_pw)
                                    print(f"[step-cache] saved post-wave cache for step {self.global_steps}", flush=True)
                                except Exception as _e:
                                    print(f"[step-cache] post-wave save failed: {_e}", flush=True)
                    else:
                        batch = _resume_batch
                        if not (_sp_q.enabled() and _resume_kind == "postgen"):
                            # RESUME SLEEP. The normal path sleeps the rollout replicas
                            # after generation/wave, BEFORE the judge wakes; the postgen+Q
                            # branch below sleeps after its wave. But a postwave/postreward
                            # (or non-Q postgen) resume skips those phases -- and the sleep
                            # inside them -- so without this the judge and the training update
                            # would run beside FULLY AWAKE policy engines holding their whole
                            # reservation, which OOMs on cache resumes (cumem wake failures,
                            # judge-inference OOM at ~200MB free) even where normal-path steps
                            # run fine at higher mem util.
                            try:
                                self.checkpoint_manager.sleep_replicas()
                                print(f"[step-cache] {_resume_kind} resume: rollout replicas "
                                      f"put to sleep before reward/training phases", flush=True)
                            except Exception as _e:
                                print(f"[step-cache] resume sleep_replicas failed ({_e}); "
                                      f"continuing", flush=True)
                        if _sp_q.enabled() and _resume_kind == "postgen":
                            # crash was inside the wave: wake the replicas at theta_0 (the
                            # restored checkpoint weights), run the wave for the first
                            # time, sleep, and drop a postwave cache for the next resume.
                            self.checkpoint_manager.update_weights(self.global_steps)
                            with marked_timer("sp_q_wave", timing_raw, color="magenta"):
                                self._sp_q_wave_maybe_separate(batch)
                            if not getattr(self, "_sp_q_wave_slept", False):
                                self.checkpoint_manager.sleep_replicas()
                            if _sc_pw:
                                try:
                                    self._sp_q_stash_wave_rows(batch)
                                    batch.save_to_disk(_sc_pw)
                                    print(f"[step-cache] saved post-wave cache for step {self.global_steps}", flush=True)
                                except Exception as _e:
                                    print(f"[step-cache] post-wave save failed: {_e}", flush=True)
                        else:
                            if _sp_q.enabled():
                                # postwave/postreward resume: the wave already completed;
                                # rehydrate its results into the harness -- never rerun it.
                                self._sp_q_rehydrate_wave_rows(batch)
                            try:
                                self.checkpoint_manager.sleep_replicas()
                            except Exception as _e:
                                print(f"[step-cache] sleep_replicas on resume: {_e}", flush=True)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # sp_dp_pad: balance the PADDED batch so the equal-size partitions are exactly
                    # the dispatch chunks, remember where the pads landed (this step's plan for
                    # the log-prob dispatches), then strip them again -- the judge, admission and
                    # the Q reward site below must only ever see real rows.
                    self._sp_dp_plan = None
                    if self.config.trainer.balance_batch:
                        batch, _dp_keep = self._sp_dp_pad_for_dispatch(batch, self.actor_rollout_wg, "actor")
                        self._balance_batch(batch, metrics=metrics)
                        if _dp_keep is not None:
                            from verl.trainer.ppo import sp_dp_pad as _sp_dp
                            batch, self._sp_dp_plan = _sp_dp.strip(batch)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    # get images_seqlens
                    images_seqlens_all = []
                    for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                        if "image_grid_thw" not in multi_modal_input.keys():
                            continue
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens_all
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            batch_reward = self._compute_reward_colocate(batch)
                            batch = batch.union(batch_reward)

                        # extract reward_tensor and reward_extra_infos_dict for training
                        reward_tensor, reward_extra_infos_dict = extract_reward(batch)
                        if _sc_pr and not os.path.exists(_sc_pr):
                            try:
                                batch.save_to_disk(_sc_pr)
                                print(f"[step-cache] saved post-reward cache for step {self.global_steps}", flush=True)
                            except Exception as _e:
                                print(f"[step-cache] post-reward save failed: {_e}", flush=True)

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            # Difficulty sampling: the global actor entropy is a per-problem
                            # EXPECTATION, so `actor/entropy` (the AEC control signal AND a logged
                            # metric) must be the UNIFORM-population entropy = the SNIS token-mean
                            # weighted by each rollout's draw-time c. Controlling / logging the raw
                            # q-sampled mean would target the difficulty-sampled distribution, not
                            # the objective. Raw kept on the side as difficulty_raw/actor/entropy.
                            # c ≡ 1 (non-difficulty) -> weighted == raw exactly, no sidecar key.
                            from verl.trainer.ppo import difficulty as _difficulty
                            entropy_ctrl = entropy_agg
                            if _difficulty.enabled() and "diff_c_raw" in batch.non_tensor_batch:
                                _cc = torch.as_tensor(
                                    batch.non_tensor_batch["diff_c_raw"].astype("float64"),
                                    device=entropys.device, dtype=entropys.dtype,
                                ).unsqueeze(-1)
                                _cden = (response_masks * _cc).sum()
                                if _cden > 0:
                                    entropy_ctrl = (entropys * response_masks * _cc).sum() / _cden
                                    metrics["difficulty_raw/actor/entropy"] = entropy_agg.detach().item()
                            # Replay-prefix training: AEC must control on the
                            # FROM-SCRATCH entropy only. Replay rows are conditioned on partial
                            # proof prefixes and are systematically lower-entropy; the batch mean
                            # would read as a fake collapse and integrate k toward kmax. The
                            # statement-row token-mean is the same quantity the parent run
                            # controlled on, so aec_k continues seamlessly across the graft.
                            # (Prefix tokens are already response_mask=0, so replay-row entropy
                            # here is continuation-only by construction.)
                            from verl.trainer.ppo import sp_replay as _sp_replay
                            if _sp_replay.enabled():
                                _sp_flags = _sp_replay.statement_row_flags(batch.non_tensor_batch)
                                if _sp_flags is not None and any(_sp_flags):
                                    _sel = torch.as_tensor(
                                        _sp_flags, device=entropys.device, dtype=torch.bool
                                    ).unsqueeze(-1)
                                    metrics["actor/entropy_full_batch"] = entropy_ctrl.detach().item()
                                    _m_orig = response_masks * _sel
                                    entropy_ctrl = (entropys * _m_orig).sum() / _m_orig.sum().clamp(min=1)
                                    _m_repl = response_masks * (~_sel)
                                    if _m_repl.sum() > 0:
                                        metrics["actor/entropy_replay"] = (
                                            (entropys * _m_repl).sum() / _m_repl.sum()
                                        ).detach().item()
                                elif _sp_flags is not None:
                                    print(
                                        "[sp_replay] WARNING: no statement rows in this batch; AEC "
                                        "falls back to full-batch entropy (composition wiring bug?)",
                                        flush=True,
                                    )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_ctrl.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            # AEC: step k on the DRIVER from the single global actor entropy
                            # (SNIS-weighted under difficulty sampling). One value -> broadcast
                            # identically to all workers below.
                            from verl.trainer.ppo import aec as _aec
                            self._aec_k = _aec.driver_step_k(float(entropy_ctrl.detach().item()))
                            metrics["actor/aec_k"] = self._aec_k
                            old_log_prob.batch.pop("entropys")
                            if "routed_experts" in batch.batch and "routed_experts" in old_log_prob.batch:
                                raise ValueError(
                                    "Detected conflicting router replay configuration: "
                                    "router_replay.mode='R2' and enable_rollout_routing_replay=True "
                                    "cannot be enabled simultaneously. "
                                    "The enable_rollout_routing_replay option is only used in R3 mode; "
                                    "it should not be set when using R2 mode."
                                )
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import (
                                    calculate_debug_metrics,
                                    calculate_train_inference_mismatch_metrics,
                                )

                                metrics.update(calculate_debug_metrics(batch))
                                # self-play: richer train/inference logprob mismatch diagnostics
                                # (tail percentiles + per-distribution perplexity + availability guard)
                                metrics.update(calculate_train_inference_mismatch_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy and self.use_ref_log_prob:
                        # compute reference log_prob (skipped for colocated OPSD: the ref model is
                        # the privileged teacher, and with both KL terms off nothing reads this)
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # Difficulty sampling: EMA-update each problem's pass rate from this step's
                        # rollouts (grouped by the stable qid, NOT uid), and log the corrected
                        # (SNIS) score mean + effective sample size. Draw-time c was locked at
                        # sampling; observe() only affects FUTURE draws, so this ordering is unbiased.
                        from verl.trainer.ppo import difficulty as _difficulty
                        if _difficulty.enabled() and "qid" in batch.non_tensor_batch:
                            # Pass signal = the BINARY judge verdict, NOT reward_tensor.sum(): with a
                            # correct-only length penalty, score<1 for a correct long proof while
                            # prover_judge_score stays 1 (prover_judge.py). Prefer the per-row judge
                            # fields (extract_reward puts reward_extra_info into non_tensor_batch);
                            # fall back to the shaped reward only when neither is present.
                            _ntb = batch.non_tensor_batch
                            _pjs = _ntb.get("prover_judge_score")
                            _cjs = _ntb.get("correctness_judge_score")
                            _seq_reward = reward_tensor.sum(-1).tolist()
                            _n = len(_ntb["qid"])
                            _passes = [
                                _difficulty.row_pass(
                                    prover_judge_score=(_pjs[i] if _pjs is not None else None),
                                    correctness_judge_score=(_cjs[i] if _cjs is not None else None),
                                    score=_seq_reward[i],
                                )
                                for i in range(_n)
                            ]
                            _per: dict = {}
                            for _q, _pass in zip(_ntb["qid"], _passes):
                                _a = _per.setdefault(_q, [0, 0])
                                _a[0] += _pass
                                _a[1] += 1
                            for _q, (_np, _nt) in _per.items():
                                _difficulty.observe(_q, _np / _nt)
                            _craw = [float(x) for x in batch.non_tensor_batch["diff_c_raw"]]
                            metrics["difficulty/ess_frac"] = _difficulty.ess(_craw) / max(len(_craw), 1)
                            metrics["difficulty/c_mean"] = sum(_craw) / max(len(_craw), 1)
                            metrics["difficulty/c_max"] = max(_craw) if _craw else 1.0
                            metrics["difficulty/score_mean_corrected"] = _difficulty.snis_mean(
                                _seq_reward, _craw
                            )
                            _smp = getattr(self.train_dataloader, "sampler", None)
                            if hasattr(_smp, "metrics"):
                                metrics.update(_smp.metrics())

                        # Replay-prefix training: success-EMA observe (from-scratch rows only) + capacity-5 buffer admission
                        # + one delta append, keyed by the LOCAL dataset step. Deliberately at the
                        # REWARD site (pre-update): _save_checkpoint fires inside the step right
                        # after update_actor, so this placement guarantees "delta k is durable
                        # before checkpoint k exists". Token ids are extracted lazily and only for
                        # admitted rows (attention-mask valid span = prefix + continuation).
                        from verl.trainer.ppo import sp_replay as _sp_replay
                        if _sp_replay.enabled():
                            _sp_resp = batch.batch["responses"]
                            _sp_attn = batch.batch["attention_mask"][:, -_sp_resp.shape[1]:]
                            _sp_valid = _sp_attn.sum(dim=-1).tolist()
                            _sp_seq_reward = reward_tensor.sum(-1).tolist()
                            # sp_q tier-1 references must be resolved BEFORE the replay update
                            # admits/evicts: eviction can remove the very entry a Q record's
                            # reference comes from, and on a postwave/postreward cache resume the
                            # proof cache is cold (only the wave RESULTS are rehydrated), so the
                            # resumed step would silently fall back to the bank or to no
                            # reference and train on different Q contexts than an uninterrupted
                            # step. Pre-warming here makes the two paths equivalent. No-op when
                            # the wave already cached them.
                            if _sp_q.enabled() and _sp_q.installed():
                                metrics.update(
                                    _sp_q.harness().prewarm_trajectory_refs(batch.non_tensor_batch)
                                )
                            metrics.update(
                                _sp_replay.observe_and_update(
                                    non_tensor_batch=batch.non_tensor_batch,
                                    seq_rewards=_sp_seq_reward,
                                    get_row_token_ids=lambda i: _sp_resp[i][: int(_sp_valid[i])].tolist(),
                                )
                            )
                            # Stream tag into the rollout dumps so offline analyses (and any
                            # future parser-side stream split) can tell from-scratch rows from
                            # prefix-replay rows without reconstructing the batch composition.
                            reward_extra_infos_dict["sp_prefix_len"] = [
                                int(ei["sp_prefix_len"]) for ei in batch.non_tensor_batch["extra_info"]
                            ]

                        # Generative-Q readiness: Q FIFO admission +
                        # readiness measurement + invalid-consumed-Q neutralization (drop
                        # from group baseline via mean-fill + response_mask zero) + the
                        # step's Q-training sample draw. Mutates reward_tensor IN PLACE
                        # (== token_level_scores) before token_level_rewards is derived.
                        from verl.trainer.ppo import sp_q_readiness as _sp_q
                        if _sp_q.enabled():
                            with marked_timer("sp_q_admit", timing_raw):
                                metrics.update(self._sp_q_reward_site(batch, reward_tensor))
                            # dump stream tags for offline analysis / dashboards
                            reward_extra_infos_dict["sp_q_route"] = [
                                str(ei.get("sp_q_route", "full"))
                                for ei in batch.non_tensor_batch["extra_info"]
                            ]
                            reward_extra_infos_dict["sp_q_route_taken"] = [
                                str(ei.get("sp_q_route_taken", ""))
                                for ei in batch.non_tensor_batch["extra_info"]
                            ]

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # ---- inflow-only drop: scratch-
                        # stream rows have now served both purposes that need them — buffer
                        # admission (sp_replay hook above) and reward/judge labels — so drop
                        # them BEFORE rollout-correction/advantage/update. The dropped slice
                        # is stashed (with its reward-extras rows) so _log_rollout_data can
                        # still dump the full generation record — dumps are the harvest
                        # source and must keep the inflow stream. ----
                        self._sp_inflow_dump_stash = None
                        if self._sp_inflow_only() and _sp_replay.enabled():
                            _extra_all = batch.non_tensor_batch.get("extra_info")
                            if (_extra_all is not None and len(_extra_all)
                                    and isinstance(_extra_all[0], dict)
                                    and "sp_source_type" in _extra_all[0]):
                                # "cold_scratch" (cold bootstrap) is a TRAINED row:
                                # a fresh problem standing in for a replay slot while the
                                # buffer is still empty. Dropping it would empty the batch,
                                # which is exactly what the assert below would then catch.
                                _KEEP_TYPES = ("replay", "cold_scratch")
                                _keep = [i for i, ei in enumerate(_extra_all)
                                         if ei.get("sp_source_type") in _KEEP_TYPES]
                                _drop = [i for i, ei in enumerate(_extra_all)
                                         if ei.get("sp_source_type") not in _KEEP_TYPES]
                                assert _keep, "inflow-only drop would empty the batch"
                                if _drop:
                                    def _sp_subset_extras(d, idxs):
                                        out = {}
                                        for k, v in d.items():
                                            vv = v.tolist() if hasattr(v, "tolist") else list(v)
                                            out[k] = [vv[i] for i in idxs]
                                        return out

                                    self._sp_inflow_dump_stash = (
                                        batch.select_idxs(_drop),
                                        _sp_subset_extras(reward_extra_infos_dict, _drop),
                                    )
                                    reward_extra_infos_dict = _sp_subset_extras(
                                        reward_extra_infos_dict, _keep
                                    )
                                    batch = batch.select_idxs(_keep)
                                    batch.meta_info["global_token_num"] = torch.sum(
                                        batch.batch["attention_mask"], dim=-1
                                    ).tolist()
                                # ALWAYS emitted, not only when rows were dropped: the
                                # dashboard reads sp_inflow/trained_rows as the run-level
                                # inflow-only marker. On burst-inflow runs nine of ten
                                # steps drop nothing (statement rows are removed
                                # pre-generation); gating this on _drop would leave those
                                # steps unmarked, and panel 1 would fall back to the
                                # route-blind rubric mean, averaging censored short rows in
                                # as zeros.
                                metrics["sp_inflow/dropped_rows"] = len(_drop)
                                metrics["sp_inflow/trained_rows"] = len(_keep)

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                        # sp_segment's per-step statistics. Merged here rather than returned from
                        # compute_advantage so the registry's 2-tuple contract is untouched.
                        _segm = batch.meta_info.pop("sp_seg_metrics", None)
                        if _segm:
                            metrics.update({k: float(v) for k, v in _segm.items()})

                        # Difficulty sampling: turn the draw-time per-row corrections into
                        # the (bsz, 1) `diff_c` tensor the worker multiplies into advantages.
                        # normalize_c_for_loss rescales over the FULL global batch (all rows visible
                        # here on the driver) so the c-weighted token count equals the raw token
                        # count -- workers keep dividing by the untouched batch_num_tokens and the
                        # aggregate is exactly the weighted token-mean. GRPO advantages themselves
                        # are already computed (intra-group; c must NOT enter them).
                        if _difficulty.enabled() and "diff_c_raw" in batch.non_tensor_batch:
                            _rmask = batch.batch["response_mask"]
                            _n_tok = _rmask.sum(-1).float().tolist()
                            _craw = [float(x) for x in batch.non_tensor_batch["diff_c_raw"]]
                            _chat = _difficulty.normalize_c_for_loss(_craw, _n_tok)
                            batch.batch["diff_c"] = torch.tensor(
                                _chat, dtype=torch.float32, device=_rmask.device
                            ).unsqueeze(-1)
                            metrics["difficulty/c_hat_max"] = max(_chat)

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup > self.global_steps:
                        # Still in critic warmup, only update weights to wake up rollout replicas.
                        self.checkpoint_manager.update_weights(self.global_steps)
                    else:
                        # Generative-Q readiness: compute the Q delta at
                        # theta_0 (captured, NOT applied) and arm the per-PPO-step norm
                        # tracker BEFORE the unchanged PPO update.
                        # Tied-Q interleaving (SP_Q_INTERLEAVE=1) instead ARMS the workers so
                        # the Q-only step lands between PPO minibatch 1 and 2 and stays
                        # applied; nothing is captured or restored.
                        _sp_q_ctx = None
                        _sp_q_il = self._sp_q_interleave() if _sp_q.enabled() else False
                        if _sp_q.enabled() and not self._sp_q_separate():
                            with marked_timer("sp_q_train", timing_raw, color="purple"):
                                _sp_q_ctx = (
                                    self._sp_q_interleave_arm_phase(metrics) if _sp_q_il
                                    else self._sp_q_train_capture_phase(metrics)
                                )

                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)

                        # Generative-Q readiness: delta_PPO_net + per-step norms, the
                        # movement cap s_t = min(1, rho_cap/rho_t), apply s_t*Delta_Q, and
                        # append the step's durable delta record -- all BEFORE the checkpoint
                        # save below, so the saved weights include the Q movement and the
                        # delta is durable before checkpoint k exists.
                        if _sp_q.enabled():
                            with marked_timer("sp_q_apply", timing_raw, color="purple"):
                                if self._sp_q_separate():
                                    # decoupled Q: 3 x mb-64 on its own theta_Q, left
                                    # applied; no theta_0 capture, no movement cap.
                                    self._sp_q_train_separate_phase(metrics)
                                elif _sp_q_il:
                                    # interleaved: the Q step already landed mid-update; this only
                                    # measures it, advances the LR ladder, and logs.
                                    self._sp_q_interleave_finish_phase(_sp_q_ctx, metrics)
                                else:
                                    self._sp_q_finish_and_apply(_sp_q_ctx, metrics)

                        # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        # Check if the conditions for saving a checkpoint are met.
                        # The conditions include a mandatory condition (1) and
                        # one of the following optional conditions (2/3/4):
                        # 1. The save frequency is set to a positive value.
                        # 2. It's the last training step.
                        # 3. The current step number is a multiple of the save frequency.
                        # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                        if self.config.trainer.save_freq > 0 and (
                            is_last_step
                            or self.global_steps % self.config.trainer.save_freq == 0
                            or esi_close_to_expiration
                        ):
                            if esi_close_to_expiration:
                                print("Force saving checkpoint: ESI instance expiration approaching.")
                            with marked_timer("save_checkpoint", timing_raw, color="green"):
                                self._save_checkpoint()

                        # update weights from trainer to rollout
                        with marked_timer("update_weights", timing_raw, color="red"):
                            self.checkpoint_manager.update_weights(self.global_steps)

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        _dump_batch, _dump_extras = batch, reward_extra_infos_dict
                        _stash = getattr(self, "_sp_inflow_dump_stash", None)
                        if _stash is not None:
                            # re-append the inflow-only dropped rows on the
                            # intersection of keys — advantage/update added keys the
                            # dropped slice never got, and the dump needs none of them.
                            _db, _de = _stash
                            _bkeys = [k for k in _db.batch.keys() if k in batch.batch.keys()]
                            _nkeys = [k for k in _db.non_tensor_batch.keys()
                                      if k in batch.non_tensor_batch.keys()]
                            _dump_batch = DataProto.concat([
                                batch.select(batch_keys=_bkeys, non_tensor_batch_keys=_nkeys),
                                _db.select(batch_keys=_bkeys, non_tensor_batch_keys=_nkeys),
                            ])
                            _dump_extras = {}
                            for k, v in reward_extra_infos_dict.items():
                                vv = v.tolist() if hasattr(v, "tolist") else list(v)
                                _dump_extras[k] = vv + list(_de.get(k, []))
                        self._log_rollout_data(_dump_batch, _dump_extras, timing_raw, rollout_data_dir)

                # validate
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                # Difficulty sampling (metric audit): overwrite the standard train-metric keys with
                # SNIS-corrected estimates (uniform-comparable = the MAIN logging); the raw
                # sampled-distribution values move to difficulty_raw/<key> (dashboard page 3).
                # Max/min (order stats) and critic/values stay raw. (actor/entropy is corrected +
                # sidecar'd at its own old_log_prob site above, since it also drives AEC.)
                if _difficulty.enabled() and "diff_c_raw" in batch.non_tensor_batch:
                    from verl.trainer.ppo.metric_utils import _compute_response_info

                    _craw = [float(x) for x in batch.non_tensor_batch["diff_c_raw"]]
                    # lengths/aborted EXACTLY as compute_data_metrics measures them (attention-mask
                    # slices via the shared helper, honoring its precomputed-fields branch) ...
                    _ri = _compute_response_info(batch)
                    _rl = _ri["response_length"].tolist()
                    _plen = _ri["prompt_length"].tolist()
                    # ... while adv/returns token-means use the response_mask FIELD, matching the
                    # masked_select in compute_data_metrics (differs in multi-turn: obs tokens).
                    _rmask = batch.batch["response_mask"].bool()
                    _max_rl = batch.batch["responses"].shape[-1]
                    _max_pl = batch.batch["prompts"].shape[-1]
                    _seqs = batch.batch["token_level_scores"].sum(-1).tolist()
                    _seqr = batch.batch["token_level_rewards"].sum(-1).tolist()
                    _abort = [1.0 if r == 0 else 0.0 for r in _rl]
                    _na_idx = [i for i, a in enumerate(_abort) if a == 0.0]
                    _na_cs = [_craw[i] for i in _na_idx]
                    _rows = {
                        "seq_score": [_seqs[i] for i in _na_idx], "seq_score__cs": _na_cs,
                        "seq_reward": [_seqr[i] for i in _na_idx], "seq_reward__cs": _na_cs,
                        "response_length": _rl,
                        "resp_clip": [1.0 if r == _max_rl else 0.0 for r in _rl],
                        "response_length_na": [_rl[i] for i in _na_idx],
                        "response_length_na__cs": _na_cs,
                        "resp_clip_na": [1.0 if _rl[i] == _max_rl else 0.0 for i in _na_idx],
                        "resp_clip_na__cs": _na_cs,
                        "aborted": _abort,
                        "prompt_length": _plen,
                        "prompt_clip": [1.0 if p == _max_pl else 0.0 for p in _plen],
                        "n_tokens": _rmask.sum(-1).float().tolist(),
                        "adv_sum": (batch.batch["advantages"] * _rmask).sum(-1).tolist(),
                        "ret_sum": (batch.batch["returns"] * _rmask).sum(-1).tolist(),
                    }
                    if "__num_turns__" in batch.non_tensor_batch:
                        _rows["num_turns"] = [float(x) for x in batch.non_tensor_batch["__num_turns__"]]
                    if "tool_call_counts" in batch.non_tensor_batch:
                        _rows["tool_calls"] = [float(x) for x in batch.non_tensor_batch["tool_call_counts"]]
                    _corr_keys = _difficulty.apply_metric_corrections(metrics, _craw, _rows)
                    metrics["difficulty/n_corrected_metrics"] = len(_corr_keys)

                # Checkpoint retention (keep last N + best-training-reward), driver-side. Runs only on
                # steps we actually checkpointed this iteration; uses the (now SNIS-corrected)
                # critic/rewards/mean as the "training reward" for the best-ckpt pointer.
                if os.environ.get("SP_KEEP_BEST_CKPT", "1") not in ("0", "false", "False", "no") \
                        and getattr(self, "_last_saved_global_step", None) == self.global_steps:
                    self._retain_recent_and_best(metrics.get("critic/rewards/mean"))
                # GDPO per-component reward metrics
                gdpo_reward_keys = self.config.algorithm.get("gdpo_reward_keys", None)
                if gdpo_reward_keys and self.config.algorithm.adv_estimator in ("gdpo", AdvantageEstimator.GDPO):
                    for key in gdpo_reward_keys:
                        if key in batch.non_tensor_batch:
                            vals = np.asarray(batch.non_tensor_batch[key], dtype=np.float32)
                            metrics[f"gdpo/{key}/mean"] = float(np.mean(vals))
                            metrics[f"gdpo/{key}/std"] = float(np.std(vals))
                            metrics[f"gdpo/{key}/max"] = float(np.max(vals))
                            metrics[f"gdpo/{key}/min"] = float(np.min(vals))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # compute variance proxy metrics
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # Per-request spec decode metrics.
                metrics.update(
                    compute_spec_decode_metrics(
                        batch.non_tensor_batch.get("spec_num_draft_tokens", None),
                        batch.non_tensor_batch.get("spec_num_accepted_tokens", None),
                        batch.non_tensor_batch.get("spec_num_verify_steps", None),
                    )
                )

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    self._shutdown_dump_executor()
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)

        # Ensure dump executor is shut down when training loop ends without reaching is_last_step
        self._shutdown_dump_executor()
