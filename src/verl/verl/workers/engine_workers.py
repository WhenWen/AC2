# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import functools
import logging
import math
import os
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from itertools import chain
from typing import Optional

import psutil
import torch
from codetiming import Timer
from omegaconf import DictConfig, open_dict
from tensordict import NonTensorData, TensorDict
from torch.distributed.device_mesh import init_device_mesh

from verl.checkpoint_engine import CheckpointEngineRegistry
from verl.single_controller.base import Worker
from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.trainer.distillation import distillation_ppo_loss, is_distillation_enabled
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_name, get_torch_device, set_expandable_segments
from verl.utils.distributed import initialize_global_process_group_ray, set_numa_affinity
from verl.utils.flops_counter import FlopsCounter
from verl.utils.import_utils import import_external_libs
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.metric.utils import Metric
from verl.utils.profiler import DistProfiler, DistProfilerExtension, ProfilerConfig, log_gpu_memory_usage
from verl.utils.py_functional import append_to_dict
from verl.utils.tensordict_utils import maybe_fix_3d_position_ids
from verl.utils.torch_functional import allgather_dict_into_dict
from verl.workers.config import (
    ActorConfig,
    DistillationConfig,
    HFModelConfig,
    MtpConfig,
    RolloutConfig,
    TrainingWorkerConfig,
)
from verl.workers.rollout.base import BaseRollout, get_rollout_class
from verl.workers.utils.losses import ppo_loss

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _with_routing_replay_flag(enabled: bool):
    """Decorator to set 'enable_routing_replay' flag on the data TensorDict."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, data: TensorDict, *args, **kwargs):
            if self.enable_routing_replay:
                tu.assign_non_tensor_data(data, "enable_routing_replay", enabled)
            return func(self, data, *args, **kwargs)

        return wrapper

    return decorator


class TrainingWorker(Worker, DistProfilerExtension):
    """
    TrainingWorker provides a Tinker-like API (https://thinkingmachines.ai/tinker/) as a RayWorkerGroup
    to a single controller. Currently, we only provide more coarse grained APIs,
    and do not provide exact APIs as Tinker does. But this can be added in the future.
    """

    def __init__(self, config: TrainingWorkerConfig):
        Worker.__init__(self)

        from verl.workers.engine import BaseEngine, EngineRegistry

        initialize_global_process_group_ray(timeout_second=None)

        set_numa_affinity()

        self.config = config
        self.model_config = self.config.model_config
        self.engine_config = self.config.engine_config
        self.optimizer_config = self.config.optimizer_config
        self.checkpoint_config = self.config.checkpoint_config
        self.device_name = get_device_name()

        if self.engine_config is None:
            assert self.optimizer_config is None
            if self.config.auto_select_engine_optim_fn is None:
                raise ValueError(
                    "engine_config is not provided and auto_select_engine_optim_fn is not set. "
                    "Cannot determine engine backend."
                )
            # Support automatically select engine backend given model config
            self.engine_config, self.optimizer_config = self.config.auto_select_engine_optim_fn(
                self.model_config, self.device_name
            )

        # we use the one defined in model
        # TODO: this is not elegant and should refactor later
        self.engine_config.use_remove_padding = self.model_config.get("use_remove_padding", False)
        self.engine_config.use_fused_kernels = self.model_config.get("use_fused_kernels", False)

        self.profiler_config = self.config.profiler_config
        if self.profiler_config is not None:
            self.profiler_tool_config = self.profiler_config.tool_config.get(self.profiler_config.tool, {})
        else:
            self.profiler_tool_config = None

        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=self.profiler_config, tool_config=self.profiler_tool_config)
        )

        self.model_config.model_type = self.config.model_type
        self.engine: BaseEngine = EngineRegistry.new(
            model_type=self.config.model_type,
            backend=self.engine_config.strategy,
            model_config=self.model_config,
            engine_config=self.engine_config,
            optimizer_config=self.optimizer_config,
            checkpoint_config=self.checkpoint_config,
        )

        # build dispatch info
        self._register_dispatch_collect_info(
            mesh_name="train",
            dp_rank=self.engine.get_data_parallel_rank(),
            is_collect=self.engine.is_mp_src_rank_with_outputs(),
        )

        if hasattr(self.model_config, "hf_config"):
            self.flops_counter = FlopsCounter(self.model_config.hf_config)
        else:
            self.flops_counter = None

        self.loss_fn = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def to(self, device, model=True, optimizer=True, grad=True):
        """Manual control of load/offload"""
        assert device in ["cpu", "device"]

        if device == "device":
            device = get_device_name()

        self.engine.to(device=device, model=model, optimizer=optimizer, grad=grad)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_loss_fn(self, loss_fn):
        self.loss_fn = loss_fn

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reset(self):
        """
        Reset the model engine to the initial state. If the engine is not initialized,
        we initialize it. Otherwise, reload ckpt and reset states
        """
        self.engine.initialize()

    def _postprocess_output(self, output, *, global_token_num, delta_time, forward_only, images_seqlens):
        """

        Args:
            output: a dictionary containing loss, model_outputs and metrics

        Returns:

        """

        metrics: dict = output.pop("metrics")
        # perform all gather in dp group to ensure that it's correct.
        # Here each metric in metrics can be a list (micro-batch metrics) or a singleton
        # we should always sum the loss of each micro-batch as we scale by global_bsz/global_token
        loss = torch.sum(torch.tensor(output.pop("loss"), device=self.device_name))
        dp_group = self.engine.get_data_parallel_group()
        if dp_group is not None:
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG, group=dp_group)
        loss = loss.item()

        # For grad_norm, we do not perform all reduce because it is already been done when clipping grad
        grad_norm = metrics.pop("grad_norm", None)
        if isinstance(grad_norm, torch.Tensor):
            grad_norm = grad_norm.detach().item()
        lr = metrics.pop("lr", None)

        # For other metrics, we perform all gather in dp group (only if DP > 1)
        if dp_group is not None:
            final_metrics = allgather_dict_into_dict(data=metrics, group=dp_group)
        else:
            final_metrics = metrics
        final_metrics["loss"] = loss
        if grad_norm is not None:
            final_metrics["grad_norm"] = grad_norm
        if lr is not None:
            final_metrics["lr"] = lr

        # log memory
        final_metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
        final_metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
        final_metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

        # TODO: confirm the mtp loss IS same across dp
        for k, v in final_metrics.items():
            if k.startswith("mtp_losses"):
                flatten_v = [sublist[0] for sublist in v]  # sublist should be single element
                final_metrics[k] = sum(flatten_v) / len(flatten_v)
        # compute mfu
        if global_token_num is not None and self.flops_counter is not None:
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(
                global_token_num, delta_time, images_seqlens=images_seqlens
            )
            final_metrics["mfu"] = estimated_flops / promised_flops / torch.distributed.get_world_size()
            if forward_only:
                final_metrics["mfu"] /= 3.0
        # model outputs
        model_output = output.pop("model_output", {})
        # We only return final_metrics
        final_output = tu.get_tensordict(tensor_dict=model_output, non_tensor_dict={"metrics": final_metrics})
        return final_output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    def train_mini_batch(self, data: TensorDict) -> TensorDict:
        """Split a batch into N mini-batches run for multiple epochs

        Args:
            data:

        Returns:

        """
        maybe_fix_3d_position_ids(data)
        batch_size_per_dp = data.shape[0]
        disable_auto_offload = tu.pop(data, key="disable_auto_offload", default=False)
        mini_batch_size = tu.pop(data, key="mini_batch_size", default=None)
        num_mini_batch = tu.pop(data, key="num_mini_batch", default=None)
        # sp_dp_pad: DRIVER-assigned minibatch membership (a per-row tensor). Present only when
        # the data-parallel size does not divide the global minibatch (e.g. 1536 real seqs per
        # minibatch over dp=56 on 7 nodes); then the rank's rows cannot be split positionally and
        # the driver's apportionment (verl.trainer.ppo.sp_dp_pad.minibatch_ids) decides which
        # rows form which minibatch so that every minibatch holds exactly ppo_mini_batch_size
        # REAL sequences globally. Absent -> the positional path below, byte for byte.
        mb_ids = tu.pop(data, key="sp_minibatch_id", default=None)
        epochs = tu.pop(data, key="epochs", default=1)
        seed = tu.pop(data, key="seed", default=42)
        dataloader_kwargs = tu.pop(data, key="dataloader_kwargs", default={})

        assert mini_batch_size is not None or num_mini_batch is not None

        if mb_ids is not None:
            assert mini_batch_size is None and num_mini_batch is not None, (
                "sp_minibatch_id needs num_mini_batch (and no mini_batch_size)"
            )
            num_mini_batch = int(num_mini_batch)
            mb_ids = torch.as_tensor(mb_ids).reshape(-1).to(torch.long).cpu()
            assert mb_ids.shape[0] == batch_size_per_dp, (mb_ids.shape[0], batch_size_per_dp)
            _counts = torch.bincount(mb_ids, minlength=num_mini_batch).tolist()
            assert len(_counts) == num_mini_batch and min(_counts) >= 1, (
                f"sp_minibatch_id: rank {self.engine.get_data_parallel_rank()} rows per minibatch "
                f"{_counts} (every minibatch needs >= 1 row on every rank)"
            )
            mini_batch_size_per_gpu = None
            print(f"[sp_dp_pad] rank {self.engine.get_data_parallel_rank()}: {batch_size_per_dp} rows -> "
                  f"{num_mini_batch} driver-assigned minibatches of {_counts} rows", flush=True)
        elif mini_batch_size is None:
            assert batch_size_per_dp % num_mini_batch == 0, f"Got {batch_size_per_dp=} and {num_mini_batch=}"
            mini_batch_size_per_gpu = batch_size_per_dp // num_mini_batch
        else:
            assert mini_batch_size % self.engine.get_data_parallel_size() == 0, (
                f"Got {mini_batch_size=} and {self.engine.get_data_parallel_size()=}"
            )
            mini_batch_size_per_gpu = mini_batch_size // self.engine.get_data_parallel_size()

        # make iterator
        if mb_ids is not None:
            dataloader = tu.make_iterator_by_ids(data, mb_ids, num_mini_batch, epochs)
            total_num_iterations = num_mini_batch * epochs
        else:
            dataloader = tu.make_iterator(
                data,
                mini_batch_size=mini_batch_size_per_gpu,
                epochs=epochs,
                seed=seed + self.engine.get_data_parallel_rank(),
                dataloader_kwargs=dataloader_kwargs,
            )
            total_num_iterations = data.shape[0] // mini_batch_size_per_gpu * epochs

        # ---- Projected Inner AdamW: the epochs x mini-batch loop below IS the inner
        # trajectory (K = epochs * num_mini_batches). Anchor theta_t before the first inner step;
        # decoupled weight decay + metrics after the last one. Inert for any other optimizer.
        _pia = getattr(self.engine, "optimizer", None)
        if not hasattr(_pia, "set_anchor"):
            _pia = None

        with (
            self.engine.train_mode(disable_auto_offload=disable_auto_offload),
            Timer(name="train_batch", logger=None),
        ):
            # MUST be inside train_mode(): with fsdp_config.param_offload=True the parameters
            # live on CPU outside this context, and an anchor cloned there would be a
            # cross-device tensor at projection time (and a per-inner-step H2D copy of a
            # parameter-sized buffer even if it worked).
            if _pia is not None:
                _pia.set_anchor()

            # update  (total_num_iterations computed with the iterator above: positional
            # rows // per-rank mini x epochs, or num_mini_batch x epochs for driver-assigned ids)
            output_lst = []

            for batch_idx, mini_batch_td in enumerate(dataloader):
                # add global token num
                if "input_ids" in mini_batch_td:
                    global_token_num = mini_batch_td["input_ids"].offsets().diff().tolist()  # (total_nnz,)
                    # allgather from dp rank
                    global_token_num_output = [None] * torch.distributed.get_world_size(
                        self.engine.get_data_parallel_group()
                    )
                    torch.distributed.all_gather_object(
                        global_token_num_output, global_token_num, self.engine.get_data_parallel_group()
                    )
                    global_token_num = [x for xs in global_token_num_output for x in xs]
                else:
                    global_token_num = None

                tu.assign_non_tensor(
                    mini_batch_td,
                    global_token_num=NonTensorData(global_token_num),
                    update_lr_scheduler=batch_idx == total_num_iterations - 1,
                    disable_auto_offload=True,
                )
                actor_output = self.train_batch(mini_batch_td)
                output_lst.append(actor_output)

            # close the inner trajectory: install decoupled weight decay once
            pia_metrics = _pia.finish_outer() if _pia is not None else {}
            if _pia is not None:
                # Primary instrument: the per-INNER-STEP loss trail. It cannot ride
                # the ordinary metric path — verl.utils.metric.reduce_metrics collapses every
                # list to its mean before metrics.jsonl — so emit one distinctly-keyed SCALAR
                # per inner step (a 1-element list survives the mean unchanged).
                for _k_idx, _out in enumerate(output_lst):
                    _m = tu.get(_out, "metrics") if _out is not None else None
                    if not _m:
                        continue
                    # NOTE the loss path emits ALREADY-PREFIXED keys ("actor/pg_loss" from
                    # losses.py:186, "actor/ppo_kl"/"actor/pg_clipfrac" from core_algos, and
                    # "actor/entropy_loss" — not "entropy"); grad_norm is the only bare one.
                    for _name in (
                        "actor/pg_loss",
                        "actor/ppo_kl",
                        "actor/pg_clipfrac",
                        "actor/pg_clipfrac_lower",
                        "actor/entropy_loss",
                        "grad_norm",
                    ):
                        _v = _m.get(_name)
                        if _v is None:
                            continue
                        _short = _name.split("/")[-1]
                        try:
                            if isinstance(_v, Metric):
                                _s = float(_v.aggregate())
                            elif isinstance(_v, list):
                                _flat = [
                                    float(x.aggregate()) if isinstance(x, Metric) else float(x) for x in _v
                                ]
                                _s = sum(_flat) / max(1, len(_flat))
                            else:
                                _s = float(_v)
                        except (TypeError, ValueError):
                            continue
                        pia_metrics[f"pia/inner_{_short}_{_k_idx}"] = _s

            if self.engine.is_mp_src_rank_with_outputs():
                actor_output = [tu.get(output, "metrics") for output in output_lst]
                metrics = {}
                for output in actor_output:
                    for key, val in output.items():
                        # flattn dp and micro batch
                        if isinstance(val, list):
                            output[key] = (
                                Metric.aggregate_dp(val)
                                if isinstance(val[0], Metric)
                                else list(chain.from_iterable(val))
                            )
                    append_to_dict(metrics, output)

                # pia/* are already reduced across ranks inside the optimizer; emit once
                for _k, _v in pia_metrics.items():
                    metrics[_k] = [_v]

                output = tu.get_tensordict(tensor_dict={}, non_tensor_dict={"metrics": metrics}).cpu()
            else:
                output = None
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    @DistProfiler.annotate(color="red", role="train_batch")
    def train_batch(self, data: TensorDict) -> TensorDict:
        assert self.loss_fn is not None, "loss function can't be None when calling train_batch"
        assert not self.engine_config.forward_only, "Can't run `train_batch` when forward_only is in the engine config."
        # global_token_num should be a list of number of tokens of each seq in this batch
        global_token_num = tu.get(data, key="global_token_num")
        disable_auto_offload = tu.get(data, key="disable_auto_offload", default=False)
        images_seqlens = tu.get(data, key="images_seqlens", default=None)

        # inject engineering parameters if not specified
        default_keys = dict(
            use_remove_padding=self.model_config.get("use_remove_padding", False),
            use_dynamic_bsz=self.engine_config.use_dynamic_bsz,
            max_token_len_per_gpu=self.engine_config.max_token_len_per_gpu,
            micro_batch_size_per_gpu=self.engine_config.micro_batch_size_per_gpu,
            use_fused_kernels=self.engine_config.use_fused_kernels,
        )

        for key, val in default_keys.items():
            if key not in data.keys():
                tu.assign_non_tensor(data, **{key: val})

        with (
            self.engine.train_mode(disable_auto_offload=disable_auto_offload),
            Timer(name="train_batch", logger=None) as timer,
        ):
            output = self.engine.train_batch(data, loss_function=self.loss_fn)
            # containing loss, model_output and metrics
            # for training, we only care about loss and metrics
        delta_time = timer.last

        update_lr_scheduler = tu.get(data, key="update_lr_scheduler", default=False)
        # update lr scheduler
        if update_lr_scheduler:
            lr = self.engine.lr_scheduler_step()
        else:
            lr = None

        if self.engine.is_mp_src_rank_with_outputs():
            # we don't need model_output in training. Maybe we change out mind later
            output.pop("model_output")
            if lr is not None:
                output["metrics"]["lr"] = lr
            final_output = self._postprocess_output(
                output,
                global_token_num=global_token_num,
                delta_time=delta_time,
                forward_only=False,
                images_seqlens=images_seqlens,
            ).cpu()
        else:
            final_output = None

        return final_output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    def infer_batch(self, data: TensorDict) -> TensorDict:
        # add mfu calculator
        global_token_num = tu.get(data, key="global_token_num")
        compute_loss = tu.get(data, key="compute_loss", default=True)
        disable_auto_offload = tu.get(data, key="disable_auto_offload", default=False)
        no_lora_adapter = tu.pop(data, key="no_lora_adapter", default=False)
        images_seqlens = tu.get(data, key="images_seqlens", default=None)

        default_keys = dict(
            use_remove_padding=self.model_config.get("use_remove_padding", False),
            use_dynamic_bsz=self.engine_config.use_dynamic_bsz,
            max_token_len_per_gpu=self.engine_config.infer_max_token_len_per_gpu,
            micro_batch_size_per_gpu=self.engine_config.infer_micro_batch_size_per_gpu,
            use_fused_kernels=self.engine_config.use_fused_kernels,
        )

        for key, val in default_keys.items():
            if key not in data.keys():
                tu.assign_non_tensor(data, **{key: val})

        # for sft training, we need to compute loss in eval
        loss_function = self.loss_fn if compute_loss else None

        with (
            self.engine.eval_mode(disable_auto_offload=disable_auto_offload),
            Timer(name="eval_batch", logger=None) as timer,
        ):
            adapter_ctx = self.engine.disable_adapter() if no_lora_adapter else nullcontext()
            with adapter_ctx:
                output = self.engine.infer_batch(data, loss_function=loss_function)
        delta_time = timer.last

        if self.engine.is_mp_src_rank_with_outputs():
            final_output = self._postprocess_output(
                output,
                global_token_num=global_token_num,
                delta_time=delta_time,
                forward_only=True,
                images_seqlens=images_seqlens,
            ).cpu()
        else:
            final_output = None

        return final_output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        return self.engine.save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        return self.engine.load_checkpoint(local_path, hdfs_path, del_local_after_load)


class ActorRolloutRefWorker(Worker, DistProfilerExtension):
    """Hybrid worker that includes actor model, rollout and optional ref model.
    For standalone actor or rollout, use ActorWorker or BaseRollout respectively.

    NOTE: ActorRolloutRefWorker no longer support spmd mode and run native server mode.
    """

    def __init__(
        self, config: DictConfig, role: str, distillation_config: Optional[DistillationConfig] = None, **kwargs
    ):
        Worker.__init__(self)
        self.config = config
        self.distillation_config = distillation_config
        self.distillation_enabled = is_distillation_enabled(distillation_config)
        self.role = role
        self.actor: TrainingWorker = None
        self.ref: TrainingWorker = None
        self.rollout: BaseRollout = None
        self._teacher_unembed = None
        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]
        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]

        if self._is_actor:
            omega_profiler_config = config.actor.get("profiler", {})
        elif self._is_rollout:
            # NOTE: In colocation mode, rollout config may not take effect (follow the actor config)
            # This is for extendability in AsyncRL cases
            omega_profiler_config = config.rollout.get("profiler", {})
        else:
            omega_profiler_config = config.ref.get("profiler", {})

        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory", "precision_debugger"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None

        # Router replay is supported on the megatron engine and on the veomni
        # engine. Both expose `router_replay` on their per-strategy engine
        # config (the field lives on the shared `EngineConfig` base).
        actor_strategy = self.config.actor.strategy
        if actor_strategy == "megatron":
            rr_mode = self.config.actor.megatron.router_replay.mode
        elif actor_strategy == "veomni":
            rr_mode = self.config.actor.veomni.router_replay.mode
        else:
            rr_mode = "disabled"
        self.enable_routing_replay = rr_mode != "disabled"

        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_loss_fn(self, loss_fn):
        self.actor.set_loss_fn(loss_fn=loss_fn)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def to(self, device, model=True, optimizer=True, grad=True):
        """Manual control of load/offload"""
        self.actor.to(device=device, model=model, optimizer=optimizer, grad=grad)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def offload_colocated_for_rm(self, empty_cache: bool = True):
        """Offload the colocated ref (OPSD frozen teacher) AND actor engines to CPU and release the
        caching-allocator reserve, so a COLOCATED reward model's vLLM ``profile_run`` sees the free
        GPU memory a no-ref run would. `to()` alone offloads only ``self.actor``; the extra ``self.ref``
        engine that OPSD adds (absent in plain GRPO) is what tips the gpt-oss-120b judge's profile into
        a cublasCreate/NVLink OOM. Params auto-onload on their next forward (param_offload)."""
        for eng in (getattr(self, "ref", None), getattr(self, "actor", None)):
            if eng is not None:
                try:
                    eng.to(device="cpu", model=True, optimizer=False, grad=False)
                except Exception as e:  # a missing optimizer/grad on the frozen ref must not break init
                    print(f"[opsd] offload_colocated_for_rm: skipped one engine: {e}", flush=True)
        if empty_cache:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model)

        # 1. build reference model
        if "ref" in self.role:
            # TODO: align ref config with actor config
            with open_dict(self.config.ref):
                self.config.ref.ppo_mini_batch_size = self.config.actor.ppo_mini_batch_size
                self.config.ref.ppo_micro_batch_size = self.config.ref.pop("log_prob_micro_batch_size", None)
                self.config.ref.ppo_micro_batch_size_per_gpu = self.config.ref.pop(
                    "log_prob_micro_batch_size_per_gpu", None
                )
                self.config.ref.use_dynamic_bsz = self.config.ref.pop("log_prob_use_dynamic_bsz", False)
                self.config.ref.ppo_max_token_len_per_gpu = self.config.ref.pop("log_prob_max_token_len_per_gpu", None)
            ref_config: ActorConfig = omega_conf_to_dataclass(self.config.ref)

            # The ref model does not need to enable MTP; force it to false.
            ref_config.model_config = deepcopy(model_config)
            ref_config.model_config.mtp = MtpConfig(enable=False)

            # OPD-aux: the colocated teacher (OPSD/nitrobrew forward-KL path) may be a DIFFERENT
            # model than the actor init. SP_OPD_TEACHER_PATH loads it into the ref slot ONLY; the
            # actor/rollout keep the init model. Rebuild the ref model_config from the teacher path
            # so HFModelConfig re-resolves the local snapshot / tokenizer / hf_config (offline-safe).
            _opd_teacher = os.environ.get("SP_OPD_TEACHER_PATH")
            if _opd_teacher:
                from omegaconf import OmegaConf

                _m = OmegaConf.create(OmegaConf.to_container(self.config.model, resolve=True))
                _m.path = _opd_teacher
                if "local_path" in _m:
                    _m.local_path = _opd_teacher
                _ref_teacher_cfg = omega_conf_to_dataclass(_m)
                _ref_teacher_cfg.mtp = MtpConfig(enable=False)
                ref_config.model_config = _ref_teacher_cfg
                print(f"[opd] ref slot loads OPD teacher {_opd_teacher} (actor init unchanged)", flush=True)

            # construct TrainingWorkerConfig
            ref_training_config = TrainingWorkerConfig(
                model_type=ref_config.model_config.get("model_type", "language_model"),
                model_config=ref_config.model_config,
                engine_config=ref_config.engine,
                optimizer_config=ref_config.optim,
                checkpoint_config=ref_config.checkpoint,
            )

            # assign engine configs
            ref_training_config.engine_config.use_dynamic_bsz = self.config.ref.use_dynamic_bsz
            ref_training_config.engine_config.infer_max_token_len_per_gpu = self.config.ref.ppo_max_token_len_per_gpu
            ref_training_config.engine_config.infer_micro_batch_size_per_gpu = (
                self.config.ref.ppo_micro_batch_size_per_gpu
            )
            ref_training_config.engine_config.use_remove_padding = model_config.get("use_remove_padding", False)

            self.ref = TrainingWorker(config=ref_training_config)
            self.ref.reset()
            self.set_dispatch_collect(mesh_name="ref", **self.ref.get_dispatch_collect())

        # 2. build actor model
        if "actor" in self.role:
            actor_config: ActorConfig = omega_conf_to_dataclass(self.config.actor)
            actor_config.model_config = model_config
            distillation_config: Optional[DistillationConfig] = (
                omega_conf_to_dataclass(self.distillation_config) if self.distillation_enabled else None
            )

            actor_training_config = TrainingWorkerConfig(
                model_type=actor_config.model_config.get("model_type", "language_model"),
                model_config=actor_config.model_config,
                engine_config=actor_config.engine,
                optimizer_config=actor_config.optim,
                checkpoint_config=actor_config.checkpoint,
            )

            assert self.config.actor.use_dynamic_bsz == self.config.rollout.log_prob_use_dynamic_bsz

            # assign engine configs
            actor_training_config.engine_config.use_dynamic_bsz = self.config.actor.use_dynamic_bsz
            actor_training_config.engine_config.infer_max_token_len_per_gpu = (
                self.config.rollout.log_prob_max_token_len_per_gpu
            )
            actor_training_config.engine_config.infer_micro_batch_size_per_gpu = (
                self.config.rollout.log_prob_micro_batch_size_per_gpu
            )
            actor_training_config.engine_config.max_token_len_per_gpu = self.config.actor.ppo_max_token_len_per_gpu
            actor_training_config.engine_config.micro_batch_size_per_gpu = (
                self.config.actor.ppo_micro_batch_size_per_gpu
            )
            actor_training_config.engine_config.use_remove_padding = model_config.get("use_remove_padding", False)

            if self.config.actor.use_dynamic_bsz:
                assert self.config.rollout.log_prob_max_token_len_per_gpu is not None
                assert self.config.actor.ppo_max_token_len_per_gpu is not None
            else:
                assert self.config.rollout.log_prob_micro_batch_size_per_gpu is not None
                assert self.config.actor.ppo_micro_batch_size_per_gpu is not None
            if self.distillation_enabled:
                self.loss_fn = partial(
                    distillation_ppo_loss, config=actor_config, distillation_config=distillation_config
                )
                # Loss knobs forwarded to the fused model forward (OPSD two-sided nitrobrew KL)
                # via non-tensor batch data in update_actor.
                self._distillation_loss_cfg = distillation_config.distillation_loss
            else:
                self.loss_fn = partial(ppo_loss, config=actor_config)
                self._distillation_loss_cfg = None
            self.actor = TrainingWorker(config=actor_training_config)
            self.actor.reset()
            self.actor.set_loss_fn(self.loss_fn)
            self.set_dispatch_collect(mesh_name="actor", **self.actor.get_dispatch_collect())

            # OPD inline teacher: stash the colocated ref engine on the actor engine so the actor's
            # per-microbatch fused forward computes teacher hidden inline (no full-batch buffer).
            _opd_on = os.environ.get("SP_OPD_ENABLE") in ("1", "True", "true")
            _opd_inline = os.environ.get("SP_OPD_INLINE_TEACHER", "1") not in ("0", "false", "False")
            if _opd_on and _opd_inline and getattr(self, "ref", None) is not None:
                self.actor.engine._opd_teacher_engine = self.ref.engine
                self._opd_inline = True
                print("[opd] inline teacher wired: ref engine stashed on actor engine", flush=True)

        # 3. build rollout engine
        if "rollout" in self.role:
            rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)

            # TODO: move rollout_device_mesh into ServerAdapter
            # 3.1 build rollout device mesh (sglang need only)
            infer_tp = rollout_config.tensor_model_parallel_size * rollout_config.data_parallel_size
            infer_pp = rollout_config.pipeline_model_parallel_size
            infer_world_size = infer_tp * infer_pp
            dp = self.world_size // infer_world_size
            assert self.world_size % infer_world_size == 0, (
                f"rollout world_size: {self.world_size} is not divisible by infer_world_size: {infer_world_size}"
            )
            rollout_device_mesh = init_device_mesh(
                get_device_name(), mesh_shape=(dp, infer_tp, infer_pp), mesh_dim_names=["dp", "infer_tp", "infer_pp"]
            )

            # 3.2 initialize rollout engine
            rollout_cls: type[BaseRollout] = get_rollout_class(rollout_config.name, rollout_config.mode)
            self.rollout = rollout_cls(
                config=rollout_config, model_config=model_config, device_mesh=rollout_device_mesh
            )

            # used for LoRA (base_sync_done is unused in merge-only mode but kept for Phase 2 adapter path)
            self.base_sync_done: bool = "dummy" not in self.config.rollout.load_format
            self.layered_summon = self.config.rollout.get("layered_summon", False)
            self.peft_merge: bool = model_config.lora.get("merge", False)

        # 4. build checkpoint engine
        if "actor" in self.role:
            checkpoint_engine_config = omega_conf_to_dataclass(self.config.rollout.checkpoint_engine)
            backend = checkpoint_engine_config.backend
            bucket_size = checkpoint_engine_config.update_weights_bucket_megabytes << 20
            engine_kwargs = checkpoint_engine_config.engine_kwargs.get(backend, {})
            # If custom_backend_module is set, import it so plugins can register
            # in CheckpointEngineRegistry before the backend is instantiated.
            import_external_libs(checkpoint_engine_config.custom_backend_module or None)
            self.checkpoint_engine = CheckpointEngineRegistry.new(
                backend, is_master=(torch.distributed.get_rank() == 0), bucket_size=bucket_size, **engine_kwargs
            )

        # Free cached GPU memory so colocated vLLM processes can see it via cudaMemGetInfo
        aggressive_empty_cache(force_sync=True)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="ref"))
    @DistProfiler.annotate(color="olive", role="ref_compute_log_prob")
    @_with_routing_replay_flag(enabled=False)
    def compute_ref_log_prob(self, data: TensorDict) -> TensorDict:
        output = self.ref.infer_batch(data=data)
        return output.cpu() if output is not None else None

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="ref"))
    @DistProfiler.annotate(color="cyan", role="ref_compute_hidden")
    @_with_routing_replay_flag(enabled=False)
    def compute_ref_hidden(self, data: TensorDict) -> TensorDict:
        """OPSD: FINAL-layer hidden states from the FROZEN ref model (colocated teacher) on the
        privileged inputs. A plain no-grad forward (NOT the rmpad loss path).

        Memory discipline: capture ONLY the last hidden via a forward
        hook on the decoder's final norm — never output_hidden_states=True (keeps every layer) and
        never full-vocab logits (logits_to_keep=1 when supported). With per-sample "hidden_start" /
        "hidden_len" (+ "hidden_out_max", constant column) in `data`, slices the response-predicting
        span [hidden_start : hidden_start+hidden_len) per sample and returns response-only
        {"ref_hidden": [B, hidden_out_max, D]} (bf16, CPU). Without them, returns the full
        {"ref_hidden": [B, L, D]} (bf16, CPU) for backward compatibility."""
        dev = get_torch_device().current_device()
        # The @register dispatch hands this method a DataProto (not a bare TensorDict); string-key
        # indexing lives on its `.batch`. Normalize so `td[...]` / `td.keys()` work for both.
        td = data.batch if hasattr(data, "batch") else data
        input_ids = td["input_ids"].to(dev)
        attention_mask = td["attention_mask"].to(dev) if "attention_mask" in td.keys() else None
        with self.ref.engine.eval_mode():
            module = self.ref.engine.module
            captured: dict = {}
            hook_handle = None

            # self.ref.engine.module is FSDP-wrapped (transformer_impl wraps then stores the
            # wrapper as self.module). FSDP.__getattr__ USUALLY forwards get_decoder()/.model, but
            # nested activation-checkpoint / FSDP units can break that — so unwrap EXPLICITLY and
            # fail loudly rather than silently regress to output_hidden_states=True (all layers,
            # which would blow the OPSD memory budget).
            inner = module
            for _ in range(8):
                nxt = getattr(inner, "_fsdp_wrapped_module", None) or getattr(
                    inner, "_checkpoint_wrapped_module", None
                )
                if nxt is None:
                    break
                inner = nxt
            decoder = None
            if hasattr(inner, "get_decoder"):
                try:
                    decoder = inner.get_decoder()
                except Exception:
                    decoder = None
            if decoder is None:
                decoder = getattr(inner, "model", inner)
            final_norm = getattr(decoder, "norm", None)
            if final_norm is None:  # some archs name it differently
                final_norm = getattr(decoder, "final_layernorm", None) or getattr(
                    decoder, "ln_f", None
                )
            if final_norm is not None:
                hook_handle = final_norm.register_forward_hook(
                    lambda _m, _inp, out: captured.__setitem__(
                        "hidden", out[0] if isinstance(out, tuple) else out
                    )
                )
            elif os.environ.get("OPSD_ALLOW_ALL_LAYER_HIDDEN") != "1":
                raise RuntimeError(
                    "OPSD compute_ref_hidden: could not locate the decoder final norm to install "
                    f"the hidden-only hook (unwrapped module type={type(inner).__name__}). Refusing "
                    "to fall back to output_hidden_states=True (all-layer capture OOMs at "
                    "no-truncation lengths). Set OPSD_ALLOW_ALL_LAYER_HIDDEN=1 to override for small "
                    "debug models."
                )
            # return_dict=True is REQUIRED: use_fused_kernels patches forward at the CLASS level,
            # so the colocated ref (same architecture as the actor) runs the patched forward too,
            # which raises on return_dict=None. Harmless for the unpatched HF forward.
            fwd_kwargs = dict(use_cache=False, return_dict=True)
            import inspect

            fwd_params = inspect.signature(module.forward).parameters
            if "nb_hidden_only" in fwd_params:
                # Patched fused forward (dense_common): decoder-only pass, ALL lm_head work
                # skipped (otherwise FusedLinearForPPO would run full-vocab chunked lm-head
                # compute over every teacher token for log_probs nothing reads).
                fwd_kwargs["nb_hidden_only"] = True
            if hook_handle is not None:
                if "nb_hidden_only" not in fwd_params:
                    # Unpatched HF forward: lm_head only on the last position.
                    if "logits_to_keep" in fwd_params:
                        fwd_kwargs["logits_to_keep"] = 1
                    elif "num_logits_to_keep" in fwd_params:
                        fwd_kwargs["num_logits_to_keep"] = 1
            else:
                fwd_kwargs["output_hidden_states"] = True  # fallback: no norm hook available

            # Micro-batch the teacher forward to bound activation memory. A single forward over the
            # full [B, Lt] batch OOMs at no-truncation lengths — the qwen3 RMSNorm fp32 upcast of
            # [B, Lt, D] alone is tens of GiB (observed 61 GiB). Process SP_OPD_REF_MICRO_BS rows at
            # a time, trimmed to the chunk's real max length (right padding), assembling the
            # response-only slice directly so no full [B, Lt, D] hidden ever materializes.
            B = input_ids.shape[0]
            have_slice = "hidden_start" in td.keys() and "hidden_len" in td.keys()
            ref_mb = max(1, int(os.environ.get("SP_OPD_REF_MICRO_BS", "2")))
            try:
                if have_slice:
                    starts = td["hidden_start"].tolist()
                    lens = td["hidden_len"].tolist()
                    out_max = (
                        int(td["hidden_out_max"][0].item())
                        if "hidden_out_max" in td.keys()
                        else max(lens)
                    )
                    hidden = None
                    with torch.no_grad():
                        for lo in range(0, B, ref_mb):
                            hi = min(lo + ref_mb, B)
                            if attention_mask is not None:
                                am_c = attention_mask[lo:hi]
                                clen = max(1, int(am_c.sum(dim=1).max().item()))
                                am_c = am_c[:, :clen]
                            else:
                                clen = input_ids.shape[1]
                                am_c = None
                            captured.pop("hidden", None)
                            fk = dict(fwd_kwargs, input_ids=input_ids[lo:hi, :clen])
                            if am_c is not None:
                                fk["attention_mask"] = am_c
                            out = module(**fk)
                            h = captured.get("hidden")
                            if h is None:
                                h = out.hidden_states[-1]
                            h = h.to(torch.bfloat16)  # [chunk_B, clen, D]
                            if hidden is None:
                                # Response buffer lives on CPU: at no-truncation lengths it is
                                # [B, out_max, D] ~ tens of GiB. Keeping it on GPU plus a single
                                # ~16 GiB GPU->host .cpu() at the end hard-kills the worker (EOF,
                                # no catchable CUDA OOM). Copy each chunk's response slice to host
                                # incrementally so the big tensor never sits on GPU.
                                hidden = torch.zeros(
                                    (B, out_max, h.shape[-1]), dtype=torch.bfloat16, device="cpu"
                                )
                            for j in range(hi - lo):
                                s = int(starts[lo + j])
                                n = min(int(lens[lo + j]), out_max, clen - s)
                                if n > 0:
                                    hidden[lo + j, :n] = h[j, s : s + n].to("cpu")
                            del h, out
                else:
                    # backward-compat (no per-sample slice info): single full-batch forward
                    captured.pop("hidden", None)
                    fk = dict(fwd_kwargs, input_ids=input_ids)
                    if attention_mask is not None:
                        fk["attention_mask"] = attention_mask
                    with torch.no_grad():
                        out = module(**fk)
                    hidden = captured.get("hidden")
                    if hidden is None:
                        hidden = out.hidden_states[-1]
                    hidden = hidden.to(torch.bfloat16)  # [B, L, D]
            finally:
                if hook_handle is not None:
                    hook_handle.remove()
        result = tu.get_tensordict({"ref_hidden": hidden})
        # .cpu() lives on the TensorDict, not the DataProto wrapper (which has no .cpu()); move the
        # host transfer before wrapping. Caller reads out.batch["ref_hidden"], so still return a DataProto.
        return DataProto.from_tensordict(result.cpu())

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
    @_with_routing_replay_flag(enabled=True)
    def compute_log_prob(self, data: TensorDict) -> TensorDict:
        output = self.actor.infer_batch(data)

        return output.cpu() if output is not None else None

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_update")
    @_with_routing_replay_flag(enabled=True)
    def update_actor(self, data: TensorDict) -> TensorDict:
        if self._teacher_unembed is not None:
            data["teacher_unembed"] = NonTensorData(self._teacher_unembed)
        if getattr(self, "_distillation_loss_cfg", None) is not None:
            # Forward the loss knobs to the fused model forward (OPSD two-sided nitrobrew KL);
            # read back in transformer_impl.prepare_model_inputs.
            tu.assign_non_tensor(
                data,
                nb_token_clip=getattr(self._distillation_loss_cfg, "token_clip", None),
                nb_kd_temperature=self._distillation_loss_cfg.kd_temperature,
                nb_log_prob_min_clamp=self._distillation_loss_cfg.log_prob_min_clamp,
            )
        if getattr(self, "_opd_inline", False):
            # Bring the param_offloaded ref (OPD teacher) onto GPU for the whole update so the
            # inline per-microbatch teacher forward can run; offload again after (frozen, ~1 GB/rank
            # FSDP-sharded, no grad/optimizer).
            from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu

            load_fsdp_model_to_gpu(self.ref.engine.module)
            try:
                output = self.actor.train_mini_batch(data=data)
            finally:
                offload_fsdp_model_to_cpu(self.ref.engine.module)
        else:
            output = self.actor.train_mini_batch(data=data)
        return output.cpu() if output is not None else None

    # ------------------------------------------------------------------
    # sp_q: generative-Q readiness worker surface. A SECOND AdamW (O_Q)
    # over the same weights, whose per-step
    # delta is computed at theta_0 BEFORE the PPO update, held in a shard-
    # sized buffer, and applied AFTER PPO scaled by the driver-computed
    # movement cap s_t = min(1, rho_cap/rho_t). All methods are inert unless
    # the driver calls them (gated by SP_Q_ENABLE on the driver side).
    # ------------------------------------------------------------------

    def _sp_q_local_tensors(self):
        """[(param, local_tensor)] over the actor module; DTensor(fsdp2)-safe."""
        out = []
        for p in self.actor.engine.module.parameters():
            d = p.data
            lt = d._local_tensor if hasattr(d, "_local_tensor") else d
            out.append((p, lt))
        return out

    def _sp_q_replication(self) -> int:
        """How many ranks hold a COPY of each local shard (1 for full-shard FSDP)."""
        ws = torch.distributed.get_world_size()
        fsdp_size = int(getattr(self.actor.engine.engine_config, "fsdp_size", -1) or -1)
        if fsdp_size <= 0 or fsdp_size >= ws:
            return 1
        return ws // fsdp_size

    def _sp_q_allreduce_sum(self, value: float) -> float:
        t = torch.tensor(float(value), dtype=torch.float64, device=get_torch_device().current_device())
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        return t.item() / self._sp_q_replication()

    def _sp_q_optim_state_to(self, device):
        """Move O_Q's Adam moment tensors between GPU and CPU (they are as big as the
        model, so they only live on GPU during the Q phase).

        `device` may be a str, a torch.device, or the bare integer index that
        get_torch_device().current_device() returns -- an int has no `.type`, so it must
        be resolved through the accelerator's device name rather than dereferenced. This
        only ever fires once O_Q holds state (i.e. from the SECOND Q phase onward, and on
        any resume that restores O_Q), which is why an unresolved int is invisible on the
        first step and fatal on the next one."""
        opt = getattr(self, "_sp_q_optimizer", None)
        if opt is None:
            return
        if isinstance(device, torch.device):
            dev = device
        elif isinstance(device, int):
            dev = torch.device(get_device_name(), device)
        else:
            dev = torch.device(device)
        for param_state in opt.state.values():
            for k, v in param_state.items():
                if torch.is_tensor(v) and v.device.type != dev.type:
                    param_state[k] = v.to(dev, non_blocking=True)

    def _sp_q_ensure_optimizer(self, lr: float = 2e-6):
        """Create O_Q (fresh AdamW, betas (0.9,0.999), eps 1e-8, wd 0, no
        scheduler) if it does not exist yet, restoring any state stashed by
        sp_q_load_optim. Shared by the Q phase and sp_q_save_optim so that EVERY
        post-branch checkpoint carries O_Q shards -- including steps whose Q phase was
        skipped, which would otherwise write a checkpoint that is fatal to resume."""
        opt = getattr(self, "_sp_q_optimizer", None)
        if opt is not None:
            return opt
        opt = torch.optim.AdamW(
            self.actor.engine.module.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8,
            weight_decay=0.0,
        )
        self._sp_q_optimizer = opt
        pending = getattr(self, "_sp_q_optim_pending_state", None)
        if pending is not None:
            # MLP-Q resume: checkpoints past the first BCE step carry a
            # TWO-group O_Q (backbone + head), but this function builds one group, and
            # torch refuses a group-count mismatch at load. Rebuild the head from the
            # checkpoint stash (sp_q_load_optim stashes qmlp_head.pt before O_Q) and add
            # its group BEFORE loading; the load then restores each group's lr/tags, and
            # the per-step lr-set loops reassign them anyway.
            _want = len(pending.get("param_groups", []) or [])
            if _want == len(opt.param_groups) + 1:
                _head = self._sp_q_mlp_ensure_head("")
                opt.add_param_group({"params": list(_head.parameters()), "lr": 1e-4,
                                     "sp_q_group": "head"})
                print("[sp_q-mlp] O_Q resume: head group added pre-load (ckpt has "
                      f"{_want} groups)", flush=True)
            opt.load_state_dict(pending)
            self._sp_q_optim_pending_state = None
            self._sp_q_optim_state_to("cpu")
            print("[sp_q] O_Q state restored from checkpoint stash", flush=True)
        return opt

    # ------------------------------------------------------------------
    # sp_q MLP surface (MLP-head ablation): Q is a small MLP head on the
    # actor backbone's final hidden state instead of generated grid text.
    # Same tied backbone, same interleave slot, same O_Q machinery; the Q
    # loss becomes row-weighted BCE on sigmoid(head(h_last)), and the head
    # trains at its own fixed lr in a separate optimizer param group.
    # ------------------------------------------------------------------

    def _sp_q_mlp_ensure_head(self, head_path: str = ""):
        """Build/load the MLP head (norm+fc1+fc2+fc3) once.

        Priority: checkpoint stash (resume) > head_path file (branch first attach).
        The head is a bare replicated module -- NOT FSDP-wrapped -- so its gradients
        must be manually all-reduced across dp before the optimizer step; FSDP only
        reduces what it wraps, and a missed reduce silently forks the head per rank."""
        head = getattr(self, "_sp_q_mlp_head", None)
        if head is not None:
            return head
        stash = getattr(self, "_sp_q_mlp_head_stash", None)
        if stash is None:
            if not head_path or not os.path.exists(head_path):
                raise FileNotFoundError(
                    f"sp_q-mlp: no head checkpoint stash and no head file at {head_path!r}; "
                    "the MLP surface cannot start without weights")
            stash = torch.load(head_path, map_location="cpu", weights_only=False)
            print(f"[sp_q-mlp] head loaded from file {head_path}", flush=True)
        mid = int(stash.get("head_mid", 1024))
        drop = float(stash.get("dropout", 0.1))
        hidden = int(stash["head"]["fc1.weight"].shape[1])

        class _SPQMlpHead(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = torch.nn.LayerNorm(hidden)
                self.fc1 = torch.nn.Linear(hidden, mid)
                self.fc2 = torch.nn.Linear(mid, mid // 2)
                self.fc3 = torch.nn.Linear(mid // 2, 1)
                self.act = torch.nn.GELU()
                self.drop = torch.nn.Dropout(drop)

            def forward(self, h):
                x = self.norm(h)
                x = self.drop(self.act(self.fc1(x)))
                x = self.act(self.fc2(x))
                return self.fc3(x).squeeze(-1)

        head = _SPQMlpHead()
        missing, unexpected = head.load_state_dict(stash["head"], strict=False)
        assert not missing and not unexpected, (missing, unexpected)
        head = head.to(torch.bfloat16).to(get_torch_device().current_device())
        self._sp_q_mlp_head = head
        self._sp_q_mlp_head_stash = None
        return head

    def _sp_q_mlp_hidden_last(self, ids_1d):
        """One exact-length forward through the FSDP actor; returns the final-norm
        hidden state at the last position. A forward hook on model.norm captures the
        tensor instead of output_hidden_states=True, which would materialise every
        layer (36 x 53K x 2560 bf16 ~ 10 GB/row); logits_to_keep=1 keeps the lm_head
        slice at [1,1,V] instead of the 16 GB full-vocab logits."""
        eng = self.actor.engine
        dev = get_torch_device().current_device()
        ids = ids_1d.to(dev, non_blocking=True).unsqueeze(0).long()
        attn = torch.ones_like(ids)
        pos = torch.arange(ids.shape[1], device=ids.device, dtype=torch.long).unsqueeze(0)
        # verl's patched forward (forward_with_torch_backend) REQUIRES return_dict=True;
        # nb_hidden_only=True skips all lm_head work AND returns the final hidden. Using the
        # RETURNED tensor (not a norm hook) is load-bearing: FSDP1 arms its pre-backward
        # hooks on the forward's returned tensors, so a loss that flows only from a captured
        # intermediate leaves the state machine IDLE and every post-backward hook asserts.
        out = eng.module(input_ids=ids, attention_mask=attn, position_ids=pos,
                         use_cache=False, return_dict=True, nb_hidden_only=True)
        hs = out.hidden_states[-1]
        return hs[0, -1]                      # [hidden]

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="purple", role="sp_q_mlp_score")
    def sp_q_mlp_score(self, data: TensorDict) -> TensorDict:
        """Forward-only Q values for a chunk of wave contexts. Rows arrive padded
        ([n, L] + lengths); each is forwarded at exact length, micro-batch 1. Every
        rank runs the same row count by construction (driver pads to the dp divisor),
        so the per-layer FSDP all-gathers stay in lockstep."""
        assert "actor" in self.role
        eng = self.actor.engine
        head = self._sp_q_mlp_ensure_head(
            str(tu.get_non_tensor_data(data, key="sp_q_head_path", default="")))
        ids_all = data["sp_q_ctx_ids"]
        lens = data["sp_q_ctx_len"].tolist()
        head.eval()
        vals = []
        with eng.train_mode():                # pages FSDP params in; grads unused
            with torch.no_grad():
                for i in range(ids_all.shape[0]):
                    hid = self._sp_q_mlp_hidden_last(ids_all[i, : int(lens[i])])
                    vals.append(torch.sigmoid(head(hid).float()).item())
        out = torch.tensor(vals, dtype=torch.float32)
        return tu.get_tensordict(tensor_dict={"sp_q_values": out})

    def _sp_q_mlp_step_inline(self, data: TensorDict) -> dict:
        """One Q-only step, MLP flavour: row-weighted BCE on sigmoid(head(h_last)),
        backward through head AND backbone (tied update, applied in place), backbone
        lr from the ladder, head lr fixed. Mirrors _sp_q_optimizer_step_inline's
        contract: same optimizer, same clip, same non-finite skip, same return shape."""
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_

        eng = self.actor.engine
        lr = float(tu.get_non_tensor_data(data, key="sp_q_lr", default=2e-6))
        head_lr = float(tu.get_non_tensor_data(data, key="sp_q_head_lr", default=1e-4))
        clip = float(tu.get_non_tensor_data(data, key="sp_q_clip", default=0.2))
        denom = float(tu.get_non_tensor_data(data, key="sp_q_denom", default=1))
        head = self._sp_q_mlp_ensure_head(
            str(tu.get_non_tensor_data(data, key="sp_q_head_path", default="")))

        opt = self._sp_q_ensure_optimizer(lr)
        head_ids = {id(p) for g in opt.param_groups for p in g["params"]
                    if g.get("sp_q_group") == "head"}
        if not head_ids:
            opt.add_param_group({"params": list(head.parameters()), "lr": head_lr,
                                 "sp_q_group": "head"})
        for g in opt.param_groups:
            g["lr"] = head_lr if g.get("sp_q_group") == "head" else lr
        self._sp_q_optim_state_to(get_torch_device().current_device())

        ids_all = data["sp_q_ctx_ids"]
        lens = data["sp_q_ctx_len"].tolist()
        zs = data["sp_q_bce_z"]
        ws = data["sp_q_row_w"]
        is_ref = data["sp_q_is_ref"].to(bool)
        dp_size = float(eng.get_data_parallel_size())

        eng.optimizer_zero_grad()
        head.train()                          # dropout active, matching the SFT
        loss_sum = 0.0
        bce_ref = bce_noref = 0.0
        n_ref = n_noref = 0
        # NO nested eng.train_mode() here: this runs INSIDE the wrapped optimizer_step of an
        # active update, where the engine context is already established -- the generative
        # inline step at this exact site does not nest it either. contextlib.nullcontext
        # keeps the block shape.
        import contextlib
        with contextlib.nullcontext():
            for i in range(ids_all.shape[0]):
                w = float(ws[i])
                hid = self._sp_q_mlp_hidden_last(ids_all[i, : int(lens[i])])
                logit = head(hid).float()
                z = zs[i].to(logit.device).float()
                bce = torch.nn.functional.binary_cross_entropy_with_logits(logit, z)
                # pad rows (w=0) still forward+backward so every rank issues the same
                # FSDP collectives; their zero weight keeps them out of the loss.
                loss = bce * w / denom * dp_size
                loss.backward()
                loss_sum += float(loss.item())
                if w > 0:
                    if bool(is_ref[i]):
                        bce_ref += float(bce.item()) * w; n_ref += 1
                    else:
                        bce_noref += float(bce.item()) * w; n_noref += 1

            # the head is replicated, not FSDP-wrapped: reduce its grads by hand or
            # the ranks silently diverge from the first step
            for p in head.parameters():
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.SUM)
                p.grad.div_(dp_size)

            if isinstance(eng.module, FSDP):
                grad_norm = eng.module.clip_grad_norm_(clip)
            elif isinstance(eng.module, FSDPModule):
                grad_norm = fsdp2_clip_grad_norm_(eng.module.parameters(), max_norm=clip)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(eng.module.parameters(), max_norm=clip)
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor()
            head_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=clip)

            _loss_bad = 1.0 if not math.isfinite(loss_sum) else 0.0
            loss_non_finite = self._sp_q_allreduce_sum(_loss_bad) > 0.0
            grad_non_finite = (not torch.isfinite(grad_norm).all().item()
                               or not torch.isfinite(head_norm).all().item())
            skipped = bool(loss_non_finite or grad_non_finite)
            if skipped:
                print(f"[sp_q-mlp] non-finite (loss={loss_sum}, grad={grad_norm}, "
                      f"head={head_norm}); skipping O_Q step", flush=True)
            else:
                opt.step()
            for p in eng.module.parameters():
                p.grad = None
            for p in head.parameters():
                p.grad = None
            self._sp_q_optim_state_to("cpu")
        head.eval()

        return {
            "loss_sum": loss_sum,
            # ce4_* key names kept on purpose: the driver maps them to q/loss_ref|noref
            # with qm.get(..). Same names -> the dashboard panels stay live, now showing
            # the row-weighted BCE sums (the analogous per-record quantity).
            "msum": {"sp_q_ce4_sum_ref": bce_ref, "sp_q_ce4_sum_noref": bce_noref,
                     "sp_q_rows_ref": n_ref, "sp_q_rows_noref": n_noref,
                     "sp_q_head_grad_norm": float(head_norm)},
            "grad_norm": float(grad_norm) if not grad_non_finite else float("nan"),
            "skipped": bool(skipped),
            "loss_non_finite": bool(loss_non_finite),
            "grad_non_finite": bool(grad_non_finite),
        }

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="purple", role="sp_q_train_capture")
    def sp_q_train_capture(self, data: TensorDict) -> TensorDict:
        """Q phase step 1: snapshot theta_0, run the teacher-forced CE
        forward/backward over the Q batch, clip at sp_q_clip, advance O_Q, CAPTURE
        Delta_Q (CPU fp32) and RESTORE theta_0. Returns reduced metrics."""
        assert "actor" in self.role
        from verl.workers.utils.losses import sp_q_ce_loss

        eng = self.actor.engine
        lr = float(tu.get_non_tensor_data(data, key="sp_q_lr", default=2e-6))
        clip = float(tu.get_non_tensor_data(data, key="sp_q_clip", default=0.2))

        # engineering defaults, mirroring TrainingWorker.train_batch (max_token_len
        # comes from the driver: Q prompts can reach C_Q ~53.3K > the PPO 51.2K cap)
        defaults = dict(
            use_remove_padding=self.actor.model_config.get("use_remove_padding", False),
            use_dynamic_bsz=eng.engine_config.use_dynamic_bsz,
            max_token_len_per_gpu=int(
                tu.get_non_tensor_data(data, key="sp_q_max_token_len", default=53360)
            ),
            micro_batch_size_per_gpu=eng.engine_config.micro_batch_size_per_gpu,
            use_fused_kernels=eng.engine_config.use_fused_kernels,
            temperature=1.0,  # true CE: never the rollout sampling temperature
        )
        for key, val in defaults.items():
            if key not in data.keys():
                tu.assign_non_tensor(data, **{key: val})

        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_

        metrics = {}
        with eng.train_mode():
            self._sp_q_ensure_optimizer(lr)
            for g in self._sp_q_optimizer.param_groups:
                if g.get("sp_q_group") != "head":
                    g["lr"] = lr
            self._sp_q_optim_state_to(get_torch_device().current_device())

            # snapshot theta_0 (local shards, CPU) -- held across the PPO update
            self._sp_q_theta0 = [lt.detach().to("cpu", copy=True) for _, lt in self._sp_q_local_tensors()]

            eng.optimizer_zero_grad()
            output = eng.forward_backward_batch(data, partial(sp_q_ce_loss, config=None), forward_only=False)

            # clip at the Q phase's own norm (0.2), same branches as engine.optimizer_step
            if isinstance(eng.module, FSDP):
                grad_norm = eng.module.clip_grad_norm_(clip)
            elif isinstance(eng.module, FSDPModule):
                grad_norm = fsdp2_clip_grad_norm_(eng.module.parameters(), max_norm=clip)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(eng.module.parameters(), max_norm=clip)
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor()

            skipped = not torch.isfinite(grad_norm).all().item()
            if skipped:
                print(f"[sp_q] non-finite Q grad norm {grad_norm}; skipping O_Q step (Delta_Q = 0)",
                      flush=True)
            else:
                self._sp_q_optimizer.step()
            # clear Q grads so nothing leaks into the PPO backward
            for p in eng.module.parameters():
                p.grad = None

            # capture Delta_Q and restore theta_0
            delta_sumsq = 0.0
            deltas = []
            for (th0, (_, lt)) in zip(self._sp_q_theta0, self._sp_q_local_tensors()):
                d = lt.detach().to("cpu", copy=True).float() - th0.float()
                deltas.append(d)
                delta_sumsq += float(d.double().pow(2).sum().item())
                lt.copy_(th0.to(lt.device, non_blocking=True))
            self._sp_q_delta = deltas
            self._sp_q_optim_state_to("cpu")

            loss_sum = float(sum(output["loss"]))
            msum = {k: float(sum(v)) for k, v in output["metrics"].items()
                    if k.startswith("sp_q_")}

        # reduce across ranks (identical values returned everywhere)
        delta_q_sumsq = self._sp_q_allreduce_sum(delta_sumsq)
        dp_size = eng.get_data_parallel_size()
        reduced = {k: self._sp_q_allreduce_sum(v) * self._sp_q_replication() for k, v in msum.items()}
        # loss: each rank's sum over microbatches is dp*(its slice of the global mean);
        # averaging over ranks (SUM/dp) recovers the global L_Q.
        loss_global = self._sp_q_allreduce_sum(loss_sum) * self._sp_q_replication() / dp_size
        out = {
            "sp_q/loss": loss_global,
            "sp_q/grad_norm": float(grad_norm) if not skipped else float("nan"),
            "sp_q/skipped": int(skipped),
            "sp_q/delta_q_norm": math.sqrt(max(delta_q_sumsq, 0.0)),
            **{f"sp_q/{k}": v for k, v in reduced.items()},
        }
        return tu.get_tensordict(tensor_dict={}, non_tensor_dict={"metrics": out}) \
            if self.actor.engine.is_mp_src_rank_with_outputs() else None

    # ------------------------------------------------------------------
    # sp_q INTERLEAVED surface (tied Q; the update order
    # PPO(M1) -> Q-only -> PPO(M2)). Same tied weights and same O_Q as the
    # fused surface above, but the Q optimizer step is taken INSIDE the actor
    # minibatch loop and LEFT APPLIED -- no theta_0 snapshot, no captured
    # Delta_Q buffer, no movement cap. rho is measured after the fact and
    # only prices the NEXT step's LR (the ladder lives on the driver).
    # Gated by SP_Q_INTERLEAVE=1; inert unless the driver arms it.
    # ------------------------------------------------------------------

    def _sp_q_optimizer_step_inline(self, data: TensorDict) -> dict:
        """One Q-only optimizer step on the SHARED weights, applied in place.

        The forward/backward/clip/step sequence is the fused path's, minus the theta_0
        snapshot + restore: here the update stays. Returns local (not yet all-reduced)
        sums so the caller can reduce once."""
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
        from verl.workers.utils.losses import sp_q_ce_loss

        if str(tu.get_non_tensor_data(data, key="sp_q_param", default="gen")) == "mlp":
            # MLP-Q: same interleave slot, BCE-on-head loss instead of generative CE.
            return self._sp_q_mlp_step_inline(data)

        eng = self.actor.engine
        lr = float(tu.get_non_tensor_data(data, key="sp_q_lr", default=2e-6))
        clip = float(tu.get_non_tensor_data(data, key="sp_q_clip", default=0.2))

        self._sp_q_ensure_optimizer(lr)
        for g in self._sp_q_optimizer.param_groups:
            if g.get("sp_q_group") != "head":
                g["lr"] = lr
        self._sp_q_optim_state_to(get_torch_device().current_device())

        # PPO's minibatch grads were consumed by the optimizer step that just ran; clear
        # them so the Q backward accumulates alone, and clear the Q grads afterwards so
        # nothing leaks into PPO minibatch 2.
        eng.optimizer_zero_grad()
        output = eng.forward_backward_batch(data, partial(sp_q_ce_loss, config=None),
                                            forward_only=False)
        loss_sum = float(sum(output["loss"]))
        if isinstance(eng.module, FSDP):
            grad_norm = eng.module.clip_grad_norm_(clip)
        elif isinstance(eng.module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(eng.module.parameters(), max_norm=clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(eng.module.parameters(), max_norm=clip)
        if hasattr(grad_norm, "full_tensor"):
            grad_norm = grad_norm.full_tensor()

        # The ladder contract: a non-finite Q LOSS OR GRADIENT skips that Q step and
        # triggers the same halving. A non-finite loss does not always produce a non-finite
        # grad_norm (a NaN can be confined to a masked/zero-weighted row, or the clip can be
        # computed over parameters whose grads stayed finite), so checking grad_norm alone
        # would let the contract be violated silently. Both are checked here.
        #
        # The loss check must be a COLLECTIVE decision: loss_sum is this rank's slice, so one
        # rank seeing NaN while others do not would have ranks disagree about whether to call
        # optimizer.step() -- a hang or a torn update. All-reduce the flag first. grad_norm is
        # already a global (FSDP clip reduces across ranks), so it needs no extra reduction.
        _loss_bad_local = 1.0 if not math.isfinite(loss_sum) else 0.0
        loss_non_finite = self._sp_q_allreduce_sum(_loss_bad_local) > 0.0
        grad_non_finite = not torch.isfinite(grad_norm).all().item()
        skipped = bool(loss_non_finite or grad_non_finite)
        if skipped:
            _why = ("loss" if loss_non_finite and not grad_non_finite
                    else "grad" if grad_non_finite and not loss_non_finite else "loss+grad")
            print(f"[sp_q] non-finite Q {_why} (loss_sum={loss_sum}, grad_norm={grad_norm}); "
                  "skipping O_Q step (interleaved: no weight movement, ladder halves)",
                  flush=True)
        else:
            self._sp_q_optimizer.step()
        for p in eng.module.parameters():
            p.grad = None
        self._sp_q_optim_state_to("cpu")

        return {
            "loss_sum": loss_sum,
            "msum": {k: float(sum(v)) for k, v in output["metrics"].items()
                     if k.startswith("sp_q_")},
            "grad_norm": float(grad_norm) if not grad_non_finite else float("nan"),
            "skipped": bool(skipped),
            "loss_non_finite": bool(loss_non_finite),
            "grad_non_finite": bool(grad_non_finite),
        }

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="purple", role="sp_q_arm_interleave")
    def sp_q_arm_interleave(self, data: TensorDict) -> TensorDict:
        """Stash this rank's Q batch and wrap engine.optimizer_step so the Q-only step
        fires INSIDE the upcoming update_actor, right after PPO minibatch
        `sp_q_after_minibatch` (default 1). Per-minibatch PPO displacement norms are
        tracked by the same wrapper, re-baselined across the Q step so delta_PPO2
        excludes the Q movement."""
        assert "actor" in self.role
        eng = self.actor.engine

        defaults = dict(
            use_remove_padding=self.actor.model_config.get("use_remove_padding", False),
            use_dynamic_bsz=eng.engine_config.use_dynamic_bsz,
            max_token_len_per_gpu=int(
                tu.get_non_tensor_data(data, key="sp_q_max_token_len", default=53360)
            ),
            micro_batch_size_per_gpu=eng.engine_config.micro_batch_size_per_gpu,
            use_fused_kernels=eng.engine_config.use_fused_kernels,
            temperature=1.0,  # true CE: never the rollout sampling temperature
        )
        for key, val in defaults.items():
            if key not in data.keys():
                tu.assign_non_tensor(data, **{key: val})

        self._sp_q_il_data = data
        self._sp_q_il_after = int(tu.get_non_tensor_data(data, key="sp_q_after_minibatch",
                                                         default=1))
        self._sp_q_il_result = None
        self._sp_q_il_delta_sumsq = 0.0
        self._sp_q_step_sumsqs = []
        # displacement baseline = the weights entering the actor update (there is no
        # theta_0 snapshot in this surface, so prev IS the only held copy).
        self._sp_q_prev = [lt.detach().to("cpu", copy=True).float()
                           for _, lt in self._sp_q_local_tensors()]
        orig_step = eng.optimizer_step

        def wrapped_step():
            gn = orig_step()
            ssq = 0.0
            for prev, (_, lt) in zip(self._sp_q_prev, self._sp_q_local_tensors()):
                cur = lt.detach().to("cpu", copy=True).float()
                ssq += float((cur - prev).double().pow(2).sum().item())
                prev.copy_(cur)
            self._sp_q_step_sumsqs.append(ssq)
            if len(self._sp_q_step_sumsqs) == self._sp_q_il_after and self._sp_q_il_result is None:
                self._sp_q_il_result = self._sp_q_optimizer_step_inline(self._sp_q_il_data)
                dssq = 0.0
                for prev, (_, lt) in zip(self._sp_q_prev, self._sp_q_local_tensors()):
                    cur = lt.detach().to("cpu", copy=True).float()
                    dssq += float((cur - prev).double().pow(2).sum().item())
                    prev.copy_(cur)   # re-baseline: delta_PPO2 must exclude Delta_Q
                self._sp_q_il_delta_sumsq = dssq
            return gn

        self._sp_q_orig_optimizer_step = orig_step
        eng.optimizer_step = wrapped_step
        return tu.get_tensordict(tensor_dict={}, non_tensor_dict={"metrics": {}}) \
            if eng.is_mp_src_rank_with_outputs() else None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def sp_q_finish_interleave(self):
        """After update_actor: unpatch, all-reduce the Q metrics and the per-minibatch
        displacement norms, and report whether the Q step actually fired.

        If the actor ran FEWER minibatches than `sp_q_after_minibatch`, the Q step has
        not fired yet -- it is taken here (after the full PPO update) rather than
        silently dropping the drawn sample, and `late` is reported so the driver can
        alert."""
        assert "actor" in self.role
        eng = self.actor.engine
        if getattr(self, "_sp_q_orig_optimizer_step", None) is not None:
            eng.optimizer_step = self._sp_q_orig_optimizer_step
            self._sp_q_orig_optimizer_step = None

        late = False
        if getattr(self, "_sp_q_il_result", None) is None and getattr(self, "_sp_q_il_data", None) is not None:
            late = True
            with eng.train_mode():
                self._sp_q_il_result = self._sp_q_optimizer_step_inline(self._sp_q_il_data)
            dssq = 0.0
            for prev, (_, lt) in zip(self._sp_q_prev, self._sp_q_local_tensors()):
                cur = lt.detach().to("cpu", copy=True).float()
                dssq += float((cur - prev).double().pow(2).sum().item())
                prev.copy_(cur)
            self._sp_q_il_delta_sumsq = dssq

        res = getattr(self, "_sp_q_il_result", None) or {
            "loss_sum": 0.0, "msum": {}, "grad_norm": float("nan"), "skipped": True,
            "loss_non_finite": False, "grad_non_finite": False}
        step_sumsqs = list(getattr(self, "_sp_q_step_sumsqs", []) or [])
        delta_q_sumsq = float(getattr(self, "_sp_q_il_delta_sumsq", 0.0))
        self._sp_q_il_data = None
        self._sp_q_il_result = None
        self._sp_q_prev = None
        self._sp_q_step_sumsqs = []

        dp_size = eng.get_data_parallel_size()
        repl = self._sp_q_replication()
        steps = [math.sqrt(max(self._sp_q_allreduce_sum(s), 0.0)) for s in step_sumsqs]
        delta_q = math.sqrt(max(self._sp_q_allreduce_sum(delta_q_sumsq), 0.0))
        reduced = {k: self._sp_q_allreduce_sum(v) * repl for k, v in res["msum"].items()}
        loss_global = self._sp_q_allreduce_sum(res["loss_sum"]) * repl / dp_size
        return {
            "delta_q": delta_q,
            "delta_ppo_steps": steps,
            "delta_ppo_sum": math.fsum(steps),
            "q_non_finite": bool(res["skipped"]),
            # split out so the ladder's halving reason is auditable (the ladder halves on
            # either, but "which one" is the difference between a bad batch and a bad update)
            "q_loss_non_finite": bool(res.get("loss_non_finite", False)),
            "q_grad_non_finite": bool(res.get("grad_non_finite", False)),
            "late": late,
            "metrics": {
                "sp_q/loss": loss_global,
                "sp_q/grad_norm": res["grad_norm"],
                "sp_q/skipped": int(res["skipped"]),
                **{f"sp_q/{k}": v for k, v in reduced.items()},
            },
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def sp_q_arm_ppo_norms(self):
        """Wrap engine.optimizer_step for the upcoming update_actor so each applied
        PPO step's displacement norm (delta_PPO1, delta_PPO2) is captured."""
        assert "actor" in self.role
        eng = self.actor.engine
        assert getattr(self, "_sp_q_theta0", None) is not None, "arm called before capture"
        self._sp_q_step_sumsqs = []
        self._sp_q_prev = [t.float().clone() for t in self._sp_q_theta0]
        orig_step = eng.optimizer_step

        def wrapped_step():
            gn = orig_step()
            ssq = 0.0
            for prev, (_, lt) in zip(self._sp_q_prev, self._sp_q_local_tensors()):
                cur = lt.detach().to("cpu", copy=True).float()
                ssq += float((cur - prev).double().pow(2).sum().item())
                prev.copy_(cur)
            self._sp_q_step_sumsqs.append(ssq)
            return gn

        self._sp_q_orig_optimizer_step = orig_step
        eng.optimizer_step = wrapped_step
        return {"armed": True}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def sp_q_finish_update(self):
        """After update_actor: unpatch, compute delta_PPO_net = ||theta_now - theta_0||
        plus the per-step norms, all-reduced. theta_0 stays held until sp_q_apply."""
        assert "actor" in self.role
        eng = self.actor.engine
        if getattr(self, "_sp_q_orig_optimizer_step", None) is not None:
            eng.optimizer_step = self._sp_q_orig_optimizer_step
            self._sp_q_orig_optimizer_step = None
        net_sumsq = 0.0
        for th0, (_, lt) in zip(self._sp_q_theta0, self._sp_q_local_tensors()):
            net_sumsq += float(
                (lt.detach().to("cpu", copy=True).float() - th0.float()).double().pow(2).sum().item()
            )
        step_sumsqs = list(getattr(self, "_sp_q_step_sumsqs", []) or [])
        self._sp_q_prev = None
        self._sp_q_step_sumsqs = []
        net = self._sp_q_allreduce_sum(net_sumsq)
        steps = [self._sp_q_allreduce_sum(s) for s in step_sumsqs]
        return {
            "delta_ppo_net": math.sqrt(max(net, 0.0)),
            "delta_ppo_steps": [math.sqrt(max(s, 0.0)) for s in steps],
        }

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def sp_q_apply(self, scale: float):
        """Apply s_t * Delta_Q to the (possibly CPU-offloaded) weights and free the
        held buffers. theta_final = theta_0 + Delta_PPO_net + s_t * Delta_Q."""
        assert "actor" in self.role
        applied_sumsq = 0.0
        deltas = getattr(self, "_sp_q_delta", None)
        assert deltas is not None, "sp_q_apply called without a captured Delta_Q"
        if scale > 0.0:
            for d, (_, lt) in zip(deltas, self._sp_q_local_tensors()):
                step = (d * float(scale)).to(dtype=lt.dtype, device=lt.device)
                lt.add_(step)
                applied_sumsq += float((d.double() * float(scale)).pow(2).sum().item())
        self._sp_q_delta = None
        self._sp_q_theta0 = None
        applied = self._sp_q_allreduce_sum(applied_sumsq)
        return {"applied_delta_q_norm": math.sqrt(max(applied, 0.0))}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def sp_q_save_optim(self, ckpt_dir: str):
        """Persist O_Q's per-rank shard state (every post-branch checkpoint must contain
        O_Q; same-topology resume). Creates O_Q first if a skipped Q phase left it
        unbuilt, and writes via tmp+rename so a torn write can never look complete. The
        driver verifies the full shard set before publishing the checkpoint."""
        assert "actor" in self.role
        opt = self._sp_q_ensure_optimizer()
        d = os.path.join(ckpt_dir, "sp_q_optim")
        os.makedirs(d, exist_ok=True)
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        path = os.path.join(d, f"rank_{rank}.pt")
        tmp = path + ".tmp"
        torch.save({"state": opt.state_dict(), "world_size": world_size}, tmp)
        os.replace(tmp, path)
        head = getattr(self, "_sp_q_mlp_head", None)
        if head is not None and rank == 0:
            hp = os.path.join(ckpt_dir, "qmlp_head.pt")
            htmp = hp + ".tmp"
            torch.save({"head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                        "head_mid": head.fc1.out_features,
                        "dropout": float(head.drop.p)}, htmp)
            os.replace(htmp, hp)
        return {"saved": True, "rank": rank, "world_size": world_size, "path": path}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def sp_q_load_optim(self, ckpt_dir: str, strict: bool = True):
        """Stash O_Q shard state for lazy load at the next Q phase. strict=True ->
        missing shards are FATAL; strict=False allows the branch-point
        first attach (fresh O_Q by design)."""
        assert "actor" in self.role
        rank = torch.distributed.get_rank()
        d = os.path.join(ckpt_dir, "sp_q_optim")
        path = os.path.join(d, f"rank_{rank}.pt")
        hp = os.path.join(ckpt_dir, "qmlp_head.pt")
        if os.path.exists(hp):
            self._sp_q_mlp_head_stash = torch.load(hp, map_location="cpu", weights_only=False)
            print(f"[sp_q-mlp] head state stashed from {hp}", flush=True)
        if not os.path.exists(path):
            if strict:
                raise FileNotFoundError(
                    f"sp_q: checkpoint {ckpt_dir} has no O_Q shard for rank {rank}; a resume "
                    "past the branch point must carry O_Q -- refusing to "
                    "silently reinitialize"
                )
            print(f"[sp_q] no O_Q shard at {path}; fresh O_Q (branch-point attach)", flush=True)
            return {"loaded": False}
        # The driver writes _complete.json only after every rank's shard landed; without it
        # this shard set is a torn save (or a partially pruned one) and must not be trusted.
        marker = os.path.join(d, "_complete.json")
        if strict and not os.path.exists(marker):
            raise FileNotFoundError(
                f"sp_q: {d} has shards but no _complete.json completeness marker; the save "
                "was interrupted or the set was partially pruned -- refusing to resume from "
                "an incomplete O_Q. Resume the previous checkpoint, or (if you have verified "
                f"all {torch.distributed.get_world_size()} rank_*.pt are present and intact) "
                "write the marker by hand."
            )
        blob = torch.load(path, map_location="cpu", weights_only=False)
        ws = int(blob.get("world_size", -1))
        assert ws == torch.distributed.get_world_size(), (
            f"sp_q: O_Q shards were saved at world_size {ws} but this run has "
            f"{torch.distributed.get_world_size()}; same-topology resume required"
        )
        if getattr(self, "_sp_q_optimizer", None) is not None:
            self._sp_q_optimizer.load_state_dict(blob["state"])
            self._sp_q_optim_state_to("cpu")
        else:
            self._sp_q_optim_pending_state = blob["state"]
        return {"loaded": True}

    # ==================================================================
    # sp_q SEPARATE-Q surface. theta_Q is a
    # persistent CPU buffer of the actor module's LOCAL FSDP shards,
    # time-shared into the single FSDP module (+ the single vLLM engine
    # for the wave) via in-place data swaps. O_Q evolves theta_Q on its
    # OWN trajectory (3 x minibatch-64 per RL step, left applied) and never
    # touches the policy. No theta_0 capture/restore, no rho movement cap.
    # Gated by SP_Q_SEPARATE=1 on the driver side.
    # ==================================================================

    def _sp_q_snapshot_module(self):
        """CPU copy of the module's current local shards (bf16/dtype preserved)."""
        return [lt.detach().to("cpu", copy=True) for _, lt in self._sp_q_local_tensors()]

    def _sp_q_load_module(self, buf):
        """In-place copy a snapshot back into the module's local shards. In-place data
        copies preserve param identity, so O_Q's per-param state stays valid across swaps."""
        for b, (_, lt) in zip(buf, self._sp_q_local_tensors()):
            lt.copy_(b.to(lt.device, non_blocking=True))

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def sp_q_init_separate(self, q_hf_dir: str, q_optim_dir: str = "", lr: float = 1e-5):
        """Init theta_Q from the SFT endpoint (full hf weights, sharded into the FSDP
        module via set_model_state_dict) and O_Q from the SFT optimizer shards
        (continuity). The module is left holding the POLICY weights unchanged."""
        assert "actor" in self.role
        from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
        from transformers import AutoModelForCausalLM

        from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu

        eng = self.actor.engine
        rank = torch.distributed.get_rank()
        ws = torch.distributed.get_world_size()
        allow_fresh = os.environ.get("SP_Q_ALLOW_FRESH_OQ", "0") in ("1", "true", "True")
        theta_pi = self._sp_q_snapshot_module()          # hold policy
        load_fsdp_model_to_gpu(eng.module)
        full = AutoModelForCausalLM.from_pretrained(
            q_hf_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        ).state_dict()
        info = set_model_state_dict(
            eng.module, full,
            options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True, strict=False),
        )
        del full
        # Fail-CLOSED model load: UNEXPECTED keys mean the SFT checkpoint does
        # not match the actor module -> abort rather than run a wrong theta_Q. A few missing
        # buffers (rotary inv_freq, etc.) are benign; a large fraction means the load did not
        # take. `info` may be None on some torch builds — then skip (best effort).
        unexpected = list(getattr(info, "unexpected_keys", []) or []) if info is not None else []
        missing = list(getattr(info, "missing_keys", []) or []) if info is not None else []
        if unexpected:
            raise RuntimeError(
                f"sp_q-sep: SFT->Q model load has {len(unexpected)} UNEXPECTED keys "
                f"(e.g. {unexpected[:5]}) — SFT checkpoint {q_hf_dir} is incompatible with the "
                "actor module; aborting."
            )
        if len(missing) > 32:
            raise RuntimeError(
                f"sp_q-sep: SFT->Q model load left {len(missing)} params UNSET "
                f"(e.g. {missing[:5]}) — theta_Q would be largely random; aborting."
            )
        self._sp_q_theta_q = self._sp_q_snapshot_module()  # capture theta_Q
        self._sp_q_load_module(theta_pi)                   # restore policy into the module
        # O_Q with continuity from the SFT optimizer. FAIL-CLOSED: any load failure aborts
        # rather than silently cold-starting Adam (which would change the optimization). SP_Q_ALLOW_FRESH_OQ=1 explicitly opts into a fresh O_Q.
        opt = self._sp_q_ensure_optimizer(lr)
        for g in opt.param_groups:
            g["lr"] = lr
        if q_optim_dir:
            from torch.distributed.checkpoint.state_dict import set_optimizer_state_dict
            self._sp_q_load_module(self._sp_q_theta_q)     # map optimizer state onto the Q params
            osd_path = os.path.join(q_optim_dir, f"optim_world_size_{ws}_rank_{rank}.pt")
            if not os.path.exists(osd_path):
                if allow_fresh:
                    print(f"[sp_q-sep] SFT O_Q shard missing {osd_path}; O_Q FRESH (SP_Q_ALLOW_FRESH_OQ=1)", flush=True)
                else:
                    raise FileNotFoundError(
                        f"sp_q-sep: SFT O_Q shard missing {osd_path}; the critic optimizer state "
                        "O_Q must be carried over. Set SP_Q_ALLOW_FRESH_OQ=1 to intentionally cold-start."
                    )
            else:
                osd = torch.load(osd_path, map_location="cpu", weights_only=False)
                loaded = False
                # producer format is not statically known: try the distributed API, then a
                # raw optimizer.state_dict() ({"state","param_groups"} or {"state":...}).
                try:
                    set_optimizer_state_dict(eng.module, opt, osd,
                                             options=StateDictOptions(full_state_dict=False, strict=False))
                    loaded = True
                except Exception as e1:  # noqa: BLE001
                    try:
                        raw = osd.get("state", osd) if isinstance(osd, dict) and "param_groups" not in osd and "state" in osd and isinstance(osd["state"], dict) and "param_groups" in osd.get("state", {}) else osd
                        opt.load_state_dict(raw)
                        loaded = True
                    except Exception as e2:  # noqa: BLE001
                        if allow_fresh:
                            print(f"[sp_q-sep] SFT O_Q load failed (dcp: {e1}; raw: {e2}); O_Q FRESH "
                                  "(SP_Q_ALLOW_FRESH_OQ=1)", flush=True)
                        else:
                            raise RuntimeError(
                                f"sp_q-sep: SFT O_Q load failed (dcp: {e1}; raw: {e2}); the critic optimizer "
                                "state O_Q must be carried over — aborting rather than silently cold-starting. "
                                "Set SP_Q_ALLOW_FRESH_OQ=1 to override."
                            ) from e2
                if loaded:
                    print("[sp_q-sep] O_Q initialized from SFT optimizer (continuity)", flush=True)
            self._sp_q_load_module(theta_pi)
            self._sp_q_optim_state_to("cpu")
        if eng._is_offload_param:
            offload_fsdp_model_to_cpu(eng.module)
        self._sp_q_separate = True
        return {"initialized": True, "has_theta_q": self._sp_q_theta_q is not None}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def sp_q_swap_in_q(self):
        """Load theta_Q into the module (policy stashed) so the driver can update_weights
        it into vLLM for the Q wave. Idempotent-guarded by _sp_q_swapped."""
        assert "actor" in self.role
        if getattr(self, "_sp_q_swapped", False):
            return {"swapped": True}
        self._sp_q_policy_stash = self._sp_q_snapshot_module()
        self._sp_q_load_module(self._sp_q_theta_q)
        self._sp_q_swapped = True
        return {"swapped": True}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def sp_q_swap_out_q(self):
        """Restore the policy weights into the module after the Q wave."""
        assert "actor" in self.role
        if not getattr(self, "_sp_q_swapped", False):
            return {"swapped": False}
        self._sp_q_load_module(self._sp_q_policy_stash)
        self._sp_q_policy_stash = None
        self._sp_q_swapped = False
        return {"swapped": False}

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="purple", role="sp_q_train_separate")
    def sp_q_train_separate(self, data: TensorDict) -> TensorDict:
        """Separate-Q update: swap theta_Q into the module, run n_steps x minibatch-64
        teacher-forced CE optimizer steps on O_Q (left applied), save the updated weights
        back to theta_Q, restore the policy. No theta_0 capture, no rho cap."""
        assert "actor" in self.role
        from verl.workers.utils.losses import sp_q_ce_loss

        eng = self.actor.engine
        lr = float(tu.get_non_tensor_data(data, key="sp_q_lr", default=1e-5))
        clip = float(tu.get_non_tensor_data(data, key="sp_q_clip", default=1.0))
        n_steps = int(tu.get_non_tensor_data(data, key="sp_q_grad_steps", default=3))
        mb = int(tu.get_non_tensor_data(data, key="sp_q_mb", default=64))
        defaults = dict(
            use_remove_padding=self.actor.model_config.get("use_remove_padding", False),
            use_dynamic_bsz=eng.engine_config.use_dynamic_bsz,
            max_token_len_per_gpu=int(tu.get_non_tensor_data(data, key="sp_q_max_token_len", default=53360)),
            micro_batch_size_per_gpu=eng.engine_config.micro_batch_size_per_gpu,
            use_fused_kernels=eng.engine_config.use_fused_kernels,
            temperature=1.0,
        )
        for k, v in defaults.items():
            if k not in data.keys():
                tu.assign_non_tensor(data, **{k: v})

        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_

        from verl.utils.device import get_device_name

        dp_size = eng.get_data_parallel_size()
        bs = data.batch_size[0] if hasattr(data, "batch_size") else data.shape[0]  # this rank's rows
        mb_per_dp = max(1, mb // dp_size)                 # per-rank rows per global minibatch of `mb`
        n_real = int(tu.get_non_tensor_data(data, key="sp_q_n_real", default=bs * dp_size))
        dev = get_device_name()
        losses, gnorms, n_done = [], [], 0
        with eng.train_mode():
            # Peak-memory invariant: model on GPU for the Q forward, but O_PPO
            # OFFLOADED to CPU so only O_Q is GPU-resident during the Q phase. eng.to()
            # executes irrespective of offload config (manual control).
            eng.to(dev, model=True, optimizer=False, grad=False)     # ensure model on GPU
            ppo_was_resident = not eng._is_offload_optimizer
            eng.to("cpu", model=False, optimizer=True, grad=False)   # O_PPO -> CPU (grad=False: eng asserts model=True whenever grad=True; PPO grads are unused/cleared in the Q phase)
            policy_stash = self._sp_q_snapshot_module()
            self._sp_q_load_module(self._sp_q_theta_q)               # train on theta_Q
            self._sp_q_ensure_optimizer(lr)
            for g in self._sp_q_optimizer.param_groups:
                if g.get("sp_q_group") != "head":
                    g["lr"] = lr
            self._sp_q_optim_state_to(get_torch_device().current_device())  # O_Q -> GPU
            for s in range(n_steps):
                lo = s * mb_per_dp
                if lo >= bs or (s * mb) >= n_real:       # no wrap, no resampling: fewer steps if <n_steps*mb records
                    break
                hi = min(lo + mb_per_dp, bs)
                # index_select (gather), NOT data[lo:hi]: the packed Q batch stores input_ids as a
                # jagged NestedTensor (pad_mode=no_padding), and slicing a NestedTensor on dim 0 is
                # unsupported by torch ("slice(): not supported for NestedTensor on dim=0"). This
                # mirrors verl's make_iterator, which chunks packed batches via index_select_tensor_dict.
                mbatch = tu.index_select_tensor_dict(data, torch.arange(lo, hi))
                # dynamic-bsz needs the per-sequence token counts all-gathered across DP so
                # forward_backward_batch can balance micro-batches (mirrors train_mini_batch:284-295).
                if "input_ids" in mbatch.keys():
                    _gtn = mbatch["input_ids"].offsets().diff().tolist()
                    _gathered = [None] * torch.distributed.get_world_size(eng.get_data_parallel_group())
                    torch.distributed.all_gather_object(_gathered, _gtn, eng.get_data_parallel_group())
                    tu.assign_non_tensor(mbatch, global_token_num=NonTensorData([x for xs in _gathered for x in xs]))
                # EXACT real-record count in THIS minibatch via the per-row real flag,
                # all-reduced — robust to seq-balancing scattering zero-weight padding across
                # slices. Falls back to slice size only if the flag is absent.
                try:
                    real_local = float(mbatch["sp_q_real"].sum().item())
                except Exception:  # noqa: BLE001
                    real_local = float(hi - lo)
                denom_g = int(round(self._sp_q_allreduce_sum(real_local) * self._sp_q_replication()))
                if denom_g <= 0:                            # all-padding minibatch -> stop
                    break
                tu.assign_non_tensor(mbatch, sp_q_denom=int(denom_g))
                eng.optimizer_zero_grad()
                output = eng.forward_backward_batch(mbatch, partial(sp_q_ce_loss, config=None), forward_only=False)
                if isinstance(eng.module, FSDP):
                    gn = eng.module.clip_grad_norm_(clip)
                elif isinstance(eng.module, FSDPModule):
                    gn = fsdp2_clip_grad_norm_(eng.module.parameters(), max_norm=clip)
                else:
                    gn = torch.nn.utils.clip_grad_norm_(eng.module.parameters(), max_norm=clip)
                if hasattr(gn, "full_tensor"):
                    gn = gn.full_tensor()
                if torch.isfinite(gn).all().item():
                    self._sp_q_optimizer.step()
                    n_done += 1
                for p in eng.module.parameters():
                    p.grad = None
                losses.append(float(sum(output["loss"])))
                gnorms.append(float(gn))
            self._sp_q_theta_q = self._sp_q_snapshot_module()  # persist updated theta_Q
            self._sp_q_load_module(policy_stash)               # restore policy
            self._sp_q_optim_state_to("cpu")                   # O_Q -> CPU
            if ppo_was_resident:
                eng.to(dev, model=False, optimizer=True, grad=False)  # restore O_PPO (grad=False: same assertion; grads were cleared)
        loss_mean = self._sp_q_allreduce_sum(sum(losses) / max(len(losses), 1)) * self._sp_q_replication() / dp_size
        out = {"sp_q/loss": loss_mean, "sp_q/grad_steps": n_done,
               "sp_q/grad_norm": (gnorms[-1] if gnorms else float("nan"))}
        return tu.get_tensordict(tensor_dict={}, non_tensor_dict={"metrics": out}) \
            if eng.is_mp_src_rank_with_outputs() else None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def sp_q_save_q_model(self, ckpt_dir: str):
        """Persist theta_Q as per-rank shards (q_model/rank_R.pt); the driver writes the
        _complete.json marker after all ranks land."""
        assert "actor" in self.role
        d = os.path.join(ckpt_dir, "q_model")
        os.makedirs(d, exist_ok=True)
        rank = torch.distributed.get_rank()
        path = os.path.join(d, f"rank_{rank}.pt")
        tmp = path + ".tmp"
        ws = torch.distributed.get_world_size()
        torch.save({"theta_q": self._sp_q_theta_q, "world_size": ws}, tmp)
        os.replace(tmp, path)
        return {"saved": True, "rank": rank, "world_size": ws}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def sp_q_load_q_model(self, ckpt_dir: str):
        """Restore theta_Q from q_model/rank_R.pt on resume. Fail-CLOSED: shards without a
        `_complete.json` marker (torn/partially-pruned save) or a world-size mismatch are
        FATAL, never silently ignored."""
        assert "actor" in self.role
        rank = torch.distributed.get_rank()
        d = os.path.join(ckpt_dir, "q_model")
        path = os.path.join(d, f"rank_{rank}.pt")
        if not os.path.exists(path):
            print(f"[sp_q-sep] no theta_Q shard at {path}; expecting sp_q_init_separate", flush=True)
            return {"loaded": False}
        marker = os.path.join(d, "_complete.json")
        if not os.path.exists(marker):
            raise FileNotFoundError(
                f"sp_q-sep: {d} has shards but no _complete.json completeness marker; the "
                "theta_Q save was interrupted or partially pruned -- refusing to resume from "
                "an incomplete Q model."
            )
        blob = torch.load(path, map_location="cpu", weights_only=False)
        ws = int(blob.get("world_size", -1))
        assert ws == torch.distributed.get_world_size(), (
            f"sp_q-sep: theta_Q shards saved at world_size {ws} but this run has "
            f"{torch.distributed.get_world_size()}; same-topology resume required"
        )
        self._sp_q_theta_q = blob["theta_q"]
        self._sp_q_separate = True
        return {"loaded": True}

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_teacher_unembed(self, W: torch.Tensor) -> None:
        """Store teacher lm_head.weight for Nitrobrew loss. TP ranks take their vocab shard."""
        assert "actor" in self.role, "set_teacher_unembed is only valid for actor workers"
        strategy = self.config.actor.get("strategy", "fsdp")
        if strategy == "megatron":
            from megatron.core.parallel_state import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )

            tp_rank = get_tensor_model_parallel_rank()
            tp_size = get_tensor_model_parallel_world_size()
            V = W.shape[0]
            shard = V // tp_size
            W_local = W[tp_rank * shard : (tp_rank + 1) * shard]
        else:
            W_local = W
        self._teacher_unembed = W_local.to(get_torch_device().current_device())

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        assert "actor" in self.role, "load_checkpoint only support actor role"
        self.actor.load_checkpoint(local_path, hdfs_path, del_local_after_load)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        assert "actor" in self.role, "save_checkpoint only support actor role"
        self.actor.save_checkpoint(local_path, hdfs_path, global_step, max_ckpt_to_keep)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps: int = None, mode: str = "auto"):
        """Update weights from trainer to rollout.

        1. For sync training with colocated trainer and rollout, update rollout directly from model engine.
           - before update_weights: rollout should be in sleep mode.
           - after update_weights: rollout should be in wake_up mode.
        2. For async training with disaggregated trainer and rollout, send_weights only by checkpoint engine.

        LoRA handling: when model.lora.merge=True (peft_merge), LoRA is merged into
        base weights before sync. The engine returns full HF-keyed params with
        peft_config=None, so the rollout receives a standard weight update.

        Args:
            global_steps: Current global training step count, passed to rollout for logging/tracking.
            mode: Weight update strategy. Supported values:
                - ``"auto"``: Automatically resolve to the backend configured in
                  ``config.rollout.checkpoint_engine.backend`` (default).
                - ``"naive"``: Direct in-process weight sync between colocated trainer
                  and rollout. Used for synchronous training where both share the same
                  process. Rollout must be in sleep mode before this call.
                - Any other value: Delegates to
                  :meth:`checkpoint_engine.send_weights` for asynchronous weight
                  transfer via checkpoint engine, suitable for disaggregated
                  trainer/rollout deployments.
        """

        # Resolve mode: "auto" falls back to config, explicit values take precedence
        effective_mode = mode if mode != "auto" else self.config.rollout.checkpoint_engine.backend

        # 0. send_weights only for async training with disaggregated trainer and rollout
        if effective_mode != "naive":
            per_tensor_param, _ = self.actor.engine.get_per_tensor_param()
            await self.checkpoint_engine.send_weights(per_tensor_param, global_steps=global_steps)
            return

        set_expandable_segments(False)
        log_gpu_memory_usage("Before resume weights", logger=logger)

        # 1. resume rollout memory (weights were released during sleep)
        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["weights"])
        log_gpu_memory_usage("After resume weights", logger=logger)

        # 2. determine if we need a base weight sync (adapter path only)
        per_tensor_param, peft_config = self.actor.engine.get_per_tensor_param(
            layered_summon=self.layered_summon, base_sync_done=True
        )

        do_lora_base_sync = False
        if not self.peft_merge and peft_config is not None:
            self.rollout.sleep_level = 1
            do_lora_base_sync = not self.base_sync_done

        # 3. sync weights: For SGLang, we need base first (when needed), then adapter/merged
        if do_lora_base_sync:
            per_tensor_param_base, peft_config = self.actor.engine.get_per_tensor_param(
                layered_summon=self.layered_summon, base_sync_done=False
            )
            await self.rollout.update_weights(
                per_tensor_param_base, peft_config=peft_config, base_sync_done=False, global_steps=global_steps
            )

        await self.rollout.update_weights(
            per_tensor_param, peft_config=peft_config, base_sync_done=True, global_steps=global_steps
        )

        log_gpu_memory_usage("After update_weights", logger=logger)

        # 3. offload model to cpu
        if self.actor.engine.is_param_offload_enabled:
            self.actor.engine.to("cpu", model=True, optimizer=False, grad=False)
        aggressive_empty_cache(force_sync=True)

        # 4. resume kv_cache
        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["kv_cache"])
        log_gpu_memory_usage("After resume kv_cache", logger=logger)

        self.base_sync_done = True
        set_expandable_segments(True)

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def execute_checkpoint_engine(self, method: str, *args, **kwargs):
        """Execute checkpoint engine method.

        Args:
            method (str): Checkpoint engine method name.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        """
        return getattr(self.checkpoint_engine, method)(*args, **kwargs)
