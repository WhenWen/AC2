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

import asyncio
import logging
import os

from verl.single_controller.ray.base import RayResourcePool, split_resource_pool
from verl.workers.config import HFModelConfig, RewardModelConfig
from verl.workers.rollout.replica import get_rollout_replica_class

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class RewardModelManager:
    """Reward model manager."""

    def __init__(
        self,
        config: RewardModelConfig,
        resource_pool: RayResourcePool = None,
    ):
        """
        Initialize the reward model manager.

        Args:
            config (RewardModelConfig): Reward model configuration.
            resource_pool (RayResourcePool, optional): Resource pool. Defaults to None.
        """
        self.config = config
        self.resource_pool = resource_pool
        self._initialize_llm_servers()
        self._initialize_router()
        assert self.config.rollout.skip_tokenizer_init is False, "Reward model should not skip tokenizer init."
        if self.config.rollout.free_cache_engine:
            self.sleep()

    def _initialize_llm_servers(self):
        rollout_config = self.config.rollout
        rollout_world_size = (
            rollout_config.tensor_model_parallel_size
            * rollout_config.data_parallel_size
            * rollout_config.pipeline_model_parallel_size
        )
        world_size = (
            self.resource_pool.world_size
            if self.resource_pool  # colocate mode
            else self.config.n_gpus_per_node * self.config.nnodes  # standalone mode
        )
        num_replicas = world_size // rollout_world_size
        assert num_replicas > 0, (
            f"Not enough GPUs to run the reward model. "
            f"world_size ({world_size}) < rollout_world_size ({rollout_world_size}). "
            f"Check your resource pool or standalone config (n_gpus_per_node, nnodes)."
        )

        rollout_replica_class = get_rollout_replica_class(rollout_config.name)
        model_config = HFModelConfig(path=self.config.model_path)
        self.tokenizer = model_config.get_processor()
        self.rollout_replicas = [
            rollout_replica_class(
                replica_rank=replica_rank,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=self.config.n_gpus_per_node,
                is_reward_model=True,
            )
            for replica_rank in range(num_replicas)
        ]
        import os as _os
        _stagger = _os.environ.get("VERL_STAGGER_ENGINE_INIT", "0") == "1"
        # Wave-based stagger, mirroring llm_server.py: serialize engine init WITHIN a node,
        # parallel ACROSS nodes. Wave k inits the k-th engine of every node simultaneously.
        # Serializing ALL replicas across nodes is not needed as a deadlock backstop: the init
        # hangs seen on this path had other root causes (prover TP=2 comm creation, fixed by
        # SP_ROLLOUT_TP=1; an intra-replica flashinfer JIT desync, fixed by
        # fuse_allreduce_rms=False), and full cross-node serialization did NOT help the latter,
        # so it only cost wall-clock (~9 min/replica at DS4-Flash scale = ~27 min of idle per
        # attach at 4 nodes). With 1 replica per node (judge TP = gpus per node) this is a
        # single all-parallel wave.
        _rpn = max(1, self.config.n_gpus_per_node // rollout_world_size)
        if self.resource_pool:
            split_resource_pools = split_resource_pool(self.resource_pool, split_size=rollout_world_size)
            assert len(split_resource_pools) == len(self.rollout_replicas)
            _pairs = list(zip(self.rollout_replicas, split_resource_pools, strict=True))
            if _stagger and _rpn > 1:
                print(
                    f"[stagger-init][reward_model] {len(_pairs)} colocated replica(s), "
                    f"{_rpn} per node -> {_rpn} waves",
                    flush=True,
                )
                for _wave in range(_rpn):
                    _group = [_p for _i, _p in enumerate(_pairs) if _i % _rpn == _wave]
                    if _group:
                        self._run_all([_server.init_colocated(_rp) for _server, _rp in _group])
            else:
                if _stagger:
                    print(
                        f"[stagger-init][reward_model] 1 replica/node ({len(_pairs)} node(s)) "
                        f"-> single parallel wave",
                        flush=True,
                    )
                self._run_all([_server.init_colocated(_rp) for _server, _rp in _pairs])
        else:
            if _stagger and _rpn > 1:
                print(
                    f"[stagger-init][reward_model] {len(self.rollout_replicas)} standalone replica(s), "
                    f"{_rpn} per node -> {_rpn} waves",
                    flush=True,
                )
                for _wave in range(_rpn):
                    _group = [_s for _i, _s in enumerate(self.rollout_replicas) if _i % _rpn == _wave]
                    if _group:
                        self._run_all([_server.init_standalone() for _server in _group])
            else:
                if _stagger:
                    print(
                        f"[stagger-init][reward_model] 1 replica/node ({len(self.rollout_replicas)} node(s)) "
                        f"-> single parallel wave",
                        flush=True,
                    )
                self._run_all([server.init_standalone() for server in self.rollout_replicas])
        self.server_handles = [server._server_handle for server in self.rollout_replicas]
        self.server_addresses = [server._server_address for server in self.rollout_replicas]

    def _initialize_router(self):
        worker_urls = [f"http://{server_address}" for server_address in self.server_addresses]

        # TODO (dyy): sglang router is not ready yet.
        # if self.config.rollout.name == "sglang":
        #     from .router.inner_sglang_router import launch_router_process
        # else:
        #     from .router.naive_router import launch_router_process

        from .router.naive_router import launch_router_process

        # LOCAL PATCH: high max_connections so the all-at-once reward fan-out
        # (bs * rollout.n concurrent judge calls) is not throttled by the default 1024 and
        # NaiveRouter's limit_per_host = max_connections // 4 = 256 cap -> HTTP 503 / retry
        # exhaustion / hangs (val passes at low rollout count, training crashes at high).
        self.router_address, _ = launch_router_process(worker_urls=worker_urls, max_connections=65536)

    def get_router_address(self):
        return self.router_address

    def wake_up(self):
        """Wake up all rollout replica instances."""
        self._run_all([replica.wake_up() for replica in self.rollout_replicas])

    def sleep(self):
        """Sleep all rollout replica instances."""
        self._run_all([replica.sleep() for replica in self.rollout_replicas])

    def _run_all(self, tasks: list[asyncio.Task]):
        async def run_all():
            await asyncio.gather(*tasks)

        asyncio.run(run_all())
