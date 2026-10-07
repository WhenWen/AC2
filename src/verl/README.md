
# VERL Vendor 

This is a vendored version of VERL release 0.8.0.

This is commit `19c6af5de10de2b5272c83c0e82aa715c8c621f3` on branch `releases/v0.8.0`. 

For original VERL READE, see `./README_ORIGINAL.md`.

## Modifications by the AC2 authors

This copy of verl is distributed under the Apache License, Version 2.0 (see `LICENSE` and
`Notice.txt` in this directory). The AC2 authors modified and added the files below to implement
AC2 (replay buffer, critic, readiness, action-chunk rollouts and related launch features). Each
modified file carries a header line stating that it was changed. All other files are identical
to upstream release 0.8.0.

Modified files:

- `tests/utils/debug/test_metrics.py`
- `verl/experimental/agent_loop/agent_loop.py`
- `verl/experimental/agent_loop/single_turn_agent_loop.py`
- `verl/experimental/agent_loop/utils.py`
- `verl/experimental/reward_loop/reward_loop.py`
- `verl/experimental/reward_loop/reward_manager/naive.py`
- `verl/experimental/reward_loop/reward_model.py`
- `verl/experimental/reward_loop/router/naive_router.py`
- `verl/models/transformers/dense_common.py`
- `verl/trainer/distillation/fsdp/losses.py`
- `verl/trainer/distillation/losses.py`
- `verl/trainer/main_ppo.py`
- `verl/trainer/ppo/core_algos.py`
- `verl/trainer/ppo/ray_trainer.py`
- `verl/trainer/ppo/utils.py`
- `verl/utils/checkpoint/fsdp_checkpoint_manager.py`
- `verl/utils/debug/metrics.py`
- `verl/utils/ray_utils.py`
- `verl/utils/tensordict_utils.py`
- `verl/utils/tracking.py`
- `verl/workers/config/distillation.py`
- `verl/workers/config/rollout.py`
- `verl/workers/engine/fsdp/transformer_impl.py`
- `verl/workers/engine_workers.py`
- `verl/workers/reward_manager/naive.py`
- `verl/workers/rollout/llm_server.py`
- `verl/workers/rollout/vllm_rollout/utils.py`
- `verl/workers/rollout/vllm_rollout/vllm_async_server.py`
- `verl/workers/utils/losses.py`
- `verl/workers/utils/padding.py`

Added files:

- `verl/experimental/agent_loop/prefix_agent_loop.py`
- `verl/experimental/agent_loop/sp_q_agent_loop.py`
- `verl/experimental/teacher_loop/nitrobrew_teacher.py`
- `verl/trainer/distillation/fsdp/nitrobrew_loss.py`
- `verl/trainer/distillation/megatron/nitrobrew_loss.py`
- `verl/trainer/ppo/aec.py`
- `verl/trainer/ppo/difficulty.py`
- `verl/trainer/ppo/sp_dp_pad.py`
- `verl/trainer/ppo/sp_dyn_group.py`
- `verl/trainer/ppo/sp_q_readiness.py`
- `verl/trainer/ppo/sp_replay.py`
