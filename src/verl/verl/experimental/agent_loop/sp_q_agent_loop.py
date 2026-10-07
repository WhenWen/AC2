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
"""sp_q_agent: one raw-token-ids generation call for the Q wave.

The driver builds the FULL Q-prompt token ids (templated statement + attempt +
turn close + Q instruction + forced '<think>\\n\\n</think>\\n\\nQ value: ' prefill --
see sp_q_readiness.build_q_context_ids) and ships them per row as
``sp_q_ctx_token_ids``. This loop sends them verbatim to the engine with

  * ``max_tokens`` = ``sp_q_max_tokens`` (4: <=3 value tokens + end-of-turn),
  * GREEDY decoding (temperature 0, top_p 1, top_k -1): the prediction that would
    actually be consumed,
  * NO constrained/guided decoding (the value is parsed from free-form output).

Only the GENERATED ids are returned response-side (the context can exceed the
response tensor width; returning it would truncate the value tokens). These rows are
parse-only -- they are never merged into the training batch, so response_mask
semantics are irrelevant; mask is all-ones over the generated tokens.

The context shares its leading tokens with the just-generated rollout (same prompt,
same attempt), so the engine's prefix cache makes these ~4-token calls cheap.
"""
import logging
import os
from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.utils import sp_route_key
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("sp_q_agent")
class SPQAgentLoop(AgentLoopBase):
    """Raw-token-ids Q-value generation (greedy, max_tokens=4, free decoding)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.response_length = self.rollout_config.response_length

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        ctx_ids = [int(t) for t in kwargs["sp_q_ctx_token_ids"]]
        max_tokens = int(kwargs.get("sp_q_max_tokens", 4))
        assert ctx_ids, "sp_q_agent: empty sp_q_ctx_token_ids"

        request_sampling_params = dict(sampling_params)
        request_sampling_params.update(
            {
                "max_tokens": max_tokens,
                "temperature": 0,   # greedy: the prediction that would actually be consumed
                "top_p": 1.0,
                "top_k": -1,
            }
        )

        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=sp_route_key(ctx_ids),
                prompt_ids=ctx_ids,
                sampling_params=request_sampling_params,
                image_data=None,
                video_data=None,
                audio_data=None,
                mm_processor_kwargs=self._get_mm_processor_kwargs(None),
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1

        generated_ids = list(output.token_ids)[: self.response_length]
        # prompt_ids returned as a 1-token stub: the postprocess pads prompts to
        # rollout.prompt_length and the real context (up to ~53K tokens) must NOT be
        # forced through that 2,048-wide tensor. Only `responses` is read back.
        stub = ctx_ids[-1:]

        agent_output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=stub,
            response_ids=generated_ids,
            response_mask=[1] * len(generated_ids),
            response_logprobs=None,
            routed_experts=None,
            multi_modal_data={},
            mm_processor_kwargs=None,
            num_turns=2,
            metrics=metrics,
            extra_fields=output.extra_fields,
        )
        agent_output.extra_fields.update({"turn_scores": [], "tool_rewards": []})
        return agent_output
