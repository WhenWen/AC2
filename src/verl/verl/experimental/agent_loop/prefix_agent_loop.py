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
"""sp_prefix_agent: single-turn agent loop with a replay-prefix riding in the RESPONSE.

Replay-prefix off-policy training.
The row's ``sp_prefix_token_ids`` (a 30-80% cut of a stored judged-correct trajectory,
sampled by the SPReplayPrefixDataset via verl.trainer.ppo.sp_replay) is fed to the
engine as generation context, and returned RESPONSE-side with response_mask=0:

  * engine sees      prompt_ids = chat_template(raw_prompt) + prefix_ids
  * output           response_ids  = prefix_ids + generated_ids
                     response_mask = [0]*len(prefix) + [1]*len(generated)
                     response_logprobs = [0.0]*len(prefix) + generated_logprobs

Why response-side and not prompt-side prefill: the agent-loop postprocess pads/checks
prompts against the fixed rollout.prompt_length (2,048) and LEFT-TRUNCATES overlong
prompts, while partially-masked responses are natively supported (the multi-turn
tool-call pattern). PPO loss, entropy aggregation, and rollout_probs_diff are all
masked by response_mask, so the prefix is never trained on -- exactly like prompt
tokens. The reward path decodes the full valid response span (attention-mask based),
so the judge grades prefix+continuation and the correct-only length penalty sees the
FULL trajectory length.

The per-request completion budget is ``response_length - len(prefix_ids)``, so
prefix+continuation always fits the fixed response tensor width.

With an empty prefix this reduces exactly to SingleTurnAgentLoop, so it is safe as the
agent for every train row (statement rows carry ``sp_prefix_token_ids=[]``).
Multimodal inputs are not supported (this project is text-only); fail loud if present.
"""
import logging
import os
import asyncio
from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.experimental.agent_loop.utils import sp_route_key
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# --- SP_PREFIX_PILOT ------------------------------------------------------------------
# One gate per route key, per AgentLoopWorker process. Contiguous chunking keeps a uid's
# siblings inside one worker (except at chunk boundaries), so a process-local gate covers
# effectively all groups without a cross-actor round trip on the hot path.
_PILOT_GATES: dict[str, "asyncio.Event"] = {}
_PILOT_MAX_GATES = 4096


def _pilot_enabled() -> bool:
    return os.environ.get("SP_PREFIX_PILOT", "0") in ("1", "true", "True")


def _pilot_min_prefix() -> int:
    """Below this the prefill saved is not worth adding a round trip to the critical path."""
    return int(os.environ.get("SP_PREFIX_PILOT_MIN", "1024"))


async def _warm_shared_prefix(server_manager, route_key, engine_prompt_ids,
                              sampling_params, mm_processor_kwargs):
    """First caller for this prefix prefills it; the rest wait, then hit the cache."""
    ev = _PILOT_GATES.get(route_key)
    if ev is None:
        # Safe without a lock: asyncio is single-threaded and there is no await between the
        # check and the insert, so exactly one coroutine per key becomes the pilot.
        ev = _PILOT_GATES[route_key] = asyncio.Event()
        if len(_PILOT_GATES) > _PILOT_MAX_GATES:      # keys are content hashes; bound growth
            for k in list(_PILOT_GATES)[: _PILOT_MAX_GATES // 2]:
                if _PILOT_GATES[k].is_set():
                    _PILOT_GATES.pop(k, None)
        try:
            warm = dict(sampling_params)
            warm["max_tokens"] = 1        # returns as soon as the prefill is cached
            warm.pop("n", None)
            await server_manager.generate(
                request_id=route_key, prompt_ids=engine_prompt_ids,
                sampling_params=warm, image_data=None, video_data=None, audio_data=None,
                mm_processor_kwargs=mm_processor_kwargs, cost=1.0)
        except Exception as e:
            # A failed warm-up must not fail the rollout: the siblings simply proceed
            # unshared, exactly as they did before this flag existed.
            logger.warning("sp_prefix_pilot: warm-up failed (%s: %s); "
                           "siblings will prefill independently", type(e).__name__, e)
        finally:
            ev.set()                       # release the waiters even if the pilot failed
        return
    try:
        await asyncio.wait_for(ev.wait(), timeout=float(
            os.environ.get("SP_PREFIX_PILOT_TIMEOUT", "1800")))
    except asyncio.TimeoutError:
        logger.warning("sp_prefix_pilot: waited out the pilot for %s; proceeding unshared",
                       route_key[:16])


# --- SP_PREFIX_GROUP_SLOTS -----------------------------------------------------------
# Affinity puts a group's siblings on one replica; the pilot makes them SHARE the prefix KV.
# Neither controls WHEN they are resident together, and that is what caps grouped cascade.
# Measured in a profiling run: only 45% of a replica's resident requests were in a cascade
# group and ~2.5 groups were co-resident where 6 fit (96 seats / n=16). The cause is
# interleaving, not sibling divergence: every one of ~3.3k requests is launched as a
# concurrent task, so vLLM admits 96 in whatever order asyncio produced -- a slice across
# ~24 groups, most of them below MIN_GROUP_SIZE.
#
# This gate admits GROUPS, not requests: a bounded number of route keys may be in flight per
# worker, and a key's siblings piggyback on the permit their leader took. The resident set
# then holds whole groups.
#
# Why this does not cost occupancy, which is the obvious objection. Holding a permit until a
# group's LAST sibling finishes idles the seats its early finishers vacate. Measured on one
# training step's rollouts (192 groups of 16): siblings are nearly synchronous
# -- within-group max/min length is 1.2x median, 2.3x p90 -- so the time-averaged occupancy of
# a whole-group permit is 0.91 median. But 0.91 occupancy costs 1/0.91 = 1.10x wall while
# doubling coverage buys only ~1.08x, i.e. strict admission is a NET LOSS. What makes it win
# is the filler: the same rollouts are exactly {1: 192, 16: 192} -- the 192 inflow rows are
# singletons that can never benefit from cascade. They are deliberately NOT gated, so they
# flow into whatever seats group decay frees (24 per replica against a ~9-seat gap),
# restoring occupancy to ~100% while the gated groups stay whole.
#
# 0 disables the gate: byte-identical to the pre-existing behaviour.
_GROUP_SEM: "asyncio.Semaphore | None" = None


class _GroupPermit:
    """Refcount + readiness for one route key's permit."""
    __slots__ = ("n", "ready", "held")

    def __init__(self):
        self.n = 0
        self.ready = asyncio.Event()
        self.held = False


_GROUP_PERMITS: dict[str, _GroupPermit] = {}


def _group_slots() -> int:
    """Route keys allowed in flight per AgentLoopWorker. 0 = off."""
    return int(os.environ.get("SP_PREFIX_GROUP_SLOTS", "0"))


class _group_gate:
    """Async CM: hold one permit per route key, shared by that key's siblings.

    The leader (first sibling to arrive for a key) takes the permit and opens the gate; the
    rest wait on it rather than acquiring, so a group costs ONE permit however many siblings
    it has. The permit is returned when the last sibling leaves.
    """

    def __init__(self, route_key: str, gated: bool):
        self._key = route_key
        self._gated = gated
        self._permit = None

    async def __aenter__(self):
        global _GROUP_SEM
        if not self._gated:
            return self
        slots = _group_slots()
        if slots <= 0:
            self._gated = False
            return self
        if _GROUP_SEM is None:
            _GROUP_SEM = asyncio.Semaphore(slots)
        st = _GROUP_PERMITS.get(self._key)
        # No await between the get and the insert, so exactly one coroutine leads a key.
        leader = st is None
        if leader:
            st = _GROUP_PERMITS[self._key] = _GroupPermit()
        st.n += 1
        self._permit = st
        if leader:
            try:
                await _GROUP_SEM.acquire()
                st.held = True
            finally:
                # Release the siblings even if the acquire was cancelled: without this they
                # would wait forever on a gate no one will ever open.
                st.ready.set()
        else:
            await st.ready.wait()
        return self

    async def __aexit__(self, *exc):
        st = self._permit
        if st is None:
            return False
        st.n -= 1
        if st.n <= 0:
            if st.held and _GROUP_SEM is not None:
                _GROUP_SEM.release()
            _GROUP_PERMITS.pop(self._key, None)
        return False


@register("sp_prefix_agent")
class PrefixAgentLoop(AgentLoopBase):
    """Single turn chat completion with an optional replay-prefix (response-side, mask 0)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        prefix_ids = [int(t) for t in (kwargs.get("sp_prefix_token_ids") or [])]

        multi_modal_data = await self.process_multi_modal_info(messages)
        if multi_modal_data.get("images") or multi_modal_data.get("videos") or multi_modal_data.get("audios"):
            raise ValueError("sp_prefix_agent is text-only; got multimodal inputs")

        prompt_ids = await self.apply_chat_template(messages)

        request_sampling_params = dict(sampling_params)
        if prefix_ids:
            if len(prefix_ids) >= self.response_length:
                raise ValueError(
                    f"sp_prefix_agent: prefix length {len(prefix_ids)} >= response_length "
                    f"{self.response_length}; the seed builder must cap stored trajectories"
                )
            # Completion-budget rule: prefix + continuation <= response_length.
            request_sampling_params["max_tokens"] = self.response_length - len(prefix_ids)
        # Q-readiness short budget: a per-row NEW-token cap stamped by
        # the dataset for ready non-audit rows (m_i = min(g, response_length - prefix)).
        # Absent/<=0 -> no-op, so non-Q runs are byte-identical.
        _q_cap = int(kwargs.get("sp_q_max_new_tokens") or 0)
        if _q_cap > 0:
            request_sampling_params["max_tokens"] = min(
                int(request_sampling_params.get("max_tokens", self.response_length)), _q_cap
            )

        # Work estimate for the SP_ROLLOUT_PRIORITY global router: the completion-token
        # budget (statement -> response_length; prefix -> response_length - prefix; Q-ready
        # -> the g cap). Ignored when the router is count-based.
        _cost = float(request_sampling_params.get("max_tokens", self.response_length))

        # SP_PREFIX_AFFINITY: the rollout.n samples of this (problem, cut) share these exact
        # ids, so a content-derived sticky key lands them all on one replica and only the
        # first pays the prefill. Off -> fresh uuid, i.e. upstream least-in-flight routing.
        engine_prompt_ids = prompt_ids + prefix_ids
        # A row with no usable replay prefix can never join a cascade group (derive_groups
        # needs a shared prefix of at least SP_GC_MIN_PREFIX), so it is exactly the population
        # that keeps `identity` False for everyone else sharing its replica. With
        # SP_PREFIX_UNGROUPED_LANES>0 these rows are herded onto their own replicas; the rest
        # of the fleet then sees pure grouped batches. Same threshold the pilot uses, so the
        # two mechanisms cannot disagree about what "has a prefix" means.
        _ungrouped = len(prefix_ids) < _pilot_min_prefix()
        route_key = sp_route_key(engine_prompt_ids, ungrouped=_ungrouped)

        # SP_PREFIX_PILOT: affinity alone does NOT make the siblings share anything. vLLM
        # looks up the prefix cache once, when a request moves waiting->running
        # (get_computed_blocks is gated on num_computed_tokens == 0), and commits blocks only
        # afterwards in update_from_output. The agent loop fires all n siblings through one
        # asyncio.gather, so they are admitted together, all miss, and EACH allocates its own
        # copy of the prefix KV -- n prefills and n copies resident, ref_cnt 1 per block.
        #
        # So one sibling runs first as a pilot with max_tokens=1 (returns as soon as the
        # prefill lands) and the rest wait on it. They then hit the cache: one prefill, one
        # physical copy at ref_cnt=n. That refcount is also the precondition for grouped
        # cascade -- without it there is no shared block for the kernel to exploit.
        if _pilot_enabled() and len(prefix_ids) >= _pilot_min_prefix():
            await _warm_shared_prefix(
                self.server_manager, route_key, engine_prompt_ids,
                dict(request_sampling_params), self._get_mm_processor_kwargs(None))

        # Gate only the requests that can actually form a cascade group: a long replay prefix
        # is exactly what makes siblings share KV. Prefix-less rows (the inflow lane) stay
        # ungated on purpose -- they are the filler that keeps occupancy at ~100% while the
        # gated groups are held whole.
        _gated = len(prefix_ids) >= _pilot_min_prefix()
        metrics = {}
        async with _group_gate(route_key, _gated):
            with simple_timer("generate_sequences", metrics):
                output: TokenOutput = await self.server_manager.generate(
                    request_id=route_key,
                    prompt_ids=engine_prompt_ids,
                    sampling_params=request_sampling_params,
                    image_data=None,
                    video_data=None,
                    audio_data=None,
                    mm_processor_kwargs=self._get_mm_processor_kwargs(None),
                    cost=_cost,
                )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1

        generated_ids = list(output.token_ids)
        response_ids = (prefix_ids + generated_ids)[: self.response_length]
        response_mask = ([0] * len(prefix_ids) + [1] * len(generated_ids))[: self.response_length]
        response_logprobs = None
        if output.log_probs:
            # positional alignment with response_ids: prefix tokens carry a 0.0 placeholder
            # (they are response_mask=0, so nothing downstream reads these positions).
            response_logprobs = ([0.0] * len(prefix_ids) + list(output.log_probs))[: self.response_length]

        agent_output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            routed_experts=None,
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=None,
            num_turns=2,
            metrics=metrics,
            extra_fields=output.extra_fields,
        )
        # keeping the schema consistent with single_turn/tool agent loops
        agent_output.extra_fields.update({"turn_scores": [], "tool_rewards": []})
        return agent_output
