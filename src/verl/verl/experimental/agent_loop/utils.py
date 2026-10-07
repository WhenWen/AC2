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

import hashlib
import os
from array import array
from typing import Any
from uuid import uuid4


def sp_prefix_affinity_enabled() -> bool:
    """SP_PREFIX_AFFINITY (self-play): route rollouts that share an engine prompt to the
    SAME vLLM replica so the second..n-th reuse the first one's prefill via the engine's
    prefix cache (and hold the shared prefix KV once, by refcount, instead of n times).

    Default off -> byte-identical random routing."""
    return os.environ.get("SP_PREFIX_AFFINITY", "0") in ("1", "true", "True")


def sp_ungrouped_lanes() -> int:
    """SP_PREFIX_UNGROUPED_LANES: how many replicas the cascade-INELIGIBLE rows are herded onto.

    0 (default) = off, byte-identical to the previous behaviour.

    Why this exists. The grouped-cascade kernel's fast path is gated on ``identity``, which
    requires that EVERY request in the resident batch belongs to a group
    (``bool(active.all()) and order_list == list(range(num_reqs))``). Production batches are
    ~12-22% prefix-less -- the inflow rows plus any replay row cut near zero -- so identity is
    False on every batch, and the prefix pass loses its prebuilt fastcall arg list (rebuilt per
    layer, 36x per step), gains a ``query[perm]`` gather and loses its preallocated output.

    The fix is spatial, not temporal. The temporal alternative (SP_PREFIX_GROUP_SLOTS) throttles
    WHEN grouped rows are submitted and measured strictly worse -- bulk coverage fell 0.455 ->
    0.280 -- because it capped the grouped lane while prefix-less rows flowed freely and took the
    seats. Sending the two kinds of row to DIFFERENT replicas needs no admission control at all:
    the ineligible rows collapse onto a few sticky keys, and every other replica then sees a
    pure grouped batch.

    Sizing. Split by WORK, not by row count: measured, prefix-less rows are ~12% of requests but
    generate ~14.5k tokens each against the grouped rows' ~6.7k, so they carry ~23% of the
    decode work. Giving them one lane out of eight (12.5% of capacity) would make them the new
    tail; two lanes (25%) matches their share. Hence 2, not 1.
    """
    try:
        return max(0, int(os.environ.get("SP_PREFIX_UNGROUPED_LANES", "0")))
    except ValueError:
        return 0


def sp_route_key(engine_prompt_ids: list[int], ungrouped: bool = False) -> str:
    """Sticky-session key for ``GlobalRequestLoadBalancer.acquire_server``.

    With affinity ON the key is a digest of the exact token ids handed to the engine, so
    the ``rollout.n`` samples of one (problem, replay cut) -- which carry byte-identical
    ``prompt_ids + prefix_ids`` -- collapse onto one replica: the first acquire picks the
    least-loaded server, the rest hit the LRU and pin to it. With affinity OFF the key is
    a fresh uuid that can never hit the LRU, which is the upstream behaviour (every
    request falls through to least-in-flight selection).

    Safe to share across concurrent requests: this key is ONLY the router's sticky
    session id. ``LLMServerClient.generate`` mints a fresh ``uuid4().hex`` for the
    engine-facing request id, so two rollouts with the same key never collide in vLLM.
    """
    if not sp_prefix_affinity_enabled():
        return uuid4().hex
    digest = hashlib.blake2b(array("q", engine_prompt_ids).tobytes(), digest_size=16).hexdigest()
    lanes = sp_ungrouped_lanes()
    if ungrouped and lanes > 0:
        # A FIXED key, so the balancer's LRU pins every ineligible row to the same handful of
        # replicas instead of least-loading them across all of them. Which replicas is left to
        # the first acquire (least-loaded at that moment); only the COUNT is controlled here.
        # The digest still selects among the lanes so the ineligible rows spread evenly over
        # them rather than piling onto lane 0.
        return f"sp-ungrouped-lane-{int(digest[:8], 16) % lanes}"
    return digest


def resolve_config_path(config_path: str) -> str:
    """Resolve agent loop configuration file path.

    In multi-node Ray training, relative paths may not resolve correctly
    because the working directory on remote nodes can differ from the driver node.
    This function resolves relative paths by checking multiple locations in order:
    1. If already absolute, return as-is
    2. Try current working directory
    3. Try relative to verl package installation (project root)

    Args:
        config_path: Configuration file path (relative or absolute)

    Returns:
        Absolute path to the configuration file

    Raises:
        FileNotFoundError: If the configuration file cannot be found
    """
    # Return absolute paths unchanged
    if os.path.isabs(config_path):
        return config_path

    # Try current working directory first
    cwd = os.path.abspath(os.getcwd())
    cwd_path = os.path.abspath(os.path.join(cwd, config_path))
    if (cwd_path == cwd or cwd_path.startswith(cwd + os.sep)) and os.path.exists(cwd_path):
        return cwd_path

    # Try relative to verl project root (where verl package is installed)
    try:
        import verl

        verl_package_dir = os.path.abspath(os.path.dirname(verl.__file__))

        # Strategy 1: For development/editable installs.
        project_root = os.path.dirname(verl_package_dir)
        dev_path = os.path.abspath(os.path.join(project_root, config_path))
        if (dev_path == project_root or dev_path.startswith(project_root + os.sep)) and os.path.exists(dev_path):
            return dev_path

        # Strategy 2: For standard package installations.
        install_path = os.path.abspath(os.path.join(verl_package_dir, config_path))
        if (install_path == verl_package_dir or install_path.startswith(verl_package_dir + os.sep)) and os.path.exists(
            install_path
        ):
            return install_path
    except (ImportError, AttributeError):
        pass  # verl not installed or __file__ not available

    # File not found - raise clear error
    raise FileNotFoundError(
        f"Agent loop configuration file not found: {config_path}. Tried current directory and verl project root."
    )


# tokenizer.apply_chat_template is not working properly for gpt-oss model.
# Because the chat template requires tool call messages to parse tool response messages
# so we need to format the tool response manually.
def format_gpt_oss_tool_response_manually(tool_response: str, tool_call_name: str) -> str:
    """Format tool response for gpt-oss model.
    Args:
        tool_response: Tool response string
        tool_call_name: Name of the tool that was called

    Returns:
        Formatted tool response string
    """
    return f"<|start|>functions.{tool_call_name} to=assistant<|channel|>commentary<|message|>{tool_response}<|end|>"


def add_generation_prompt_for_gpt_oss(message_content: str) -> str:
    """Add generation prompt for gpt-oss model.
    Args:
        message_content: Message content string

    Returns:
        Message content string with generation prompt
    """
    return message_content + "<|start|>assistant"


def build_gpt_oss_tool_response_text(messages: list[dict[str, Any]], tool_call_names: list[str]) -> str:
    """Build gpt-oss tool response text (manual formatting + generation prompt)."""
    tool_response_texts: list[str] = []
    for i, tool_msg in enumerate(messages):
        actual_tool_name = tool_call_names[i]
        formatted = format_gpt_oss_tool_response_manually(tool_msg["content"], actual_tool_name)
        tool_response_texts.append(formatted)
    return add_generation_prompt_for_gpt_oss("".join(tool_response_texts))
