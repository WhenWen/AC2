"""Prover proof-judge reward: an LLM grades the actor's proof, binary 1/0.

The actor is prompted (by ``ac2.data.prepare_fineproofs``) to emit a ``<proof>...</proof>`` block; this
reward extracts that proof and asks an LLM judge (gpt-oss-120b) whether it correctly solves the
problem, parsing a final ``Score: 1`` / ``Score: 0`` line into a binary reward.

Contrast with a rule-based boxed-answer reward (LaTeX/Sympy verify): here
the "judge" is a served LLM reached over HTTP, so ``compute_score`` is ``async`` and issues an
aiohttp chat-completion request per rollout. verl 0.8.0 auto-detects the ``async def`` and awaits
it; the dict this returns is unpacked into ``reward_extra_info`` and every key is dumped into the
per-step rollout JSONL (used by ``ac2.viz``).

Judge endpoint discovery (differs from an earlier verl fork, which injected
``reward_router_address`` from its in-verl reward-model router). Stock verl does NOT inject that
argument, so we resolve the judge base URL in this order:

    1. ``reward_router_address`` kwarg  (kept for fork compatibility, if ever wired)
    2. ``reward_kwargs.judge_url``      (Hydra: ++reward.custom_reward_function.reward_kwargs.judge_url=...)
    3. ``$SELF_PLAY_JUDGE_URL``         (set by submit.sbatch after launching the standalone judge)

Wire in via the runner:
    reward.custom_reward_function.path=<...>/src/ac2/rewards/prover_judge.py
    reward.custom_reward_function.name=compute_score
    ++reward.custom_reward_function.reward_kwargs.judge_max_tokens=81920
    ++reward.custom_reward_function.reward_kwargs.judge_reasoning_effort=medium
"""

import asyncio
import logging
import os
import re
from typing import Any, Optional

import aiohttp

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

_TEMPLATE_CACHE: dict[str, str] = {}

_DEFAULT_PROVER_JUDGE_TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__), "templates", "prover_judge.txt"
)

_HTTP_SESSION: Optional[aiohttp.ClientSession] = None
_HTTP_SESSION_LOCK = asyncio.Lock()

# Bound concurrent in-flight judge calls PER reward-worker process. The training
# reward fan-out is bs*rollout.n (~2048) issued all at once; firing them all
# simultaneously overwhelmed the judge server's HTTP layer -> dropped/half-open
# connections -> the reward gather hung (compute_rm_score wedge). Capping in-flight
# calls keeps the judge saturated but never over-subscribed. Total across N workers
# is N * SP_JUDGE_MAX_INFLIGHT; size it near the judge's aggregate max_num_seqs.
_JUDGE_SEM: Optional[asyncio.Semaphore] = None


def _get_judge_sem() -> asyncio.Semaphore:
    global _JUDGE_SEM
    if _JUDGE_SEM is None:
        _JUDGE_SEM = asyncio.Semaphore(int(os.environ.get("SP_JUDGE_MAX_INFLIGHT", "16")))
    return _JUDGE_SEM

_PROOF_RE = re.compile(r"<proof>(.*?)</proof>", re.DOTALL)
_SCORE_RE = re.compile(r"\bScore:\s*([01])\b")


async def _get_session() -> aiohttp.ClientSession:
    global _HTTP_SESSION
    if _HTTP_SESSION is None or _HTTP_SESSION.closed:
        async with _HTTP_SESSION_LOCK:
            if _HTTP_SESSION is None or _HTTP_SESSION.closed:
                # Finite total so a stalled/half-open judge connection releases into
                # _post_with_retries instead of hanging the whole reward gather forever.
                # total=None wedged compute_rm_score at the training-scale ~2048-call
                # fan-out (a dropped judge connection was never timed out; hung requests
                # filled every connector slot -> deadlock). sock_connect fails fast on a
                # refused/dropped connection. Connector limit raised to cover the fan-out
                # so requests aren't queued behind a 512 cap. All overridable via env.
                _judge_total = float(os.environ.get("SP_JUDGE_HTTP_TOTAL_TIMEOUT", "1800"))
                _judge_conn = int(os.environ.get("SP_JUDGE_HTTP_CONN_LIMIT", "2048"))
                _HTTP_SESSION = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=_judge_total, sock_connect=30),
                    connector=aiohttp.TCPConnector(
                        limit=_judge_conn, limit_per_host=_judge_conn, ttl_dns_cache=300
                    ),
                )
    return _HTTP_SESSION


def _load_template(path: str, required_slots: tuple[str, ...]) -> str:
    if path not in _TEMPLATE_CACHE:
        with open(path, "r") as f:
            tmpl = f.read()
        for slot in required_slots:
            if "{" + slot + "}" not in tmpl:
                raise ValueError(
                    f"prover judge template {path} is missing the {{{slot}}} slot."
                )
        _TEMPLATE_CACHE[path] = tmpl
    return _TEMPLATE_CACHE[path]


def _resolve_judge_url(
    reward_router_address: Optional[str], judge_url: Optional[str]
) -> str:
    """Resolve the judge's chat-completions URL (see module docstring for order).

    Accepts a bare ``host:port`` (fork-style ``reward_router_address``), a full
    ``http://host:port`` base, or one already ending in ``/v1`` or the full chat path.
    """
    base = judge_url or reward_router_address or os.environ.get("SELF_PLAY_JUDGE_URL")
    if not base:
        raise RuntimeError(
            "prover_judge reward: no judge endpoint. Set $SELF_PLAY_JUDGE_URL (submit.sbatch "
            "launches the judge and exports it) or pass reward_kwargs.judge_url."
        )
    s = base.strip()
    if not s.startswith(("http://", "https://")):
        s = "http://" + s
    s = s.rstrip("/")
    if s.endswith("/chat/completions"):
        return s
    if s.endswith("/v1"):
        return s + "/chat/completions"
    return s + "/v1/chat/completions"


def _strip_thinking(text: str, close_tag: str) -> str:
    """Strip everything up to and including the last close_tag (Qwen3 ``</think>`` convention)."""
    if not close_tag:
        return text
    idx = text.rfind(close_tag)
    if idx < 0:
        return text
    return text[idx + len(close_tag):]


def _extract_proof(post_think: str) -> tuple[Optional[str], bool]:
    """Return (proof_text, proof_tag_present): the first ``<proof>...</proof>`` block, stripped."""
    m = _PROOF_RE.search(post_think)
    if m is None:
        return None, False
    return m.group(1).strip(), True


def _parse_score_from_response(response_text: str) -> Optional[int]:
    if not response_text:
        return None
    matches = _SCORE_RE.findall(response_text)
    if not matches:
        return None
    return int(matches[-1])


async def _post_with_retries(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    max_retries: int = 5,
) -> dict[str, Any]:
    """POST a chat-completion with exponential backoff. Raises on 4xx or exhaustion."""
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            async with session.post(url, json=payload) as resp:
                resp.raise_for_status()
                return await resp.json()
        except aiohttp.ClientResponseError as e:
            if 400 <= e.status < 500:
                logger.error(f"judge {url} client error HTTP {e.status}; not retrying.")
                raise
            last_exc = e
            logger.warning(
                f"judge [Attempt {attempt+1}/{max_retries}] {url} HTTP {e.status}; retrying."
            )
        except (asyncio.TimeoutError, aiohttp.ClientConnectorError) as e:
            last_exc = e
            logger.warning(
                f"judge [Attempt {attempt+1}/{max_retries}] {url} {type(e).__name__}; retrying."
            )
        except Exception as e:
            last_exc = e
            logger.warning(
                f"judge [Attempt {attempt+1}/{max_retries}] {url} {type(e).__name__}: {e}; retrying."
            )
        if attempt < max_retries - 1:
            await asyncio.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"judge: max retries ({max_retries}) reached for {url}") from last_exc


def _chat_payload(
    content: str,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    reasoning_effort: Optional[str],
    seed: Optional[int],
    payload_style: str = "gptoss",
) -> dict:
    if payload_style == "deepseek_v4":
        # DeepSeek-V4 on vLLM >=0.23: thinking + effort ride in chat_template_kwargs, the
        # request shape validated against the served judge.
        # The gpt-oss-isms (top-level reasoning_effort / include_reasoning / top_k) are
        # deliberately absent: never validated on 0.23, and mixing top-level effort with
        # chat_template_kwargs can 400.
        payload = {
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "chat_template_kwargs": {
                "thinking": True,
                "reasoning_effort": reasoning_effort or "high",
            },
        }
        if seed is not None:
            payload["seed"] = seed
        return payload
    payload = {
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "include_reasoning": True,
    }
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    if seed is not None:
        payload["seed"] = seed
    return payload


def _extract_content(chat_output: dict) -> str:
    return chat_output["choices"][0]["message"]["content"]


def _extract_reasoning(chat_output: dict) -> str:
    """Return the reasoning/analysis channel (``message.reasoning_content``), or "" if absent.

    ``message.reasoning`` is an alternate spelling some parsers emit, so accept it as a
    fallback.
    """
    msg = (chat_output.get("choices") or [{}])[0].get("message") or {}
    return msg.get("reasoning_content") or msg.get("reasoning") or ""


def _format_judge_response(*, reasoning: str, content: str) -> str:
    """Combine reasoning + final-answer channels into one persisted string (full reply)."""
    parts = []
    if reasoning:
        parts.append(f"[reasoning]\n{reasoning}")
    parts.append(f"[content]\n{content}")
    return "\n\n".join(parts)


def _extract_usage_tuple(chat_output: dict) -> tuple[int, int]:
    """Return (prompt_tokens, completion_tokens) from a chat-completion response.

    BUG FIX (vs the upstream reward): the original raised on any ``usage`` shape it didn't expect
    and the caller then silently zeroed both counts even on a *successful* judge reply, so
    ``judge_completion_tokens`` was ~always 0 in the dump. Here we accept the common OpenAI/vLLM
    spellings and only fall back to 0 when a count is genuinely absent (logged at ERROR, not
    swallowed). ``-1`` sentinel is returned for a missing count so the dump distinguishes
    "judge returned no usage" from a real zero.
    """
    usage = chat_output.get("usage") or {}

    def _pick(*keys: str) -> Optional[int]:
        for k in keys:
            v = usage.get(k)
            if v is not None:
                return int(v)
        return None

    p = _pick("prompt_tokens", "input_tokens", "prompt_token_count")
    c = _pick("completion_tokens", "output_tokens", "completion_token_count", "generation_tokens")
    if c is None and p is not None:
        total = _pick("total_tokens", "total_token_count")
        if total is not None:
            c = total - p
    if p is None or c is None:
        logger.error(
            "judge response missing token usage (prompt=%s, completion=%s); usage=%r",
            p, c, usage,
        )
    return (p if p is not None else -1), (c if c is not None else -1)


def _is_truncated(chat_output: dict) -> bool:
    choices = chat_output.get("choices") or [{}]
    return choices[0].get("finish_reason") == "length"


class _JudgeOutcome:
    """One judge call's parsed result + health flags. ``response`` is the full formatted reply."""

    __slots__ = (
        "parsed_value", "parse_failed", "http_error", "truncated",
        "prompt_tokens", "completion_tokens", "response",
    )

    def __init__(
        self, *, parsed_value, parse_failed, http_error, truncated,
        prompt_tokens, completion_tokens, response,
    ):
        self.parsed_value = parsed_value
        self.parse_failed = parse_failed
        self.http_error = http_error
        self.truncated = truncated
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.response = response


async def _run_judge_call(
    *,
    session: aiohttp.ClientSession,
    judge_url: str,
    prompt: str,
    judge_max_tokens: int,
    judge_temperature: float,
    judge_top_p: float,
    judge_top_k: int,
    judge_reasoning_effort: Optional[str],
    judge_seed: Optional[int],
    judge_payload_style: str = "gptoss",
) -> _JudgeOutcome:
    """Send one judge call; parse ``Score: 0/1``. HTTP failure -> parsed_value=None, http_error=1.

    The full judge reply (reasoning + content) is captured in ``outcome.response`` for the dump,
    and token usage is captured whenever the server reports it (see ``_extract_usage_tuple``).
    """
    payload = _chat_payload(
        content=prompt,
        max_tokens=judge_max_tokens,
        temperature=judge_temperature,
        top_p=judge_top_p,
        top_k=judge_top_k,
        reasoning_effort=judge_reasoning_effort,
        seed=judge_seed,
        payload_style=judge_payload_style,
    )
    try:
        async with _get_judge_sem():
            chat_output = await _post_with_retries(session, judge_url, payload)
    except Exception as e:
        logger.warning(f"judge HTTP failure: {type(e).__name__}: {e}")
        return _JudgeOutcome(
            parsed_value=None, parse_failed=0, http_error=1, truncated=0,
            prompt_tokens=0, completion_tokens=0, response="",
        )

    response_text = _extract_content(chat_output) or ""
    reasoning_text = _extract_reasoning(chat_output)
    prompt_tokens, completion_tokens = _extract_usage_tuple(chat_output)
    truncated = int(_is_truncated(chat_output))

    parsed = _parse_score_from_response(response_text)
    return _JudgeOutcome(
        parsed_value=parsed,
        parse_failed=int(parsed is None),
        http_error=0,
        truncated=truncated,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        response=_format_judge_response(reasoning=reasoning_text, content=response_text),
    )


def _empty_extras() -> dict:
    """Canonical prover ``reward_extra_info`` schema, every key zeroed.

    verl pins the dumped key set to the FIRST rollout's dict keys, so every return path must
    surface this exact key set (mode-specific fields stay zeroed when a proof is missing). This
    is what guarantees ``judge_completion_tokens`` / ``judge_response`` reach the JSONL dump for
    every row, not just the ones whose judge call succeeded.
    """
    return {
        "score": 0.0,
        "acc": 0.0,
        "mode_is_prover": 1,
        "proof_tag_present": 0,
        "candidate_len_chars": 0,
        "proof_len_chars": 0,
        "prover_judge_score": 0,
        "length_penalty": 0.0,
        "response_length_tokens": 0.0,
        "judge_parse_failed": 0,
        "judge_http_error": 0,
        "judge_truncated": 0,
        "judge_prompt_tokens": 0,
        "judge_completion_tokens": 0,
        "judge_response": "",
    }


async def compute_score(
    data_source: str = None,
    solution_str: str = None,
    ground_truth: str = None,
    extra_info: dict = None,
    reward_router_address: Optional[str] = None,
    *,
    prover_judge_template_path: Optional[str] = None,
    judge_url: Optional[str] = None,
    judge_max_tokens: int = 16384,
    judge_temperature: float = 1.0,
    judge_top_p: float = 1.0,
    judge_top_k: int = -1,
    judge_reasoning_effort: Optional[str] = "high",
    judge_seed: Optional[int] = 42,
    thinking_close_tag: str = "</think>",
    problem_max_chars: int = 0,
    proof_max_chars: int = 0,
    **kwargs,
) -> dict:
    """Prover proof-judge reward. Returns the flat dict verl wraps into ``reward_extra_info``.

    ``score`` is the binary ``prover_judge_score`` (1 iff the judge outputs ``Score: 1``). A
    missing ``<proof>`` short-circuits to ``score=0`` without a judge call.
    """
    extra_info = extra_info or {}
    problem = extra_info.get("theorem") or extra_info.get("question") or ""
    if not problem:
        logger.warning("prover_judge: extra_info missing 'theorem'/'question'; score=0, no judge call.")
        return _empty_extras()

    post_think = _strip_thinking(solution_str or "", thinking_close_tag)
    proof, proof_tag_present = _extract_proof(post_think)
    proof_len = len(proof) if proof else 0

    extras = _empty_extras()
    extras.update(
        {
            "proof_tag_present": int(proof_tag_present),
            "candidate_len_chars": proof_len,
            "proof_len_chars": proof_len,
        }
    )
    if not proof:
        return extras  # no proof to judge -> score stays 0

    proof_for_judge = proof[:proof_max_chars] if proof_max_chars else proof
    problem_for_judge = problem[:problem_max_chars] if problem_max_chars else problem

    template = _load_template(
        prover_judge_template_path or _DEFAULT_PROVER_JUDGE_TEMPLATE_PATH,
        required_slots=("problem", "proof"),
    )
    prompt = template.format(problem=problem_for_judge, proof=proof_for_judge)

    session = await _get_session()
    outcome = await _run_judge_call(
        session=session,
        judge_url=_resolve_judge_url(reward_router_address, judge_url),
        prompt=prompt,
        judge_max_tokens=judge_max_tokens,
        judge_temperature=judge_temperature,
        judge_top_p=judge_top_p,
        judge_top_k=judge_top_k,
        judge_reasoning_effort=judge_reasoning_effort,
        judge_seed=judge_seed,
    )

    binary = 1 if outcome.parsed_value == 1 else 0
    # Correct-only length penalty (env-tunable): penalty is 0 below SP_LENPEN_START
    # response tokens and grows linearly to 1.0 at SP_LENPEN_END (default 20000 ->
    # 50000). Applied ONLY to correct rollouts (binary==1); wrong rollouts stay 0.
    # Correct reward = 1 - penalty in [0, 1] (1 at <=20k tokens, 0 at >=50k). Uses
    # the true response token length injected into extra_info by the reward manager.
    # OFF by default (the baseline has no length penalty); enable the ablation with
    # SP_LENPEN_ENABLE=1. When on: correct-only, 0 below START tokens -> 1.0 at END.
    # SP_LENPEN_MAX (default 1.0) is the max penalty COEFFICIENT: penalty saturates at
    # this value once L>=END, so a correct rollout's reward floor is (1 - SP_LENPEN_MAX).
    # With START=0, END=L_max, MAX=0.9 this is exactly R = 1 - 0.9*clamp(L/L_max,0,1).
    # Default 1.0 reproduces the prior behaviour (penalty 0->1, reward floor 0).
    _lp_enable = os.environ.get("SP_LENPEN_ENABLE", "0") in ("1", "true", "True")
    _lp_start = float(os.environ.get("SP_LENPEN_START", "20000"))
    _lp_end = float(os.environ.get("SP_LENPEN_END", "50000"))
    _lp_max = float(os.environ.get("SP_LENPEN_MAX", "1.0"))
    _resp_len = float(extra_info.get("valid_response_length", 0) or 0)
    _length_penalty = 0.0
    if _lp_enable and binary == 1 and _lp_end > _lp_start:
        _length_penalty = _lp_max * min(max((_resp_len - _lp_start) / (_lp_end - _lp_start), 0.0), 1.0)
    _reward = float(binary) * (1.0 - _length_penalty)
    extras.update(
        {
            "score": _reward,
            "acc": float(binary),
            "prover_judge_score": binary,
            "length_penalty": _length_penalty,
            "response_length_tokens": _resp_len,
            "judge_parse_failed": outcome.parse_failed,
            "judge_http_error": outcome.http_error,
            "judge_truncated": outcome.truncated,
            "judge_prompt_tokens": outcome.prompt_tokens,
            "judge_completion_tokens": outcome.completion_tokens,
            "judge_response": outcome.response,
        }
    )
    return extras
