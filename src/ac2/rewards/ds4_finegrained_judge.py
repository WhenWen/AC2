"""DeepSeek-V4-Flash fine-grained judge reward: a "fine-grained without reference"
prompt graded 0-7, served by a colocated DS4-Flash judge (vLLM >=0.23).

Replaces ``qednano_rubric_judge`` as the training reward for runs that use the DS4-Flash
judge. Differences vs that module:

  * TRAIN prompt = ``templates/finegrained_noref_judge.txt`` (sha-asserted at load), the
    exact template a 1010-request throughput benchmark validated (1010/1010). Slots: ``{problem}``/``{proof}`` only — **no per-problem rubric map is needed**.
  * Judge payload style defaults to ``deepseek_v4`` (thinking + reasoning_effort via
    ``chat_template_kwargs``; no gpt-oss-isms) — see ``prover_judge._chat_payload``.
  * Grade parse accepts, in order: ``<points>N out of 7</points>`` (the fine-grained
    format; FIRST match, per the prompt's own instruction), ``<points>N</points>`` (the
    rubric/ProofAutoGrader format, so the val branch and judge drift stay parseable),
    then a bare ``N out of 7``. Content channel only — reasoning may contain drafts.
  * ``judge_strict_parse_ok`` is added to the dump: the benchmark's stricter criterion
    (exactly one fine-grained marker AND it is the final line). Telemetry only — the
    reward uses the lenient parse.

Reward semantics: ``score = points/7`` (menu {0,1,6,7} -> support {0,1/7,6/7,1}) and
``acc = points/7`` as before. ``prover_judge_score`` — the difficulty-sampling
``row_pass`` class — is ``1 iff points >= pass_points_min`` with **default 6**: under
the fine-grained menu both 6 ("almost correct") and 7 ("correct") count as class-1 for
the per-problem p-hat. Override via
``reward_kwargs.pass_points_min`` / ``$SP_PASS_POINTS_MIN``. VAL rows keep the
IMO-ProofBench ProofAutoGrader protocol (reference solution + guidelines) — only the
judge model changes. Endpoint discovery, HTTP session/retry/semaphore,
proof extraction, and the dump-schema discipline are reused from ``prover_judge``.
"""

import hashlib
import json
import logging
import os
import re
import sys
from typing import Optional

# verl loads custom rewards by FILE path (spec_from_file_location), so guarantee the
# package import below works even if the loading process lacks src/ on sys.path.
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from ac2.rewards.prover_judge import (  # noqa: F401  (shared plumbing)
    _extract_proof,
    _get_session,
    _load_template,
    _resolve_judge_url,
    _run_judge_call,
    _strip_thinking,
)
from ac2.rewards import prover_judge as _pj
from ac2.rewards.qednano_rubric_judge import (  # reuse the val-branch machinery
    _get_val_map,
    _is_val_row,
    _problem_key,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

_DEFAULT_TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__), "templates", "finegrained_noref_judge.txt"
)
# Byte-identity with the benchmarked prompt (474fc5f4... = the sha the benchmark
# pinned). A drifted template invalidates the throughput/quality evidence.
_FINEGRAINED_TEMPLATE_SHA256 = "474fc5f44191ebcee4bb9e3bd1d3cd3f407d059dfda316439dc4fd930ee2fbe6"
_DEFAULT_VAL_TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__), "templates", "imo_proofautograder.txt"
)

# Parse order: fine-grained tagged -> plain tagged (val/ProofAutoGrader spelling) -> bare.
_POINTS_OUTOF_RE = re.compile(r"<points>\s*([0-7])\s*out of 7\s*</points>")
_POINTS_PLAIN_RE = re.compile(r"<points>\s*([0-7])\s*</points>")
_POINTS_BARE_RE = re.compile(r"\b([0-7])\s*out of 7\b")
# The benchmark's strict criterion: one legal marker in the whole content, and that
# marker is the final line.
_STRICT_FULL_RE = re.compile(r"<points>\s*(-?\d+)\s*out of 7\s*</points>")

_TEMPLATE_SHA_CHECKED = False
_CUSTOM_TEMPLATE_WARNED: set = set()


def _content_channel(response_text: str) -> str:
    """prover_judge formats replies as "[reasoning]\\n...\\n\\n[content]\\n..."; grade ONLY
    the final-answer channel — the reasoning channel can contain draft <points> blocks."""
    return (response_text or "").rsplit("[content]\n", 1)[-1]


def _parse_points(response_text: str) -> Optional[int]:
    content = _content_channel(response_text)
    for rx in (_POINTS_OUTOF_RE, _POINTS_PLAIN_RE, _POINTS_BARE_RE):
        m = rx.search(content)
        if m is not None:
            return int(m.group(1))
    return None


def _strict_parse_ok(response_text: str) -> int:
    """Benchmark-parity strict check (telemetry only): exactly one fine-grained marker,
    as the final line, value in 0..7."""
    stripped = _content_channel(response_text).strip()
    if not stripped:
        return 0
    matches = _STRICT_FULL_RE.findall(stripped)
    if len(matches) != 1:
        return 0
    final = re.fullmatch(
        r"<points>\s*(-?\d+)\s*out of 7\s*</points>", stripped.splitlines()[-1].strip()
    )
    if not final:
        return 0
    return int(0 <= int(final.group(1)) <= 7)


def _load_finegrained_template(path: Optional[str]) -> str:
    """Load the train template; on the DEFAULT path also assert byte-identity with the
    benchmarked prompt (an explicit override skips the sha pin, loudly)."""
    global _TEMPLATE_SHA_CHECKED
    resolved = path or _DEFAULT_TEMPLATE_PATH
    if resolved == _DEFAULT_TEMPLATE_PATH and not _TEMPLATE_SHA_CHECKED:
        with open(resolved, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        if digest != _FINEGRAINED_TEMPLATE_SHA256:
            raise ValueError(
                f"finegrained judge template {resolved} sha256={digest} != benchmarked "
                f"{_FINEGRAINED_TEMPLATE_SHA256}; refusing to grade with a drifted prompt."
            )
        _TEMPLATE_SHA_CHECKED = True
    elif path and path not in _CUSTOM_TEMPLATE_WARNED:
        # once per process, not once per rollout (otherwise logged ~1024x per step)
        _CUSTOM_TEMPLATE_WARNED.add(path)
        logger.warning("ds4_finegrained_judge: custom template %s (sha pin skipped)", path)
    return _load_template(resolved, required_slots=("problem", "proof"))


def _empty_extras() -> dict:
    """prover_judge's dump schema + the grade fields, every key zeroed (verl pins the
    dumped key set to the FIRST rollout's dict, so all return paths carry the full set).
    ``rubric_points``/``rubric_found`` keep their names for dashboard compatibility;
    for train rows rubric_found is always 0 (this prompt uses no rubric)."""
    extras = _pj._empty_extras()
    extras.update({"rubric_points": 0, "rubric_found": 0, "judge_strict_parse_ok": 0,
                   # sp_q: consumed-Q flags, zeroed on every non-Q path so the
                   # dumped key set (pinned to the FIRST rollout's dict) stays uniform.
                   "sp_q_consumed": 0, "sp_q_invalid": 0})
    return extras


async def compute_score(
    data_source: str = None,
    solution_str: str = None,
    ground_truth: str = None,
    extra_info: dict = None,
    reward_router_address: Optional[str] = None,
    *,
    finegrained_template_path: Optional[str] = None,
    val_judge_template_path: Optional[str] = None,
    val_map_path: Optional[str] = None,
    judge_url: Optional[str] = None,
    judge_max_tokens: int = 40000,
    judge_temperature: float = 1.0,
    judge_top_p: float = 1.0,
    judge_top_k: int = -1,
    judge_reasoning_effort: Optional[str] = "high",
    judge_seed: Optional[int] = 42,
    judge_payload_style: str = "deepseek_v4",
    pass_points_min: Optional[int] = None,
    thinking_close_tag: str = "</think>",
    problem_max_chars: int = 0,
    proof_max_chars: int = 0,
    train_rubric_template_path: Optional[str] = None,
    rubric_map_path: Optional[str] = None,
    **kwargs,
) -> dict:
    """Fine-grained judge reward (0-7 -> reward points/7). Missing ``<proof>`` -> 0."""
    extra_info = extra_info or {}

    # Generative-Q readiness: a ready non-audit rollout that hit the
    # short budget with no extractable proof was scored by the actor's own Q function in
    # the pre-judge Q wave. The driver stamps sp_q_route_taken/sp_q_value; NO judge call
    # is made (that is the judge-cost win). An invalid consumed-Q call is returned as
    # score 0 here and DROPPED from the group baseline/advantages/PPO by the driver
    # (survivor-mean fill + response_mask 0), never trained on. prover_judge_score stays
    # 0 so these partials can never be admitted to the policy replay buffer. Inert for
    # every run that doesn't stamp the key (e.g. the GRPO baselines and all validation rows).
    if extra_info.get("sp_q_route_taken") == "q_consumed":
        extras = _empty_extras()
        qv = extra_info.get("sp_q_value")
        invalid = int(extra_info.get("sp_q_invalid", 1)) or qv is None
        extras.update(
            {
                "score": 0.0 if invalid else float(qv),
                "acc": 0.0 if invalid else float(qv),
                "sp_q_consumed": 1,
                "sp_q_invalid": int(bool(invalid)),
            }
        )
        return extras

    problem = extra_info.get("theorem") or extra_info.get("question") or ""
    if not problem:
        logger.warning("ds4_finegrained_judge: extra_info missing 'theorem'/'question'; score=0.")
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
        return extras  # no proof to judge -> reward stays 0 (benchmark short-circuit parity)

    proof_for_judge = proof[:proof_max_chars] if proof_max_chars else proof
    problem_for_judge = problem[:problem_max_chars] if problem_max_chars else problem

    if _is_val_row(data_source, extra_info):
        # VAL: IMO-ProofBench ProofAutoGrader protocol (reference solution + guidelines),
        # unchanged from qednano_rubric_judge — only the judge model differs.
        entry = _get_val_map(val_map_path).get(_problem_key(problem)) or {}
        rubric_found = int(bool(entry))
        template = _load_template(
            val_judge_template_path or _DEFAULT_VAL_TEMPLATE_PATH,
            required_slots=("problem", "solution", "guidelines", "proof"),
        )
        prompt = template.format(
            problem=problem_for_judge,
            solution=entry.get("solution") or "(no reference solution available)",
            guidelines=entry.get("guidelines")
            or "(No problem-specific grading guidelines available; apply the General Scoring Rubric.)",
            proof=proof_for_judge,
        )
    elif train_rubric_template_path:
        # TRAIN, rubric-scheme variant (smooth-reward ablation): the QED-Nano marking-
        # scheme grader — FULL integer 0-7 scale against the per-problem FineProofs rubric
        # (smooth reward), instead of the fine-grained menu {0,1,6,7}. Kwargs-gated: runs
        # that don't pass train_rubric_template_path are byte-identical. The rubric map and
        # generic-scheme fallback reuse qednano_rubric_judge's own loader/constant (same
        # sha1(normalized problem) key convention as the map builder).
        from ac2.rewards.qednano_rubric_judge import (
            _GENERIC_MARKING_SCHEME,
            _get_rubric_map,
        )
        entry = _get_rubric_map(rubric_map_path).get(_problem_key(problem)) or ""
        rubric_found = int(bool(entry))
        template = _load_template(
            train_rubric_template_path,
            required_slots=("problem", "marking_scheme", "solution"),
        )
        prompt = template.format(
            problem=problem_for_judge,
            marking_scheme=entry or _GENERIC_MARKING_SCHEME,
            solution=proof_for_judge,
        )
    else:
        # TRAIN: the benchmarked fine-grained no-reference prompt. No rubric map.
        rubric_found = 0
        template = _load_finegrained_template(finegrained_template_path)
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
        judge_payload_style=judge_payload_style,
    )
    points = _parse_points(outcome.response)
    parse_failed = int(points is None and outcome.http_error == 0)
    if points is None:
        points = 0

    frac = points / 7.0
    # Difficulty row_pass class: {6,7} -> 1 by default (see module docstring). kwarg wins,
    # then $SP_PASS_POINTS_MIN, then 6.
    _pass_min = pass_points_min if pass_points_min is not None else int(
        os.environ.get("SP_PASS_POINTS_MIN", "6"))
    pass_flag = 1 if points >= _pass_min else 0

    # Env-gated correct-only length penalty, config-parity with prover_judge (same
    # SP_LENPEN_* envs; multiplicative on the normalized grade). Default OFF.
    _lp_enable = os.environ.get("SP_LENPEN_ENABLE", "0") in ("1", "true", "True")
    _lp_start = float(os.environ.get("SP_LENPEN_START", "20000"))
    _lp_end = float(os.environ.get("SP_LENPEN_END", "50000"))
    _lp_max = float(os.environ.get("SP_LENPEN_MAX", "1.0"))
    _resp_len = float(extra_info.get("valid_response_length", 0) or 0)
    _length_penalty = 0.0
    if _lp_enable and points > 0 and _lp_end > _lp_start:
        _length_penalty = _lp_max * min(max((_resp_len - _lp_start) / (_lp_end - _lp_start), 0.0), 1.0)
    _reward = frac * (1.0 - _length_penalty)

    extras.update(
        {
            "score": _reward,
            "acc": frac,
            "prover_judge_score": pass_flag,  # difficulty row_pass: points >= pass_points_min (default 6)
            "rubric_points": points,
            "rubric_found": rubric_found,
            "judge_strict_parse_ok": _strict_parse_ok(outcome.response),
            "length_penalty": _length_penalty,
            "response_length_tokens": _resp_len,
            "judge_parse_failed": parse_failed,
            "judge_http_error": outcome.http_error,
            "judge_truncated": outcome.truncated,
            "judge_prompt_tokens": outcome.prompt_tokens,
            "judge_completion_tokens": outcome.completion_tokens,
            "judge_response": outcome.response,
        }
    )
    return extras
