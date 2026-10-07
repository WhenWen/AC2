"""QED-Nano-style rubric judge reward: an LLM grades the actor's proof 0-7 against a
per-problem marking scheme (non-binary).

Reproduces the training grader of QED-Nano (arXiv:2604.04898): their RL reward is a
rubric-based 0-7 grade from the "ProofBench (Strict) Grader Prompt" (their Appendix,
``grader_prompt_template`` — adapted from ProofBench/ma2025reliable), *augmented with a
problem-specific grading scheme* generated per training question. The schemes ship in the
``rubrics`` column of ``lm-provers/FineProofs-RL`` — the SAME dataset our train parquet is
built from (``ac2.data.prepare_fineproofs``), so rubrics join on raw problem text
(``extra_info.theorem``). The paper used gpt-oss-20b (medium) for latency; we keep our
gpt-oss-120b judge (their reported quality ceiling) — only the prompt/scoring changes.

Contrast with ``prover_judge`` (binary Score: 1/0):
  * ``score``  (training reward)      = points / 7          (non-binary, in [0, 1])
  * ``acc``    (val accuracy metric)  = points / 7          (mean == avg grade fraction)
  * ``prover_judge_score``            = 1 iff points == 7   (STRICT: difficulty sampling's
    ``row_pass`` reads this field, so only a full-marks proof counts as "correct" for the
    per-qid p-hat.)
  * ``rubric_points`` (0-7 int) and ``rubric_found`` (0/1) are added to the dump schema.

The marking scheme comes from a JSON map {sha1(normalized problem): rubric_text} built by
``ac2.data.build_rubric_map`` (FineProofs-RL rubrics + IMO-ProofBench grading
guidelines for the val set). Path: ``reward_kwargs.rubric_map_path`` or ``$SP_RUBRIC_MAP``.
A problem with no rubric falls back to a generic 0-7 IMO marking scheme (rubric_found=0).

Endpoint discovery, HTTP session/retry/semaphore, proof extraction, and the health-field
dump schema are all reused from ``prover_judge`` so the reward path behaves identically
under training-scale fan-out.
"""

import hashlib
import json
import logging
import os
import re
import sys
from typing import Any, Optional

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

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

_DEFAULT_TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__), "templates", "qednano_rubric_judge.txt"
)
# VALIDATION rows are graded with the IMO-ProofBench ProofAutoGrader prompt instead (the
# benchmark's own protocol: GT solution + specific grading guidelines, {0,1,6,7} score menu) so
# the in-loop val metric is directly comparable to paper-prompt evals — mirroring QED-Nano's own
# split (rubric judge for TRAINING reward, benchmark autograder for EVALUATION).
_DEFAULT_VAL_TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__), "templates", "imo_proofautograder.txt"
)

# First <points>N</points> block wins (the prompt instructs exactly one). Fallback accepts
# the IMO-ProofBench autograder spelling "N out of 7" in case the judge drifts formats.
_POINTS_RE = re.compile(r"<points>\s*([0-7])\s*</points>")
_POINTS_FALLBACK_RE = re.compile(r"\b([0-7])\s*out of 7\b")

_GENERIC_MARKING_SCHEME = (
    "(No problem-specific marking scheme is available for this problem.) Apply the standard "
    "IMO 0-7 scale with full rigor: 7 = complete, correct, fully rigorous solution; 6 = "
    "correct core argument with only minor, locally-fixable defects; 1-5 = partial credit "
    "proportional to substantial verified progress toward a complete solution (key lemmas "
    "proved, essential structure established); 0 = no substantial progress or fundamentally "
    "flawed. Unjustified claims earn no credit."
)

_RUBRIC_MAP: Optional[dict] = None
_RUBRIC_MAP_PATH_LOADED: Optional[str] = None
_VAL_MAP: Optional[dict] = None
_VAL_MAP_PATH_LOADED: Optional[str] = None


def _norm_problem(text: str) -> str:
    return re.sub(r"\s+", "", text or "").lower()


def _problem_key(text: str) -> str:
    return hashlib.sha1(_norm_problem(text).encode()).hexdigest()


def _get_rubric_map(path: Optional[str]) -> dict:
    """Load {sha1(norm(problem)): rubric} once per process. Empty map when no file is set —
    every lookup then falls back to the generic scheme (loudly, once)."""
    global _RUBRIC_MAP, _RUBRIC_MAP_PATH_LOADED
    resolved = path or os.environ.get("SP_RUBRIC_MAP") or ""
    if _RUBRIC_MAP is not None and _RUBRIC_MAP_PATH_LOADED == resolved:
        return _RUBRIC_MAP
    if not resolved:
        logger.error(
            "qednano_rubric_judge: no rubric map (reward_kwargs.rubric_map_path / $SP_RUBRIC_MAP); "
            "EVERY problem will use the generic marking scheme."
        )
        _RUBRIC_MAP, _RUBRIC_MAP_PATH_LOADED = {}, resolved
        return _RUBRIC_MAP
    with open(os.path.expanduser(resolved)) as f:
        _RUBRIC_MAP = json.load(f)
    _RUBRIC_MAP_PATH_LOADED = resolved
    logger.info(f"qednano_rubric_judge: rubric map loaded from {resolved} ({len(_RUBRIC_MAP)} entries)")
    return _RUBRIC_MAP


def _get_val_map(path: Optional[str]) -> dict:
    """Load {sha1(norm(problem)): {"solution":..., "guidelines":...}} once per process (the
    IMO-ProofBench reference solutions + grading guidelines for the ProofAutoGrader val branch)."""
    global _VAL_MAP, _VAL_MAP_PATH_LOADED
    resolved = path or os.environ.get("SP_VAL_MAP") or ""
    if _VAL_MAP is not None and _VAL_MAP_PATH_LOADED == resolved:
        return _VAL_MAP
    if not resolved:
        logger.error(
            "qednano_rubric_judge: no val map (reward_kwargs.val_map_path / $SP_VAL_MAP); "
            "val rows will grade WITHOUT reference solution/guidelines."
        )
        _VAL_MAP, _VAL_MAP_PATH_LOADED = {}, resolved
        return _VAL_MAP
    with open(os.path.expanduser(resolved)) as f:
        _VAL_MAP = json.load(f)
    _VAL_MAP_PATH_LOADED = resolved
    logger.info(f"qednano_rubric_judge: val map loaded from {resolved} ({len(_VAL_MAP)} entries)")
    return _VAL_MAP


def _is_val_row(data_source: Optional[str], extra_info: dict) -> bool:
    """Validation rows carry data_source='imoproofbench' / extra_info.split='test'
    (prepare_fineproofs.to_verl_record). Either signal routes to the ProofAutoGrader branch."""
    if (extra_info or {}).get("split") == "test":
        return True
    return (data_source or "").lower() == "imoproofbench"


def _parse_points(response_text: str) -> Optional[int]:
    """Parse the 0-7 grade. ``response_text`` is prover_judge's formatted reply
    ("[reasoning]\\n...\\n\\n[content]\\n..."); grade ONLY the final-answer channel — the
    reasoning channel can contain draft <points> that would otherwise win the first-match."""
    if not response_text:
        return None
    content = response_text.rsplit("[content]\n", 1)[-1]
    m = _POINTS_RE.search(content)
    if m is None:
        m = _POINTS_FALLBACK_RE.search(content)
    if m is None:
        return None
    return int(m.group(1))


def _empty_extras() -> dict:
    """prover_judge's dump schema + the rubric fields, every key zeroed (verl pins the dumped
    key set to the FIRST rollout's dict, so all return paths must carry the full set)."""
    extras = _pj._empty_extras()
    extras.update({"rubric_points": 0, "rubric_found": 0})
    return extras


async def compute_score(
    data_source: str = None,
    solution_str: str = None,
    ground_truth: str = None,
    extra_info: dict = None,
    reward_router_address: Optional[str] = None,
    *,
    rubric_judge_template_path: Optional[str] = None,
    rubric_map_path: Optional[str] = None,
    val_judge_template_path: Optional[str] = None,
    val_map_path: Optional[str] = None,
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
    """Rubric-judge reward (0-7 -> reward points/7). Missing ``<proof>`` short-circuits to 0."""
    extra_info = extra_info or {}
    problem = extra_info.get("theorem") or extra_info.get("question") or ""
    if not problem:
        logger.warning("qednano_rubric_judge: extra_info missing 'theorem'/'question'; score=0.")
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
        return extras  # no proof to judge -> reward stays 0

    proof_for_judge = proof[:proof_max_chars] if proof_max_chars else proof
    problem_for_judge = problem[:problem_max_chars] if problem_max_chars else problem

    if _is_val_row(data_source, extra_info):
        # VAL: IMO-ProofBench ProofAutoGrader protocol (benchmark-comparable {0,1,6,7} scale).
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
    else:
        # TRAIN: QED-Nano rubric grader (dense 0-7 partial credit -> the RL reward).
        rubric = _get_rubric_map(rubric_map_path).get(_problem_key(problem))
        rubric_found = int(rubric is not None)
        marking_scheme = rubric if rubric is not None else _GENERIC_MARKING_SCHEME
        template = _load_template(
            rubric_judge_template_path or _DEFAULT_TEMPLATE_PATH,
            required_slots=("problem", "marking_scheme", "solution"),
        )
        prompt = template.format(
            problem=problem_for_judge, marking_scheme=marking_scheme, solution=proof_for_judge
        )

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
    # _run_judge_call parses the binary "Score:" convention (absent here) -> re-parse 0-7
    # points from the full response text it captured.
    points = _parse_points(outcome.response)
    parse_failed = int(points is None and outcome.http_error == 0)
    if points is None:
        points = 0

    frac = points / 7.0
    strict_correct = 1 if points == 7 else 0

    # Env-gated correct-only length penalty, kept for config parity with prover_judge (same
    # SP_LENPEN_* envs; applied multiplicatively on the normalized grade). Default OFF for
    # QED-Nano alignment (their reward is the raw grade fraction).
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
            "prover_judge_score": strict_correct,  # difficulty row_pass: only 7/7 counts
            "rubric_points": points,
            "rubric_found": rubric_found,
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
