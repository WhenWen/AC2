#!/usr/bin/env python3
"""Regression fixture for the exponential-length-penalty parser path.

Runs with the *exponential* length penalty (instead of a linear overlong
buffer) have the reward manager (DAPO) emit per-rollout::

    exponential_penalty_factor          (float, e.g. 0.9999^exceed_tokens)
    exponential_penalty_exceed_tokens   (int,   tokens over the free budget)
    pre_kl_post_penalty_reward          (float, the post-penalty pre-KL reward)

and DO NOT emit ``overlong_reward`` / ``overlong`` (those belong to the linear
path). The parser must:

    1. Aggregate ``exponential_penalty_factor`` and
       ``exponential_penalty_exceed_tokens`` into per-step means for BOTH
       prover and proposer families.
    2. Compute ``optimized_reward_mean`` from ``pre_kl_post_penalty_reward``
       (the post-penalty pre-KL scalar) without depending on the
       (now-absent) ``overlong_reward`` field.
    3. Surface ``length_penalty_mean`` for prover rows AND proposer rows
       (proposer-only emission is wrong here — both modes carry it).
    4. Surface the new manifest knobs in ``fig_data["config"]``:
           - exponential_penalty
           - exponential_penalty_base_free_tokens
           - exponential_penalty_gamma
           - no_drop_except_true_zero_std
           - dapo_filter_metric

Runs both as pytest::

    cd <repo-root> \\
        && python -m pytest src/ac2/viz/tests/test_parse_fig_data_exponential.py -x -q

and as a standalone script::

    python3 src/ac2/viz/tests/test_parse_fig_data_exponential.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARSER_PATH = HERE.parent / "parse_fig_data.py"

_spec = importlib.util.spec_from_file_location("parse_fig_data", PARSER_PATH)
assert _spec is not None and _spec.loader is not None, (
    f"could not load parse_fig_data from {PARSER_PATH}"
)
PF = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = PF
_spec.loader.exec_module(PF)


# ---------------------------------------------------------------------------
# Synthetic fixture
# ---------------------------------------------------------------------------
# Inputs must contain THEOREM_PROBLEM_PREFIX + one of the THEOREM_INSTRUCTION
# sentinels so _theorem_key() returns a stable hash (see _theorem_key in
# parse_fig_data.py).
_PROVER_INPUT = (
    "Consider the following mathematical problem:\n"
    "Prove that for every prime p > 2, p^2 is odd.\n\n"
    "Solve the problem. Write a complete solution/proof."
)
_PROPOSER_INPUT = (
    "Consider the following mathematical problem:\n"
    "Prove that for every prime p > 2, p^2 is odd.\n\n"
    "Think briefly about how to solve the problem."
)

# Exponential penalty parameters (manifest defaults).
_GAMMA = 0.9999
_BASE_FREE_TOKENS = 2000
# Two prover rollouts: one short (under budget), one long (over budget).
_SHORT_RESP_LEN = 1800  # under free budget, factor = 1.0, exceed = 0
_LONG_RESP_LEN = 2500   # exceeds by 500, factor = 0.9999^500
# Score for the long rollout is 1; for the short rollout is 0 (gives binary mix).
_LONG_FACTOR = _GAMMA ** (_LONG_RESP_LEN - _BASE_FREE_TOKENS)  # ~0.9512
_LONG_EXCEED = _LONG_RESP_LEN - _BASE_FREE_TOKENS              # 500
_SHORT_FACTOR = 1.0
_SHORT_EXCEED = 0


def _prover_row(
    *,
    score: float,
    pre_kl_post_penalty_reward: float,
    length_penalty: float,
    exponential_penalty_factor: float,
    exponential_penalty_exceed_tokens: int,
    response_length: int,
) -> dict:
    """Synthetic prover rollout row with the exponential-penalty fields.

    Note: ``overlong_reward`` is EXPLICITLY OMITTED (exponential-penalty runs do not emit it).
    """
    return {
        "mode_is_prover": 1,
        "mode_is_proposer": 0,
        "input": _PROVER_INPUT,
        "step": 1,
        "score": score,
        # post-penalty pre-KL reward (new canonical scalar for optimized_reward_mean).
        "pre_kl_post_penalty_reward": pre_kl_post_penalty_reward,
        "length_penalty": length_penalty,
        "exponential_penalty_factor": exponential_penalty_factor,
        "exponential_penalty_exceed_tokens": exponential_penalty_exceed_tokens,
        # tags / judge plumbing so the row is treated as a clean judged rollout.
        "candidate_tag_present": 1,
        "proof_tag_present": 1,
        "proof_len_chars": 150,
        "prover_judge_score": int(score >= 0.5),
        "response_length": response_length,
        "total_tokens": response_length + 300,
        "judge_prompt_tokens": 100,
        "judge_completion_tokens": 50,
        # NOTE: overlong_reward and overlong are absent on purpose.
    }


def _proposer_row(
    *,
    score: float,
    pre_kl_post_penalty_reward: float,
    length_penalty: float,
    exponential_penalty_factor: float,
    exponential_penalty_exceed_tokens: int,
    response_length: int,
) -> dict:
    """Synthetic proposer rollout row with the exponential-penalty fields.

    Family detection (_resolve_family) routes proposer rows into the
    proposer family regardless of data_source.
    """
    return {
        "mode_is_proposer": 1,
        "mode_is_prover": 0,
        "input": _PROPOSER_INPUT,
        "step": 1,
        "score": score,
        "pre_kl_post_penalty_reward": pre_kl_post_penalty_reward,
        "length_penalty": length_penalty,
        "exponential_penalty_factor": exponential_penalty_factor,
        "exponential_penalty_exceed_tokens": exponential_penalty_exceed_tokens,
        # proposer-specific decomposition (so the proposer accumulator is well-formed).
        "candidate_tag_present": 1,
        "proposition_tag_present": 1,
        "proof_tag_present": 0,
        "correctness_judge_score": int(score >= 0.5),
        "impact_judge_score": 2,
        "impact_applied": 1.0 if score >= 0.5 else 0.0,
        "proposer_pre_length_score": pre_kl_post_penalty_reward + length_penalty,
        "proposer_reward": pre_kl_post_penalty_reward,
        "response_length": response_length,
        "total_tokens": response_length + 200,
        "judge_prompt_tokens": 80,
        "judge_completion_tokens": 40,
        # NOTE: overlong_reward and overlong are absent on purpose.
    }


_DAPO_FILTER_METRIC = "seq_final_reward"


def _build_fixture(td: Path) -> tuple[Path, Path]:
    """Build a synthetic per-step rollout dump and manifest under ``td``.

    Returns (rollouts_dir, manifest_path).
    """
    rollouts_dir = td / "rollouts" / "train"
    rollouts_dir.mkdir(parents=True)

    rows = [
        # Two prover rollouts (one short clean win, one long over-budget fail).
        _prover_row(
            score=1.0,
            pre_kl_post_penalty_reward=1.0 * _SHORT_FACTOR,   # 1.0
            length_penalty=1.0 - _SHORT_FACTOR,               # 0.0
            exponential_penalty_factor=_SHORT_FACTOR,
            exponential_penalty_exceed_tokens=_SHORT_EXCEED,
            response_length=_SHORT_RESP_LEN,
        ),
        _prover_row(
            score=1.0,
            pre_kl_post_penalty_reward=1.0 * _LONG_FACTOR,    # ~0.9512
            length_penalty=1.0 - _LONG_FACTOR,                # ~0.0488
            exponential_penalty_factor=_LONG_FACTOR,
            exponential_penalty_exceed_tokens=_LONG_EXCEED,
            response_length=_LONG_RESP_LEN,
        ),
        # Two proposer rollouts (parallel: short clean, long over-budget).
        _proposer_row(
            score=1.0,
            pre_kl_post_penalty_reward=0.95,
            length_penalty=1.0 - 0.95,   # 0.05
            exponential_penalty_factor=0.95,
            exponential_penalty_exceed_tokens=500,
            response_length=_LONG_RESP_LEN,
        ),
        _proposer_row(
            score=0.0,
            pre_kl_post_penalty_reward=0.0,
            length_penalty=0.0,
            exponential_penalty_factor=1.0,
            exponential_penalty_exceed_tokens=0,
            response_length=_SHORT_RESP_LEN,
        ),
    ]
    (rollouts_dir / "1.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )

    # Manifest with the exponential-penalty + DAPO-filter knobs.
    manifest_path = td / "manifest.txt"
    manifest_lines = [
        "run_id=test-056-exponential",
        "experiment=test-056-exponential",
        "reward=v7_gptoss120b_two_mode_judge",
        "cohort=v7_twomode",
        "nnodes=2",
        "total_steps=10",
        "train_problem_batch_size=8",
        "gen_problem_batch_size=8",
        "ppo_mini_batch_size=8",
        "actor_lr=1e-6",
        "actor_kl_loss_coef=0.0",
        "save_freq=1",
        "reward_num_workers=4",
        "pi1_init=/path/to/actor",
        "reward_model=/path/to/judge",
        # exponential-penalty knobs (the parser must surface these in fig_data["config"]).
        "exponential_penalty=true",
        f"exponential_penalty_base_free_tokens={_BASE_FREE_TOKENS}",
        f"exponential_penalty_gamma={_GAMMA}",
        "no_drop_except_true_zero_std=True",
        f"dapo_filter_metric={_DAPO_FILTER_METRIC}",
    ]
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    return rollouts_dir, manifest_path


def _run_parser(td: Path) -> dict:
    """Drive the parser over the fixture and return the fig_data dict."""
    rollouts_dir, manifest_path = _build_fixture(td)
    return PF.parse_run(
        run_id="test-056-exponential",
        manifest_path=manifest_path,
        metrics_path=None,
        train_patterns=[str(rollouts_dir / "*.jsonl")],
        val_patterns=[],
        source={},
        start_step=None,
    )


# ---------------------------------------------------------------------------
# Per-step lookup helper
# ---------------------------------------------------------------------------

def _g(per_step: dict, key: str):
    """Return the first per-step value for ``key`` (or None if absent)."""
    return (per_step.get(key) or [None])[0]


# ---------------------------------------------------------------------------
# Pytest-style importable tests
# ---------------------------------------------------------------------------

def test_exponential_penalty_factor_and_exceed_tokens_emitted_for_prover():
    """exponential_penalty_factor + _exceed_tokens must aggregate for prover."""
    with tempfile.TemporaryDirectory() as td:
        data = _run_parser(Path(td))
    ps = data["per_step"]

    factor = _g(ps, "v7__train__prover__exponential_penalty_factor_mean")
    exceed = _g(ps, "v7__train__prover__exponential_penalty_exceed_tokens_mean")
    assert factor is not None, (
        "v7__train__prover__exponential_penalty_factor_mean missing — the "
        "056 parser must aggregate row.get('exponential_penalty_factor') "
        "into a per-step mean for prover rows."
    )
    assert exceed is not None, (
        "v7__train__prover__exponential_penalty_exceed_tokens_mean missing — "
        "the 056 parser must aggregate row.get('exponential_penalty_exceed_tokens') "
        "into a per-step mean for prover rows."
    )
    # NaN guard.
    assert factor == factor, f"exponential_penalty_factor_mean is NaN: {factor}"
    assert exceed == exceed, f"exponential_penalty_exceed_tokens_mean is NaN: {exceed}"


def test_exponential_penalty_factor_and_exceed_tokens_emitted_for_proposer():
    """exponential_penalty_factor + _exceed_tokens must aggregate for proposer too."""
    with tempfile.TemporaryDirectory() as td:
        data = _run_parser(Path(td))
    ps = data["per_step"]

    factor = _g(ps, "v7__train__proposer__exponential_penalty_factor_mean")
    exceed = _g(ps, "v7__train__proposer__exponential_penalty_exceed_tokens_mean")
    assert factor is not None, (
        "v7__train__proposer__exponential_penalty_factor_mean missing — the "
        "056 parser must aggregate proposer rows' exponential_penalty_factor."
    )
    assert exceed is not None, (
        "v7__train__proposer__exponential_penalty_exceed_tokens_mean missing — "
        "the 056 parser must aggregate proposer rows' exponential_penalty_exceed_tokens."
    )
    assert factor == factor, f"proposer exponential_penalty_factor_mean is NaN: {factor}"
    assert exceed == exceed, f"proposer exponential_penalty_exceed_tokens_mean is NaN: {exceed}"


def test_optimized_reward_mean_emitted_without_overlong_reward():
    """optimized_reward_mean must be present and non-NaN even with overlong_reward absent."""
    with tempfile.TemporaryDirectory() as td:
        data = _run_parser(Path(td))
    ps = data["per_step"]

    prov_opt = _g(ps, "v7__train__prover__optimized_reward_mean")
    prop_opt = _g(ps, "v7__train__proposer__optimized_reward_mean")
    assert prov_opt is not None, (
        "v7__train__prover__optimized_reward_mean missing — the 056 parser "
        "must compute optimized_reward_mean from pre_kl_post_penalty_reward, "
        "not from overlong_reward (which is absent in 056 dumps)."
    )
    assert prop_opt is not None, (
        "v7__train__proposer__optimized_reward_mean missing — the 056 parser "
        "must compute optimized_reward_mean from pre_kl_post_penalty_reward "
        "for proposer rows too."
    )
    assert prov_opt == prov_opt, f"prover optimized_reward_mean is NaN: {prov_opt}"
    assert prop_opt == prop_opt, f"proposer optimized_reward_mean is NaN: {prop_opt}"


def test_length_penalty_mean_emitted_for_both_prover_and_proposer():
    """length_penalty_mean must surface for prover rows AND proposer rows."""
    with tempfile.TemporaryDirectory() as td:
        data = _run_parser(Path(td))
    ps = data["per_step"]

    prov_lp = _g(ps, "v7__train__prover__length_penalty_mean")
    prop_lp = _g(ps, "v7__train__proposer__length_penalty_mean")
    assert prov_lp is not None, (
        "v7__train__prover__length_penalty_mean missing — under the 056 "
        "exponential-penalty schema, length_penalty is emitted by the reward "
        "manager for prover rollouts and the parser must aggregate it."
    )
    assert prop_lp is not None, (
        "v7__train__proposer__length_penalty_mean missing — the proposer "
        "length_penalty aggregate is the existing path and must remain emitted."
    )
    assert prov_lp == prov_lp, f"prover length_penalty_mean is NaN: {prov_lp}"
    assert prop_lp == prop_lp, f"proposer length_penalty_mean is NaN: {prop_lp}"


def test_length_penalty_mean_also_emitted_for_prover_family():
    """Family-level prover (prover_original) must also carry length_penalty_mean."""
    with tempfile.TemporaryDirectory() as td:
        data = _run_parser(Path(td))
    ps = data["per_step"]

    fam_lp = _g(ps, "v7__train__family__prover_original__length_penalty_mean")
    assert fam_lp is not None, (
        "v7__train__family__prover_original__length_penalty_mean missing — "
        "the 056 parser must emit length_penalty_mean for the prover family "
        "block too (replay-aware namespace)."
    )
    assert fam_lp == fam_lp, f"family prover_original length_penalty_mean is NaN: {fam_lp}"


def test_config_surfaces_exponential_penalty_manifest_fields():
    """fig_data['config'] must carry the five exponential-penalty manifest knobs."""
    with tempfile.TemporaryDirectory() as td:
        data = _run_parser(Path(td))
    cfg = data.get("config") or {}

    # exponential_penalty: truthy (manifest had "true").
    ep = cfg.get("exponential_penalty")
    assert ep, (
        "config.exponential_penalty missing or falsy — the 056 parser's "
        "_derive_config() must surface manifest.get('exponential_penalty')."
    )

    # exponential_penalty_gamma: numeric, equal to _GAMMA.
    gamma = cfg.get("exponential_penalty_gamma")
    assert gamma is not None, "config.exponential_penalty_gamma missing"
    assert float(gamma) == _GAMMA, (
        f"config.exponential_penalty_gamma={gamma!r}, expected {_GAMMA}"
    )

    # exponential_penalty_base_free_tokens: int, equal to _BASE_FREE_TOKENS.
    base_free = cfg.get("exponential_penalty_base_free_tokens")
    assert base_free is not None, (
        "config.exponential_penalty_base_free_tokens missing"
    )
    assert int(base_free) == _BASE_FREE_TOKENS, (
        f"config.exponential_penalty_base_free_tokens={base_free!r}, "
        f"expected {_BASE_FREE_TOKENS}"
    )

    # no_drop_except_true_zero_std: truthy (manifest had "True").
    ndez = cfg.get("no_drop_except_true_zero_std")
    assert ndez, (
        "config.no_drop_except_true_zero_std missing or falsy — must be "
        "truthy when the manifest has no_drop_except_true_zero_std=True."
    )

    # dapo_filter_metric: string, equal to the provided value.
    dfm = cfg.get("dapo_filter_metric")
    assert dfm == _DAPO_FILTER_METRIC, (
        f"config.dapo_filter_metric={dfm!r}, expected {_DAPO_FILTER_METRIC!r}"
    )


# ---------------------------------------------------------------------------
# Standalone script entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Run every test_* function above with print + assert.

    Returns 0 on success, 1 on any failure. Exit code is honored under the
    script entry point so CI / `python3 ...` produce non-zero status.
    """
    tests = [
        test_exponential_penalty_factor_and_exceed_tokens_emitted_for_prover,
        test_exponential_penalty_factor_and_exceed_tokens_emitted_for_proposer,
        test_optimized_reward_mean_emitted_without_overlong_reward,
        test_length_penalty_mean_emitted_for_both_prover_and_proposer,
        test_length_penalty_mean_also_emitted_for_prover_family,
        test_config_surfaces_exponential_penalty_manifest_fields,
    ]
    n_pass = 0
    n_fail = 0
    failures: list[str] = []
    for t in tests:
        name = t.__name__
        try:
            t()
            print(f"  PASS  {name}")
            n_pass += 1
        except AssertionError as exc:
            print(f"  FAIL  {name}")
            print(f"        {exc}")
            failures.append(f"{name}: {exc}")
            n_fail += 1
        except Exception as exc:  # pragma: no cover - defensive
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            n_fail += 1
    print(f"\n{n_pass} passed, {n_fail} failed")
    if n_fail == 0:
        print("ALL TESTS PASSED")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
