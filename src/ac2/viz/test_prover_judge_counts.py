#!/usr/bin/env python3
"""Regression test for the prover judge-count semantics in parse_fig_data.py.

The per-step prover counts must be MECE: ``attempts`` (judge ran) + ``missing``
(could not run) == ``rows_generated``, with ``failed`` (judge ran but errored) a
subset of attempts. In particular: an empty-proof row counts as MISSING (not
silently dropped), and a row with overlapping http/parse/trunc flags counts as a
single failure (no double counting).

Run::  python3 src/ac2/viz/test_prover_judge_counts.py
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("parse_fig_data", HERE / "parse_fig_data.py")
PF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PF)

_INPUT = ("Consider the following mathematical problem:\nT\n\n"
          "Solve the problem. Write a complete solution/proof.")


def _row(**kw):
    base = dict(mode_is_prover=1, mode_is_proposer=0, input=_INPUT, step=1)
    base.update(kw)
    return base


def main() -> None:
    rows = [
        _row(proof_tag_present=1, proof_len_chars=100, prover_judge_score=1, score=1.0),  # clean success
        _row(proof_tag_present=1, proof_len_chars=0, prover_judge_score=0, score=0.0),    # empty proof -> missing
        _row(proof_tag_present=0, proof_len_chars=0, prover_judge_score=0, score=0.0),    # no tag -> missing
        _row(proof_tag_present=1, proof_len_chars=100, prover_judge_score=0, score=0.0, judge_http_error=1),  # failed
        _row(proof_tag_present=1, proof_len_chars=100, prover_judge_score=0, score=0.0,
             judge_truncated=1, judge_parse_failed=1),  # overlapping flags -> ONE failure
    ]
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "rollouts" / "train"
        d.mkdir(parents=True)
        (d / "1.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        data = PF.parse_run(run_id="t", manifest_path=None, metrics_path=None,
                            train_patterns=[str(d / "*.jsonl")], val_patterns=[],
                            source={}, start_step=None)
    ps = data["per_step"]
    def g(k):
        return (ps.get(k) or [None])[0]
    gen = g("v7__train__prover__rows_generated")
    att = g("v7__train__prover__judge_attempts")
    mis = g("v7__train__prover__judge_missing")
    fail = g("v7__train__prover__judge_failed")
    print(f"  rows_generated={gen} attempts={att} missing={mis} failed={fail}")
    assert gen == 5, gen
    assert att == 3, f"attempts should be 3 (judged rows), got {att}"
    assert mis == 2, f"missing should be 2 (empty-proof + no-tag), got {mis}"
    assert fail == 2, f"failed should be 2 (overlapping flags counted once), got {fail}"
    assert att + mis == gen, "attempts + missing must equal rows_generated (MECE)"
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
