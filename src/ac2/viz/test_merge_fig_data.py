#!/usr/bin/env python3
"""Regression tests for merge_fig_data.py (stdlib only).

Covers:
- the strict train-rollout overlap guard (reject a disjoint parse-output that
  would leave the boundary unverified / introduce a gap), and that a real
  one-step overlap still merges;
- n / n_step_files being recomputed from the MERGED per_step (full history),
  not just the latest parse delta.

Run::  python3 src/ac2/viz/test_merge_fig_data.py
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MERGE = HERE / "merge_fig_data.py"


def _fig(steps, gens, n, nfiles, overlap=None):
    doc = {
        "schema": "riemann_v7_fig_data", "schema_version": 1,
        "per_step": {"steps": steps, "v7__train__mode_mix__total_generated": gens},
        "train_jsonl": {"n": n, "n_step_files": nfiles, "examples": {}, "pooled": []},
        "val_jsonl": {"n": 0, "n_step_files": 0, "examples": {}, "pooled": []},
        "warnings": [],
    }
    if overlap is not None:
        doc["refresh"] = {"overlap_step": overlap}
    return doc


def _write(p, doc):
    Path(p).write_text(json.dumps(doc))


def _run(parse, canon):
    return subprocess.run([sys.executable, str(MERGE), "--parse-output", parse,
                           "--canonical", canon], capture_output=True, text=True)


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        canon = _fig([9, 10], [800, 820], 1620, 2)

        print("P1 overlap guard:")
        # No shared train-rollout step -> reject, canonical untouched.
        c1 = tmp / "c1.json"; _write(c1, canon)
        _write(tmp / "gap.json", _fig([11], [830], 830, 1, overlap=10))
        r = _run(str(tmp / "gap.json"), str(c1))
        _assert(r.returncode != 0, "disjoint parse-output (gap) is rejected")
        _assert("no train-rollout overlap" in (r.stdout + r.stderr), "error explains the gap")
        _assert(json.loads(c1.read_text())["per_step"]["steps"] == [9, 10],
                "canonical left UNCHANGED on reject")

        # Real one-step overlap (re-includes step 10) -> merges.
        print("P1 overlap present + P2a counts:")
        c2 = tmp / "c2.json"; _write(c2, canon)
        _write(tmp / "ok.json", _fig([10, 11], [820, 830], 830, 1, overlap=10))
        r = _run(str(tmp / "ok.json"), str(c2))
        _assert(r.returncode == 0, "overlapping parse-output merges cleanly")
        m = json.loads(c2.read_text())
        _assert(m["per_step"]["steps"] == [9, 10, 11], "steps appended (9,10,11)")
        _assert(m["train_jsonl"]["n"] == 2450,
                f"train n = full-history 800+820+830=2450 (not parse-only 830), got {m['train_jsonl']['n']}")
        _assert(m["train_jsonl"]["n_step_files"] == 3,
                f"n_step_files = 3 merged steps (not parse-only 1), got {m['train_jsonl']['n_step_files']}")

        # A declared overlap_step that isn't actually shared is rejected.
        print("P1 declared-overlap sanity:")
        c3 = tmp / "c3.json"; _write(c3, canon)
        _write(tmp / "badov.json", _fig([10, 11], [820, 830], 830, 1, overlap=7))
        r = _run(str(tmp / "badov.json"), str(c3))
        _assert(r.returncode != 0, "overlap_step not present in both files is rejected")

        # Fresh canonical (none yet): parse-output becomes canonical (no guard).
        print("fresh canonical:")
        c4 = tmp / "missing.json"  # does not exist
        _write(tmp / "fresh.json", _fig([0, 1], [100, 110], 210, 2))
        r = _run(str(tmp / "fresh.json"), str(c4))
        _assert(r.returncode == 0 and c4.exists(), "fresh experiment seeds canonical from parse-output")

        print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
