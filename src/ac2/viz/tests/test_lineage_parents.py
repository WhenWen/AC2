#!/usr/bin/env python3
"""Regression tests for the fig_data parent chain: single-seed promotion + immutable
`parents` across routine refreshes (stdlib only).

The contract this pins:
- Each canonical fig_data carries top-level `parents` (list[dict]) — the
  chronological ancestor chain. Empty for root runs.
- Seeding a child from one parent P promotes P's run_id (with P's OWN range)
  into the parents list and clears `run_id`.
- The OWN range is what P contributed beyond its own ancestors.
- Routine strict-overlap merges never mutate `parents` (canonical-wins).

Run::  python3 src/ac2/viz/tests/test_lineage_parents.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_VIZ = _HERE.parent  # the viz package dir (modules are siblings one level up from tests/)

spec = importlib.util.spec_from_file_location(
    "merge_fig_data", _VIZ / "merge_fig_data.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

dc_spec = importlib.util.spec_from_file_location(
    "dashboard_common", _VIZ / "dashboard_common.py")
DC = importlib.util.module_from_spec(dc_spec)
dc_spec.loader.exec_module(DC)


# ---------------------------------------------------------------------------
# _seed_own_range — the range a seed CONTRIBUTED beyond its own ancestors.
# ---------------------------------------------------------------------------

def test_root_own_range():
    doc = {"run_id": "G", "per_step": {"steps": list(range(0, 61))}}
    assert M._seed_own_range(doc) == (0, 60)


def test_one_ancestor_own_range():
    # P had grandparent G covering 0..60; P's own contribution starts at 61.
    doc = {
        "run_id": "P",
        "parents": [{"run_id": "G", "start_step": 0, "end_step": 60}],
        "per_step": {"steps": list(range(0, 96))},
    }
    assert M._seed_own_range(doc) == (61, 95)


def test_two_ancestors_own_range():
    doc = {
        "run_id": "R",
        "parents": [
            {"run_id": "G", "start_step": 0, "end_step": 60},
            {"run_id": "P", "start_step": 61, "end_step": 95},
        ],
        "per_step": {"steps": list(range(0, 130))},
    }
    assert M._seed_own_range(doc) == (96, 129)


def test_empty_steps_returns_none():
    doc = {"run_id": "X", "per_step": {"steps": []}}
    assert M._seed_own_range(doc) is None


# ---------------------------------------------------------------------------
# _promote_seed_into_parents — the parent-promotion mutator.
# ---------------------------------------------------------------------------

def test_promote_root_appends_self():
    doc = {"run_id": "G", "parents": [], "per_step": {"steps": list(range(0, 61))}}
    M._promote_seed_into_parents(doc)
    assert doc["run_id"] == ""
    assert doc["parents"] == [{"run_id": "G", "start_step": 0, "end_step": 60}]


def test_promote_chained_preserves_prior():
    doc = {
        "run_id": "P",
        "parents": [{"run_id": "G", "start_step": 0, "end_step": 60}],
        "per_step": {"steps": list(range(0, 96))},
    }
    M._promote_seed_into_parents(doc)
    assert doc["run_id"] == ""
    assert doc["parents"] == [
        {"run_id": "G", "start_step": 0, "end_step": 60},
        {"run_id": "P", "start_step": 61, "end_step": 95},
    ]


def test_promote_no_run_id_is_noop():
    # A seed without a run_id (e.g. an already-promoted half-state) should not
    # be re-promoted.
    doc = {"run_id": "", "parents": [{"run_id": "G", "start_step": 0, "end_step": 5}],
           "per_step": {"steps": [6, 7]}}
    M._promote_seed_into_parents(doc)
    assert doc["parents"] == [{"run_id": "G", "start_step": 0, "end_step": 5}]
    assert doc["run_id"] == ""


# ---------------------------------------------------------------------------
# _merge_parents — canonical-wins (the parent chain is immutable across routine merges).
# ---------------------------------------------------------------------------

def test_merge_canonical_wins():
    c = [{"run_id": "G", "start_step": 0, "end_step": 60}]
    p = []
    assert M._merge_parents(c, p) == c


def test_merge_parse_output_used_when_canonical_missing():
    p = [{"run_id": "G", "start_step": 0, "end_step": 60}]
    assert M._merge_parents(None, p) == p


def test_merge_empty_when_both_missing():
    assert M._merge_parents(None, None) == []


def test_merge_canonical_wins_even_when_parse_has_different():
    # A parse-output that mistakenly carries a non-empty parents list must NOT
    # overwrite the canonical's parent chain. This is the immutability guarantee.
    c = [{"run_id": "G", "start_step": 0, "end_step": 60}]
    p = [{"run_id": "MALICIOUS", "start_step": 0, "end_step": 999}]
    assert M._merge_parents(c, p) == c


# ---------------------------------------------------------------------------
# DC.lineage_boundaries — renderer-side marker generation.
# ---------------------------------------------------------------------------

class _MockF:
    def __init__(self, doc):
        self.d = doc
        self.steps = doc["per_step"]["steps"]
        self.dashboard_annotations = {}


def test_lineage_boundaries_empty_for_root():
    F = _MockF({"per_step": {"steps": list(range(0, 30))}, "parents": []})
    assert DC.lineage_boundaries(F) == []


def test_lineage_boundaries_missing_parents_field():
    F = _MockF({"per_step": {"steps": list(range(0, 30))}})
    assert DC.lineage_boundaries(F) == []


def test_lineage_boundaries_one_per_parent_BETWEEN_segments():
    # Marker x is end_step + 0.5 so the line falls visually BETWEEN the last
    # point of parent i (at end_step) and the first point of parent i+1
    # (at start_step = end_step + 1).
    F = _MockF({
        "per_step": {"steps": list(range(0, 130))},
        "parents": [
            {"run_id": "fineproofs-v7-gg-parent", "start_step": 0, "end_step": 60},
            {"run_id": "fineproofs-v7-parent-P", "start_step": 61, "end_step": 95},
        ],
    })
    bs = DC.lineage_boundaries(F)
    assert len(bs) == 2
    assert bs[0]["x"] == 60.5
    assert bs[1]["x"] == 95.5
    # Labels strip the common run-id prefix.
    assert bs[0]["label"] == "gg-parent"
    assert bs[1]["label"] == "parent-P"


# ---------------------------------------------------------------------------
# Contiguity contract — every transition must satisfy
# parents[i+1].start_step == parents[i].end_step + 1.
# ---------------------------------------------------------------------------

def test_contiguity_passes_for_contiguous_chain():
    chain = [
        {"run_id": "G", "start_step": 0, "end_step": 60},
        {"run_id": "P", "start_step": 61, "end_step": 95},
        {"run_id": "Q", "start_step": 96, "end_step": 120},
    ]
    # Should not raise.
    M._validate_parents_contiguity(chain, context="test")


def test_contiguity_passes_for_single_or_empty():
    M._validate_parents_contiguity([], context="test")
    M._validate_parents_contiguity(
        [{"run_id": "X", "start_step": 5, "end_step": 7}], context="test")


def test_contiguity_raises_on_GAP():
    # P.start_step = 70 but G.end_step = 60 -> gap of 9 steps (61..69 missing)
    chain = [
        {"run_id": "G", "start_step": 0, "end_step": 60},
        {"run_id": "P", "start_step": 70, "end_step": 95},
    ]
    try:
        M._validate_parents_contiguity(chain, context="gap-test")
    except SystemExit as e:
        assert "GAP" in str(e), f"expected GAP error, got: {e}"
        assert "70" in str(e) and "60" in str(e)
        return
    raise AssertionError("expected SystemExit for non-contiguous chain")


def test_contiguity_raises_on_OVERLAP():
    # P starts at 55 but G ended at 60 -> overlap of 6 steps
    chain = [
        {"run_id": "G", "start_step": 0, "end_step": 60},
        {"run_id": "P", "start_step": 55, "end_step": 95},
    ]
    try:
        M._validate_parents_contiguity(chain, context="overlap-test")
    except SystemExit as e:
        assert "OVERLAP" in str(e), f"expected OVERLAP error, got: {e}"
        return
    raise AssertionError("expected SystemExit for overlapping chain")


def test_contiguity_raises_on_malformed_step():
    chain = [
        {"run_id": "G", "start_step": 0, "end_step": "sixty"},  # non-int
        {"run_id": "P", "start_step": 61, "end_step": 95},
    ]
    try:
        M._validate_parents_contiguity(chain, context="malformed-test")
    except SystemExit as e:
        assert "malformed" in str(e), f"expected malformed error, got: {e}"
        return
    raise AssertionError("expected SystemExit for malformed chain")


def test_promote_refuses_seed_with_bad_existing_chain():
    # Seed already has a non-contiguous parents list — refuse to promote.
    doc = {
        "run_id": "P",
        "parents": [
            {"run_id": "A", "start_step": 0, "end_step": 10},
            {"run_id": "B", "start_step": 20, "end_step": 30},  # gap
        ],
        "per_step": {"steps": list(range(0, 50))},
    }
    try:
        M._promote_seed_into_parents(doc)
    except SystemExit:
        return  # expected
    raise AssertionError("expected SystemExit for seed with non-contiguous parents")


def test_merge_parents_raises_on_corrupt_chain():
    # If the canonical was hand-edited into a bad state, the merger refuses
    # to use it. This is the routine-refresh defense.
    c = [
        {"run_id": "G", "start_step": 0, "end_step": 60},
        {"run_id": "P", "start_step": 100, "end_step": 130},  # gap
    ]
    try:
        M._merge_parents(c, None)
    except SystemExit:
        return  # expected
    raise AssertionError("expected SystemExit when canonical's `parents` is bad")


def test_short_run_id_truncates_long_labels():
    long = "fineproofs-v7-" + "a" * 50
    assert DC._short_run_id(long, keep=10) == "aaaaaaaaa…"


# ---------------------------------------------------------------------------
# Runner.
# ---------------------------------------------------------------------------

def _run():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = []
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed.append(t.__name__)
            print(f"FAIL  {t.__name__}: {e!r}")
        except Exception as e:
            failed.append(t.__name__)
            print(f"ERROR {t.__name__}: {e!r}")
    print()
    print(f"{len(tests) - len(failed)}/{len(tests)} passed")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(_run())
