"""prefix_cuts_td — the TD slot's n-cut draw.

Contract: a pinned permutation over the cut grid covers every grid point once before any
repeats (distinctness when the grid allows it, full coverage when it does not), the draw
is deterministic in (seed, step, rslot, entry_id), the existing "cut" stream is never
consumed, and the fallback/clamp behavior matches prefix_cut.

CPU only. The harness is not constructed: prefix_cuts_td only touches cut_low/cut_high/
cut_grain/rng_seed and the two metric lists, so a SimpleNamespace stands in.
"""
from types import SimpleNamespace

from verl.trainer.ppo.sp_replay import ReplayHarness


def _stub(cut_low=0.0, cut_high=0.90, cut_grain=10000, rng_seed=831001):
    return SimpleNamespace(
        cut_low=cut_low, cut_high=cut_high, cut_grain=cut_grain, rng_seed=rng_seed,
        _last_prefix_lens=[], _last_cut_fracs=[],
    )


def _entry(t, entry_id="e0"):
    return {"response_token_ids": list(range(t)), "entry_id": entry_id}


def _cuts(stub, entry, step=7, rslot=3, n=16, grain=None):
    return ReplayHarness.prefix_cuts_td(stub, entry, step, rslot, n, grain=grain)


def test_distinct_when_grid_is_large_enough():
    # t=30000 at grain 1000 over [0, 0.9t]: grid = {0, 1000, ..., 27000} = 28 points >= 16
    cuts = _cuts(_stub(cut_grain=1000), _entry(30000), n=16)
    assert len(cuts) == 16
    assert len(set(cuts)) == 16, cuts
    assert all(c % 1000 == 0 for c in cuts), cuts
    assert all(0 <= c <= 27000 for c in cuts), cuts


def test_full_coverage_before_any_repeat_when_grid_is_small():
    # t=30000 at grain 10000: grid = {0, 10000, 20000} -> 16 cuts must cover all 3 points
    # and each point appears either 5 or 6 times (16 = 3*5 + 1).
    cuts = _cuts(_stub(cut_grain=10000), _entry(30000), n=16)
    assert set(cuts) == {0, 10000, 20000}, cuts
    counts = {c: cuts.count(c) for c in set(cuts)}
    assert sorted(counts.values()) == [5, 5, 6], counts
    # the first 3 draws are the full grid (permutation prefix)
    assert set(cuts[:3]) == {0, 10000, 20000}, cuts[:3]


def test_deterministic_and_keyed():
    a = _cuts(_stub(), _entry(50000), step=11, rslot=5, n=16)
    b = _cuts(_stub(), _entry(50000), step=11, rslot=5, n=16)
    assert a == b
    c = _cuts(_stub(), _entry(50000), step=11, rslot=6, n=16)
    d = _cuts(_stub(), _entry(50000, entry_id="e9"), step=11, rslot=5, n=16)
    assert a != c or a != d  # a different slot or entry re-keys the draw


def test_grain_override_and_clamp():
    # explicit grain overrides the stub's; every cut leaves >= 1 token to generate toward
    stub = _stub(cut_grain=10000)
    cuts = _cuts(stub, _entry(1500), n=8, grain=1000)
    assert all(c <= 1499 for c in cuts), cuts
    # t=1500, [0, 1350] at grain 1000 -> grid {0, 1000}
    assert set(cuts) <= {0, 1000}, cuts
    assert set(cuts) == {0, 1000}


def test_short_trajectory_fallback_matches_prefix_cut():
    # cut_low high enough that no grain multiple fits in [lo, hi]: fall back to the
    # largest multiple respecting cut_high (0 here), exactly like prefix_cut.
    cuts = _cuts(_stub(cut_low=0.5, cut_high=0.6, cut_grain=10000), _entry(800), n=4)
    assert cuts == [0, 0, 0, 0], cuts


def test_metrics_appended_per_cut():
    stub = _stub(cut_grain=1000)
    _cuts(stub, _entry(30000), n=16)
    assert len(stub._last_prefix_lens) == 16
    assert len(stub._last_cut_fracs) == 16
