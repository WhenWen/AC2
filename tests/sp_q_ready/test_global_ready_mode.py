"""sp_q_ready_mode=global (global readiness only, the paper's local-readiness ablation):
routing follows the GLOBAL gate, not the per-problem table. These tests build a QHarness
skeleton without seeds/tokenizer (only the routing + state fields) so they run on a login
node in well under a second."""
import pytest

from verl.trainer.ppo.sp_q_readiness import QHarness


class _Replay:
    def __init__(self, qids):
        self._qids = list(qids)

    def replay_plan_for_step(self, step):
        return list(self._qids), 0


def _harness(mode, *, latch=1, ready=None, audit_den=4, qids=("a", "b", "c", "d")):
    h = QHarness.__new__(QHarness)  # skip __init__: no seeds, no tokenizer
    h.ready_mode = mode
    h.ready_global_latch = bool(latch)
    h.ready = dict(ready or {})
    h.global_gate_open = False
    h.global_ready_latched = False
    h.audit_frac_den = audit_den
    h.rng_seed = 804001
    h.mae_window = 5
    h.ready_thresh_global = 0.2
    h._route_cache = None
    h.replay = _Replay(qids)
    return h


def _routes(h, step=7):
    h._route_cache = None
    return h.route_plan_for_step(step)


def test_problem_mode_is_the_lineage_behaviour():
    h = _harness("problem", ready={"a": True, "c": True}, audit_den=0)
    r = _routes(h)
    assert r == {0: "short", 1: "full", 2: "short", 3: "full"}
    # the global flags are tracked but do NOT decide anything here
    h.global_gate_open = h.global_ready_latched = True
    assert _routes(h) == r


def test_global_mode_ignores_the_per_problem_table():
    h = _harness("global", ready={"a": True, "c": True}, audit_den=0)
    # gate never opened -> nothing is ready, even flagged problems
    assert set(_routes(h).values()) == {"full"}
    # gate open -> every slot is ready, flagged or not
    h.global_gate_open = h.global_ready_latched = True
    assert set(_routes(h).values()) == {"short"}
    assert all(h.is_ready(q) for q in ("a", "b", "zzz-never-seen"))


def test_global_mode_keeps_the_audit_draw():
    h = _harness("global", audit_den=4, qids=[f"q{i}" for i in range(16)])
    h.global_gate_open = h.global_ready_latched = True
    r = _routes(h)
    assert sum(v == "audit" for v in r.values()) == 4  # ceil(16/4)
    assert sum(v == "short" for v in r.values()) == 12
    assert "full" not in r.values()


def test_latch_vs_live_gate():
    latched = _harness("global", latch=1, audit_den=0)
    live = _harness("global", latch=0, audit_den=0)
    for h in (latched, live):
        h.global_ready_latched = True   # opened once ...
        h.global_gate_open = False      # ... and closed again
    assert set(_routes(latched).values()) == {"short"}  # monotone: stays ready
    assert set(_routes(live).values()) == {"full"}       # live: follows the gate


def test_flags_from_a_legacy_state_derive_from_the_saved_window():
    # parent checkpoint written before the fields existed: derive from mae_tail
    closed_tail = [(s, [0.25, 0.15, 0.22]) for s in range(15, 20)]   # pooled 0.2067 > 0.2
    open_tail = [(s, [0.10, 0.15, 0.22]) for s in range(15, 20)]     # pooled 0.1567 < 0.2
    short_tail = open_tail[:3]                                        # window incomplete
    f = QHarness._global_flags_from_state
    assert f({}, mae_tail=closed_tail, thresh=0.2, window=5) == (False, False)
    assert f({}, mae_tail=open_tail, thresh=0.2, window=5) == (True, True)
    assert f({}, mae_tail=short_tail, thresh=0.2, window=5) == (False, False)
    # a state that carries the fields wins over the window
    st = {"global_gate_open": False, "global_ready_latched": True}
    assert f(st, mae_tail=open_tail, thresh=0.2, window=5) == (False, True)


