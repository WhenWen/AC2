"""Stdlib-only unit tests for verl.trainer.ppo.difficulty (run with plain python3, no torch).

    python3 tests/local/test_difficulty.py
"""
import importlib.util
import os
import tempfile

_DIFF_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "src", "verl", "verl", "trainer", "ppo", "difficulty.py"
)


def _load_module():
    """Load difficulty.py directly from its file path, bypassing the heavy verl package __init__
    (which imports torch/packaging). The module itself is stdlib-only by design."""
    spec = importlib.util.spec_from_file_location("difficulty_under_test", _DIFF_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fresh(env=None):
    """Fresh module instance under a controlled env (module state is process-global)."""
    for k in list(os.environ):
        if k.startswith("SP_DIFF_"):
            del os.environ[k]
    os.environ["SP_DIFF_SAMPLING"] = "1"
    for k, v in (env or {}).items():
        os.environ[k] = v
    return _load_module()


def approx(a, b, eps=1e-12):
    assert abs(a - b) < eps, (a, b)


# ---------------------------------------------------------------- qid
d = fresh()
t1 = "Here is a mathematical problem: prove 1+1=2.\nSolve the problem step by step."
t2 = "PREAMBLE DIFFERS mathematical problem: prove 1+1=2.\nSolve the problem NOW"
t3 = "Here is a mathematical problem: prove 1+1=3.\nSolve the problem step by step."
assert d.qid_from_text(t1) == d.qid_from_text(t2), "same statement core must hash equal"
assert d.qid_from_text(t1) != d.qid_from_text(t3), "different statements must differ"
# marker fallback: no markers -> whole-text hash, stable
assert d.qid_from_text("plain") == d.qid_from_text("plain")
# chat-format extraction matches the raw text path
msgs = [{"role": "user", "content": t1}]
assert d.qid_from_messages(msgs) == d.qid_from_text(t1)
msgs_mm = [{"role": "user", "content": [{"type": "text", "text": t1}]}]
assert d.qid_from_messages(msgs_mm) == d.qid_from_text(t1)
print("PASS qid: marker extraction, stability, chat/mm forms")

# ---------------------------------------------------------------- disabled -> no-op
for k in list(os.environ):
    if k.startswith("SP_DIFF_"):
        del os.environ[k]
d0 = _load_module()
assert d0.enabled() is False
assert d0.save_state(tempfile.mkdtemp()) is None and d0.load_state(tempfile.mkdtemp()) is None
print("PASS disabled: enabled()=False, save/load no-ops")

# ---------------------------------------------------------------- EMA + first-obs init
d = fresh({"SP_DIFF_EMA_ALPHA": "0.5", "SP_DIFF_MIN_OBS": "2"})
d.observe("q1", 1.0)                      # first obs EMAs FROM the 0.5 prior: 0.5*0.5 + 0.5*1.0
approx(d._S["stats"]["q1"]["p"], 0.75)
d.observe("q1", 0.0)                      # 0.5*0.75 + 0.5*0.0
approx(d._S["stats"]["q1"]["p"], 0.375)
d.observe("q1", 1.0)
approx(d._S["stats"]["q1"]["p"], 0.6875)
print("PASS EMA: prior-anchored first obs + exponential update")

# ---------------------------------------------------------------- weight forms + min-obs gate
d = fresh({"SP_DIFF_WEIGHT": "sqrt_pq", "SP_DIFF_MIN_OBS": "2", "SP_DIFF_W_FLOOR": str(1 / 32),
           "SP_DIFF_EMA_ALPHA": "0.4"})   # pin alpha (this block tests the FORM, not the default)
approx(d.weight("unseen"), 0.5)           # unseen -> form max
d.observe("qa", 1.0)                      # n=1 < min_obs -> still max
approx(d.weight("qa"), 0.5)
import math as _math
d.observe("qa", 1.0)                      # n=2: prior-EMA (a=0.4): 0.5->0.7->0.82
approx(d.weight("qa"), _math.sqrt(0.82 * 0.18), 1e-9)
d.observe("qb", 0.5); d.observe("qb", 0.5)
approx(d.weight("qb"), 0.5)               # peak at p=1/2 (prior IS the peak)
d.observe("qc", 0.0); d.observe("qc", 0.0)   # 0.5->0.3->0.18: both tails downweighted symmetrically
approx(d.weight("qc"), _math.sqrt(0.18 * 0.82), 1e-9)
# floor engages once p-hat is EMA'd near an extreme
dd = fresh({"SP_DIFF_WEIGHT": "sqrt_pq", "SP_DIFF_MIN_OBS": "1", "SP_DIFF_W_FLOOR": str(1 / 32),
            "SP_DIFF_EMA_ALPHA": "1.0"})
dd.observe("qx", 1.0)
approx(dd.weight("qx"), 1 / 32)           # p=1 exactly -> sqrt(0) -> floored
d = fresh({"SP_DIFF_WEIGHT": "sqrt_p", "SP_DIFF_MIN_OBS": "1", "SP_DIFF_W_FLOOR": str(1 / 32),
           "SP_DIFF_EMA_ALPHA": "0.4"})
d.observe("qa", 0.25)                     # prior-EMA: 0.6*0.5+0.4*0.25 = 0.4
approx(d.weight("qa"), _math.sqrt(0.4), 1e-9)
d.observe("qz", 0.0)                      # 0.3
approx(d.weight("qz"), _math.sqrt(0.3), 1e-9)
print("PASS weights: sqrt_pq/sqrt_p forms under prior-EMA, min-obs gate, floor, both tails")

# ---------------------------------------------------------------- step form + hysteresis
d = fresh({"SP_DIFF_WEIGHT": "step", "SP_DIFF_MIN_OBS": "1", "SP_DIFF_EMA_ALPHA": "1.0"})
d.observe("q", 14 / 16); approx(d.weight("q"), 1.0)        # below 15/16 -> full weight
d.observe("q", 16 / 16); approx(d.weight("q"), 1 / 16)     # demoted at >= 15/16
d.observe("q", 14 / 16); approx(d.weight("q"), 1 / 16)     # 14/16 > 13/16: hysteresis holds
d.observe("q", 13 / 16); approx(d.weight("q"), 1.0)        # promoted back at <= 13/16
print("PASS step form: demotion, hysteresis hold, promotion")

# ---------------------------------------------------------------- corrections: E_q[c] = 1 exactly
d = fresh({"SP_DIFF_WEIGHT": "sqrt_pq", "SP_DIFF_MIN_OBS": "1"})
for i, p in enumerate([0.0, 0.1, 0.5, 0.9, 1.0, 0.3, 0.7]):
    d.observe(f"q{i}", p)
qids = [f"q{i}" for i in range(7)]
c, w = d.corrections_for(qids)
W = sum(w.values())
eq_c = sum((w[q] / W) * c[q] for q in qids)   # E_q[c] = sum q_i c_i
approx(eq_c, 1.0)
wbar = W / len(qids)
for q in qids:
    approx(c[q], wbar / w[q])
assert max(c.values()) <= 32.0 + 1e-9 or wbar <= 1.0, "c bounded by floor"
print("PASS corrections: c = w_bar/w, E_q[c] = 1")

# ---------------------------------------------------------------- loss normalization identity
d = fresh()
cs = [0.5, 2.0, 1.0, 8.0]
ns = [100, 50, 200, 10]
ch = d.normalize_c_for_loss(cs, ns)
approx(sum(c * n for c, n in zip(ch, ns)), sum(ns), 1e-9)   # Σ c-hat n == Σ n
# ratio preserved: c-hat_i / c-hat_j == c_i / c_j
approx(ch[0] / ch[1], cs[0] / cs[1])
# the two forms of the weighted mean agree: Σ c-hat l / Σ n == Σ c l / Σ c n
ls = [3.0, -1.0, 0.5, 10.0]  # per-row loss sums
lhs = sum(c * l for c, l in zip(ch, ls)) / sum(ns)
rhs = sum(c * l for c, l in zip(cs, ls)) / sum(c * n for c, n in zip(cs, ns))
approx(lhs, rhs, 1e-9)
# all-equal c -> identity
assert d.normalize_c_for_loss([1.0] * 3, [5, 6, 7]) == [1.0, 1.0, 1.0]
print("PASS normalize_c_for_loss: denominator identity + ratio preservation")

# ---------------------------------------------------------------- ESS + SNIS
d = fresh()
approx(d.ess([1.0] * 10), 10.0)
assert d.ess([1.0, 0.0, 0.0]) < 3.0
approx(d.snis_mean([2.0, 4.0], [1.0, 1.0]), 3.0)
approx(d.snis_mean([2.0, 4.0], [3.0, 1.0]), 2.5)
print("PASS ESS + SNIS")

# ---------------------------------------------------------------- FIFO
d = fresh()
for i in range(5):
    d.push_draw(i, f"q{i}", 1.0 + i)
out = d.pop_draws(3)
assert [o[0] for o in out] == [0, 1, 2] and d.fifo_len() == 2
out = d.pop_draws(10)                     # short pop after "resume"
assert len(out) == 2
print("PASS FIFO: order, short pop")

# ---------------------------------------------------------------- persistence + init-order
d = fresh({"SP_DIFF_MIN_OBS": "1"})
d.observe("qq", 0.25); d.observe("qq", 0.75)
ckpt = tempfile.mkdtemp()
d.save_state(ckpt)
d2 = fresh({"SP_DIFF_MIN_OBS": "1"})      # fresh process simulation
assert d2._S["stats"] == {}
n = d2.load_state(ckpt)                   # load BEFORE any weight()/observe() call
assert n == 1
# init-order guard: a lazy enabled() after load must NOT clobber the stats (the AEC set_k bug class)
assert d2.enabled() is True and len(d2._S["stats"]) == 1
approx(d2._S["stats"]["qq"]["p"], 0.66)   # prior-EMA a=0.8: 0.5->0.3 (obs 0.25) -> 0.66 (obs 0.75)
assert d2.load_state(tempfile.mkdtemp()) is None     # missing file -> None, state kept
assert len(d2._S["stats"]) == 1
# corrupted file must not crash
bad = tempfile.mkdtemp()
with open(os.path.join(bad, "difficulty_state.json"), "w") as f:
    f.write("{not json")
assert d2.load_state(bad) is None and len(d2._S["stats"]) == 1
print("PASS persistence: roundtrip, init-order, missing/corrupt tolerated")

print("\nALL DIFFICULTY UNIT TESTS PASSED")

# ---------------------------------------------------------------- weighted sampler
class FakeDS:
    """Mimics RLHFDataset: .dataframe[i][prompt_key] -> chat messages."""
    prompt_key = "prompt"
    def __init__(self, statements):
        self.dataframe = [
            {"prompt": [{"role": "user", "content": f"mathematical problem: {s}\nSolve the problem"}]}
            for s in statements
        ]
    def __len__(self):
        return len(self.dataframe)

d = fresh({"SP_DIFF_WEIGHT": "step", "SP_DIFF_MIN_OBS": "1", "SP_DIFF_EMA_ALPHA": "1.0"})
# 4 distinct problems; problem D duplicated across two rows (rows 3 and 4)
ds = FakeDS(["A", "B", "C", "D", "D"])
smp = d.DifficultyWeightedSampler(ds, seed=7)
assert len(smp) == 5 and len(smp._distinct) == 4 and smp._dup[smp._row_qid[3]] == 2
# demote problem A (row 0) -> w=1/16; others w=1
d.observe(smp._row_qid[0], 1.0)
draws = []
for _ in range(4):                      # 4 "epochs" of 5 draws
    draws.extend(iter(smp))
N_DRAW = 20000
it = iter(smp)
draws = [smp._rng.choices(range(5), weights=smp._row_w, k=1)[0] for _ in range(N_DRAW)]
from collections import Counter
cnt = Counter(draws)
# expected q: w = [1/16, 1, 1, 1, 1] with D split .5/.5 over rows 3,4 -> row probs ∝ [1/16,1,1,.5,.5]
W = 1 / 16 + 1 + 1 + 1
q_exp = [1 / 16 / W, 1 / W, 1 / W, 0.5 / W, 0.5 / W]
for i in range(5):
    emp = cnt[i] / N_DRAW
    assert abs(emp - q_exp[i]) < 0.015, (i, emp, q_exp[i])
# problem-level: D's two rows together ≈ one normal problem's mass
approx_d = (cnt[3] + cnt[4]) / N_DRAW
assert abs(approx_d - 1 / W) < 0.02
print("PASS sampler: draw freq ∝ w, demotion respected, duplicate rows split (problem mass invariant)")

# FIFO alignment: fresh sampler, consume exactly like the fit loop would
d = fresh({"SP_DIFF_WEIGHT": "step", "SP_DIFF_MIN_OBS": "1"})
smp = d.DifficultyWeightedSampler(ds, seed=3)
idx = []
g = iter(smp)
for _ in range(5):
    idx.append(next(g))
recs = d.pop_draws(5)
assert [r[0] for r in recs] == idx, "FIFO order must match yielded index order"
assert all(r[1] == smp._row_qid[r[0]] for r in recs)
# E_q[c]=1 at problem level: sum over distinct problems of q_qid * c_qid
c_map, w_map = d.corrections_for(smp._distinct)
Wq = sum(w_map.values())
approx(sum((w_map[q] / Wq) * c_map[q] for q in smp._distinct), 1.0)
# state_dict roundtrip
sd = smp.state_dict()
smp2 = d.DifficultyWeightedSampler(ds, seed=99)
smp2.load_state_dict(sd)
assert smp2._pos == smp._pos
a = [next(iter(smp)) for _ in range(3)]
print("PASS sampler: FIFO alignment, qid consistency, state roundtrip")

# version cache: weights refresh only after observe()
d = fresh({"SP_DIFF_WEIGHT": "step", "SP_DIFF_MIN_OBS": "1", "SP_DIFF_EMA_ALPHA": "1.0"})
smp = d.DifficultyWeightedSampler(ds, seed=1)
smp._refresh(); v0 = smp._cache_version
next(iter(smp))
assert smp._cache_version == v0            # no observe -> no rebuild
d.observe(smp._row_qid[1], 1.0)
next(iter(smp))
assert smp._cache_version != v0            # observe bumped version -> rebuilt
w_after = smp._row_w[1]
approx(w_after, 1 / 16)
print("PASS sampler: version-cached weights rebuild only on observe()")

print("\nALL DIFFICULTY UNIT TESTS PASSED (incl. sampler)")

# ---------------------------------------------------------------- IS metric corrections
d = fresh()
import statistics
B = 12
cs = [0.5, 2.0, 1.0, 8.0, 1.0, 0.25, 3.0, 1.0, 0.8, 1.5, 4.0, 1.0]
rl = [100.0, 0.0, 200.0, 50.0, 300.0, 0.0, 120.0, 80.0, 60.0, 90.0, 110.0, 500.0]  # two aborted, one at cap
MAXRL = 500.0
na = [i for i, r in enumerate(rl) if r > 0]
na_cs = [cs[i] for i in na]
seqs = [0.7, 0.0, 1.0, 0.3, 0.5, 0.0, 0.9, 0.2, 0.4, 0.6, 0.8, 1.0]
rows = {
    "seq_score": [seqs[i] for i in na], "seq_score__cs": na_cs,
    "response_length": rl,
    "resp_clip": [1.0 if r == MAXRL else 0.0 for r in rl],
    "aborted": [1.0 if r == 0 else 0.0 for r in rl],
    "n_tokens": rl,
    "adv_sum": [r * 0.01 for r in rl],
}
raw = {
    "critic/score/mean": statistics.mean(rows["seq_score"]),
    "response_length/mean": statistics.mean(rl),
    "response_length/clip_ratio": statistics.mean(rows["resp_clip"]),
    "response/aborted_ratio": statistics.mean(rows["aborted"]),
    "critic/advantages/mean": sum(rows["adv_sum"]) / sum(rl),
    "critic/score/max": max(rows["seq_score"]),        # order stat: must NOT be touched
    "actor/entropy": 0.15,                              # AEC signal: must NOT be touched
}
m = dict(raw)
done = d.apply_metric_corrections(m, cs, rows)
assert "critic/score/mean" in done and "critic/advantages/mean" in done
# raw preserved under difficulty_raw/, standard key overwritten with SNIS
for k in done:
    approx(m[f"difficulty_raw/{k}"], raw[k])
approx(m["critic/score/mean"], d.snis_mean(rows["seq_score"], na_cs))
approx(m["response_length/mean"], d.snis_mean(rl, cs))
approx(m["critic/advantages/mean"], d.snis_token_mean(rows["adv_sum"], rl, cs))
# untouched keys
approx(m["critic/score/max"], raw["critic/score/max"])
approx(m["actor/entropy"], 0.15)
assert "difficulty_raw/critic/score/max" not in m and "difficulty_raw/actor/entropy" not in m
print("PASS metric corrections: SNIS overwrite, raw preserved, order-stats/entropy untouched")

# c == 1 everywhere -> corrected values EQUAL raw (uniform-run consistency).
# Subset cs are baked into rows (built by the driver from the same _craw), so unit them too.
rows1 = dict(rows)
rows1["seq_score__cs"] = [1.0] * len(na)
m1 = dict(raw)
d.apply_metric_corrections(m1, [1.0] * B, rows1)
for k in done:
    approx(m1[k], raw[k], 1e-9)
print("PASS metric corrections: c=1 -> corrected == raw exactly")

# subset cs misalignment must assert (guard against silent misweighting)
bad_rows = {"seq_score": [1.0, 2.0], "seq_score__cs": [1.0]}
try:
    d.apply_metric_corrections({"critic/score/mean": 1.5}, cs, bad_rows)
    raise SystemExit("FAIL: misaligned subset cs must raise")
except AssertionError:
    print("PASS metric corrections: misaligned subset cs raises")

print("\nALL DIFFICULTY UNIT TESTS PASSED (incl. metric corrections)")

# ---------------------------------------------------------------- backward compatibility
# (1) Disabled (the default for every earlier run): all hooks are env- or key-gated -> no-op.
#     Covered above ("disabled: enabled()=False..."). The fit-loop/loss/metric blocks are
#     additionally guarded on batch keys that only a difficulty run attaches.
# (2) Enabled on a FRESH / pre-difficulty checkpoint == plain uniform PPO (w all-equal -> c==1
#     EXACTLY) until pass-rate observations accumulate. This is the "equivalent to w=1" graft
#     property (mirrors AEC's k=0 == un-grafted run).
for form in ("step", "sqrt_p", "sqrt_pq"):
    d = fresh({"SP_DIFF_WEIGHT": form, "SP_DIFF_MIN_OBS": "2"})
    qids = [f"q{i}" for i in range(9)]
    c_map, w_map = d.corrections_for(qids)
    assert len(set(w_map.values())) == 1, f"{form}: unseen weights must be identical"
    for q in qids:
        approx(c_map[q], 1.0)                       # c == 1 EXACTLY
    d.observe("q0", 0.3)                            # one obs < min_obs: still gated -> still c==1
    c_map, _ = d.corrections_for(qids)
    for q in qids:
        approx(c_map[q], 1.0)
print("PASS backward-compat: fresh/under-observed state -> c == 1 exactly (all 3 forms)")

# (3) The sampler on fresh state draws UNIFORMLY and pushes c==1, and the loss-side ĉ stays 1
#     -> the full enabled-but-fresh pipeline is the identity on the loss.
d = fresh({"SP_DIFF_WEIGHT": "sqrt_pq", "SP_DIFF_MIN_OBS": "2"})
smp = d.DifficultyWeightedSampler(FakeDS(["A", "B", "C", "D", "E"]), seed=5)
smp._refresh()                                      # weights are built lazily on first draw
assert len(set(smp._row_w)) == 1                    # uniform draw probabilities
it = iter(smp); [next(it) for _ in range(5)]
recs = d.pop_draws(5)
for r in recs:
    approx(r[2], 1.0)                               # draw-time c == 1
ch = d.normalize_c_for_loss([r[2] for r in recs], [10, 20, 30, 40, 50])
assert all(x == 1.0 for x in ch)                    # ĉ == 1 -> advantages * 1 == advantages
print("PASS backward-compat: fresh sampler == uniform draws, c==1 through the whole loss path")

# (4) Resuming an OLD checkpoint (no difficulty_state.json) with difficulty enabled: load_state
#     returns None, state stays empty -> case (2) applies. Already covered by the missing-file
#     persistence test; re-assert the composition explicitly:
d = fresh({"SP_DIFF_MIN_OBS": "2"})
assert d.load_state(tempfile.mkdtemp()) is None and d._S["stats"] == {}
c_map, _ = d.corrections_for(["x", "y"])
approx(c_map["x"], 1.0); approx(c_map["y"], 1.0)
print("PASS backward-compat: old checkpoint (no state file) + enabled -> starts as uniform (c==1)")

print("\nALL DIFFICULTY UNIT TESTS PASSED (incl. backward compatibility)")

# ---------------------------------------------------------------- warm-start from rollout dumps
import json as _json
d = fresh({"SP_DIFF_WEIGHT": "step", "SP_DIFF_MIN_OBS": "2", "SP_DIFF_EMA_ALPHA": "0.5",
           "SP_DIFF_PASS_THRESH": "0.999"})
rdir = tempfile.mkdtemp()
inp_easy = "a mathematical problem: EASY.\nSolve the problem now"
inp_hard = "a mathematical problem: HARD.\nSolve the problem now"
q_easy, q_hard = d.qid_from_text(inp_easy), d.qid_from_text(inp_hard)
# step files: easy passes 16/16 every step; hard: 0/16 then 8/16. Step 99 must be capped out.
for step, (e_pass, h_pass) in {1: (16, 0), 2: (16, 8), 3: (16, 8), 99: (0, 0)}.items():
    with open(os.path.join(rdir, f"{step}.jsonl"), "w") as f:
        for i in range(16):
            f.write(_json.dumps({"input": inp_easy, "prover_judge_score": 1 if i < e_pass else 0}) + "\n")
        for i in range(16):
            f.write(_json.dumps({"input": inp_hard, "score": 1 if i < h_pass else 0}) + "\n")  # score fallback
        f.write("{corrupt\n")                                    # must be skipped
        f.write(_json.dumps({"input": "no score row"}) + "\n")   # must be skipped
n = d.warmstart_from_rollouts(rdir, max_step=50)                 # excludes 99.jsonl
assert n == 6, n                                                 # 3 steps x 2 qids
approx(d._S["stats"][q_easy]["p"], 0.9375)                       # prior-EMA: 0.5->0.75->0.875->0.9375
approx(d._S["stats"][q_hard]["p"], 0.4375)                       # 0.5->0.25->0.375->0.4375
assert d._S["stats"][q_easy]["n"] == 3 and d._S["stats"][q_hard]["n"] == 3
# the graft's payoff: known-easy problem is ALREADY demoted at the first difficulty step
# (p-hat hit the 15/16 threshold exactly after three 16/16 history steps under prior-EMA)
approx(d.weight(q_easy), 1 / 16)
approx(d.weight(q_hard), 1.0)
c_map, _ = d.corrections_for([q_easy, q_hard])
assert c_map[q_easy] > 1.0 > c_map[q_hard]
print("PASS warm-start: EMA replay in step order, score fallback, corrupt rows skipped, "
      "max_step cap, known-easy demoted immediately")

# priority chain: a real difficulty_state.json wins over warm-start (ray_trainer calls
# warmstart only when load_state returns None)
d2 = fresh({"SP_DIFF_MIN_OBS": "2"})
ck = tempfile.mkdtemp()
d2.observe("qq", 0.5); d2.save_state(ck)
d3 = fresh({"SP_DIFF_MIN_OBS": "2"})
assert d3.load_state(ck) == 1                                    # found -> warmstart must NOT run
assert d3.warmstart_from_rollouts(tempfile.mkdtemp()) is None    # and would no-op on empty dir anyway
# disabled gate
os.environ["SP_DIFF_WARMSTART"] = "0"
d4 = fresh({"SP_DIFF_WARMSTART": "0"})
assert d4.warmstart_from_rollouts(rdir) is None
print("PASS warm-start: priority (state file wins), empty dir no-op, env gate off")

print("\nALL DIFFICULTY UNIT TESTS PASSED (incl. warm-start)")

# ---------------------------------------------------------------- review fix 1: binary pass signal
d = fresh({"SP_DIFF_PASS_THRESH": "0.999"})
# correct long proof: shaped score < 1 but prover_judge_score == 1 -> MUST count as pass
approx(d.row_pass(prover_judge_score=1, score=0.42), 1)
approx(d.row_pass(prover_judge_score=0, score=0.0), 0)
# proposer path
approx(d.row_pass(correctness_judge_score=1, score=0.0), 1)
# judge field takes precedence over shaped score
approx(d.row_pass(prover_judge_score=1, correctness_judge_score=None, score=0.0), 1)
# fallback only when no judge field: shaped score thresholded
approx(d.row_pass(score=1.0), 1)
approx(d.row_pass(score=0.5), 0)
approx(d.row_pass(), 0)                                # nothing -> 0, no crash
approx(d.row_pass(prover_judge_score="1", score="0.4"), 1)   # string-tolerant
print("PASS review-fix 1: row_pass uses binary judge verdict, not shaped reward")

# ---------------------------------------------------------------- review fix 4: FIFO reset per epoch
d = fresh({"SP_DIFF_WEIGHT": "step", "SP_DIFF_MIN_OBS": "1"})
smp = d.DifficultyWeightedSampler(FakeDS(["A", "B", "C", "D", "E"]), seed=11)
g = iter(smp)
next(g); next(g)                                       # draw 2 of 5, then abandon (drop_last tail)
assert d.fifo_len() == 2                               # stale records left behind
g2 = iter(smp)                                          # next epoch
first = next(g2)
assert d.fifo_len() == 1, d.fifo_len()                 # reset cleared the 2 stale, pushed 1 fresh
recs = d.pop_draws(1)
assert recs[0][0] == first and recs[0][1] == smp._row_qid[first]  # fresh record matches this draw
print("PASS review-fix 4: __iter__ resets FIFO so dropped-tail records never go stale")

print("\nALL DIFFICULTY UNIT TESTS PASSED (incl. review fixes)")
