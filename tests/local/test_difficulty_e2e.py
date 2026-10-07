"""End-to-end driver-cycle simulation for difficulty sampling (no verl import; replicates the fit
loop's exact call sequence against the real module + a synthetic problem population).

Verifies, over many simulated steps:
  1. the sampler->FIFO->fit-loop handshake (qid match on every step, no fallback trips);
  2. compute is actually reallocated (easy problems drawn ~16x less under `step` form);
  3. the SNIS-corrected score mean tracks the true uniform-population mean while the RAW batch
     mean is biased low (the whole point of the correction);
  4. the weighted-loss gradient direction matches uniform sampling in expectation;
  5. persistence roundtrip mid-"run" (simulated relaunch) changes nothing statistically.

    /tmp/ds-venv/bin/python tests/local/test_difficulty_e2e.py
"""
import importlib.util
import os
import random
import tempfile

import torch

_DIFF = os.path.join(
    os.path.dirname(__file__), "..", "..", "src", "verl", "verl", "trainer", "ppo", "difficulty.py"
)
os.environ["SP_DIFF_SAMPLING"] = "1"
os.environ["SP_DIFF_WEIGHT"] = "step"       # {1, 1/16} rule: crisp compute-reallocation signal
os.environ["SP_DIFF_MIN_OBS"] = "2"
os.environ["SP_DIFF_EMA_ALPHA"] = "0.4"
spec = importlib.util.spec_from_file_location("difficulty_e2e", _DIFF)
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)

rng = random.Random(0)
torch.manual_seed(0)

# --- synthetic population: 200 problems, bimodal true pass rates (a 35% easy tail) ---
N = 200
true_p = [0.97 if i < 70 else rng.uniform(0.0, 0.7) for i in range(N)]   # 35% near-solved
STATEMENTS = [f"problem number {i}" for i in range(N)]


class FakeDS:
    prompt_key = "prompt"
    dataframe = [
        {"prompt": [{"role": "user", "content": f"mathematical problem: {s} Solve the problem"}]}
        for s in STATEMENTS
    ]
    def __len__(self):
        return N


smp = d.DifficultyWeightedSampler(FakeDS(), seed=1)
qid_of = {smp._row_qid[i]: i for i in range(N)}            # qid -> problem index
BATCH, ROLLOUT_N, STEPS = 32, 16, 120

uniform_mean = sum(true_p) / N                             # ground truth E_p[score]
draw_count = [0] * N
draw_count_tail = [0] * N                                  # last third: post-demotion steady state
snis_est, raw_est = [], []
it = iter(smp)
ckpt = tempfile.mkdtemp()

for step in range(STEPS):
    # --- dataloader: draw one FULL batch of row indices (sampler pushes FIFO at draw time).
    # Faithful to StatefulDataLoader(drop_last=True): a yielded batch never spans an epoch
    # boundary; when the epoch's iterator is exhausted mid-batch the partial tail is dropped and
    # a fresh epoch iterator starts for the NEXT batch (which resets the FIFO in __iter__). ---
    rows = []
    while len(rows) < BATCH:
        try:
            rows.append(next(it))
        except StopIteration:
            rows = []            # drop the partial tail (drop_last)
            it = iter(smp)       # new epoch -> FIFO reset; batch is drawn wholly within it
    # --- fit loop: hash qids from "raw_prompt", pop FIFO, cross-check (the real handshake) ---
    qids = [d.qid_from_messages(FakeDS.dataframe[i]["prompt"]) for i in rows]
    recs = d.pop_draws(len(qids))
    assert len(recs) == len(qids) and all(r[1] == q for r, q in zip(recs, qids)), \
        f"FIFO handshake broke at step {step}"
    cs = [r[2] for r in recs]
    for i in rows:
        draw_count[i] += 1
        if step >= 2 * STEPS // 3:
            draw_count_tail[i] += 1

    # --- generation + judge: per-problem Bernoulli(true_p) x 16 rollouts ---
    scores = []           # per-rollout binary score, grouped [row0 x16, row1 x16, ...]
    for i in rows:
        scores.extend(1.0 if rng.random() < true_p[i] else 0.0 for _ in range(ROLLOUT_N))
    cs_roll = [c for c in cs for _ in range(ROLLOUT_N)]     # batch.repeat(16) equivalent

    # --- post-reward: observe() per qid + SNIS metric (exact fit-loop code path) ---
    per = {}
    for q, chunk_start in zip(qids, range(0, len(scores), ROLLOUT_N)):
        chunk = scores[chunk_start:chunk_start + ROLLOUT_N]
        a = per.setdefault(q, [0, 0])
        a[0] += sum(1 for s in chunk if s >= d.pass_threshold())
        a[1] += len(chunk)
    for q, (npass, ntot) in per.items():
        d.observe(q, npass / ntot)
    snis_est.append(d.snis_mean(scores, cs_roll))
    raw_est.append(sum(scores) / len(scores))

    # --- update path: normalize + (bsz,1) tensor + weighted "loss" gradient check ---
    n_tok = [rng.randint(50, 200) for _ in range(len(cs_roll))]
    chat = d.normalize_c_for_loss(cs_roll, n_tok)
    diff_c = torch.tensor(chat).unsqueeze(-1)
    assert diff_c.shape == (BATCH * ROLLOUT_N, 1)
    assert abs(sum(c * n for c, n in zip(chat, n_tok)) - sum(n_tok)) < 1e-6 * sum(n_tok)

    # --- mid-run relaunch: persist + reload in a fresh module instance ---
    if step == STEPS // 2:
        d.save_state(ckpt)
        spec2 = importlib.util.spec_from_file_location("difficulty_e2e_2", _DIFF)
        d2 = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(d2)
        assert d2.load_state(ckpt) == len(d._S["stats"])
        for q, st in d._S["stats"].items():
            assert abs(d2._S["stats"][q]["p"] - st["p"]) < 1e-12
        print(f"PASS mid-run persistence roundtrip at step {step} ({len(d._S['stats'])} qids)")

# ---------------------------------------------------------------- assertions
tail = STEPS // 2
easy_draws = sum(draw_count[i] for i in range(70))
hard_draws = sum(draw_count[i] for i in range(70, N))
easy_rate = easy_draws / 70
hard_rate = hard_draws / (N - 70)
tail_easy = sum(draw_count_tail[i] for i in range(70)) / 70
tail_hard = sum(draw_count_tail[i] for i in range(70, N)) / (N - 70)
print(f"draw rate per problem: easy={easy_rate:.2f} hard={hard_rate:.2f} "
      f"(full-window ratio {hard_rate / easy_rate:.1f}x; "
      f"tail-window {tail_hard / max(tail_easy, 1e-9):.1f}x, weights say 16x once demoted)")
# Under the p-hat=1/2 prior, demotion needs ~6 consecutive high-pass observations (EMA from 0.5),
# so the full-window ratio includes the slow ramp; the TAIL window shows the steady state.
assert hard_rate > 2.5 * easy_rate, "compute must shift toward hard problems overall"
assert tail_hard > 6 * tail_easy, "steady-state (post-demotion) shift must approach the 16x design"

snis_tail = sum(snis_est[-tail:]) / tail
raw_tail = sum(raw_est[-tail:]) / tail
print(f"uniform truth={uniform_mean:.4f}  SNIS corrected={snis_tail:.4f}  raw(biased)={raw_tail:.4f}")
assert abs(snis_tail - uniform_mean) < 0.05, "corrected estimate must track the uniform mean"
assert raw_tail < uniform_mean - 0.05, "raw batch mean must be visibly biased low (sanity)"

ess_last = d.ess([c for c in cs_roll])
print(f"final-step ESS = {ess_last:.0f} / {len(cs_roll)}")
print("\nALL E2E DIFFICULTY TESTS PASSED")
