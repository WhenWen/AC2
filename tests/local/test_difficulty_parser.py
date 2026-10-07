"""Parser-side SNIS weighting test: run the REAL parse_fig_data CLI on synthetic run_data.

Case A (uniform dump, no diff_c_raw): main keys equal plain means, NO *_raw keys anywhere
(old fig_data keysets unchanged -> viz backward compatible).
Case B (difficulty dump, c per problem): expectation keys equal hand-computed SNIS, *_raw
equal plain means, pass-count histogram group-weighted + rescaled, counts stay raw.

    python3 tests/local/test_difficulty_parser.py     (parser is stdlib-only)
"""
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PARSER = os.path.join(ROOT, "src", "ac2", "viz", "parse_fig_data.py")


def approx(a, b, eps=1e-9):
    assert abs(a - b) < eps, (a, b)


def make_run(tmp, with_c):
    """2 problems x 4 rollouts. Problem A: easy (4/4 pass), c=0.25. Problem B: hard
    (1/4 pass), c=4.0. Response lengths differ so weighted means move."""
    rdir = os.path.join(tmp, "run_data", "rollouts")
    os.makedirs(rdir)
    rows = []
    for uid, qtxt, c, passes, rlen in (
        ("uA", "mathematical problem: A. Solve the problem", 0.25, [1, 1, 1, 1], 100),
        ("uB", "mathematical problem: B. Solve the problem", 4.0, [1, 0, 0, 0], 300),
    ):
        for i, p in enumerate(passes):
            row = {
                "input": qtxt, "output": f"<proof>x</proof> {i}",
                "uid": uid, "mode_is_prover": 1,
                "score": float(p), "prover_judge_score": int(p),
                "proof_tag_present": 1, "proof_len_chars": 10,
                "response_length": rlen, "prompt_length": 50,
                "total_tokens": rlen + 50,
            }
            if with_c:
                row["diff_c_raw"] = c
            rows.append(row)
    with open(os.path.join(rdir, "1.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(tmp, "run_data", "metrics.jsonl"), "w") as f:
        f.write(json.dumps({"step": 1, "data": {"actor/entropy": 0.15}}) + "\n")
    return tmp


def parse(run_dir):
    out = os.path.join(run_dir, "fig.json")
    r = subprocess.run(
        [sys.executable, PARSER, "--run-dir", run_dir, "--out", out],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert r.returncode == 0, r.stderr[-3000:]
    return json.load(open(out))["per_step"]


P = "v7__train__prover__"

# ---------------------------------------------------------------- Case A: uniform
tmpA = make_run(tempfile.mkdtemp(), with_c=False)
psA = parse(tmpA)
assert not any(k.endswith("_raw") for k in psA), \
    "uniform dumps must emit NO *_raw keys (old keysets unchanged)"
approx(psA[P + "score_mean_generated"][0], 5 / 8)          # plain mean over 8 rows
approx(psA[P + "prover_judge_score_mean"][0], 5 / 8)
approx(psA[P + "resp_len_mean"][0], 200.0)                 # (4*100 + 4*300)/8
approx(psA[P + "rows_generated"][0], 8.0)
approx(psA[P + "pass_count_hist__k4"][0], 1.0)             # A: 4 passes
approx(psA[P + "pass_count_hist__k1"][0], 1.0)             # B: 1 pass
print("PASS parser uniform: plain means, raw counts, no *_raw keys")

# ---------------------------------------------------------------- Case B: difficulty
tmpB = make_run(tempfile.mkdtemp(), with_c=True)
psB = parse(tmpB)
# SNIS over rows: each row of A has c=0.25, of B c=4.0
cA, cB = 0.25, 4.0
den = 4 * cA + 4 * cB
snis_score = (cA * 4 * 1.0 + cB * 1 * 1.0) / den           # A all pass, B 1 of 4
approx(psB[P + "score_mean_generated"][0], snis_score)
approx(psB[P + "prover_judge_score_mean"][0], snis_score)
approx(psB[P + "prover_judge_pass_rate"][0], snis_score)
snis_rlen = (cA * 4 * 100 + cB * 4 * 300) / den
approx(psB[P + "resp_len_mean"][0], snis_rlen)
# raw counterparts preserved
approx(psB[P + "score_mean_generated_raw"][0], 5 / 8)
approx(psB[P + "resp_len_mean_raw"][0], 200.0)
approx(psB[P + "prover_judge_score_mean_raw"][0], 5 / 8)
# counts stay actual work
approx(psB[P + "rows_generated"][0], 8.0)
approx(psB[P + "judge_attempts"][0], 8.0)
# group-weighted histogram: 2 groups, c = {A:0.25, B:4.0}, scale = 2/4.25
scale = 2 / 4.25
approx(psB[P + "pass_count_hist__k4"][0], 0.25 * scale)    # easy group down-weighted
approx(psB[P + "pass_count_hist__k1"][0], 4.0 * scale)     # hard group up-weighted
approx(psB[P + "pass_count_hist__k4_raw"][0], 1.0)
approx(psB[P + "pass_count_hist__k1_raw"][0], 1.0)
# bars still sum to the actual group count
tot = sum(psB[P + f"pass_count_hist__k{k}"][0] for k in range(17))
approx(tot, 2.0)
# group composition: allone = A (c-weighted), mixed = B
approx(psB[P + "group_allone"][0], 0.25 * scale)
approx(psB[P + "group_mixed"][0], 4.0 * scale)
approx(psB[P + "group_allone_raw"][0], 1.0)
# percentiles + diagnostics stay raw (p50 of actual lengths [100x4,300x4])
assert psB[P + "resp_len_p50"][0] == psA[P + "resp_len_p50"][0]
print("PASS parser difficulty: SNIS means, raw preserved, group-weighted hist (sums to n_groups), counts/percentiles raw")

print("\nALL PARSER WEIGHTING TESTS PASSED")
