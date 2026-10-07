"""Validate q_monitor.jsonl and print the Q-quality headline numbers.

Two jobs. First, check the invariants that would silently corrupt any downstream read of
the dataset -- a broken step join or a mislabelled lane looks perfectly plausible in
aggregate, so the join is asserted rather than eyeballed. Second, print the monitoring
view: how much of the batch Q is scoring, whether Q's scores track the judge's, and the
audit-lane calibration that is the only unbiased check on that.

    python experiments/ac2/q_monitor_report.py [--dataset PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import Counter, defaultdict

DEFAULT = os.path.join(
    os.environ.get("AC2_REPO", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "experiments/ac2/analysis/q_monitor.jsonl",
)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def fmt(x, n=3):
    return "-" if x is None else ("%.*f" % (n, x))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT)
    args = ap.parse_args()

    recs = []
    with open(args.dataset, encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))

    # ---------------------------------------------------------------- invariants
    problems = []
    n_comp = Counter()
    for r in recs:
        lanes = set(c["lane"] for c in r["completions"])
        route = r["readiness"]["route"]
        n_comp[r["n_completions"]] += 1
        # a group is routed as a unit: audit groups are all-audit, full groups are all-judged,
        # and only short groups may mix q_consumed with judged
        if route == "audit" and lanes - {"audit_full"}:
            problems.append("step %s uid %s: audit route but lanes %s" % (r["step"], r["uid"], lanes))
        if route == "full" and lanes - {"judged"}:
            problems.append("step %s uid %s: full route but lanes %s" % (r["step"], r["uid"], lanes))
        if route == "short" and lanes - {"q_consumed", "judged"}:
            problems.append("step %s uid %s: short route but lanes %s" % (r["step"], r["uid"], lanes))
        # a q_consumed completion must carry a Q value unless it was flagged invalid
        for c in r["completions"]:
            if c["lane"] == "q_consumed" and c["q_value"] is None and not c["q_invalid"]:
                problems.append("step %s uid %s: valid q_consumed with no q_value" % (r["step"], r["uid"]))
        if r["audit_cut"] and route != "audit":
            problems.append("step %s uid %s: audit_cut on a %s group" % (r["step"], r["uid"], route))
        # readiness must agree with routing: ready_since is monotone, so a routed-off-full
        # group whose qid has no ready_since would mean the delta join missed a transition
        # ready_since_step is a per-QID fact and is carried on records from BEFORE the
        # problem went ready, where being "in the future" is correct, not a violation. The
        # real invariant is the other direction: a group actually routed off the full lane
        # must already have gone ready by this step.
        rd = r["readiness"]
        if rd["ready"] and (rd["ready_since_step"] is None or rd["ready_since_step"] > r["step"]):
            problems.append("step %s uid %s: routed %s but ready_since=%s"
                            % (r["step"], r["uid"], route, rd["ready_since_step"]))
        if (rd["steps_ready"] or 0) < 0:
            problems.append("step %s uid %s: negative steps_ready" % (r["step"], r["uid"]))

    print("=" * 78)
    print("VALIDATION")
    print("=" * 78)
    print("records                : %d" % len(recs))
    print("group sizes            : %s" % dict(n_comp))
    print("invariant violations   : %d" % len(problems))
    for p in problems[:10]:
        print("   ! %s" % p)

    # ---------------------------------------------------------------- per step
    by_step = defaultdict(list)
    for r in recs:
        by_step[r["step"]].append(r)

    print()
    print("=" * 78)
    print("PER STEP  (replay groups only; scratch/inflow rows run rollout_n=1)")
    print("=" * 78)
    print("%-5s %-7s %-7s %-7s %-8s %-8s %-8s %-8s %-8s"
          % ("step", "ready", "audit", "nQ", "Q_mean", "judge_mu", "cut_MAE", "mae5", "pairs"))
    for step in sorted(by_step):
        rs = [r for r in by_step[step] if r["is_replay_group"]]
        ready = [r for r in rs if r["readiness"]["ready"]]
        audit = [r for r in rs if r["readiness"]["route"] == "audit"]
        qv = [v for r in rs for v in r["summary"]["q_values"]]
        jv = [v for r in rs for v in r["summary"]["judge_rewards"]]
        errs = [r["audit_cut"]["abs_error"] for r in rs
                if r["audit_cut"] and r["audit_cut"].get("abs_error") is not None]
        mae5 = next((r["readiness"]["mae5_global"] for r in rs
                     if r["readiness"]["mae5_global"] is not None), None)
        print("%-5d %-7d %-7d %-7d %-8s %-8s %-8s %-8s %-8d"
              % (step, len(ready), len(audit), len(qv), fmt(mean(qv)), fmt(mean(jv)),
                 fmt(mean(errs)), fmt(mae5), len(errs)))

    # ---------------------------------------------------------------- calibration
    pairs = [(r["audit_cut"]["q_at_cut"], r["audit_cut"]["terminal_reward"], r["step"])
             for r in recs
             if r["audit_cut"] and r["audit_cut"].get("abs_error") is not None]
    print()
    print("=" * 78)
    print("AUDIT-LANE CALIBRATION  (n=%d pairs)" % len(pairs))
    print("=" * 78)
    if pairs:
        q = [p[0] for p in pairs]
        t = [p[1] for p in pairs]
        print("MAE |Q - reward|       : %s" % fmt(mean([abs(a - b) for a, b in zip(q, t)])))
        print("bias  mean(Q - reward) : %s" % fmt(mean([a - b for a, b in zip(q, t)])))
        print("mean Q / mean reward   : %s / %s" % (fmt(mean(q)), fmt(mean(t))))
        # mean Q per realized-reward bin: the calibration curve, with the bin counts panel
        # 48 hides. Sparse bins are what make that panel's mid-range interpolation.
        print("\nmean Q | realized reward:")
        bins = defaultdict(list)
        for a, b, _ in pairs:
            bins[round(b, 1)].append(a)
        for b in sorted(bins):
            v = bins[b]
            print("   reward %.1f  n=%-5d mean Q %-7s std %s"
                  % (b, len(v), fmt(mean(v)),
                     fmt(statistics.pstdev(v) if len(v) > 1 else 0.0)))
        # the operational question: of the rollouts Q scored high, how many were actually
        # correct? Q is consumed as a reward, so a high-Q/low-reward cell is what poisons PPO.
        print("\nconfusion at Q>=0.9 vs reward>=1.0:")
        hi_q = [(a, b) for a, b, _ in pairs if a >= 0.9]
        print("   Q>=0.9            : n=%d, of which reward>=1.0: %d (%.0f%%)"
              % (len(hi_q), sum(1 for _, b in hi_q if b >= 1.0),
                 100.0 * sum(1 for _, b in hi_q if b >= 1.0) / len(hi_q) if hi_q else 0))
        lo_q = [(a, b) for a, b, _ in pairs if a <= 0.1]
        print("   Q<=0.1            : n=%d, of which reward>=1.0: %d (%.0f%%)"
              % (len(lo_q), sum(1 for _, b in lo_q if b >= 1.0),
                 100.0 * sum(1 for _, b in lo_q if b >= 1.0) / len(lo_q) if lo_q else 0))

    # ---------------------------------------------------------------- readiness
    print()
    print("=" * 78)
    print("READINESS")
    print("=" * 78)
    tags = Counter(r["readiness"]["probe_source_tag"] for r in recs if r["is_replay_group"])
    print("probe target provenance : %s" % dict(tags))
    errs = [r["readiness"]["probe_error"] for r in recs if r["is_replay_group"]]
    errs = [e for e in errs if e is not None]
    if errs:
        errs_s = sorted(errs)
        print("probe |pred - z|        : mean %s  median %s  p90 %s  (gate %s)"
              % (fmt(mean(errs)), fmt(errs_s[len(errs_s) // 2]),
                 fmt(errs_s[int(0.9 * len(errs_s))]),
                 fmt(recs[0]["readiness"]["thresh_problem"])))
    ready_qids = {r["qid"] for r in recs if r["readiness"]["ready_since_step"] is not None}
    all_qids = {r["qid"] for r in recs if r["is_replay_group"] and r["qid"]}
    print("qids seen / ever ready  : %d / %d" % (len(all_qids), len(ready_qids)))
    # Q's scores vs the judge's, on the SAME steps -- not a calibration claim (different
    # rollouts), just whether the two reward streams are even on the same scale.
    qv = [v for r in recs for v in r["summary"]["q_values"]]
    jv = [v for r in recs for v in r["summary"]["judge_rewards"]]
    print("\nall q_consumed scores   : n=%d mean %s" % (len(qv), fmt(mean(qv))))
    print("all judged rewards      : n=%d mean %s" % (len(jv), fmt(mean(jv))))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
