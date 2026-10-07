"""How Q's error relates to problem difficulty and prefix length, and what Q looks like
across a group of rollouts sharing one prefix.

Reads analysis/q_monitor.jsonl. Three sections, and each carries the control that makes it
readable rather than just the raw correlation.

TWO TRAPS THIS SCRIPT IS BUILT AROUND -- both would flip the conclusions if ignored.

1. THE TARGET IS THE GROUP'S OWN MEAN. Probe error is |Q(prefix) - z| where z = grid(mean
   reward over the group's valid members). So binning error by the group's own reward is
   binning by z, and |pred - z| against z is the calibration curve, NOT a difficulty
   relationship. Difficulty here is therefore LEAVE-ONE-OUT: a problem's mean over its
   OTHER groups, never the group being scored.

2. z IS A SAMPLE MEAN, SO IT IS NOISIEST AT p~0.5. Its sampling variance peaks in the
   middle, so even a PERFECT Q shows an inverted-U of |pred - z| against difficulty. Every
   MAE below is therefore reported next to a NOISE FLOOR -- the error an oracle predicting
   the problem's own long-run rate would still incur against this noisy target -- and the
   EXCESS over that floor is the only number that says anything about Q.

Difficulty is estimated ONLY from full-lane and audit-lane groups, where all 16 members
carry a real judge reward. A short-lane group's judged members are the ones that finished
early WITHOUT being cut, which is a biased (easier) subset -- using them would make every
problem look easier than it is.

    python experiments/ac2/q_mae_analysis.py [--dataset PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict

DEFAULT = os.path.join(
    os.environ.get("AC2_REPO", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "experiments/ac2/analysis/q_monitor.jsonl",
)


def grid(x):
    """The 0.0..1.0 grid in steps of 0.1 that Q emits on and that z is snapped to."""
    return None if x is None else min(1.0, max(0.0, round(float(x) * 10.0) / 10.0))


def mean(xs):
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def pstd(xs):
    xs = [x for x in xs if x is not None]
    return statistics.pstdev(xs) if len(xs) > 1 else (0.0 if xs else None)


def f(x, n=3):
    return "  -  " if x is None else ("%.*f" % (n, x))


def bar(x, lo, hi, w=22):
    if x is None:
        return ""
    t = max(0.0, min(1.0, (x - lo) / (hi - lo))) if hi > lo else 0.0
    return "#" * int(round(t * w))


def load(path):
    recs = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["is_replay_group"]:
                recs.append(r)
    return recs


def build_difficulty(recs):
    """qid -> list of (step, unbiased group mean). Only full/audit groups qualify: every
    member there carries a real judge reward, so the mean is not selection-biased."""
    by_qid = defaultdict(list)
    for r in recs:
        if r["readiness"]["route"] not in ("full", "audit"):
            continue
        jr = [c["judge_reward"] for c in r["completions"] if c["judge_reward"] is not None]
        if len(jr) >= 8:  # same min_valid the harness uses to trust a group mean
            by_qid[r["qid"]].append((r["step"], sum(jr) / len(jr)))
    return by_qid


def p_loo(by_qid, qid, step):
    """The problem's difficulty EXCLUDING the group at `step` -- so the difficulty measure
    and the thing being explained never share a sample."""
    xs = [m for (s, m) in by_qid.get(qid, []) if s != step]
    return (sum(xs) / len(xs)) if xs else None


def section(title):
    print()
    print("=" * 86)
    print(title)
    print("=" * 86)


def binned_table(rows, keyfn, label, bins_desc):
    """rows: list of (bin_key, probe_error, z, p_loo). Prints MAE, the noise floor and the
    excess per bin."""
    groups = defaultdict(list)
    for row in rows:
        groups[keyfn(row)].append(row)
    print("%-16s %7s %8s %8s %8s %8s   %s"
          % (label, "n", "MAE", "floor", "excess", "mean|z|", bins_desc))
    out = []
    for k in sorted(groups):
        g = groups[k]
        errs = [r[1] for r in g]
        floor = [abs(grid(r[3]) - r[2]) for r in g if r[3] is not None and r[2] is not None]
        m, fl = mean(errs), mean(floor)
        ex = (m - fl) if (m is not None and fl is not None) else None
        out.append((k, len(g), m, fl, ex, mean([r[2] for r in g])))
    for k, n, m, fl, ex, mz in out:
        print("%-16s %7d %8s %8s %8s %8s   %s"
              % (k, n, f(m), f(fl), f(ex), f(mz), bar(ex, 0.0, 0.30)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT)
    ap.add_argument("--all-provenance", action="store_true",
                    help="include probes whose target came from consumed Q (default: only "
                         "terminal_group, where z is entirely judge-derived)")
    args = ap.parse_args()

    recs = load(args.dataset)
    by_qid = build_difficulty(recs)

    # probe rows usable for a difficulty/prefix regression
    rows = []
    dropped_prov = 0
    for r in recs:
        rd = r["readiness"]
        if rd["probe_error"] is None or rd["probe_z"] is None:
            continue
        if not args.all_provenance and rd["probe_source_tag"] != "terminal_group":
            dropped_prov += 1
            continue
        pl = r["completions"][0]["prefix_len"]
        rows.append((r, rd["probe_error"], rd["probe_z"], p_loo(by_qid, r["qid"], r["step"]), pl))

    section("0. INPUTS")
    print("replay groups                 : %d" % len(recs))
    print("probe rows used               : %d  (dropped %d as Q-derived-target provenance)"
          % (len(rows), dropped_prov))
    print("problems with unbiased diff   : %d" % len(by_qid))
    ngroups = [len(v) for v in by_qid.values()]
    print("full/audit groups per problem : median %d, max %d (need >=2 for leave-one-out)"
          % (statistics.median(ngroups) if ngroups else 0, max(ngroups) if ngroups else 0))
    with_loo = [x for x in rows if x[3] is not None]
    print("probe rows WITH leave-one-out : %d" % len(with_loo))
    # prefix is a group property; assert it rather than assume
    bad = sum(1 for r in recs if len({c["prefix_len"] for c in r["completions"]}) > 1)
    print("groups with non-constant prefix: %d (expect 0)" % bad)

    # ------------------------------------------------------------------ 1. difficulty
    section("1. MAE vs PROBLEM DIFFICULTY   (difficulty = leave-one-out mean judge reward)")
    print("Harder problem = LOWER mean reward. 'floor' is what a perfect oracle still pays")
    print("against a 16-sample target; 'excess' = MAE - floor is Q's own error.\n")
    d_rows = [(x[0], x[1], x[2], x[3]) for x in with_loo]
    binned_table(d_rows, lambda r: "p_loo %.1f-%.1f" % (min(0.9, (int(r[3] * 10) / 10.0)),
                                                       min(1.0, (int(r[3] * 10) / 10.0) + 0.1)),
                 "difficulty bin", "excess bar")

    # audit-lane cross-check: ground truth, no shared-sample worry at all
    ap_rows = []
    for r in recs:
        ac = r.get("audit_cut")
        if not ac or ac.get("abs_error") is None:
            continue
        pl_ = p_loo(by_qid, r["qid"], r["step"])
        if pl_ is not None:
            ap_rows.append((pl_, ac["abs_error"]))
    print("\naudit-lane cross-check (|Q_at_cut - realized terminal reward|, no shared sample):")
    print("%-16s %7s %8s" % ("difficulty bin", "n", "MAE"))
    ab = defaultdict(list)
    for p, e in ap_rows:
        ab["p_loo %.1f-%.1f" % (int(p * 10) / 10.0, int(p * 10) / 10.0 + 0.1)].append(e)
    for k in sorted(ab):
        print("%-16s %7d %8s" % (k, len(ab[k]), f(mean(ab[k]))))

    # ------------------------------------------------------------------ 2. prefix length
    section("2. MAE vs PREFIX LENGTH   (cuts are multiples of 10k tokens in [0, 0.9*t])")
    pl_rows = [(x[0], x[1], x[2], x[3], x[4]) for x in rows]
    print("%-16s %7s %8s %8s %8s %8s   %s"
          % ("prefix tokens", "n", "MAE", "floor", "excess", "mean z", ""))
    pb = defaultdict(list)
    for r, e, z, pl_, plen in pl_rows:
        pb["%6d" % (int(plen // 10000) * 10000)].append((r, e, z, pl_))
    for k in sorted(pb, key=lambda s: int(s)):
        g = pb[k]
        errs = [x[1] for x in g]
        floor = [abs(grid(x[3]) - x[2]) for x in g if x[3] is not None]
        m, fl = mean(errs), mean(floor)
        ex = (m - fl) if (m is not None and fl is not None) else None
        print("%-16s %7d %8s %8s %8s %8s   %s"
              % (k, len(g), f(m), f(fl), f(ex), f(mean([x[2] for x in g])), bar(ex, 0.0, 0.30)))

    # prefix and difficulty are entangled -- long prefixes come from long problems, which
    # are not the same population as short ones. Hold difficulty fixed and look again.
    print("\nprefix effect WITHIN difficulty terciles (controls for the two being entangled):")
    lo_rows = [x for x in pl_rows if x[3] is not None]
    if lo_rows:
        ps = sorted(x[3] for x in lo_rows)
        t1, t2 = ps[len(ps) // 3], ps[2 * len(ps) // 3]
        print("  terciles at p_loo <= %.3f, <= %.3f" % (t1, t2))
        print("  %-10s %-14s %7s %8s %8s" % ("tercile", "prefix", "n", "MAE", "excess"))
        for name, lo_, hi_ in (("hard", -1, t1), ("mid", t1, t2), ("easy", t2, 2)):
            sub = [x for x in lo_rows if lo_ < x[3] <= hi_]
            b2 = defaultdict(list)
            for r, e, z, pl_, plen in sub:
                b2["%6d" % (int(plen // 10000) * 10000)].append((e, z, pl_))
            for k in sorted(b2, key=lambda s: int(s)):
                g = b2[k]
                m = mean([x[0] for x in g])
                fl = mean([abs(grid(x[2]) - x[1]) for x in g])
                print("  %-10s %-14s %7d %8s %8s"
                      % (name, k, len(g), f(m), f(m - fl) if (m and fl) else "-"))

    # ------------------------------------------------------------------ 3. within-group Q
    section("3. DISTRIBUTION OF Q ACROSS A GROUP SHARING ONE PREFIX")
    print("Members of a group share the prefix and diverge only in their continuation, so")
    print("within-group spread is how much Q responds to the CONTINUATION rather than to")
    print("the problem+prefix alone.\n")
    qgroups = [r for r in recs if len(r["summary"]["q_values"]) >= 2]
    nq = [len(r["summary"]["q_values"]) for r in recs
          if r["readiness"]["route"] == "short"]
    print("short-lane groups             : %d" % len(nq))
    print("Q calls per short group       : mean %s, median %s (of %d members)"
          % (f(mean(nq), 1), f(statistics.median(nq) if nq else None, 1), 16))
    print("groups with >=2 Q values      : %d" % len(qgroups))

    within = [pstd(r["summary"]["q_values"]) for r in qgroups]
    gmeans = [mean(r["summary"]["q_values"]) for r in qgroups]
    allq = [v for r in qgroups for v in r["summary"]["q_values"]]
    print("\nvariance decomposition over %d Q values in %d groups:" % (len(allq), len(qgroups)))
    print("  total std                   : %s" % f(pstd(allq)))
    print("  mean WITHIN-group std       : %s" % f(mean(within)))
    print("  BETWEEN-group std of means  : %s" % f(pstd(gmeans)))
    tot = pstd(allq)
    if tot and tot > 0:
        wv = mean([w * w for w in within]) or 0.0
        print("  within-group share of var   : %.1f%%" % (100.0 * wv / (tot * tot)))

    print("\nwithin-group shape:")
    ndist = [len(set(r["summary"]["q_values"])) for r in qgroups]
    rng = [max(r["summary"]["q_values"]) - min(r["summary"]["q_values"]) for r in qgroups]
    print("  distinct grid values / group: mean %s  (1 = every member got the same Q)"
          % f(mean(ndist), 2))
    print("  groups fully collapsed (n=1): %d (%.0f%%)"
          % (sum(1 for d in ndist if d == 1),
             100.0 * sum(1 for d in ndist if d == 1) / len(ndist) if ndist else 0))
    print("  min-max range               : mean %s  median %s"
          % (f(mean(rng)), f(statistics.median(rng) if rng else None)))

    print("\n  where the Q mass sits (all %d within-group Q values):" % len(allq))
    hist = defaultdict(int)
    for v in allq:
        hist[grid(v)] += 1
    for k in sorted(hist):
        print("    Q=%.1f  n=%-6d %5.1f%%  %s"
              % (k, hist[k], 100.0 * hist[k] / len(allq),
                 "#" * int(round(60.0 * hist[k] / len(allq)))))

    print("\n  within-group std by prefix length:")
    sb = defaultdict(list)
    for r in qgroups:
        sb["%6d" % (int(r["completions"][0]["prefix_len"] // 10000) * 10000)].append(
            pstd(r["summary"]["q_values"]))
    for k in sorted(sb, key=lambda s: int(s)):
        print("    prefix %s  n=%-5d mean within-std %s" % (k, len(sb[k]), f(mean(sb[k]))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
