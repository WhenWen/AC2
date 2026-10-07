"""Q's nested structure: the numbers a pooled histogram cannot show.

Q values are CLUSTERED -- rollouts nested inside prefix-sharing groups -- and 81% of their
variance lives between clusters. A marginal histogram pools the two levels and therefore
shows neither: its four spikes are a between-group fact, and say nothing about whether any
single group is spread or collapsed. This computes the group-level quantities instead.

  * ICC by one-way random-effects ANOVA (unequal cluster sizes), not the crude
    mean-within-variance ratio -- MSW is a pooled variance with the right df, and the
    naive ratio is biased when groups are small and uneven.
  * The design effect it implies, and the EFFECTIVE independent sample size. This is the
    consequence that matters: it converts "81% between" into how much independent
    information Q's outputs actually carry.
  * Group composition. Since 99.6% of Q's mass sits on {0.0,0.1} u {0.9,1.0}, the
    operational question is per GROUP: is it entirely on the low pair, entirely on the
    high pair, or does it straddle? Only a straddling group gives PPO any within-group
    signal at all -- everywhere else Q hands all 16 members effectively the same advantage.
  * Where the spread lives: within-group std against the group's own mean.

    python experiments/ac2/q_group_structure.py
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT)
    ap.add_argument("--min-members", type=int, default=2)
    args = ap.parse_args()

    groups = []
    with open(args.dataset, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if not r["is_replay_group"]:
                continue
            qs = r["summary"]["q_values"]
            if len(qs) >= args.min_members:
                groups.append((r, qs))

    k = len(groups)
    sizes = [len(q) for _, q in groups]
    N = sum(sizes)
    allq = [v for _, q in groups for v in q]
    gmean = {i: sum(q) / len(q) for i, (_, q) in enumerate(groups)}
    grand = sum(allq) / N

    # ---- one-way random-effects ANOVA with unequal cluster sizes -------------------
    msb = sum(len(q) * (gmean[i] - grand) ** 2 for i, (_, q) in enumerate(groups)) / (k - 1)
    msw = sum((v - gmean[i]) ** 2 for i, (_, q) in enumerate(groups) for v in q) / (N - k)
    m0 = (N - sum(s * s for s in sizes) / N) / (k - 1)
    icc = (msb - msw) / (msb + (m0 - 1) * msw)
    deff = 1 + (m0 - 1) * icc
    neff = N / deff

    print("=" * 78)
    print("NESTED STRUCTURE  (rollouts within prefix-sharing groups)")
    print("=" * 78)
    print("groups k                    : %d" % k)
    print("Q values N                  : %d" % N)
    print("cluster size: mean %.2f  median %d  min %d  max %d"
          % (N / k, statistics.median(sizes), min(sizes), max(sizes)))
    print("MS between                  : %.5f" % msb)
    print("MS within                   : %.5f" % msw)
    print("m0 (ANOVA cluster size)     : %.3f" % m0)
    print("ICC (one-way random effects): %.3f" % icc)
    print("  -> between-group share    : %.1f%%" % (100 * icc))
    print("  -> within-group share     : %.1f%%" % (100 * (1 - icc)))
    print("design effect 1+(m0-1)*ICC  : %.2f" % deff)
    print("EFFECTIVE independent n     : %.0f   (from %d raw Q values)" % (neff, N))

    # ---- group composition ---------------------------------------------------------
    def band(v):
        if v <= 0.1:
            return "lo"
        if v >= 0.9:
            return "hi"
        return "mid"

    comp = Counter()
    straddle = []
    for i, (r, q) in enumerate(groups):
        bs = set(band(v) for v in q)
        if bs == {"lo"}:
            comp["all low  (Q<=0.1)"] += 1
        elif bs == {"hi"}:
            comp["all high (Q>=0.9)"] += 1
        elif bs == {"mid"}:
            comp["all mid"] += 1
        elif "mid" in bs and len(bs) > 1:
            comp["touches mid"] += 1
            straddle.append((i, r, q))
        else:
            comp["STRADDLES lo/hi"] += 1
            straddle.append((i, r, q))

    print()
    print("=" * 78)
    print("GROUP COMPOSITION  (99.6%% of Q mass is on {0.0,0.1} u {0.9,1.0})")
    print("=" * 78)
    for kk, v in comp.most_common():
        print("  %-22s %5d  %5.1f%%  %s" % (kk, v, 100.0 * v / k, "#" * int(60.0 * v / k)))
    disc = sum(v for kk, v in comp.items() if "STRADDLE" in kk or "mid" in kk)
    print("\n  groups where Q separates members at all : %d (%.1f%%)" % (disc, 100.0 * disc / k))
    print("  groups where Q gives one flat advantage : %d (%.1f%%)" % (k - disc, 100.0 * (k - disc) / k))

    # ---- distinct values per group --------------------------------------------------
    print()
    print("=" * 78)
    print("DISTINCT Q VALUES PER GROUP")
    print("=" * 78)
    dv = Counter(len(set(q)) for _, q in groups)
    for d in sorted(dv):
        print("  %d value%s  %5d groups  %5.1f%%  %s"
              % (d, " " if d == 1 else "s", dv[d], 100.0 * dv[d] / k,
                 "#" * int(60.0 * dv[d] / k)))

    # ---- where the spread lives ------------------------------------------------------
    print()
    print("=" * 78)
    print("WITHIN-GROUP STD vs GROUP MEAN   (is the spread uniform, or only at the edges?)")
    print("=" * 78)
    bins = defaultdict(list)
    for i, (_, q) in enumerate(groups):
        bins[round(gmean[i], 1)].append(statistics.pstdev(q) if len(q) > 1 else 0.0)
    print("  %-12s %7s %10s" % ("group mean", "n", "mean within-std"))
    for b in sorted(bins):
        print("  %-12.1f %7d %10.3f  %s"
              % (b, len(bins[b]), sum(bins[b]) / len(bins[b]),
                 "#" * int(round(120 * sum(bins[b]) / len(bins[b])))))

    # ---- the between-level distribution: what the histogram SHOULD show --------------
    print()
    print("=" * 78)
    print("DECOMPOSED DISTRIBUTIONS  (what replaces the pooled histogram)")
    print("=" * 78)
    print("  group MEANS (the between-group level, k=%d):" % k)
    gm = Counter(round(gmean[i], 1) for i in range(k))
    for b in sorted(gm):
        print("    %.1f  %5d  %5.1f%%  %s" % (b, gm[b], 100.0 * gm[b] / k,
                                              "#" * int(60.0 * gm[b] / k)))
    print("\n  within-group DEVIATIONS Q - group mean (the within level, N=%d):" % N)
    dev = Counter(round(v - gmean[i], 1) for i, (_, q) in enumerate(groups) for v in q)
    for b in sorted(dev):
        print("    %+.1f  %5d  %5.1f%%  %s" % (b, dev[b], 100.0 * dev[b] / N,
                                               "#" * int(60.0 * dev[b] / N)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
