"""Hint-gap conjecturer dashboard page (optional; not used by the experiments in this repo).

Adds one PDF page of hint-gap-specific panels on top of the standard dashboard (which
already covers the CONJECTURER's generic training signals from metrics.jsonl: score/reward
mean, response length, entropy, loss, grad norm, PPO KL/clipping, LR, time-per-stage,
throughput — the run is a normal verl run, parsed by parse_fig_data in single-mode).

This module covers what the stock parser cannot see:

  - the hint-gap reward decomposition per step (gap, pass_plain vs pass_hint, solvable,
    group mixedness, conjecture formation health) from the per-row reward extras dumped
    into run_data*/rollouts/N.jsonl;
  - the PROVER side from run_data*/prover_rollouts/N.jsonl (per-(C,variant) pass-count
    distributions, proof lengths, proof-tag health);
  - phase wall-times + prover-update (update-P) optimizer metrics from
    run_data*/hint_gap_metrics.jsonl (loss/grad-norm/entropy/KL/clipfrac when
    SP_PROVER_TRAIN=1; panels render N/A on frozen-prover runs);
  - per-exemplar conjecturer reward from the run_data*/conjectures/N.jsonl cache
    (falls back to hashing the prompt text when the cache is absent).

Two entry points:

  parse:   python -m ac2.viz.hintgap_dashboard parse --run-dir experiments/<exp> \
               [--run-data run_data_conjecturer] --out analysis/<exp>_hintgap.json
  render:  handled by render_dashboard --hintgap-sidecar analysis/<exp>_hintgap.json
           (appends the page to the same PDF; see render_dashboard.py)

The sidecar parse is stdlib-only (mirrors parse_fig_data's contract); the figure builder
needs matplotlib/numpy and reuses dashboard_common styling.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Sidecar parser (stdlib-only)
# ---------------------------------------------------------------------------

_PROOF_TAG_RE = re.compile(r"<proof>", re.IGNORECASE)

_EXTRA_KEYS = ("score", "gap", "pass_plain", "pass_hint", "solvable",
               "mixed_plain", "mixed_hint", "has_conjecture", "too_long")


def _step_files(d: Path) -> list[tuple[int, Path]]:
    out = []
    for p in glob.glob(str(d / "*.jsonl")):
        stem = Path(p).stem
        if stem.isdigit():
            out.append((int(stem), Path(p)))
    return sorted(out)


def _percentile(sorted_vals: list[float], q: float) -> float | None:
    if not sorted_vals:
        return None
    idx = min(int(round(q / 100.0 * (len(sorted_vals) - 1))), len(sorted_vals) - 1)
    return float(sorted_vals[idx])


def _mean(vals: list[float]) -> float | None:
    return (sum(vals) / len(vals)) if vals else None


def parse_hintgap(run_dir: Path, run_data_name: str = "run_data_conjecturer") -> dict:
    rd = run_dir / run_data_name
    if not rd.is_dir() and (run_dir / "run_data").is_dir():
        rd = run_dir / "run_data"  # symlinked layouts

    steps: list[int] = []
    per_step: dict[str, dict[int, float | None]] = defaultdict(dict)
    passdist: dict[str, dict[int, list[int]]] = {"plain": {}, "hint": {}}
    exemplar_series: dict[str, dict[int, float]] = defaultdict(dict)
    k_inferred = 0

    # --- conjectures cache: exemplar_id per row index (may be absent on older runs) ---
    conj_meta: dict[int, dict[int, dict]] = {}
    for step, f in _step_files(rd / "conjectures"):
        rows = {}
        for line in open(f, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[int(r.get("i", len(rows)))] = r
        conj_meta[step] = rows

    # --- conjecturer rollout dumps: reward extras per row ---
    for step, f in _step_files(rd / "rollouts"):
        rows = []
        for line in open(f, encoding="utf-8"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if not rows:
            continue
        steps.append(step)
        cols = {k: [float(r[k]) for r in rows if isinstance(r.get(k), (int, float))]
                for k in _EXTRA_KEYS}
        for k in ("score", "gap"):
            per_step[f"hg_{k}_mean"][step] = _mean(cols[k])
            s = sorted(cols[k])
            for q in (10, 50, 90):
                per_step[f"hg_{k}_p{q}"][step] = _percentile(s, q)
        for k in ("pass_plain", "pass_hint"):
            per_step[f"hg_{k}_mean"][step] = _mean(cols[k])
        for k in ("solvable", "mixed_plain", "mixed_hint", "has_conjecture", "too_long"):
            per_step[f"hg_{k}_frac"][step] = _mean(cols[k])
        # per-exemplar reward: prefer the conjectures cache (row order = batch order);
        # fall back to a stable hash of the prompt text.
        meta = conj_meta.get(step, {})
        by_ex: dict[str, list[float]] = defaultdict(list)
        for i, r in enumerate(rows):
            sc = r.get("score")
            if not isinstance(sc, (int, float)):
                continue
            ex = (meta.get(i) or {}).get("exemplar_id")
            if not ex:
                inp = r.get("input") or ""
                ex = "ex_" + hashlib.sha1(inp[:2000].encode()).hexdigest()[:6]
            by_ex[str(ex)].append(float(sc))
        for ex, vals in by_ex.items():
            exemplar_series[ex][step] = _mean(vals)
        # conjecture char length percentiles (from the cache when present)
        lens = sorted(len(m.get("conjecture") or "") for m in meta.values() if m.get("conjecture"))
        if lens:
            for q in (50, 90):
                per_step[f"hg_conj_chars_p{q}"][step] = _percentile([float(x) for x in lens], q)

    # --- prover rollouts: pass-count dists, proof lengths, proof-tag health ---
    for step, f in _step_files(rd / "prover_rollouts"):
        group_pass: dict[str, list[int]] = defaultdict(list)
        proof_lens: list[float] = []
        n_rows = 0
        n_tag = 0
        for line in open(f, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_rows += 1
            uid = str(r.get("uid", ""))
            variant = uid.rsplit(":", 1)[-1] if ":" in uid else "plain"
            group_pass[uid].append(1 if (r.get("score") or 0) > 0 else 0)
            sol = r.get("solution") or ""
            if _PROOF_TAG_RE.search(sol):
                n_tag += 1
            proof_lens.append(float(len(sol)))
        if not n_rows:
            continue
        if step not in steps:
            steps.append(step)
        k_step = max((len(v) for v in group_pass.values()), default=0)
        k_inferred = max(k_inferred, k_step)
        hists = {"plain": defaultdict(int), "hint": defaultdict(int)}
        for uid, passes in group_pass.items():
            variant = uid.rsplit(":", 1)[-1]
            if variant not in hists:
                continue
            hists[variant][sum(passes)] += 1
        for variant in ("plain", "hint"):
            passdist[variant][step] = [hists[variant].get(c, 0) for c in range(k_step + 1)]
        s = sorted(proof_lens)
        for q in (50, 90, 99):
            per_step[f"hg_proof_chars_p{q}"][step] = _percentile(s, q)
        per_step["hg_proof_tag_frac"][step] = n_tag / n_rows

    # --- per-step aggregates + update-P metrics ---
    hm = rd / "hint_gap_metrics.jsonl"
    if hm.exists():
        for line in open(hm, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = r.get("step")
            if not isinstance(step, int):
                continue
            if step not in steps:
                steps.append(step)
            for k in ("t_prover_s", "t_judge_s", "t_update_p_s",
                      "n_valid", "n_solvable", "reward_mean"):
                v = r.get(k)
                if isinstance(v, (int, float)):
                    per_step[f"hg_{k}"][step] = float(v)
            pu = r.get("prover_update")
            if isinstance(pu, dict):
                for k, v in pu.items():
                    if isinstance(v, (int, float)):
                        per_step[f"hg_pu_{k.replace('/', '_')}"][step] = float(v)

    steps = sorted(set(steps))

    def as_list(series: dict[int, float | None]) -> list[float | None]:
        return [series.get(s) for s in steps]

    out = {
        "schema": "hintgap_v1",
        "steps": steps,
        "k": k_inferred,
        "series": {k: as_list(v) for k, v in per_step.items()},
        "exemplar_reward": {ex: as_list(v) for ex, v in sorted(exemplar_series.items())},
        "passdist": {
            variant: {str(s): passdist[variant].get(s) for s in steps}
            for variant in ("plain", "hint")
        },
        "run_data": str(rd),
    }
    return out


# ---------------------------------------------------------------------------
# Figure builder (matplotlib; called from render_dashboard)
# ---------------------------------------------------------------------------

def build_hintgap_figure(F, HG: dict):
    import matplotlib.pyplot as plt
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import dashboard_common as DC

    steps = HG.get("steps") or []
    x = np.array(steps, dtype=float)
    S = HG.get("series") or {}
    K = int(HG.get("k") or 0)

    def arr(key: str) -> "np.ndarray":
        vals = S.get(key) or []
        return np.array([np.nan if v is None else float(v) for v in vals], dtype=float)

    fig, axes = plt.subplots(4, 4, figsize=(26, 22))
    fig.subplots_adjust(top=0.90, hspace=0.5, wspace=0.28)
    try:
        DC.draw_header(fig, F, "hint-gap conjecturer panels")
    except Exception:
        fig.suptitle("hint-gap conjecturer panels", fontsize=13, fontweight="bold")
    A = axes.ravel()

    def title(i, name, ylabel="", note=""):
        DC.panel_title(A[i], i + 1, name, ylabel=ylabel)
        if note:
            A[i].set_xlabel(note, fontsize=6, color="gray")

    # (1) hint-gap reward (the conjecturer's actual training signal)
    y = arr("hg_score_mean")
    if DC.has(y):
        DC.line(A[0], x, y, "#2ca02c", "reward mean (= gap)")
        DC.line(A[0], x, DC.ema(y), "#2ca02c", "EMA", linestyle="--", alpha=0.5)
        lo, hi = arr("hg_score_p10"), arr("hg_score_p90")
        if DC.has(lo) and DC.has(hi):
            A[0].fill_between(x, lo, hi, color="#2ca02c", alpha=0.12, label="p10-p90")
        A[0].axhline(0.0, color="gray", lw=0.6)
        title(0, "Hint-gap reward", "reward",
              "reward = pass(C+hint)/K - pass(C)/K, judged on bare C")
    else:
        DC.na(A[0], 1, "Hint-gap reward")

    # (2) pass rates plain vs hinted
    yp, yh = arr("hg_pass_plain_mean"), arr("hg_pass_hint_mean")
    if DC.has(yp) or DC.has(yh):
        kk = float(K) if K else 1.0
        DC.line(A[1], x, yp / kk, "#d62728", "pass rate C (cold)")
        DC.line(A[1], x, yh / kk, "#1f77b4", "pass rate C+hint")
        A[1].set_ylim(0, 1)
        title(1, f"Prover pass rates (K={K})", "pass rate")
    else:
        DC.na(A[1], 2, "Prover pass rates")

    # (3) solvable + mixedness
    drew = False
    for key, color, label in (("hg_solvable_frac", "#9467bd", "solvable (pass_plain>0)"),
                              ("hg_mixed_plain_frac", "#d62728", "mixed plain (0<p<K)"),
                              ("hg_mixed_hint_frac", "#1f77b4", "mixed hint (0<p<K)")):
        y = arr(key)
        if DC.has(y):
            DC.line(A[2], x, y, color, label)
            drew = True
    if drew:
        A[2].set_ylim(0, 1)
        title(2, "Solvable & group mixedness", "fraction",
              "mixed groups are the only prover-gradient carriers (update-P)")
    else:
        DC.na(A[2], 3, "Solvable & group mixedness")

    # (4) conjecture formation health
    drew = False
    for key, color, label in (("hg_has_conjecture_frac", "#2ca02c", "has <conjecture> tag"),
                              ("hg_too_long_frac", "#d62728", "rejected too-long")):
        y = arr(key)
        if DC.has(y):
            DC.line(A[3], x, y, color, label)
            drew = True
    if drew:
        A[3].set_ylim(0, 1.05)
        title(3, "Conjecture formation health", "fraction")
    else:
        DC.na(A[3], 4, "Conjecture formation health")

    # (5)/(6) pass-count distributions (stacked), plain and hinted
    for idx, variant, name in ((4, "plain", "Pass-count dist — C (cold)"),
                               (5, "hint", "Pass-count dist — C+hint")):
        pd = (HG.get("passdist") or {}).get(variant) or {}
        counts = [pd.get(str(s)) for s in steps]
        kmax = max((len(c) - 1 for c in counts if c), default=0)
        if kmax > 0:
            cmap = DC.passcount_colormap(kmax + 1)
            specs = []
            for c in range(kmax + 1):
                y = np.array([(cc[c] if (cc and c < len(cc)) else 0) for cc in counts], dtype=float)
                specs.append((y, f"{c}/{kmax}", cmap(c / max(kmax, 1))))
            DC.stacked(A[idx], x, specs)
            DC.panel_title(A[idx], idx + 1, name, ylabel="conjectures", ncol=3)
        else:
            DC.na(A[idx], idx + 1, name)

    # (7) reward percentiles across the batch
    drew = False
    for q, color in ((10, "#1f77b4"), (50, "#2ca02c"), (90, "#d62728")):
        y = arr(f"hg_gap_p{q}")
        if DC.has(y):
            DC.line(A[6], x, y, color, f"gap p{q}")
            drew = True
    if drew:
        A[6].axhline(0.0, color="gray", lw=0.6)
        title(6, "Gap percentiles across conjectures", "gap")
    else:
        DC.na(A[6], 7, "Gap percentiles across conjectures")

    # (8) conjecture length
    drew = False
    for q, color in ((50, "#1f77b4"), (90, "#d62728")):
        y = arr(f"hg_conj_chars_p{q}")
        if DC.has(y):
            DC.line(A[7], x, y, color, f"chars p{q}")
            drew = True
    if drew:
        title(7, "Conjecture length", "chars")
    else:
        DC.na(A[7], 8, "Conjecture length", "needs run_data/conjectures cache")

    # (9) per-exemplar mean reward
    ex = HG.get("exemplar_reward") or {}
    if ex:
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
                   "#e377c2", "#7f7f7f"]
        for j, (name, vals) in enumerate(list(ex.items())[:8]):
            y = np.array([np.nan if v is None else float(v) for v in vals], dtype=float)
            DC.line(A[8], x, y, palette[j % len(palette)], str(name)[:22])
        A[8].axhline(0.0, color="gray", lw=0.6)
        title(8, "Reward by reference exemplar", "reward mean")
    else:
        DC.na(A[8], 9, "Reward by reference exemplar")

    # (10) prover proof length
    drew = False
    for q, color in ((50, "#1f77b4"), (90, "#ff7f0e"), (99, "#d62728")):
        y = arr(f"hg_proof_chars_p{q}")
        if DC.has(y):
            DC.line(A[9], x, y, color, f"proof chars p{q}")
            drew = True
    if drew:
        title(9, "Prover proof length", "chars")
    else:
        DC.na(A[9], 10, "Prover proof length")

    # (11) proof-tag health
    y = arr("hg_proof_tag_frac")
    if DC.has(y):
        DC.line(A[10], x, y * 100.0, "#2ca02c", "% solutions with <proof>")
        A[10].set_ylim(0, 105)
        title(10, "Prover format health", "%")
    else:
        DC.na(A[10], 11, "Prover format health")

    # (12) phase wall-times
    tp, tj, tu = arr("hg_t_prover_s"), arr("hg_t_judge_s"), arr("hg_t_update_p_s")
    if DC.has(tp) or DC.has(tj):
        specs = [(np.nan_to_num(tp), "prover gen", "#1f77b4"),
                 (np.nan_to_num(tj), "judge", "#ff7f0e"),
                 (np.nan_to_num(tu), "update-P", "#d62728")]
        DC.stacked(A[11], x, specs)
        DC.panel_title(A[11], 12, "Reward-phase wall time", ylabel="seconds")
    else:
        DC.na(A[11], 12, "Reward-phase wall time", "needs hint_gap_metrics.jsonl")

    # (13-15) update-P optimizer metrics (SP_PROVER_TRAIN=1 only). Key names come from the
    # worker's raw metric dict prefixed "prover/" and slash-sanitized (hg_pu_prover_*); exact
    # names vary across verl worker versions, so match by substring.
    pu_keys = [k for k in S if k.startswith("hg_pu_")]

    def pu_find(*needles):
        hits = []
        for k in pu_keys:
            kl = k.lower()
            if any(n in kl for n in needles):
                hits.append(k)
        return hits

    pu_panels = [
        (12, "Prover update: loss / grad norm", pu_find("loss", "grad_norm")),
        (13, "Prover update: entropy", pu_find("entropy")),
        (14, "Prover update: KL / clipfrac", pu_find("kl", "clipfrac")),
    ]
    palette2 = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
    for idx, name, keys in pu_panels:
        drew = False
        for j, key in enumerate(keys[:6]):
            y = arr(key)
            if DC.has(y):
                DC.line(A[idx], x, y, palette2[j % len(palette2)],
                        key.replace("hg_pu_prover_", ""))
                drew = True
        if drew:
            DC.panel_title(A[idx], idx + 1, name)
        else:
            DC.na(A[idx], idx + 1, name, "prover frozen (SP_PROVER_TRAIN=0)")

    # (16) identity / provenance text panel
    A[15].axis("off")
    n_valid = arr("hg_n_valid")
    lines = [
        f"steps with data: {len(steps)} ({steps[0]}-{steps[-1]})" if steps else "no steps",
        f"K (proofs per variant): {K or '?'}",
        f"run_data: {HG.get('run_data', '?')}",
        f"exemplars seen: {len(ex)}",
        f"last n_valid: {n_valid[~np.isnan(n_valid)][-1]:.0f}" if DC.has(n_valid) else "",
        "judge never sees the hint; all proofs judged on bare C",
    ]
    A[15].text(0.02, 0.95, "\n".join([ln for ln in lines if ln]),
               va="top", ha="left", fontsize=9, family="monospace",
               transform=A[15].transAxes)
    A[15].set_title("(16) Hint-gap identity", fontweight="bold", fontsize=10)

    return fig


# ---------------------------------------------------------------------------
# CLI (parse mode)
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("parse", help="build the hint-gap sidecar JSON")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--run-data", default=os.environ.get("SP_RUN_DATA_DIR", "run_data_conjecturer"))
    p.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "parse":
        out = parse_hintgap(args.run_dir, args.run_data)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out))
        print(f"hintgap sidecar: {args.out} ({len(out['steps'])} steps, K={out['k']})")


if __name__ == "__main__":
    main()
