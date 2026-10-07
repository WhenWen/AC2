#!/usr/bin/env python3
"""Diagnostics dashboard page (panels 1-18, plus replay-buffer panels).

Usage:
  python3 src/ac2/viz/diagnostics_dashboard.py fig_data.json "run title" out.png

More operational than the main page: it should make obvious whether a bad
curve is caused by the proposer, prover, judge, filtering, length
shaping, or infrastructure. Input is produced by ``parse_fig_data.py``.
Normally rendered through ``render_dashboard.py``.

Standalone: no imports from other viz packages.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import dashboard_common as DC  # noqa: E402
from dashboard_common import (  # noqa: E402
    CO, has, impact_color, line, na, panel_title, present, stacked,
)

plt.rcParams.update(
    {
        "font.size": 8.3,
        "axes.grid": True,
        "grid.alpha": 0.24,
        "axes.axisbelow": True,
        "axes.titlepad": 5,
    }
)


def _text_panel(
    ax, n: int, title: str, lines: list[str], *,
    mono: bool = True, fontsize: float = 7.0,
) -> None:
    ax.set_title(f"({n}) {title}", fontweight="bold", fontsize=10, loc="left")
    ax.axis("off")
    family = "monospace" if mono else None
    body = "\n".join(lines) if lines else "(none)"
    ax.text(
        0.0, 1.0, body, transform=ax.transAxes, va="top", ha="left",
        fontsize=fontsize, family=family, linespacing=1.25, parse_math=False,
    )


def _wrap(text: str, width: int) -> str:
    text = " ".join(str(text).split())
    return text[:width]


def _example_lines(
    examples: dict, categories: list[tuple[str, str]], max_rows: int, width: int,
) -> list[str]:
    """Render a list of slim-example rows into a text block.

    Each row is rendered with mode-aware fields: proposer rows include the
    proposition excerpt; prover rows show the proof. Judge excerpts on
    proposer rows are the "[correctness] X / [impact] Y" concatenation
    written by the two-mode reward path.
    """
    lines: list[str] = []
    for cat_key, cat_label in categories:
        rows = examples.get(cat_key) or []
        if not rows:
            continue
        lines.append(f"— {cat_label} ({len(rows)}) —")
        for r in rows[-max_rows:]:
            step = r.get("step")
            score = r.get("score")
            mode = r.get("mode", "?")
            corr = r.get("correctness_judge_score")
            imp = r.get("impact_judge_score")
            pjs = r.get("prover_judge_score")
            preward = r.get("proposer_reward")
            plen = r.get("proof_len_chars")
            prob = _wrap(r.get("input_excerpt", ""), width)
            if mode == "proposer":
                lines.append(
                    f"s{step} [{mode}] corr={corr} imp={imp} prew={preward} "
                    f"plen={plen}: {prob}"
                )
                prop = r.get("proposition_excerpt") or ""
                if prop:
                    lines.append(f"    Prop: {_wrap(prop, width)}")
            else:
                lines.append(
                    f"s{step} [{mode}] pjs={pjs} sc={score} plen={plen}: {prob}"
                )
            proof = r.get("proof_excerpt") or ""
            if proof:
                lines.append(f"    Proof: {_wrap(proof, width)}")
            response = r.get("judge_response") or r.get("judge_response_excerpt") or ""
            if response:
                lines.append(f"    > {_wrap(response, width)}")
    return lines or ["(no matching example rows)"]


def _lineage_dash_step(F: "DC.FigData") -> float | None:
    """Last parent's ``end_step`` (the dashed-history boundary), or None
    for a root run. Derived from ``DC.lineage_boundaries(F)`` (markers at
    ``end_step + 0.5``) so root runs stay visually unchanged."""
    bnds = DC.lineage_boundaries(F)
    if not bnds:
        return None
    try:
        return float(bnds[-1]["x"]) - 0.5
    except (TypeError, ValueError, KeyError):
        return None


def build_diag_figure(F: "DC.FigData", *, prover_only: bool | None = None) -> "plt.Figure":
    """Dispatch: render the 17-panel baseline diag layout, OR — when the
    parsed fig_data contains any ``prover_additional`` rows — the
    28-panel replay layout that splits the prover-side diagnostics
    into adjacent original/additional family panels (and adds four
    replay-only drilldowns).

    ``prover_only`` — when True, the 4 proposer-only diagnostics panels
    (6 proposer correctness group composition, 7 proposer impact histogram,
    8 proposer reward decomposition, 12 proposer examples) are omitted and
    the remaining panels reflow. When ``None`` (default), auto-detected via
    ``DC.is_prover_only(F)``.
    """
    if prover_only is None:
        prover_only = DC.is_prover_only(F)
    DC.set_lineage_dash_boundary(_lineage_dash_step(F))
    try:
        if _has_replay_diag(F):
            return _build_diag_replay(F, prover_only=prover_only)
        return _build_diag_baseline(F, prover_only=prover_only)
    finally:
        DC.set_lineage_dash_boundary(None)


# Panel numbers that read proposer-only fig_data keys and are dropped in
# prover-only runs.
_PROPOSER_ONLY_DIAG_PANELS = frozenset({6, 7, 8, 12})


def _build_diag_baseline(F: "DC.FigData", *, prover_only: bool = False) -> "plt.Figure":
    R = F.steps
    cfg = F.config
    have_steps = bool(len(R))

    # Panel specs in panel-number order.
    specs: list[tuple[int, callable]] = [
        (1, lambda ax, n: _panel_1_run_identity(ax, F, cfg)),
        (2, lambda ax, n: _panel_2_artifact_completeness(ax, F)),
        (3, lambda ax, n: _panel_3_parser_warnings(ax, F)),
        (4, lambda ax, n: _panel_4_generated_trained_counts(ax, R, F, have_steps)),
        (5, lambda ax, n: _panel_5_prover_group_composition(ax, R, F, have_steps)),
        (6, lambda ax, n: _panel_6_proposer_correctness_group_composition(ax, R, F, have_steps)),
        (7, lambda ax, n: _panel_7_proposer_impact_dist_histogram(ax, R, F, have_steps)),
        (8, lambda ax, n: _panel_8_proposer_reward_decomposition(ax, R, F)),
        (9, lambda ax, n: _panel_9_filter_composition_by_mode_and_score(ax, R, F, have_steps)),
        (10, lambda ax, n: _panel_10_length_overlong_by_mode(ax, R, F)),
        (11, lambda ax, n: _panel_11_judge_failure_examples(ax, F)),
        (12, lambda ax, n: _panel_12_proposer_examples(ax, F)),
        (13, lambda ax, n: _panel_13_prover_examples(ax, F)),
        (14, lambda ax, n: _panel_14_validation_drilldown(ax, R, F, have_steps, cfg)),
        (15, lambda ax, n: _panel_15_paired_theorem_view(ax, F)),
        (16, lambda ax, n: _panel_16_optimizer_infra_detail(ax, R, F)),
        (17, lambda ax, n: _panel_17_prover_judge_missing_failed_per_group(ax, R, F)),
        # 18: judge health buckets, relocated from main-page slot 9 (whose position now holds the
        # SNIS-weighted mean length). Local import avoids a module cycle.
        (18, lambda ax, n: __import__("ac2.viz.single_run_dashboard", fromlist=["x"])._panel_judge_health(ax, n, R, F)),
    ]
    # Replay-prefix runs: buffer-state panel (pinned 40, relocated from the main
    # page — buffer plumbing is a diagnostic; the pass-rate split stays on page 1 as 39).
    if has(F.arr("replay__coverage")):
        specs.append(
            (40, lambda ax, n: __import__(
                "ac2.viz.single_run_dashboard", fromlist=["x"]
            )._panel_replay_buffer(ax, n, R, F))
        )
        # 41: coverage-only companion, placed immediately right of 40 (own autoscaled
        # axis so the modest coverage growth isn't flattened by the ~30x-larger buffer size).
        specs.append(
            (41, lambda ax, n: __import__(
                "ac2.viz.single_run_dashboard", fromlist=["x"]
            )._panel_replay_coverage(ax, n, R, F))
        )
    if prover_only:
        specs = [s for s in specs if s[0] not in _PROPOSER_ONLY_DIAG_PANELS]

    n_active = len(specs)
    ncols = 4
    nrows = (n_active + ncols - 1) // ncols
    # Per-row figure height tracks the original baseline (27 in / 5 rows = 5.4 in/row).
    fig = plt.figure(figsize=(24, 5.4 * nrows))
    page_label = "v7 diagnostics" + (" (prover-only)" if prover_only else "")
    top = DC.draw_header(fig, F, page_label)
    gs = GridSpec(nrows, ncols, figure=fig, hspace=0.42, wspace=0.22, top=top)
    A = [fig.add_subplot(gs[i // ncols, i % ncols]) for i in range(n_active)]

    for slot, (n, render_fn) in enumerate(specs):
        render_fn(A[slot], n)

    DC.apply_vertical_markers(A, F)
    fig.subplots_adjust(left=0.04, right=0.975, bottom=0.04)
    return fig


# ---------------------------------------------------------------------------
# Replay (family-aware) diagnostics: used for runs with prover_additional rows.
# Splits diag panels 4, 5, 9, 10, 11, 13 into per-family variants and adds
# four replay-only drilldowns: replay provenance breakdown, seed drilldown,
# additional-statement token-count drilldown, additional-statement entropy
# drilldown. Baseline runs (no prover_additional rows) render the 17-panel
# layout above unchanged.
# ---------------------------------------------------------------------------

_FAMILY_INFO_DIAG = {
    "original": {
        "key_prefix": "v7__train__family__prover_original",
        "color": CO["prover_original"],
        "color_dark": CO["prover_original_dark"],
        "color_light": CO["prover_original_light"],
        "color_accent": CO["prover_original_accent"],
        "title_prefix": "Original-statement prover",
    },
    "additional": {
        "key_prefix": "v7__train__family__prover_additional",
        "color": CO["prover_additional"],
        "color_dark": CO["prover_additional_dark"],
        "color_light": CO["prover_additional_light"],
        "color_accent": CO["prover_additional_accent"],
        "title_prefix": "Additional-statement prover",
    },
    "proposer": {
        "key_prefix": "v7__train__family__proposer",
        "color": CO["proposer"],
        "color_dark": CO["proposer_dark"],
        "color_light": CO["proposer_light"],
        "color_accent": CO["proposer_accent"],
        "title_prefix": "Proposer",
    },
}


def _has_replay_diag(F: "DC.FigData") -> bool:
    """Detect replay runs by the presence of any ``prover_additional`` rows.
    Same logic as single_run_dashboard._has_replay; duplicated here to avoid a
    cross-module import that would tie diagnostics_dashboard's runtime to the
    main builder."""
    y = F.arr("v7__train__family__prover_additional__rows_generated")
    if not has(y):
        return False
    try:
        return bool(np.any(np.asarray(y, dtype=float) > 0))
    except (TypeError, ValueError):
        return False


def _default_family_for_cat(cat_key: str) -> str | None:
    """Backwards-compat fallback for slim examples written before the parser
    emitted a family field: proposer mode -> proposer family; prover mode ->
    prover_original family (assume not additional for old example data)."""
    if cat_key.startswith("proposer__"):
        return "proposer"
    if cat_key.startswith("prover__"):
        return "prover_original"
    return None


def _examples_for_family(
    examples_dict: dict, family: str,
) -> dict[str, list[dict]]:
    """Return the {cat_key: [rows]} subset of `examples_dict` whose rows are in
    `family`. Filters per-row via the parser's slim-sample ``family`` field,
    falling back to category-based assignment for old slim rows."""
    out: dict[str, list[dict]] = {}
    for cat_key, rows in (examples_dict or {}).items():
        kept = []
        for row in rows or []:
            row_family = row.get("family") or _default_family_for_cat(cat_key)
            if row_family == family:
                kept.append(row)
        out[cat_key] = kept
    return out


def _build_diag_replay(F: "DC.FigData", *, prover_only: bool = False) -> "plt.Figure":
    # NOTE: prover_only filtering not yet wired for replay; baseline-only for now.
    _ = prover_only
    """28-panel diagnostics layout (7x4) for replay runs."""
    R = F.steps
    fig = plt.figure(figsize=(24, 38))
    top = DC.draw_header(fig, F, "v7 diagnostics (replay)")
    gs = GridSpec(7, 4, figure=fig, hspace=0.42, wspace=0.22, top=top)
    A = [fig.add_subplot(gs[i // 4, i % 4]) for i in range(28)]

    cfg = F.config
    have_steps = bool(len(R))

    # Identity / parser provenance (kept).
    _panel_1_run_identity(A[0], F, cfg)
    _panel_2_artifact_completeness(A[1], F)
    _panel_3_parser_warnings(A[2], F)
    # Per-step counts by family (replaces panel 4).
    _panel_per_step_counts_by_family(A[3], R, F, have_steps)
    # Prover group composition split (replaces 5; adds 5b).
    _panel_prover_group_composition_family(A[4], "5a", R, F, have_steps, "original")
    _panel_prover_group_composition_family(A[5], "5b", R, F, have_steps, "additional")
    # Proposer composition / impact / reward (kept).
    _panel_6_proposer_correctness_group_composition(A[6], R, F, have_steps)
    _panel_7_proposer_impact_dist_histogram(A[7], R, F, have_steps)
    _panel_8_proposer_reward_decomposition(A[8], R, F)
    # Filter composition split into 3 families (replaces 9; adds 9b, 9c).
    _panel_filter_composition_family(A[9], "9a", R, F, have_steps, "proposer")
    _panel_filter_composition_family(A[10], "9b", R, F, have_steps, "original")
    _panel_filter_composition_family(A[11], "9c", R, F, have_steps, "additional")
    # Length & overlong split into 3 families (replaces 10; adds 10b, 10c).
    _panel_length_overlong_family(A[12], "10a", R, F, "proposer")
    _panel_length_overlong_family(A[13], "10b", R, F, "original")
    _panel_length_overlong_family(A[14], "10c", R, F, "additional")
    # Judge failure examples split (modifies 11 into 3 family sections).
    _panel_judge_failure_examples_family(A[15], "11a", F, "proposer")
    _panel_judge_failure_examples_family(A[16], "11b", F, "original")
    _panel_judge_failure_examples_family(A[17], "11c", F, "additional")
    # Proposer examples (kept).
    _panel_12_proposer_examples(A[18], F)
    # Prover examples split into original + additional (replaces 13; adds 13b).
    _panel_prover_examples_family(A[19], "13a", F, "original")
    _panel_prover_examples_family(A[20], "13b", F, "additional")
    # Validation + optimizer + per-group judge counts (kept; renumbered).
    _panel_14_validation_drilldown(A[21], R, F, have_steps, cfg)
    _panel_16_optimizer_infra_detail(A[22], R, F)
    _panel_17_prover_judge_missing_failed_per_group(A[23], R, F)
    # Replay-only drilldowns.
    _panel_replay_provenance_breakdown(A[24], F)
    _panel_replay_seed_drilldown(A[25], F)
    _panel_additional_token_drilldown(A[26], R, F, have_steps)
    _panel_additional_entropy_drilldown(A[27], R, F)

    DC.apply_vertical_markers(A, F)
    fig.subplots_adjust(left=0.04, right=0.975, bottom=0.03)
    return fig


# Per-family variants of the splittable panels.

def _panel_per_step_counts_by_family(ax, R, F, have_steps) -> None:
    """3-way per-step counts (proposer / prover_original / prover_additional)
    of generated / trained / filtered rows. Replaces baseline diag panel 4
    (mode-only)."""
    if not have_steps:
        na(ax, 4, "Per-step counts by family")
        return
    drew = False
    for fam, info in _FAMILY_INFO_DIAG.items():
        pre = info["key_prefix"]
        g = F.arr(f"{pre}__rows_generated")
        t = F.arr(f"{pre}__rows_trained")
        f_ = F.arr(f"{pre}__rows_filtered")
        if has(g):
            line(ax, R, g, info["color_light"], f"{fam} gen", marker=".", lw=1.0, ls="--")
            drew = True
        if has(t):
            line(ax, R, t, info["color"], f"{fam} trained", marker=".", lw=1.5)
            drew = True
        if has(f_):
            line(ax, R, f_, info["color_dark"], f"{fam} filtered", marker=".", lw=1.0, ls=":")
            drew = True
    if drew:
        ax.set_ylim(bottom=0)
        panel_title(ax, 4, "Per-step counts by family", "rows", loc="best", ncol=3)
    else:
        na(ax, 4, "Per-step counts by family", "no family__* rows_* keys")


def _panel_prover_group_composition_family(ax, n, R, F, have_steps, family) -> None:
    info = _FAMILY_INFO_DIAG[family]
    pre = info["key_prefix"]
    title = f"{info['title_prefix']} group composition"
    if not have_steps:
        na(ax, n, title)
        return
    if stacked(ax, R, [
        (F.arr(f"{pre}__group_allzero"), "all-zero (0/n)", CO["navy"]),
        (F.arr(f"{pre}__group_mixed"), "mixed (DAPO-trainable)", CO["gold"]),
        (F.arr(f"{pre}__group_allone"), "all-one (n/n)", CO["red"]),
    ], all_in_legend=True):
        panel_title(ax, n, title, "prompt groups", loc="upper left")
    else:
        na(ax, n, title, f"no {family}-family prover groups")


def _panel_filter_composition_family(ax, n, R, F, have_steps, family) -> None:
    info = _FAMILY_INFO_DIAG[family]
    pre = info["key_prefix"]
    title = f"{info['title_prefix']} filter composition"
    if not have_steps:
        na(ax, n, title)
        return
    kept = present(F.arr(f"{pre}__groups_trained"))
    filt = present(F.arr(f"{pre}__groups_filtered"))
    mixed = present(F.arr(f"{pre}__groups_mixed_de"))
    drew = False
    if np.any(kept + filt + mixed):
        width = 0.7
        ax.bar(R, kept, width=width, color=info["color"], label="kept")
        ax.bar(R, filt, width=width, color=info["color_light"],
               bottom=kept, label="filtered")
        if np.any(mixed):
            ax.bar(R, mixed, width=width, color=CO["red"],
                   bottom=kept + filt, label="mixed_de (!)")
        drew = True
    if drew:
        panel_title(ax, n, title, "groups", loc="upper left", ncol=2)
    else:
        na(ax, n, title, f"no {family}-family groups")


def _panel_length_overlong_family(ax, n, R, F, family) -> None:
    info = _FAMILY_INFO_DIAG[family]
    pre = info["key_prefix"]
    title = f"{info['title_prefix']} length & overlong"
    # Length percentiles on left axis: candidate (proposer) or proof (prover).
    if family == "proposer":
        drew_len = DC.percentile_family(ax, R, F, f"{pre}__candidate_len",
                                        "cand", info["color_accent"])
    else:
        drew_len = DC.percentile_family(ax, R, F, f"{pre}__proof_len",
                                        "proof", info["color_accent"])
    # Overlong rate on right axis.
    ax2 = ax.twinx(); ax2.grid(False)
    drew_ov = False
    over = F.arr(f"{pre}__overlong_rate", 100)
    if has(over):
        ax2.plot(R, over, color=info["color_dark"], ls=":", marker=".", lw=1.0,
                 label="overlong %")
        drew_ov = True
    if drew_ov:
        ax2.set_ylim(0, 105); ax2.set_ylabel("% overlong")
        ax2.legend(fontsize=6.0, loc="upper right", framealpha=0.85)
    if drew_len or drew_ov:
        ax.set_ylim(bottom=0)
        panel_title(ax, n, title, "chars", loc="upper left", ncol=2)
    else:
        na(ax, n, title, f"no {family}-family length/overlong data")


def _panel_judge_failure_examples_family(ax, n, F, family) -> None:
    """One judge-failure-examples section per family. Replaces/modifies
    baseline diag panel 11. For prover_additional rows,
    replay_provenance is appended to each example line."""
    info = _FAMILY_INFO_DIAG[family]
    title = f"{info['title_prefix']} judge failure examples"
    ex_all = (F.train_jsonl.get("examples") or {})
    ex_fam = _examples_for_family(ex_all, _family_for_filter(family))
    # Failure categories for this family.
    if family == "proposer":
        cats = [
            ("proposer__judge_http_error", "HTTP error"),
            ("proposer__judge_truncated", "truncated"),
            ("proposer__judge_parse_failed", "parse failed"),
            ("proposer__missing_proposition", "missing proposition"),
        ]
    else:
        cats = [
            ("prover__judge_http_error", "HTTP error"),
            ("prover__judge_truncated", "truncated"),
            ("prover__judge_parse_failed", "parse failed"),
            ("prover__missing_proof", "missing proof"),
        ]
    lines = _family_example_lines(ex_fam, cats, family=family,
                                  max_rows=1, width=58)
    _text_panel(ax, n, title, lines, fontsize=5.8)


def _panel_prover_examples_family(ax, n, F, family) -> None:
    """Per-family prover examples. Replaces baseline panel 13 (original) and
    adds an adjacent additional panel. Additional-
    statement examples include replay provenance (proposition_uid,
    seed_statement_uid, source_step, source_completion_index)."""
    info = _FAMILY_INFO_DIAG[family]
    title = f"{info['title_prefix']} examples"
    ex_all = (F.train_jsonl.get("examples") or {})
    ex_fam = _examples_for_family(ex_all, _family_for_filter(family))
    cats = [
        ("prover__correct", "correct (prover_judge_score=1)"),
        ("prover__incorrect_wellformed", "incorrect well-formed (prover_judge_score=0)"),
    ]
    lines = _family_example_lines(ex_fam, cats, family=family,
                                  max_rows=3, width=58)
    _text_panel(ax, n, title, lines, fontsize=5.8)


def _family_for_filter(family: str) -> str:
    """Map the panel-side family arg ('proposer'/'original'/'additional') to
    the slim-sample family value ('proposer'/'prover_original'/'prover_additional')."""
    return {"proposer": "proposer",
            "original": "prover_original",
            "additional": "prover_additional"}[family]


def _family_example_lines(
    examples_dict: dict, cats: list[tuple[str, str]], *,
    family: str, max_rows: int, width: int,
) -> list[str]:
    """Like _example_lines but, for family == "additional" rows, appends a
    one-line replay-provenance summary after each example. Renders a clear
    header that says how many examples per category and the family."""
    out: list[str] = []
    base = _example_lines(examples_dict, cats, max_rows=max_rows, width=width)
    out.extend(base)
    if family == "additional":
        # Add a provenance trailer per row.
        prov_lines: list[str] = []
        for cat_key, _ in cats:
            for row in (examples_dict.get(cat_key) or [])[:max_rows]:
                prov = row.get("replay_provenance") or {}
                if not prov:
                    continue
                ssu = prov.get("seed_statement_uid", "?")
                srid = prov.get("source_run_id", "?")
                sstep = prov.get("source_step", "?")
                sci = prov.get("source_completion_index", "?")
                puid = prov.get("proposition_uid")
                if puid:
                    # Compact representation if proposition_uid is the canonical
                    # `<seed>/run=<rid>/step=<step>/completion=<ci>` form.
                    prov_lines.append(f"  prov: prop_uid={puid[:60]}")
                else:
                    prov_lines.append(
                        f"  prov: seed={ssu[:40]} run={srid[:34]} "
                        f"step={sstep} ci={sci}")
        if prov_lines:
            out.append("")
            out.append("[replay provenance]")
            out.extend(prov_lines[:max_rows * len(cats)])
    return out


# Replay-only diagnostic panels.

def _panel_replay_provenance_breakdown(ax, F) -> None:
    """Top-K table of (source_run_id, source_step) breakdown.
    Columns: source_step | n | pass@1 | overlong | resp_len mean.
    Reads global.replay.by_source_run_step (parser stage 1)."""
    r = (F.glob.get("replay") or {}).get("by_source_run_step") or []
    if not r:
        na(ax, "P1", "Replay provenance breakdown",
           "no global.replay.by_source_run_step\n(rebuild sidecar with provenance + re-parse)")
        return
    # Sort by step then source_run_id; keep up to MAX_ROWS.
    MAX_ROWS = 24
    r2 = sorted(r, key=lambda x: (x.get("source_step") or -1, x.get("source_run_id") or ""))
    if len(r2) > MAX_ROWS:
        r2 = r2[:MAX_ROWS]
        truncated = True
    else:
        truncated = False
    header = f"{'src_step':>9}  {'n':>5}  {'pass@1':>7}  {'overlong%':>9}  {'resp_p50':>9}"
    rows = [header, "-" * len(header)]
    for x in r2:
        p1 = x.get("pass_at_1")
        ovr = x.get("overlong_rate")
        rlm = x.get("resp_len_mean")
        rows.append(
            f"{(x.get('source_step') or '?'):>9}  "
            f"{x.get('n', 0):>5}  "
            f"{(f'{p1*100:6.1f}%' if isinstance(p1,(int,float)) else '   n/a'):>7}  "
            f"{(f'{ovr*100:7.1f}%' if isinstance(ovr,(int,float)) else '     n/a'):>9}  "
            f"{(f'{rlm:9.0f}' if isinstance(rlm,(int,float)) else '      n/a'):>9}"
        )
    if truncated:
        rows.append(f"... ({len(r) - MAX_ROWS} more rows truncated; full table in fig_data.global.replay)")
    distinct_runs = sorted({(x.get("source_run_id") or "?")[:32] for x in r})
    title = f"Replay provenance breakdown (by source_step; runs: {', '.join(distinct_runs[:2])}{'...' if len(distinct_runs) > 2 else ''})"
    _text_panel(ax, "P1", title, rows, fontsize=5.6)


def _panel_replay_seed_drilldown(ax, F) -> None:
    """Top-K table of seed_statement_uid drilldown.
    Shows top failure-rate seeds (rows where the prover never solved the
    replayed conjecture)."""
    r = (F.glob.get("replay") or {}).get("by_seed_statement_uid") or {}
    rows_list = r.get("top_by_failure") or []
    n_distinct = r.get("n_distinct", 0)
    if not rows_list:
        na(ax, "P2", "Replay seed drilldown",
           "no global.replay.by_seed_statement_uid\n(rebuild sidecar with provenance + re-parse)")
        return
    MAX_ROWS = 20
    show = rows_list[:MAX_ROWS]
    header = f"{'fail%':>6}  {'n':>3}  {'overlong%':>9}  seed_statement_uid"
    out = [f"Seeds with highest fail-rate ({n_distinct} distinct seeds in run):",
           "", header, "-" * len(header)]
    for x in show:
        out.append(
            f"{x.get('fail_rate', 0) * 100:5.1f}%  "
            f"{x.get('n', 0):>3}  "
            f"{x.get('overlong_rate', 0) * 100:7.1f}%  "
            f"{(x.get('seed_statement_uid') or '?')[:48]}"
        )
    if len(rows_list) > MAX_ROWS:
        out.append(f"... ({len(rows_list) - MAX_ROWS} more in fig_data.global.replay.by_seed_statement_uid.top_by_failure)")
    _text_panel(ax, "P2", "Replay seed drilldown (top failures)", out, fontsize=5.6)


def _panel_additional_token_drilldown(ax, R, F, have_steps) -> None:
    """Per-step response-token p50/p90/p99 for prover_additional rows.

    Candidate character length is included as non-token context.
    """
    title = "Additional-statement response-token drilldown"
    if not have_steps:
        na(ax, "P3", title); return
    pre = "v7__train__family__prover_additional"
    drew = False
    drew |= DC.percentile_family(ax, R, F, f"{pre}__resp_len",
                                 "resp", CO["prover_additional_dark"])
    drew |= DC.percentile_family(ax, R, F, f"{pre}__candidate_len",
                                 "candidate_chars", CO["gray"])
    if drew:
        ax.set_ylim(bottom=0)
        panel_title(ax, "P3", title, "tokens / chars", loc="upper left", ncol=2)
    else:
        na(ax, "P3", title,
           "no prover_additional token/length percentiles\n(sidecar absent, or no token fields in dump)")


def _panel_additional_entropy_drilldown(ax, R, F) -> None:
    """Per-step rollout-entropy for prover_additional rows.
    The source-specific entropy fields
    (rollout/entropy_prover_additional_statements) are NOT in current dumps,
    so this panel renders N/A and names exactly which entropy fields are
    missing."""
    na(ax, "P4", "Additional-statement entropy drilldown",
       "no source-specific entropy fields in dump\n"
       "(rollout/entropy_prover_additional_statements not logged)")


# ---------------------------------------------------------------------------
# Panel implementations.
# ---------------------------------------------------------------------------


def _panel_1_run_identity(ax, F, cfg) -> None:
    def cv(k, default="?"):
        v = cfg.get(k)
        return default if v in (None, "") else v

    id_lines = [
        f"run_id       {F.run_id}",
        f"experiment   {cv('experiment')}",
        f"cohort       {cv('cohort')}",
        f"nodes        {cv('nnodes')}   total_steps {cv('total_steps')}",
        f"problem b t/g {cv('train_problem_batch_size')}/{cv('gen_problem_batch_size')}",
        f"row     b t/g/m  {cv('train_batch_size')}/{cv('gen_batch_size')}/{cv('ppo_mini_batch_size')}",
        f"rollout n    train {cv('train_rollout_n', cv('rollout_n'))}   "
        f"val {cv('val_rollout_n', cv('val_n_default'))}",
        f"reward wkrs  {cv('reward_num_workers')}   save_freq {cv('save_freq')}   test_freq {cv('test_freq_default')}",
        f"actor lr     {cv('actor_lr')}   hot_fix_lr {cv('hot_fix_lr', '(unset)')}   kl coef {cv('actor_kl_loss_coef')}",
        f"resp cap     {cv('max_response_length_default')}   judge max tok {cv('judge_max_tokens_default')}",
        f"overlong     len {cv('overlong_buffer_len_default')}  factor {cv('overlong_penalty_factor_default')}",
        f"actor model  {cv('actor_model')}",
        f"judge model  {cv('judge_model')}",
        f"verl head    {cv('verl_head')}",
        f"code head    {cv('riemann_head')}",
        f"started_at   {cv('started_at')}   host {cv('host')}",
        f"W&B url      {cv('wandb_url', 'N/A')}",
    ]
    _text_panel(ax, 1, "Run identity / config", id_lines, fontsize=6.4)


def _panel_2_artifact_completeness(ax, F) -> None:
    src = F.d.get("source", {}) if isinstance(F.d.get("source"), dict) else {}
    tj, vj = F.train_jsonl, F.val_jsonl
    art = F.d.get("artifacts", {}) if isinstance(F.d.get("artifacts"), dict) else {}
    union_steps = F.per_step.get("steps", [])
    span = f"{union_steps[0]}-{union_steps[-1]}" if union_steps else "—"
    n_metric = art.get("n_metric_steps", "?")
    metric_last = art.get("metric_last_step")
    metric_last = "—" if metric_last is None else metric_last
    has_mode_entropy = "yes" if art.get("has_mode_entropy_metrics") else "no (showing global only)"
    art_lines = [
        f"metrics.jsonl   rows={n_metric}  last_step={metric_last}",
        f"train rollouts  files={tj.get('n_step_files', 0)}  rows={tj.get('n', 0)}",
        f"val rollouts    files={vj.get('n_step_files', 0)}  rows={vj.get('n', 0)}",
        f"step span       {span}  ({len(union_steps)} steps shown)",
        f"manifest        {'yes' if F.manifest else 'MISSING'}",
        f"mode entropy    {has_mode_entropy}",
        "",
        "sources:",
        f"  run dir  {_wrap(src.get('durable_run_dir', ''), 64)}",
        f"  metrics  {_wrap(src.get('metrics_jsonl', ''), 64)}",
        f"  train    {_wrap(src.get('train_rollout_glob', ''), 64)}",
        f"  val      {_wrap(src.get('val_rollout_glob', ''), 64)}",
    ]
    _text_panel(ax, 2, "Artifact completeness", art_lines, fontsize=6.4)


def _panel_3_parser_warnings(ax, F) -> None:
    warn_lines: list[str] = []
    for w in (F.warnings or []):
        words = str(w).split()
        cur = ""
        for word in words:
            if len(cur) + len(word) + 1 > 58:
                warn_lines.append(("• " + cur) if not cur.startswith(("•", " ")) else "  " + cur)
                cur = word
                continue
            cur = f"{cur} {word}".strip()
        if cur:
            warn_lines.append(("• " + cur))
    _text_panel(ax, 3, "Parser warnings", warn_lines or ["(no warnings)"], fontsize=6.2)


def _panel_4_generated_trained_counts(ax, R, F, have_steps) -> None:
    """Bars: per-step generated / trained / filtered, by mode."""
    if not have_steps:
        na(ax, 4, "Generated/trained counts by mode")
        return
    line(ax, R, F.arr("v7__train__proposer__rows_generated"),
         CO["proposer_light"], "prop gen", marker=".", lw=1.1, ls="--")
    line(ax, R, F.arr("v7__train__prover__rows_generated"),
         CO["prover_light"], "prov gen", marker=".", lw=1.1, ls="--")
    line(ax, R, F.arr("v7__train__proposer__rows_trained"),
         CO["proposer"], "prop trained", marker=".", lw=1.5)
    line(ax, R, F.arr("v7__train__prover__rows_trained"),
         CO["prover"], "prov trained", marker=".", lw=1.5)
    line(ax, R, F.arr("v7__train__proposer__rows_filtered"),
         CO["red"], "prop filtered", marker=".", lw=1.0, ls=":")
    line(ax, R, F.arr("v7__train__prover__rows_filtered"),
         CO["salmon"], "prov filtered", marker=".", lw=1.0, ls=":")
    if ax.get_legend_handles_labels()[0]:
        ax.set_ylim(bottom=0)
        panel_title(ax, 4, "Per-step counts by mode", "rows", loc="best", ncol=2)
    else:
        na(ax, 4, "Generated/trained counts by mode")


def _panel_5_prover_group_composition(ax, R, F, have_steps) -> None:
    if have_steps and stacked(
        ax, R,
        [
            (F.arr("v7__train__prover__group_allzero"), "all-zero (0/n)", CO["navy"]),
            (F.arr("v7__train__prover__group_mixed"), "mixed (DAPO-trainable)", CO["gold"]),
            (F.arr("v7__train__prover__group_allone"), "all-one (n/n)", CO["red"]),
        ],
        all_in_legend=True,
    ):
        panel_title(ax, 5, "Prover group composition", "prompt groups", loc="upper left")
        ax.text(0.02, -0.20,
                "Groups by prover_judge_score variation. Mixed groups give DAPO useful "
                "signal; all-zero / all-one groups are dropped by std>0 filter.",
                transform=ax.transAxes, fontsize=6.0, color="gray", va="top")
    else:
        na(ax, 5, "Prover group composition")


def _panel_6_proposer_correctness_group_composition(ax, R, F, have_steps) -> None:
    if have_steps and stacked(
        ax, R,
        [
            (F.arr("v7__train__proposer__group_allzero"), "all-zero (0/n)", CO["navy"]),
            (F.arr("v7__train__proposer__group_mixed"), "mixed", CO["gold"]),
            (F.arr("v7__train__proposer__group_allone"), "all-one (n/n)", CO["red"]),
        ],
        all_in_legend=True,
    ):
        panel_title(ax, 6, "Proposer correctness group composition", "prompt groups", loc="upper left")
    else:
        na(ax, 6, "Proposer correctness group composition")


def _panel_7_proposer_impact_dist_histogram(ax, R, F, have_steps) -> None:
    """Distribution view (vs main panel 10's % stack): plot impact COUNTS,
    so dashboards see absolute volume of each impact level. Complements the
    main page's per-step fraction view."""
    if not have_steps:
        na(ax, 7, "Proposer impact level counts")
        return
    # Drive the loop from fig_data["global"]["impact_levels"] (the rubric
    # width, 0..4 for the current impact judge). Fall back to the 0..3 tuple
    # for older fig_data files that do not record it.
    impact_levels = F.glob.get("impact_levels", [0, 1, 2, 3])
    specs = [
        (F.arr(f"v7__train__proposer__impact_level_count__l{lvl}"),
         f"Impact {lvl}", impact_color(lvl))
        for lvl in impact_levels
    ]
    if stacked(ax, R, specs, all_in_legend=True):
        panel_title(ax, 7, "Proposer impact level counts", "rollouts", loc="upper left", ncol=2)
        ax.text(0.02, -0.20,
                "Absolute counts (vs main panel 10 fractions). Counts include rows with "
                "short-circuit impact=0; see panel 8 for the reward-decomposition view.",
                transform=ax.transAxes, fontsize=6.0, color="gray", va="top")
    else:
        na(ax, 7, "Proposer impact level counts")


def _panel_8_proposer_reward_decomposition(ax, R, F) -> None:
    """Means side-by-side with shaded length-penalty band."""
    correctness = F.arr("v7__train__proposer__correctness_judge_score_mean")
    impact_app = F.arr("v7__train__proposer__impact_applied_mean")
    pre_len = F.arr("v7__train__proposer__proposer_pre_length_score_mean")
    final = F.arr("v7__train__proposer__proposer_reward_mean")
    line(ax, R, correctness, CO["proposer"], "correctness 0/1", marker=".", lw=1.2)
    line(ax, R, impact_app, CO["impact2"], "impact_applied", marker=".", lw=1.2)
    line(ax, R, pre_len, CO["proposer_accent"], "pre-length", marker="o", ms=3, lw=1.5)
    line(ax, R, final, CO["proposer_dark"], "final reward", marker="o", ms=4, lw=2.0)
    # Shade pre_length -> final as the length-penalty consumption.
    if has(pre_len) and has(final):
        ax.fill_between(R, present(pre_len), present(final),
                        color=CO["red"], alpha=0.15, label="length penalty band")
    if ax.get_legend_handles_labels()[0]:
        ax.axhline(0, color="black", lw=0.6)
        panel_title(ax, 8, "Proposer reward decomposition", "reward", loc="best", ncol=2)
    else:
        na(ax, 8, "Proposer reward decomposition")


def _panel_9_filter_composition_by_mode_and_score(ax, R, F, have_steps) -> None:
    """For each mode: filtered vs trained vs mixed_de stacked."""
    if not have_steps:
        na(ax, 9, "Filter composition by mode")
        return
    # Stack proposer above prover by offsetting x slightly.
    width = 0.4
    xp = R - width / 2
    xv = R + width / 2

    p_filt = present(F.arr("v7__train__proposer__groups_filtered"))
    p_kept = present(F.arr("v7__train__proposer__groups_trained"))
    p_mixed = present(F.arr("v7__train__proposer__groups_mixed_de"))
    r_filt = present(F.arr("v7__train__prover__groups_filtered"))
    r_kept = present(F.arr("v7__train__prover__groups_trained"))
    r_mixed = present(F.arr("v7__train__prover__groups_mixed_de"))

    drew = False
    if np.any(p_filt + p_kept + p_mixed):
        ax.bar(xp, p_kept, width=width, color=CO["proposer"], label="prop kept")
        ax.bar(xp, p_filt, width=width, color=CO["proposer_light"],
               bottom=p_kept, label="prop filtered")
        ax.bar(xp, p_mixed, width=width, color=CO["red"],
               bottom=p_kept + p_filt, label="prop mixed_de (!)")
        drew = True
    if np.any(r_filt + r_kept + r_mixed):
        ax.bar(xv, r_kept, width=width, color=CO["prover"], label="prov kept")
        ax.bar(xv, r_filt, width=width, color=CO["prover_light"],
               bottom=r_kept, label="prov filtered")
        ax.bar(xv, r_mixed, width=width, color=CO["salmon"],
               bottom=r_kept + r_filt, label="prov mixed_de (!)")
        drew = True
    if drew:
        panel_title(ax, 9, "Filter composition by mode", "groups", loc="upper left", ncol=2)
        ax.text(0.02, -0.20,
                "side-by-side per step: left=proposer, right=prover. "
                "mixed_de means a group has both kept+filtered rows (shouldn't happen).",
                transform=ax.transAxes, fontsize=6.0, color="gray", va="top")
    else:
        na(ax, 9, "Filter composition by mode")


def _panel_10_length_overlong_by_mode(ax, R, F) -> None:
    """Length percentiles + overlong rate, both modes.

    Prover proof_len shows the extended distribution-body set (p50/60/70/80/
    90/99) with per-percentile viridis colors — same treatment as the main
    panel 5 (prover resp_len). The proposer candidate_len side keeps the
    historical 3-percentile (p50/p90/p99) single-color look so mixed-mode
    runs stay legible (no 12-color clash with the prover proof palette)."""
    drew = False
    drew |= DC.percentile_family(
        ax, R, F, "v7__train__proposer__candidate_len", "prop cand", CO["proposer_accent"],
    )
    drew |= DC.percentile_family(
        ax, R, F, "v7__train__prover__proof_len", "prov proof", CO["prover_accent"],
        percentiles=(50, 60, 70, 80, 90, 99),
        color_per_percentile=True,
    )
    ax2 = ax.twinx()
    ax2.grid(False)
    drew2 = False
    pol = F.arr("v7__train__proposer__overlong_rate", 100)
    rol = F.arr("v7__train__prover__overlong_rate", 100)
    if has(pol):
        ax2.plot(R, pol, color=CO["proposer_dark"], ls=":", marker=".", lw=1.0,
                 label="prop overlong %")
        drew2 = True
    if has(rol):
        ax2.plot(R, rol, color=CO["prover_dark"], ls=":", marker=".", lw=1.0,
                 label="prov overlong %")
        drew2 = True
    if drew2:
        ax2.set_ylim(0, 105)
        ax2.set_ylabel("% overlong")
        ax2.legend(fontsize=6.0, loc="upper right", framealpha=0.85)
    if drew or drew2:
        ax.set_ylim(bottom=0)
        panel_title(ax, 10, "Length & overlong by mode", "chars", loc="upper left", ncol=2)
    else:
        na(ax, 10, "Length & overlong by mode")


def _panel_11_judge_failure_examples(ax, F) -> None:
    ex = (F.train_jsonl.get("examples") or {})
    # Aggregate failures across both modes; the row's own mode field tells
    # the reader which one each example came from.
    cats = [
        ("proposer__judge_http_error", "[prop] HTTP error"),
        ("prover__judge_http_error", "[prov] HTTP error"),
        ("proposer__judge_truncated", "[prop] truncated"),
        ("prover__judge_truncated", "[prov] truncated"),
        ("proposer__judge_parse_failed", "[prop] parse failed"),
        ("prover__judge_parse_failed", "[prov] parse failed"),
        ("proposer__missing_proposition", "[prop] missing proposition"),
        ("prover__missing_proof", "[prov] missing proof"),
    ]
    _text_panel(
        ax, 11, "Judge failure examples",
        _example_lines(ex, cats, max_rows=1, width=58),
        fontsize=5.8,
    )


def _panel_12_proposer_examples(ax, F) -> None:
    ex = (F.train_jsonl.get("examples") or {})
    cats = [
        ("proposer__correct_high_impact", "correct high-impact (corr=1, imp>=2)"),
        ("proposer__correct_low_impact", "correct low-impact (corr=1, imp<2)"),
        ("proposer__incorrect_wellformed", "incorrect well-formed (corr=0)"),
    ]
    _text_panel(
        ax, 12, "Proposer examples",
        _example_lines(ex, cats, max_rows=2, width=58),
        fontsize=5.8,
    )


def _panel_13_prover_examples(ax, F) -> None:
    ex = (F.train_jsonl.get("examples") or {})
    cats = [
        ("prover__correct", "correct (prover_judge_score=1)"),
        ("prover__incorrect_wellformed", "incorrect well-formed (prover_judge_score=0)"),
    ]
    _text_panel(
        ax, 13, "Prover examples",
        _example_lines(ex, cats, max_rows=3, width=58),
        fontsize=5.8,
    )


def _panel_14_validation_drilldown(ax, R, F, have_steps, cfg) -> None:
    """Prover validation drill-down. The val parquet is prover-only by
    design (see ``ac2.data.prepare_fineproofs``); the launcher's
    `val-core/<ds>/acc/mean@N` headline is therefore a clean binary signal
    and these per-step lines show the prover-side detail."""
    if not have_steps:
        na(ax, 14, "Prover validation drill-down")
        return
    r_val = F.arr("v7__val__prover__score_mean_generated", 100)
    r_pjs = F.arr("v7__val__prover__prover_judge_score_mean", 100)
    # Also include proposer val curves in case a run opts into proposer
    # validation; they render as empty in the default prover-only setup.
    p_val = F.arr("v7__val__proposer__score_mean_generated", 100)
    p_corr = F.arr("v7__val__proposer__correctness_judge_score_mean", 100)
    line(ax, R, r_val, CO["prover_light"], "prover val score", marker="o", ms=3, ls="--")
    line(ax, R, r_pjs, CO["prover"], "prover val pjs", marker="o", ms=4)
    line(ax, R, p_val, CO["proposer_light"], "proposer val score (opt-in)",
         marker="o", ms=3, ls="--")
    line(ax, R, p_corr, CO["proposer"], "proposer val correctness (opt-in)",
         marker="o", ms=4)
    if ax.get_legend_handles_labels()[0]:
        ax.set_ylim(bottom=0)
        panel_title(ax, 14, "Prover validation drill-down", "%", loc="best", ncol=2)
        tf = cfg.get("test_freq_default", 10)
        ax.text(0.02, -0.20,
                f"val runs every test_freq={tf}. v7 val parquet is prover-only "
                "by design; proposer lines appear only if --val_modes was "
                "overridden to include proposer.",
                transform=ax.transAxes, fontsize=6.0, color="gray", va="top")
    else:
        na(ax, 14, "Prover validation drill-down",
           f"no validation dumps yet\n(runs at test_freq={cfg.get('test_freq_default', 10)})")


def _panel_15_paired_theorem_view(ax, F) -> None:
    """N/A by default — current launcher does not dump uid / extra_info
    source keys. Activates when the dump writer is extended."""
    src = F.d.get("source", {}) if isinstance(F.d.get("source"), dict) else {}
    msg = (
        "N/A — paired view needs uid + data_source + extra_info.{split,index}\n"
        "in the train rollout dump. Current launcher does not write these.\n"
        "\n"
        "Once added, this panel joins proposer/prover rows for the same\n"
        "source theorem and shows: prop high-impact + prov success/failure,\n"
        "prop low-impact + prov success/failure, both-fail."
    )
    _text_panel(ax, 15, "Paired theorem view", [msg], mono=False, fontsize=7.0)


def _panel_16_optimizer_infra_detail(ax, R, F) -> None:
    """Compact optimizer summary + training/rollout/judge throughput.

    Main page already carries the standalone LR/grad/loss/KL/clip/entropy
    panels (15-20). This diagnostics panel adds the mini-batch token stats
    and tokens/sec/node phase throughput that doesn't have a main-page
    slot."""
    line(ax, R, F.arr("actor__grad_norm"), CO["red"], "grad norm", marker=".")
    line(ax, R, F.arr_first(["actor__loss", "actor__pg_loss"]),
         CO["blue"], "loss", marker=".")
    line(ax, R, F.arr("actor__entropy"), CO["teal"], "entropy", marker=".", ls="--")
    ax2 = ax.twinx()
    ax2.grid(False)
    drew2 = False
    try:
        node_count = float(F.cfg("nnodes") or 1)
    except (TypeError, ValueError):
        node_count = 1.0
    node_count = max(node_count, 1.0)

    def per_node_rate(tokens: np.ndarray, seconds: np.ndarray) -> np.ndarray:
        return np.divide(
            tokens,
            seconds * node_count,
            out=np.full(tokens.shape, np.nan, dtype=float),
            where=seconds > 0,
        )

    train_tps = per_node_rate(
        F.arr("perf__total_num_tokens"),
        F.arr_first(["timing_s__update_actor"]),
    )
    tps_roll = F.arr_first(["perf__tokens_per_sec__pi1_rollout"]) / node_count
    tps_R = F.arr_first(["perf__tokens_per_sec__R"]) / node_count
    if has(train_tps):
        ax2.plot(R, train_tps, color=CO["red"], ls=":", marker=".", lw=1.0,
                 label="training tok/s/node")
        drew2 = True
    if has(tps_roll):
        ax2.plot(R, tps_roll, color=CO["purple"], ls=":", marker=".", lw=1.0,
                 label="rollout tok/s/node")
        drew2 = True
    if has(tps_R):
        ax2.plot(R, tps_R, color=CO["orange"], ls=":", marker=".", lw=1.0,
                 label="judge tok/s/node")
        drew2 = True
    if drew2:
        ax2.set_ylabel("tokens/sec/node")
        ax2.legend(fontsize=6.0, loc="upper right", framealpha=0.85)
    if (
        has(F.arr("actor__grad_norm"))
        or has(F.arr("actor__entropy"))
        or has(F.arr_first(["actor__loss", "actor__pg_loss"]))
        or drew2
    ):
        panel_title(ax, 16, "Optimizer + infra detail", "value", loc="upper left", ncol=2)
    else:
        na(ax, 16, "Optimizer + infra detail")


def _panel_17_prover_judge_missing_failed_per_group(ax, R, F) -> None:
    """Per-prompt-group p50/p90/p99 of MISSING (no proof tag) and FAILED
    (http/parse/truncated) prover judge counts. These two families are computed
    alongside the attempts/success per-group families but routed here, off the
    main prover-judge-count panel (11), to keep that panel legible."""
    drew_mis = DC.percentile_family(
        ax, R, F, "v7__train__prover__judge_missing_per_group", "missing/grp", CO["orange"])
    drew_fail = DC.percentile_family(
        ax, R, F, "v7__train__prover__judge_failed_per_group", "failed/grp", CO["red"])
    if drew_mis or drew_fail:
        ax.set_ylim(bottom=0)
        panel_title(ax, 17, "Prover judge missing/failed per-group", "count / group",
                    loc="upper left", ncol=2)
        ax.text(0.02, -0.18,
                "per-prompt-group p50/p90/p99. missing = no proof tag (judge could not "
                "run); failed = http/parse/truncated. Companion to the main panel-11 "
                "attempts/success per-group families.",
                transform=ax.transAxes, fontsize=6.0, color="gray", va="top")
    else:
        na(ax, 17, "Prover judge missing/failed per-group",
           "no per-group missing/failed metrics\n(older fig_data; re-parse to populate)")


def main() -> None:
    fig_data = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("FIG_DATA", "fig_data.json")
    _title = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("TITLE", "diagnostics")
    out = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("OUT", "v7_dashboard_diagnostics.png")

    d = DC.load_fig_data(fig_data)
    F = DC.FigData(d, fig_data)
    fig = build_diag_figure(F)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
