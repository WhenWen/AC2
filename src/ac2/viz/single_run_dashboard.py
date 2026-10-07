#!/usr/bin/env python3
"""Main dashboard page (29 baseline panels, plus optional run-specific panels).

Usage:
  python3 src/ac2/viz/single_run_dashboard.py fig_data.json "run title" out.png
  python3 src/ac2/viz/single_run_dashboard.py fig_data.json "run title" out.pdf

When the output path ends in ``.pdf`` this writes a 2-page PDF (main + diag);
otherwise it writes just the main page as a PNG. The input is produced by
``parse_fig_data.py``; normally this page is rendered through
``render_dashboard.py``. Training correctness is split into prover (1) and
conjecture (2) panels. Each ``_panel_*`` takes its display number ``n`` from
``build_main_figure`` so the order lives in one place.

The fig_data format supports a two-mode setup: ``mode_is_proposer`` /
``mode_is_prover`` tag every rollout, the parser splits per-step metrics into
proposer / prover, and every panel that has a mode-natural answer is split
here (prover-only runs drop the proposer panels).

This file is **standalone**: nothing here imports from any other viz package.
``dashboard_common`` is the shared helper module sitting next to this file.
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
    CO, ema, has, impact_color, line, na, panel_title, present, stacked,
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


def _lineage_dash_step(F: "DC.FigData") -> float | None:
    """Last parent's ``end_step`` (the dashed-history boundary), or None
    for a root run. Derived from ``DC.lineage_boundaries(F)`` (whose markers sit
    at ``end_step + 0.5``); returns None when there are no parents so root runs
    are visually unchanged."""
    bnds = DC.lineage_boundaries(F)
    if not bnds:
        return None
    try:
        return float(bnds[-1]["x"]) - 0.5
    except (TypeError, ValueError, KeyError):
        return None


def build_main_figure(F: "DC.FigData", *, prover_only: bool | None = None) -> "plt.Figure":
    """Dispatch: render the 29-panel baseline layout, OR — when the parsed
    fig_data contains any ``prover_additional`` rows — the 44-panel
    replay layout that splits each prover-side panel into adjacent
    original/additional family panels with matched axes.

    ``prover_only`` — when True, the 9 proposer-only main panels (2 conjecture
    correctness, 4 proposer impact, 6 proposer resp_len, 8 proposer reward
    components, 10 proposer judge tokens, 18 proposer rollout entropy,
    21 proposer-correctness pass-count, 22 proposer impact dist, 25 proposer
    judge token budget) are omitted and the remaining panels reflow into a
    smaller grid. When ``None`` (default), auto-detected from fig_data via
    ``DC.is_prover_only(F)`` (run has 0 proposer rows generated throughout).
    """
    if prover_only is None:
        prover_only = DC.is_prover_only(F)
    DC.set_lineage_dash_boundary(_lineage_dash_step(F))
    try:
        if _has_replay(F):
            return _build_main_replay(F, prover_only=prover_only)
        return _build_main_baseline(F, prover_only=prover_only)
    finally:
        DC.set_lineage_dash_boundary(None)


# Panel numbers that read proposer-only fig_data keys and are dropped in
# prover-only runs. Pinned by number so renumbering is impossible — each panel
# keeps its display number (e.g. "(5)") even when adjacent slots are dropped.
# Panel 12 (mode mix) is also dropped in prover-only because the proposer/prover
# split is trivially 0%/100% — the panel carries no signal.
_PROPOSER_ONLY_MAIN_PANELS = frozenset({2, 4, 6, 8, 10, 12, 18, 21, 22, 25})


def _build_main_baseline(F: "DC.FigData", *, prover_only: bool = False) -> "plt.Figure":
    R = F.steps
    train_ds = F.cfg("train_data_source", "fineproofs-rl")
    val_ds = F.cfg("val_data_source", train_ds)
    train_n = int(F.cfg("train_rollout_n") or F.cfg("train_best_at_n") or F.cfg("rollout_n") or 8)
    # judge_max_tokens — prefer the launch manifest (the actual cap used in
    # this run, e.g. 81920); fall back to the config default (typically
    # 16384) only if missing. Used for the dotted cap line on panels 10/25/26.
    _jmt = (F.d.get("manifest") or {}).get("judge_max_tokens") or F.cfg(
        "judge_max_tokens_default"
    )
    judge_max_tokens = int(_jmt) if _jmt not in (None, "", "None") else None
    _rc = F.cfg("max_response_length_default")
    resp_cap = int(_rc) if _rc not in (None, "", "None") else None

    # Panel specs in display order: (panel_num, render_fn). render_fn takes (ax, n).
    specs: list[tuple[int, callable]] = [
        (1, lambda ax, n: _panel_prover_train_correctness(ax, n, R, F, train_n)),
        (2, lambda ax, n: _panel_conjecture_train_correctness(ax, n, R, F, train_n)),
        (3, lambda ax, n: _panel_val_correctness(ax, n, R, F, val_ds)),
        (4, lambda ax, n: _panel_proposer_impact_score(ax, n, R, F)),
        (5, lambda ax, n: _panel_prover_resp_len(ax, n, R, F, resp_cap)),
        (6, lambda ax, n: _panel_proposer_resp_len(ax, n, R, F, resp_cap)),
        (7, lambda ax, n: _panel_prover_reward_components(ax, n, R, F)),
        (8, lambda ax, n: _panel_proposer_reward_components(ax, n, R, F)),
        # 38 sits in panel 9's old slot: judge health moved to the diagnostics page (18); the
        # SNIS-weighted mean length gets the prime page-1 position instead.
        (38, lambda ax, n: _panel_prover_resp_len_snis_mean(ax, n, R, F, resp_cap)),
        (10, lambda ax, n: _panel_proposer_judge_tokens(ax, n, R, F, judge_max_tokens)),
        (11, lambda ax, n: _panel_prover_judge_count(ax, n, R, F)),
        (12, lambda ax, n: _panel_mode_mix(ax, n, R, F)),
        (13, lambda ax, n: _panel_policy_loss(ax, n, R, F)),
        (14, lambda ax, n: _panel_policy_kl(ax, n, R, F)),
        (15, lambda ax, n: _panel_ppo_clip(ax, n, R, F)),
        (16, lambda ax, n: _panel_grad_norm(ax, n, R, F)),
        (17, lambda ax, n: _panel_prover_rollout_entropy(ax, n, R, F)),
        (18, lambda ax, n: _panel_proposer_rollout_entropy(ax, n, R, F)),
        (19, lambda ax, n: _panel_dapo_refill_by_mode(ax, n, R, F)),
        (20, lambda ax, n: _panel_prover_passcount(ax, n, R, F, train_n)),
        (21, lambda ax, n: _panel_proposer_correctness_passcount(ax, n, R, F, train_n)),
        (22, lambda ax, n: _panel_proposer_impact_dist(ax, n, R, F)),
        (23, lambda ax, n: _panel_generated_vs_trained(ax, n, R, F)),
        (24, lambda ax, n: _panel_format_length_health(ax, n, R, F, resp_cap)),
        (25, lambda ax, n: _panel_proposer_judge_token_budget(ax, n, R, F, judge_max_tokens)),
        (26, lambda ax, n: _panel_prover_judge_token_budget(ax, n, R, F, judge_max_tokens)),
        (27, lambda ax, n: _panel_actor_lr(ax, n, R, F)),
        (28, lambda ax, n: _panel_global_rollout_entropy(ax, n, R, F)),
        (29, lambda ax, n: _panel_time_per_stage(ax, n, R, F)),
        (30, lambda ax, n: _panel_length_penalty_config(ax, n, F)),
        (31, lambda ax, n: _panel_throughput(ax, n, R, F)),
        (32, lambda ax, n: _panel_judge_rollout_length(ax, n, R, F, judge_max_tokens)),
        (33, lambda ax, n: _panel_judge_input_length(ax, n, R, F)),
    ]
    # AEC (MAI-Thinking-1): surface the clip-constant panel only for runs that drove it
    # (actor/aec_k logged); non-AEC runs are unaffected. Placed right after the prover rollout-entropy
    # panel (17) — AEC acts on entropy, so they read together — while keeping its pinned number 34.
    if has(F.arr("actor__aec_k")):
        _i17 = next((i for i, s in enumerate(specs) if s[0] == 17), len(specs) - 1)
        specs.insert(_i17 + 1, (34, lambda ax, n: _panel_aec_k(ax, n, R, F)))
    # OPD distillation aux: panels only for runs that logged the aux
    # forward-KL (SP_OPD_ENABLE=1); other runs are unaffected. Placed right after policy loss
    # (13) -- actor/loss = pg_loss + coef*distill_loss, so they read together. Pinned 36/37.
    if has(F.arr("actor__distillation__loss")):
        _i13 = next((i for i, s in enumerate(specs) if s[0] == 13), len(specs) - 1)
        specs.insert(_i13 + 1, (36, lambda ax, n: _panel_opd_distill(ax, n, R, F)))
        specs.insert(_i13 + 2, (37, lambda ax, n: _panel_opd_region(ax, n, R, F)))
    # Difficulty p-hat distribution (35): only when the sampler is active; placed right after the
    # prover pass-count dist (20) so the two difficulty views read together. Pinned number.
    if F.glob.get("difficulty_phat"):
        _i20 = next((i for i, s in enumerate(specs) if s[0] == 20), len(specs) - 1)
        specs.insert(_i20 + 1, (35, lambda ax, n: _panel_difficulty_phat_dist(ax, n, R, F)))
    # Replay-prefix training: the stream-split pass-rate panel,
    # only for runs whose sp_replay hook logged replay/* keys; other runs are unaffected.
    # Placed right after the prover train-correctness panel (1) so the from-scratch vs
    # prefix-conditioned split reads next to it. Pinned number 39. The buffer-state panel
    # (40) lives on the diagnostics page (page 1 stays comparison-focused; buffer
    # plumbing is a diagnostic).
    if has(F.arr("replay__pass_rate_orig")):
        _i1 = next((i for i, s in enumerate(specs) if s[0] == 1), 0)
        specs.insert(_i1 + 1, (39, lambda ax, n: _panel_replay_stream_pass(ax, n, R, F)))
    # Stream-split response length: REPLACE the pooled panel 5 with 5a (from-scratch)
    # + 5b (replay, short vs full lanes) when the split keys exist. Pooling hides that the
    # g-capped short lane is ~45% of replay slots. Runs without the keys keep the single
    # panel 5 unchanged.
    if has(F.arr("v7__train__prover__resp_len_replay_p50")):
        _i5 = next((i for i, s in enumerate(specs) if s[0] == 5), None)
        if _i5 is not None:
            specs[_i5:_i5 + 1] = [
                ("5a", lambda ax, n: _panel_resp_len_stream(ax, n, R, F, resp_cap, "scratch")),
                ("5b", lambda ax, n: _panel_gen_len_all(ax, n, R, F, resp_cap)),
            ]
    # Generative-Q readiness: the critic panel block (pinned 41-50), appended as a
    # trailing group only for runs that logged q/* keys; other runs are unaffected.
    specs.extend(_sp_q_panel_specs(R, F))
    if prover_only:
        specs = [s for s in specs if s[0] not in _PROPOSER_ONLY_MAIN_PANELS]

    n_active = len(specs)
    ncols = 4
    nrows = (n_active + ncols - 1) // ncols
    # Per-row figure height tracks the original baseline (42 in / 8 rows ≈ 5.25 in/row).
    fig = plt.figure(figsize=(24, 5.25 * nrows))
    page_label = "v7 main dashboard" + (" (prover-only)" if prover_only else "")
    if has_difficulty_raw(F):
        # difficulty-sampling run: standard metric keys carry the SNIS-corrected values
        page_label += "  ·  difficulty-sampling: metrics SNIS-corrected (raw → page 3)"
    top = DC.draw_header(fig, F, page_label)
    gs = GridSpec(nrows, ncols, figure=fig, hspace=0.58, wspace=0.27, top=top)
    A = [fig.add_subplot(gs[i // ncols, i % ncols]) for i in range(n_active)]

    if not len(R):
        na(A[0], specs[0][0], "No steps", "fig_data has no steps")
        for i in range(1, n_active):
            na(A[i], specs[i][0], "")
        fig.subplots_adjust(left=0.045, right=0.985, bottom=0.02)
        return fig

    for slot, (n, render_fn) in enumerate(specs):
        render_fn(A[slot], n)

    DC.apply_vertical_markers(A, F)
    fig.subplots_adjust(left=0.045, right=0.975, bottom=0.02)
    return fig


def _build_main_replay(F: "DC.FigData", *, prover_only: bool = False) -> "plt.Figure":
    # NOTE: prover_only filtering is not yet implemented for the replay layout
    # (split-family panel numbering "1a/1b/etc." needs its own skip map).
    # Prover-only + replay is unusual; render the full layout for now.
    _ = prover_only  # accepted for API compatibility
    """Replay (family-aware) layout. 44 panels in an 11x4 grid: every prover-side panel from the baseline
    splits into adjacent original/additional family panels with matched axes
    (and into a 3-way proposer+original+additional split for judge health,
    filter pressure, generated-vs-trained, and format/length-health). Panel 12
    becomes a 3-family mix; panels 13-16 (optimizer), 27-29 (LR / global
    entropy / time) stay global.

    Display labels use "Na"/"Nb"/"Nc" for the split variants of baseline panel N.
    """
    R = F.steps
    # 12x4 grid: 44 panels + the length-penalty config strip in the
    # first cell of an appended row, leaving 3 cells of trailing whitespace.
    fig = plt.figure(figsize=(24, 63))
    top = DC.draw_header(fig, F, "v7 main dashboard (replay)")
    gs = GridSpec(12, 4, figure=fig, hspace=0.58, wspace=0.27, top=top)
    A = [fig.add_subplot(gs[i // 4, i % 4]) for i in range(45)]

    if not len(R):
        na(A[0], 1, "No steps", "fig_data has no steps")
        for i in range(1, 45):
            na(A[i], i + 1, "")
        fig.subplots_adjust(left=0.045, right=0.985, bottom=0.02)
        return fig

    train_ds = F.cfg("train_data_source", "fineproofs-rl")
    val_ds = F.cfg("val_data_source", train_ds)
    train_n = int(F.cfg("train_rollout_n") or F.cfg("train_best_at_n") or F.cfg("rollout_n") or 8)
    # judge_max_tokens — prefer the launch manifest (the actual cap used in
    # this run, e.g. 81920); fall back to the config default (typically
    # 16384) only if missing. Used for the dotted cap line on panels 10/25/26.
    _jmt = (F.d.get("manifest") or {}).get("judge_max_tokens") or F.cfg(
        "judge_max_tokens_default"
    )
    judge_max_tokens = int(_jmt) if _jmt not in (None, "", "None") else None
    _rc = F.cfg("max_response_length_default")
    resp_cap = int(_rc) if _rc not in (None, "", "None") else None

    # Display order with split labels.
    _panel_prover_train_correctness_family(A[0], "1a", R, F, train_n, "original")
    _panel_prover_train_correctness_family(A[1], "1b", R, F, train_n, "additional")
    _panel_conjecture_train_correctness(A[2], 2, R, F, train_n)
    _panel_val_correctness(A[3], 3, R, F, val_ds)
    _panel_proposer_impact_score(A[4], 4, R, F)
    _panel_prover_resp_len_family(A[5], "5a", R, F, resp_cap, "original")
    _panel_prover_resp_len_family(A[6], "5b", R, F, resp_cap, "additional")
    _panel_proposer_resp_len(A[7], 6, R, F, resp_cap)
    _panel_prover_reward_components_family(A[8], "7a", R, F, "original")
    _panel_prover_reward_components_family(A[9], "7b", R, F, "additional")
    _panel_proposer_reward_components(A[10], 8, R, F)
    _panel_judge_health_family(A[11], "9a", R, F, "proposer")
    _panel_judge_health_family(A[12], "9b", R, F, "original")
    _panel_judge_health_family(A[13], "9c", R, F, "additional")
    _panel_proposer_judge_tokens(A[14], 10, R, F, judge_max_tokens)
    _panel_prover_judge_count_family(A[15], "11a", R, F, "original")
    _panel_prover_judge_count_family(A[16], "11b", R, F, "additional")
    _panel_family_mix(A[17], 12, R, F)
    _panel_policy_loss(A[18], 13, R, F)
    _panel_policy_kl(A[19], 14, R, F)
    _panel_ppo_clip(A[20], 15, R, F)
    _panel_grad_norm(A[21], 16, R, F)
    _panel_prover_rollout_entropy_family(A[22], "17a", R, F, "original")
    _panel_prover_rollout_entropy_family(A[23], "17b", R, F, "additional")
    _panel_proposer_rollout_entropy(A[24], 18, R, F)
    _panel_filter_pressure_family(A[25], "19a", R, F, "proposer")
    _panel_filter_pressure_family(A[26], "19b", R, F, "original")
    _panel_filter_pressure_family(A[27], "19c", R, F, "additional")
    _panel_prover_passcount_family(A[28], "20a", R, F, train_n, "original")
    _panel_prover_passcount_family(A[29], "20b", R, F, train_n, "additional")
    _panel_proposer_correctness_passcount(A[30], 21, R, F, train_n)
    _panel_proposer_impact_dist(A[31], 22, R, F)
    _panel_generated_vs_trained_reward_family(A[32], "23a", R, F, "proposer")
    _panel_generated_vs_trained_reward_family(A[33], "23b", R, F, "original")
    _panel_generated_vs_trained_reward_family(A[34], "23c", R, F, "additional")
    _panel_format_length_health_family(A[35], "24a", R, F, resp_cap, "proposer")
    _panel_format_length_health_family(A[36], "24b", R, F, resp_cap, "original")
    _panel_format_length_health_family(A[37], "24c", R, F, resp_cap, "additional")
    _panel_proposer_judge_token_budget(A[38], 25, R, F, judge_max_tokens)
    _panel_prover_judge_token_budget_family(A[39], "26a", R, F, judge_max_tokens, "original")
    _panel_prover_judge_token_budget_family(A[40], "26b", R, F, judge_max_tokens, "additional")
    _panel_actor_lr(A[41], 27, R, F)
    _panel_global_rollout_entropy(A[42], 28, R, F)
    _panel_time_per_stage(A[43], 29, R, F)
    _panel_length_penalty_config(A[44], 30, F)

    DC.apply_vertical_markers(A, F)
    fig.subplots_adjust(left=0.045, right=0.975, bottom=0.015)
    return fig


# ---------------------------------------------------------------------------
# Replay (family-aware) helpers. When the parser sees any prover_additional
# rows (controlled by the additional-data sidecar), the main figure swaps the
# "all prover" panels for adjacent original/additional pairs. _has_replay() drives
# build_main_figure's dispatch; _FAMILY_INFO centralises the per-family color +
# key-prefix conventions so each split panel is one parameterized function.
# ---------------------------------------------------------------------------

_FAMILY_INFO = {
    "original": {
        "key_prefix": "v7__train__family__prover_original",
        "color": CO["prover_original"],
        "color_dark": CO["prover_original_dark"],
        "color_light": CO["prover_original_light"],
        "color_accent": CO["prover_original_accent"],
        "title_prefix": "Original-statement prover",
        "short": "orig",
    },
    "additional": {
        "key_prefix": "v7__train__family__prover_additional",
        "color": CO["prover_additional"],
        "color_dark": CO["prover_additional_dark"],
        "color_light": CO["prover_additional_light"],
        "color_accent": CO["prover_additional_accent"],
        "title_prefix": "Additional-statement prover",
        "short": "add",
    },
    "proposer": {
        "key_prefix": "v7__train__family__proposer",
        "color": CO["proposer"],
        "color_dark": CO["proposer_dark"],
        "color_light": CO["proposer_light"],
        "color_accent": CO["proposer_accent"],
        "title_prefix": "Proposer",
        "short": "prop",
    },
}


def _has_replay(F: "DC.FigData") -> bool:
    """True iff the parsed fig_data contains any ``prover_additional`` rows.

    Drives the adaptive split: replay runs render an expanded layout
    (per-family adjacent panels with matched axes), baseline runs render
    the 29-panel layout (no behavior change). Detection is robust against
    a key existing but being all-null/zero.
    """
    y = F.arr("v7__train__family__prover_additional__rows_generated")
    if not has(y):
        return False
    try:
        return bool(np.any(np.asarray(y, dtype=float) > 0))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Family-aware panel variants. One parameterized function per concept; called
# with family ∈ {"original", "additional"} (or "proposer" for the 3-way
# panels). The baseline panel functions further below are kept unchanged so
# non-replay runs render exactly as before.
# ---------------------------------------------------------------------------


def _cap_axhline(ax, cap, label):
    """Dotted cap reference line, drawn only when the cap is known. A None cap
    (config didn't specify it) draws nothing rather than a made-up default line."""
    if cap is None:
        return
    ax.axhline(cap, color="black", ls=":", lw=1.0, label=f"{label} ({cap})")


def _floor10_bottom(ax, *series):
    """Anchor the y-axis bottom at the largest multiple of 10 at or below the
    smallest finite plotted value (clamped to >=0), instead of pinning it to 0.
    For percent-scale correctness / pass@k panels whose curves sit well above
    zero, this spends the vertical space on signal rather than empty margin.
    Top is left to matplotlib autoscale; no-ops to bottom=0 with no finite data."""
    finite = []
    for s in series:
        if s is None:
            continue
        arr = np.asarray(s, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            finite.append(arr)
    if not finite:
        ax.set_ylim(bottom=0)
        return
    lo = float(np.concatenate(finite).min())
    ax.set_ylim(bottom=max(0.0, float(np.floor(lo / 10.0) * 10.0)))


def _panel_prover_train_correctness_family(ax, n, R, F, train_n, family):
    """Per-family prover correctness: mean@n + EMA + best@n. Family in
    {"original", "additional"}. Uses v7__train__family__prover_<family>__*."""
    info = _FAMILY_INFO[family]
    pre = info["key_prefix"]
    mean_raw = F.arr_first([f"{pre}__rubric_grade_mean",
                            f"{pre}__prover_judge_score_mean",
                            f"{pre}__score_mean_generated"], 100)
    groups = F.arr(f"{pre}__groups_generated")
    allzero = F.arr(f"{pre}__group_allzero")
    passn = np.where(groups > 0, 100.0 * (groups - allzero) / np.maximum(groups, 1e-9), np.nan)
    drew = False
    if has(mean_raw):
        line(ax, R, mean_raw, info["color"], f"mean@{train_n}", marker=".", lw=1.3, alpha=0.8)
        line(ax, R, _ema_alpha(mean_raw, 0.95), info["color"], "mean EMA (a=0.95)", lw=2.0)
        drew = True
    if has(passn):
        line(ax, R, passn, info["color_light"], f"best@{train_n} (>=1 of {train_n})",
             marker=".", ls="--", lw=1.1)
        drew = True
    title = f"{info['title_prefix']} training correctness"
    if drew:
        _floor10_bottom(ax, mean_raw, passn)
        panel_title(ax, n, title, "% correct", loc="upper left")
    else:
        na(ax, n, title, f"no {family}-family prover rows")


def _panel_prover_resp_len_family(ax, n, R, F, resp_cap, family):
    info = _FAMILY_INFO[family]
    pre = info["key_prefix"]
    drew = DC.percentile_family(ax, R, F, f"{pre}__resp_len", "resp", info["color_dark"])
    title = f"{info['title_prefix']} rollout response length"
    if drew:
        _cap_axhline(ax, resp_cap, "resp cap")
        ax.set_ylim(bottom=0)
        panel_title(ax, n, title, "tokens", loc="upper left")
    else:
        na(ax, n, title, "no actor response_length field in dump")


def _panel_prover_reward_components_family(ax, n, R, F, family):
    info = _FAMILY_INFO[family]
    pre = info["key_prefix"]
    correctness = _arr_coalesce(F, [f"{pre}__rubric_grade_mean", f"{pre}__prover_judge_score_mean", f"{pre}__score_mean_generated"])
    lp = F.arr_first([f"{pre}__length_penalty_mean", f"{pre}__overlong_reward_mean"])
    opt = F.arr(f"{pre}__optimized_reward_mean")
    line(ax, R, correctness, info["color"], "correctness", marker=".", lw=1.2)
    line(ax, R, lp, CO["red"], "length penalty", marker=".", ls="--", lw=1.1)
    line(ax, R, opt, info["color_dark"], "final reward", marker="o", ms=4, lw=2.2)
    title = f"{info['title_prefix']} reward components"
    if any(has(y) for y in (correctness, lp, opt)):
        ax.axhline(0, color="black", lw=0.6)
        panel_title(ax, n, title, "reward", loc="best")
    else:
        na(ax, n, title, f"no {family}-family prover rows")


def _panel_prover_passcount_family(ax, n, R, F, train_n, family):
    info = _FAMILY_INFO[family]
    pre = info["key_prefix"]
    cmap = DC.passcount_colormap(train_n + 1)
    specs = []
    for k in range(train_n + 1):
        y = F.arr(f"{pre}__pass_count_hist__k{k}")
        specs.append((y, f"{k}/{train_n}", cmap(k / max(1, train_n))))
    drew = stacked(ax, R, specs, all_in_legend=True)
    title = f"{info['title_prefix']} pass-count dist."
    if drew:
        panel_title(ax, n, title, "prompt groups", loc="upper left", ncol=3)
    else:
        na(ax, n, title, "requires explicit uid in train rollout dump")


def _panel_prover_judge_count_family(ax, n, R, F, family):
    info = _FAMILY_INFO[family]
    pre = info["key_prefix"]
    gen = F.arr(f"{pre}__rows_generated")
    att = F.arr(f"{pre}__judge_attempts")
    suc = F.arr(f"{pre}__judge_success")
    mis = F.arr(f"{pre}__judge_missing")
    fail = F.arr(f"{pre}__judge_failed")
    line(ax, R, gen, CO["light_gray"], "generated (total)", marker=".", lw=1.6)
    line(ax, R, att, CO["gray"], "attempts (total)", marker=".", lw=1.4)
    line(ax, R, suc, info["color"], "success (total)", marker=".", lw=1.7)
    line(ax, R, mis, CO["orange"], "missing (total)", marker=".", ls="--", lw=1.1)
    line(ax, R, fail, CO["red"], "failed (total)", marker=".", ls=":", lw=1.1)
    drew_tot = any(has(y) for y in (gen, att, suc, mis, fail))
    ax2 = ax.twinx(); ax2.grid(False)
    drew_att = DC.percentile_family(ax2, R, F, f"{pre}__judge_attempts_per_group",
                                    "attempts/grp", CO["navy"])
    drew_suc = DC.percentile_family(ax2, R, F, f"{pre}__judge_success_per_group",
                                    "success/grp", info["color_dark"])
    if drew_att or drew_suc:
        ax2.set_ylim(bottom=0); ax2.set_ylabel("count / group", fontsize=7)
        ax2.legend(fontsize=5.6, loc="upper right", framealpha=0.85, ncol=2)
    else:
        ax2.set_yticks([])
    title = f"{info['title_prefix']} judge count"
    if drew_tot or drew_att or drew_suc:
        ax.set_ylim(bottom=0)
        panel_title(ax, n, title, "rows / step", loc="upper left", ncol=2)
    else:
        na(ax, n, title)


def _panel_prover_judge_token_budget_family(ax, n, R, F, judge_max_tokens, family):
    info = _FAMILY_INFO[family]
    pre = info["key_prefix"]
    drew = False
    drew |= DC.percentile_family(ax, R, F, f"{pre}__judge_prompt_tokens",
                                 "prompt", info["color_accent"])
    drew |= DC.percentile_family(ax, R, F, f"{pre}__judge_completion_tokens",
                                 "compl", info["color_dark"])
    title = f"{info['title_prefix']} judge token budget"
    if drew:
        _cap_axhline(ax, judge_max_tokens, "max tok")
        ax.set_ylim(bottom=0)
        panel_title(ax, n, title, "tokens", loc="upper left", ncol=2)
    else:
        na(ax, n, title)


def _panel_prover_rollout_entropy_family(ax, n, R, F, family):
    """Per-family prover entropy. Prefer
    `rollout/entropy_prover_<family>_statements`; fall back to
    `actor/entropy_prover_<family>_statements` with an explicit update-batch
    label. Global entropy is deliberately NOT copied into these
    panels, so we render N/A when neither source-specific field exists
    in the dump.

    Most runs don't log either source-specific key, so these panels show
    N/A — but the wiring is here so any run that logs them renders
    correctly without a code change."""
    info = _FAMILY_INFO[family]
    title = f"{info['title_prefix']} rollout entropy"
    col = info["color_dark"]
    # family is "original" / "additional" in the renderer; the spec keys are
    # `..._prover_<family>_statements`.
    roll = F.arr(f"rollout__entropy_prover_{family}_statements")
    actor = F.arr(f"actor__entropy_prover_{family}_statements")
    if has(roll):
        line(ax, R, roll, col,
             f"rollout/entropy_prover_{family}_statements", marker=".", lw=1.6)
        panel_title(ax, n, title, "entropy", loc="best")
    elif has(actor):
        line(ax, R, actor, col,
             f"actor/entropy_prover_{family}_statements (update-batch)",
             marker=".", lw=1.6)
        panel_title(ax, n, title, "entropy", loc="best")
        ax.text(0.02, -0.16,
                "update-batch fallback: actor/entropy_prover_*_statements is "
                "over KEPT training rows, not generated rollouts. Log "
                "rollout/entropy_prover_*_statements for the true signal.",
                transform=ax.transAxes, fontsize=6.0, color="gray", va="top")
    else:
        # Deliberate N/A: never substitute the global all-prover entropy
        # here (would silently mix original + additional rows).
        na(ax, n, title,
           f"no rollout/entropy_prover_{family}_statements or "
           f"actor/entropy_prover_{family}_statements\n"
           f"(global all-prover entropy is intentionally NOT substituted)")


def _panel_judge_health_family(ax, n, R, F, family):
    """Per-family MECE judge health buckets, stacked. family in
    {"proposer", "original", "additional"}."""
    info = _FAMILY_INFO[family]
    pre = info["key_prefix"]
    cats = [
        ("truncated", "truncated (judge cap)", CO["orange"]),
        ("parse_failed", "parse failed (not trunc)", CO["red"]),
        ("http_error", "http error", CO["purple"]),
        ("no_proof", "no proof/proposition", CO["gray"]),
    ]
    gen = present(F.arr(f"{pre}__rows_generated"))
    specs = []
    any_data = False
    for cat, label, col in cats:
        cnt = present(F.arr(f"{pre}__judge_cat__{cat}"))
        pct = np.where(gen > 0, 100.0 * cnt / np.maximum(gen, 1e-9), 0.0)
        if np.any(cnt > 0):
            any_data = True
        specs.append((pct, label, col))
    title = f"{info['title_prefix']} judge health buckets (% of generated)"
    if any_data or float(np.nansum(gen)) > 0:
        # Percentage stack — lock the y-axis to [0, 100] (see panel 9 main).
        stacked(ax, R, specs, all_in_legend=True, y_max=100)
        panel_title(ax, n, title, "% unhealthy", loc="upper left", ncol=2)
    else:
        na(ax, n, title, f"no {family}-family rows")


def _panel_filter_pressure_family(ax, n, R, F, family):
    """Per-family filter pressure: generated / trained / filtered groups + the
    filtered fraction. family in {"proposer", "original", "additional"}."""
    info = _FAMILY_INFO[family]
    pre = info["key_prefix"]
    g_gen = F.arr(f"{pre}__groups_generated")
    g_tra = F.arr(f"{pre}__groups_trained")
    g_fil = F.arr(f"{pre}__groups_filtered")
    g_frac = F.arr(f"{pre}__de_filtered_fraction", 100)
    drew = False
    if has(g_gen):
        ax.bar(R, present(g_gen), color=CO["light_gray"], width=0.7, label="generated")
        drew = True
    if has(g_tra):
        ax.bar(R, present(g_tra), color=info["color"], width=0.7, label="trained", alpha=0.85)
        drew = True
    if has(g_fil):
        ax.bar(R, -present(g_fil), color=CO["red"], width=0.7, label="filtered (-)",
               alpha=0.7)
        drew = True
    ax.axhline(0, color="black", lw=0.5)
    if drew:
        ax.set_ylabel("groups / step")
    ax2 = ax.twinx(); ax2.grid(False)
    drew2 = False
    if has(g_frac):
        ax2.plot(R, g_frac, color=info["color_dark"], marker=".", lw=1.4,
                 label="filtered %")
        drew2 = True
    if drew2:
        ax2.set_ylim(0, 105); ax2.set_ylabel("% filtered")
        ax2.legend(fontsize=6.3, loc="upper right", framealpha=0.85)
    title = f"{info['title_prefix']} filter pressure"
    if drew or drew2:
        panel_title(ax, n, title, "groups", loc="upper left")
        ax.set_xlabel("step")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=6.3, loc="upper left", framealpha=0.85)
    else:
        na(ax, n, title, f"no {family}-family rows")


def _panel_generated_vs_trained_reward_family(ax, n, R, F, family):
    """Per-family generated-vs-trained reward (one line generated, one trained).
    family in {"proposer", "original", "additional"}."""
    info = _FAMILY_INFO[family]
    pre = info["key_prefix"]
    if family == "proposer":
        gen = F.arr(f"{pre}__proposer_reward_mean")
        tra = F.arr(f"{pre}__score_mean_trained")
        if not has(gen):
            gen = F.arr(f"{pre}__score_mean_generated")
    else:
        gen = F.arr_first([f"{pre}__rubric_grade_mean", f"{pre}__prover_judge_score_mean", f"{pre}__score_mean_generated"])
        tra = F.arr(f"{pre}__score_mean_trained")
    line(ax, R, gen, info["color"], "generated (mean)", marker=".", lw=1.2)
    line(ax, R, tra, info["color_dark"], "trained (after DAPO filter)",
         marker="o", ms=3, lw=1.8)
    title = f"{info['title_prefix']} generated vs trained reward"
    if has(gen) or has(tra):
        panel_title(ax, n, title, "reward", loc="best")
    else:
        na(ax, n, title, f"no {family}-family rows")


def _panel_format_length_health_family(ax, n, R, F, resp_cap, family):
    """Per-family format & character-length health. family in
    {"proposer", "original", "additional"}."""
    info = _FAMILY_INFO[family]
    pre = info["key_prefix"]
    if family == "proposer":
        # Tag presence.
        ppt = F.arr(f"{pre}__proposition_tag_present_rate", 100)
        pft = F.arr(f"{pre}__proof_tag_present_rate", 100)
        both = F.arr(f"{pre}__both_tags_present_rate", 100)
        if has(ppt): ax.plot(R, ppt, color=info["color_light"], label="prop tag", lw=1.1, ls=":")
        if has(pft): ax.plot(R, pft, color=info["color_accent"], label="proof tag", lw=1.1, ls="--")
        if has(both): ax.plot(R, both, color=info["color_dark"], label="both tags", marker=".", lw=1.6)
        # Candidate (proposition+proof) char-length percentiles on right axis.
        ax2 = ax.twinx(); ax2.grid(False)
        drew = DC.percentile_family(ax2, R, F, f"{pre}__candidate_len", "cand chars", info["color_dark"])
        if drew:
            ax2.set_ylabel("chars / candidate"); ax2.set_ylim(bottom=0)
            ax2.legend(fontsize=5.6, loc="upper right", framealpha=0.85, ncol=1)
    else:
        # Prover: proof format buckets + proof char-length + overlong rate.
        pn = F.arr_first([
            f"{pre}__proof_nonempty_rate",
            f"{pre}__proof_tag_present_rate",
        ], 100)
        pm = F.arr(f"{pre}__proof_missing_rate", 100)
        pe = F.arr(f"{pre}__proof_empty_rate", 100)
        over = F.arr(f"{pre}__overlong_rate", 100)
        if has(pn): ax.plot(R, pn, color=info["color"], label="nonempty <proof> block", marker=".", lw=1.4)
        if has(pm): ax.plot(R, pm, color=CO["red"], label="no <proof> tag", marker=".", ls=":", lw=1.1)
        if has(pe): ax.plot(R, pe, color=CO["orange"], label="empty <proof> block", marker=".", ls=":", lw=1.1)
        if has(over): ax.plot(R, over, color=CO["orange"], label="overlong", marker=".", ls="--", lw=1.2)
        ax2 = ax.twinx(); ax2.grid(False)
        drew = DC.percentile_family(ax2, R, F, f"{pre}__proof_len", "proof chars", info["color_dark"])
        if drew:
            ax2.set_ylabel("chars / proof"); ax2.set_ylim(bottom=0)
            ax2.legend(fontsize=5.6, loc="upper right", framealpha=0.85, ncol=1)
    ax.set_ylim(0, 105)
    title = f"{info['title_prefix']} format & text-length health"
    if ax.get_legend_handles_labels()[0] or ax2.get_legend_handles_labels()[0]:
        ax.set_ylabel("%")
        panel_title(ax, n, title, "%", loc="upper left")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=6.3, loc="upper left", framealpha=0.85)
    else:
        na(ax, n, title, f"no {family}-family rows")


def _panel_family_mix(ax, n, R, F):
    """Three-family generated and trained fractions (replay variant of panel 12).
    Reads v7__train__family_mix__*. Annotates absolute row counts."""
    fams = ("proposer", "prover_original", "prover_additional")
    colors_g = {"proposer": CO["proposer_light"],
                "prover_original": CO["prover_original_light"],
                "prover_additional": CO["prover_additional_light"]}
    colors_t = {"proposer": CO["proposer_dark"],
                "prover_original": CO["prover_original_dark"],
                "prover_additional": CO["prover_additional_dark"]}
    drew = False
    for f in fams:
        gf = F.arr(f"v7__train__family_mix__{f}__generated_fraction", 100)
        tf = F.arr(f"v7__train__family_mix__{f}__trained_fraction", 100)
        if has(gf):
            ax.plot(R, gf, color=colors_g[f], lw=1.1, ls=":", marker=".",
                    label=f"{f} (gen)")
            drew = True
        if has(tf):
            ax.plot(R, tf, color=colors_t[f], lw=1.7, marker=".",
                    label=f"{f} (trained)")
            drew = True
    if drew:
        ax.set_ylim(0, 100)
        # Annotate latest absolute counts so a small additional dataset doesn't
        # vanish in fractional view.
        counts = []
        for f in fams:
            g = F.arr(f"v7__train__family_mix__{f}__generated")
            if has(g):
                vals = [v for v in g if v is not None and not (isinstance(v, float) and np.isnan(v))]
                if vals: counts.append(f"{f}={int(vals[-1])}")
        if counts:
            ax.text(0.02, -0.16, "latest counts: " + ", ".join(counts),
                    transform=ax.transAxes, fontsize=6.0, color="gray", va="top")
        panel_title(ax, n, "Row family mix (3-family)", "% of generated/trained",
                    loc="upper left", ncol=2)
    else:
        na(ax, n, "Row family mix (3-family)",
           "no family_mix__* keys (re-parse to populate)")


# ---------------------------------------------------------------------------
# Panel implementations. Each function takes (ax, R, F[, extras]) and either
# draws a complete panel or calls na(...) with a reason.
# ---------------------------------------------------------------------------


def _derive_length_penalty_mode(F: "DC.FigData") -> str:
    """Resolve the active length-penalty mode from fig_data config.

    Precedence: exponential > linear > none. Truthy
    ``exponential_penalty`` wins; else a positive
    ``overlong_buffer_len`` (or ``overlong_buffer_len_default``) means
    linear-overlong; else no penalty is active.
    """
    if F.cfg("exponential_penalty"):
        return "exponential"
    overlong_len = F.cfg("overlong_buffer_len", F.cfg("overlong_buffer_len_default"))
    try:
        if overlong_len is not None and float(overlong_len) > 0:
            return "linear"
    except (TypeError, ValueError):
        pass
    return "none"


def _panel_length_penalty_config(ax, n, F):
    """Diagnostic strip: active length-penalty mode, gamma, base_free_tokens,
    NO_DROP_EXCEPT_TRUE_ZERO_STD, resolved DAPO filter metric.

    Mirrors the diagnostics page's config-strip style (no axes, monospace
    text block) so it reads as a config readout rather than a time series.
    Sits in one of the empty cells of the 8x4 grid so it does not disturb
    any existing panel.
    """
    mode = _derive_length_penalty_mode(F)
    gamma_raw = F.cfg("exponential_penalty_gamma")
    base_raw = F.cfg("exponential_penalty_base_free_tokens")
    gamma = "—" if mode != "exponential" or gamma_raw in (None, "") else gamma_raw
    base_free = "—" if mode != "exponential" or base_raw in (None, "") else base_raw
    no_drop_raw = F.cfg("no_drop_except_true_zero_std")
    if no_drop_raw is None or no_drop_raw == "":
        no_drop = "(unset)"
    else:
        s = str(no_drop_raw).strip().lower()
        no_drop = "true" if s in ("true", "1", "yes") else (
            "false" if s in ("false", "0", "no") else str(no_drop_raw)
        )
    dapo_metric = F.cfg("dapo_filter_metric")
    dapo_metric = "(unset)" if dapo_metric in (None, "") else dapo_metric

    lines = [
        f"length penalty mode   {mode}",
        f"gamma                 {gamma}",
        f"base_free_tokens      {base_free}",
        f"NO_DROP_EXCEPT_TRUE_ZERO_STD  {no_drop}",
        f"dapo_filter_metric    {dapo_metric}",
    ]
    ax.set_title(f"({n}) Length-penalty config", fontweight="bold", fontsize=10, loc="left")
    ax.axis("off")
    ax.text(
        0.0, 1.0, "\n".join(lines), transform=ax.transAxes, va="top", ha="left",
        fontsize=7.2, family="monospace", linespacing=1.3, parse_math=False,
    )


def _panel_time_per_stage(ax, n, R, F):
    timing_specs = [
        ("rollout gen", ["timing_s__gen"], CO["blue"]),
        ("reward/judge", ["timing_s__reward"], CO["green"]),
        ("old log-prob", ["timing_s__old_log_prob"], CO["teal"]),
        ("ref log-prob", ["timing_s__ref"], CO["mint"]),
        ("advantage", ["timing_s__adv"], CO["pink"]),
        ("actor update", ["timing_s__update_actor"], CO["red"]),
        ("update weights", ["timing_s__update_weights"], CO["orange"]),
        ("val/test", ["timing_s__testing", "timing_s__validation"], CO["purple"]),
        ("checkpoint", ["timing_s__save_checkpoint"], CO["navy"]),
        # Generative-Q readiness: the Q phases as named stages so the
        # mechanism's cost is attributable at a glance instead of hiding in "other".
        ("Q wave", ["timing_s__sp_q_wave"], CO["salmon"]),
        ("Q train", ["timing_s__sp_q_train"], CO["brown"]),
        ("Q apply", ["timing_s__sp_q_apply"], CO["light_gray"]),
    ]
    # sp_q_admit runs NESTED inside the "advantage" marked_timer (the reward-site
    # hook), so its seconds are already in that bar — consume it silently or the
    # stack double-counts vs the step-total line.
    consumed = {"timing_s__step", "timing_s__start_profile", "timing_s__stop_profile",
                "timing_s__sp_q_admit"}
    specs = []
    for label, keys, color in timing_specs:
        consumed.update(keys)
        specs.append((F.arr_first(keys), label, color))
    other = np.zeros(len(R))
    for k in F.per_step:
        if (
            isinstance(k, str)
            and k.startswith("timing_s__")
            and k not in consumed
            and "agent_loop" not in k
            and "slowest" not in k
        ):
            other = other + present(F.arr(k))
    specs.append((other, "other", CO["gold"]))
    drew = stacked(ax, R, specs)
    total = F.arr_first(["timing_s__step"])
    line(ax, R, total, "black", "step total", ls="--", marker=".", lw=1.1)
    if drew or has(total):
        panel_title(ax, n, "Time per stage", "seconds", loc="upper left", ncol=2)
    else:
        na(ax, n, "Time per stage")


def _panel_mode_mix(ax, n, R, F):
    """Pre-filter (generated) vs post-filter (trained) mode mix.

    Lines, not stacked bars: a stacked-bar version hides the trained-mix
    drift, which is the panel's whole point. Pre-filter is data-prep-
    guaranteed 50/50 so it should be flat; the trained-mix line moving
    away from 0.5 is the headline signal.
    """
    pg = F.arr("v7__train__mode_mix__proposer_generated_fraction", 100)
    rg = F.arr("v7__train__mode_mix__prover_generated_fraction", 100)
    pt_raw = F.arr("v7__train__mode_mix__proposer_trained_fraction", 100)
    rt_raw = F.arr("v7__train__mode_mix__prover_trained_fraction", 100)
    # Prefer the trainer's explicit post-filter metric when present
    # (trained_batch/mode_is_*). Fall back to the parser's mode-
    # mix-from-de_filtered split if the metric is absent in this run.
    pt_metric = F.arr("trained_batch__mode_is_proposer__mean", 100)
    rt_metric = F.arr("trained_batch__mode_is_prover__mean", 100)
    pt = pt_metric if has(pt_metric) else pt_raw
    rt = rt_metric if has(rt_metric) else rt_raw
    line(ax, R, pg, CO["proposer_light"], "proposer (generated)", marker=".", ls="--", lw=1.1)
    line(ax, R, rg, CO["prover_light"], "prover (generated)", marker=".", ls="--", lw=1.1)
    line(ax, R, pt, CO["proposer"], "proposer (trained)", marker="o", ms=3, lw=1.8)
    line(ax, R, rt, CO["prover"], "prover (trained)", marker="o", ms=3, lw=1.8)
    ax.axhline(50.0, color=CO["gray"], ls=":", lw=0.8, label="50% (data-prep mix)")
    if has(pg) or has(rg) or has(pt) or has(rt):
        ax.set_ylim(0, 100)
        panel_title(ax, n, "Mode mix: generated vs trained", "%", loc="best", ncol=2)
        ax.text(0.02, -0.20,
                "trained-mix drift from 50% means DAPO filtering keeps proposer/prover at "
                "different rates — every other panel must be interpreted accordingly.",
                transform=ax.transAxes, fontsize=6.0, color="gray", va="top")
    else:
        na(ax, n, "Mode mix: generated vs trained")


def _panel_proposer_reward_components(ax, n, R, F):
    """Optimized reward vs correctness vs impact (the three headline proposer
    reward components). The pre-length-score / length-penalty decomposition lives
    in the diagnostics 'proposer reward decomposition' panel, not here."""
    correctness = F.arr("v7__train__proposer__correctness_judge_score_mean")
    impact_app = F.arr("v7__train__proposer__impact_applied_mean")
    final = F.arr("v7__train__proposer__proposer_reward_mean")
    line(ax, R, final, CO["proposer_dark"], "optimized (proposer_reward)", marker="o", ms=4, lw=2.2)
    line(ax, R, correctness, CO["proposer"], "correctness 0/1", marker=".", lw=1.3)
    line(ax, R, impact_app, CO["impact2"], "impact_applied", marker=".", lw=1.3)
    if any(has(y) for y in (correctness, impact_app, final)):
        ax.axhline(0, color="black", lw=0.6)
        panel_title(ax, n, "Proposer reward components", "reward", loc="best", ncol=2)
        ax.text(0.02, -0.16,
                "impact_applied rising at flat correctness = USEFUL but easy lemmas; "
                "rising correctness at flat impact = correct but uninfluential propositions. "
                "Pre-length score + length penalty: see diagnostics decomposition panel.",
                transform=ax.transAxes, fontsize=6.0, color="gray", va="top")
    else:
        na(ax, n, "Proposer reward components")


def _vline(ax, R, y, color, label, **kw):
    """Plot a SPARSE validation series over its present points only, so the
    markers are joined by line segments. Validation runs every ~10 steps, so
    plotting over all steps (as ``line``) leaves a NaN between every pair of val
    points and matplotlib draws disconnected markers with no line (and a
    marker-less EMA becomes invisible). Masking to finite points connects
    consecutive val runs without interpolating their values."""
    y = np.asarray(y, dtype=float)
    m = np.isfinite(y)
    if not m.any():
        return
    xm = np.asarray(R, dtype=float)[m]
    ym = y[m]
    # Dashed-history split (shared with dashboard_common.line): the
    # inherited <=boundary points render dashed, the current-run >=boundary
    # points solid. Only splits when the finite points straddle the boundary.
    b = DC.get_lineage_dash_boundary()
    if b is None or not (np.any(xm <= b) and np.any(xm > b)):
        ax.plot(xm, ym, color=color, label=label, **kw)
        return
    before = xm <= b
    after = xm >= b
    dkw = dict(kw)
    dkw.pop("ls", None)  # drop caller's ls= alias so it can't clash with linestyle
    dkw["linestyle"] = "--"
    dkw["alpha"] = 0.55
    ax.plot(xm[before], ym[before], color=color, label="_nolegend_", **dkw)
    ax.plot(xm[after], ym[after], color=color, label=label, **kw)


def _panel_val_correctness(ax, n, R, F, val_ds):
    """Direct-proof validation: best@n headline + pass@1 (prover only).

    Two curves (verl val-core only — no exact
    (groups-allzero)/groups cross-check):
      * **best@{vn}** (headline) -- verl's official `val-core/<ds>/acc/best@n/mean`
        (bootstrap-with-replacement estimate of "solved by >=1 of n proofs").
      * **pass@1** -- `val-core/<ds>/acc/mean@n`, the average correctness of an
        individual sampled proof.
    The EMA(0.8) is on the headline best@n curve. Validation step gaps preserved.
    """
    _vn = F.cfg("val_best_at_n") or F.cfg("val_n_default")
    vn = int(_vn) if _vn not in (None, "", "None") else None
    vbest = F.arr(f"val-core__{val_ds}__acc__best@{vn}__mean", 100)
    vpass1 = F.arr(f"val-core__{val_ds}__acc__mean@{vn}", 100)
    _vline(ax, R, vbest, CO["prover_dark"], f"best@{vn} (val-core)", marker="o", ms=4, lw=2.4)
    _vline(ax, R, vpass1, CO["prover"], f"pass@1 (mean@{vn})", marker="o", ms=3, lw=1.4)
    # Short EMA (alpha=0.95 new-point weight) on the verl best@n curve, gaps
    # preserved (present val points only, not interpolated).
    _vline(ax, R, _ema_alpha(vbest, 0.95), CO["teal"], f"best@{vn} EMA (a=0.95)", lw=2.0)
    # Reference: QED-Nano (the released 4B prover) on this exact IMO-ProofBench + DS4-Flash
    # protocol, from a matched evaluation (bare IMO-ProofBench prompt + full-output
    # judging + 229K budget, n=4): avg@4 = 0.304, best@4 = 0.426. These are plain mean/max
    # over 4 samples; our val curves above use verl's bootstrap-with-replacement estimator, so
    # treat as approximate reference lines, not like-for-like.
    if val_ds == "imoproofbench":
        ax.axhline(30.4, color=CO["gray"], ls=(0, (5, 2)), lw=1.1, label="QED-Nano avg@4 (0.304)")
        ax.axhline(42.6, color=CO["gray"], ls=(0, (1, 1)), lw=1.1, label="QED-Nano best@4 (0.426)")
    test_freq = F.cfg("test_freq_default")
    if has(vbest) or has(vpass1):
        _floor10_bottom(ax, vbest, vpass1)
        panel_title(ax, n, "Direct-proof validation best@n + pass@1", "% correct",
                    loc="upper left")
        ax.text(0.02, -0.16,
                f"best@{vn} = verl val-core bootstrap pass@{vn} (headline); pass@1 = "
                f"mean@{vn}, avg correctness of one proof. Val runs every "
                f"test_freq={test_freq} steps (+step 0 if val_before_train).",
                transform=ax.transAxes, fontsize=6.0, color="gray", va="top")
    else:
        na(ax, n, "Direct-proof validation best@n + pass@1",
           f"no val best@{vn} yet\n(runs at test_freq={test_freq})")


def _panel_dapo_refill_by_mode(ax, n, R, F):
    """num_gen_batches (global) + per-mode de_filtered fraction."""
    ngb = F.arr("train__num_gen_batches")
    pg_filt = F.arr("v7__train__proposer__de_filtered_fraction", 100)
    rg_filt = F.arr("v7__train__prover__de_filtered_fraction", 100)
    drew = False
    if has(ngb):
        ax.bar(R, present(ngb), color=CO["light_gray"], width=0.7, label="num_gen_batches")
        ax.set_ylabel("gen batches")
        drew = True
    ax2 = ax.twinx()
    ax2.grid(False)
    drew2 = False
    if has(pg_filt):
        ax2.plot(R, pg_filt, color=CO["proposer"], marker=".", lw=1.4, label="proposer filtered %")
        drew2 = True
    if has(rg_filt):
        ax2.plot(R, rg_filt, color=CO["prover"], marker=".", lw=1.4, label="prover filtered %")
        drew2 = True
    if drew2:
        ax2.set_ylabel("% filtered")
        ax2.set_ylim(0, 105)
        ax2.legend(fontsize=6.3, loc="upper right", framealpha=0.85)
    if drew or drew2:
        ax.set_title(f"({n}) DAPO refill & filter pressure by mode", fontweight="bold", fontsize=10)
        ax.set_xlabel("step")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=6.3, loc="upper left", framealpha=0.85)
        ax.text(0.02, -0.22,
                "num_gen_batches is per-step (not per-mode). de_filtered % split by mode "
                "shows whether proposer or prover groups are being dropped disproportionately.",
                transform=ax.transAxes, fontsize=6.0, color="gray", va="top")
    else:
        na(ax, n, "DAPO refill & filter pressure by mode")


def _panel_prover_passcount(ax, n, R, F, rollout_n):
    cmap = DC.passcount_colormap(rollout_n + 1)
    specs = []
    for k in range(rollout_n + 1):
        y = F.arr(f"v7__train__prover__pass_count_hist__k{k}")
        specs.append((y, f"{k}/{rollout_n}", cmap(k / max(1, rollout_n))))
    drew = stacked(ax, R, specs, all_in_legend=True)
    if drew:
        panel_title(ax, n, "Prover pass-count dist.", "prompt groups", loc="upper left", ncol=3)
        ax.text(0.02, -0.22,
                "grouped by explicit uid using prover_judge_score. bottom 0/n = "
                "all-wrong, top n/n = saturated, middle = useful DAPO signal.",
                transform=ax.transAxes, fontsize=6.0, color="gray", va="top")
    else:
        na(ax, n, "Prover pass-count dist.", "requires explicit uid in train rollout dump")


def _panel_difficulty_phat_dist(ax, n, R, F):
    """Snapshot histogram of the difficulty sampler's per-problem p-hat (EMA of the binary-judge
    pass rate). Companion to the pass-count dist, but over the sampler's tracked difficulty across
    the whole problem set rather than one step: left = hard (rarely solved -> up-weighted), right =
    easy/saturated (down-weighted). Data from fig_data['global']['difficulty_phat']."""
    blk = F.glob.get("difficulty_phat")
    if not blk or not blk.get("counts"):
        na(ax, n, "Difficulty p̂ distribution",
           "difficulty sampling inactive\n(no difficulty_state / rollouts)")
        return
    counts = blk["counts"]
    B = blk.get("n_bins", len(counts))
    centers = [(i + 0.5) / B for i in range(B)]
    cmap = DC.passcount_colormap(B)  # red=hard(0) .. green=easy(1), same scale as pass-count dist
    colors = [cmap(i / max(1, B - 1)) for i in range(B)]
    ax.bar(centers, counts, width=(1.0 / B) * 0.92, color=colors, edgecolor="none")
    ax.set_xlim(0, 1)
    ax.set_xlabel("p̂  (EMA pass rate)")
    panel_title(ax, n, "Difficulty p̂ distribution", "problems", loc="upper right")
    N = blk.get("n", sum(counts))
    sub = f"N={N}"
    if blk.get("mean") is not None:
        sub += f"  mean={blk['mean']:.2f}"
    if blk.get("median") is not None:
        sub += f"  median={blk['median']:.2f}"
    ax.text(0.02, 0.97, sub, transform=ax.transAxes, fontsize=6.5, va="top", color="dimgray")
    hard, easy, fl = blk.get("frac_le_1_16"), blk.get("frac_ge_15_16"), blk.get("frac_at_w_floor")
    cap = ("per-problem p̂ = EMA(a=0.9) of the binary judge pass rate the sampler tracks; "
           "weight w=max(√(p̂(1-p̂)),1/32). left=hard/up-weighted, right=easy/down-weighted. ")
    if hard is not None:
        cap += f"{100*hard:.0f}% at p̂≤1/16, "
    if easy is not None:
        cap += f"{100*easy:.0f}% at p̂≥15/16, "
    if fl is not None:
        cap += f"{100*fl:.0f}% pinned at the 1/32 weight floor. "
    if blk.get("source"):
        cap += f"[{blk['source']}]"
    ax.text(0.02, -0.22, cap, transform=ax.transAxes, fontsize=6.0, color="gray", va="top")


def _panel_proposer_correctness_passcount(ax, n, R, F, rollout_n):
    cmap = DC.passcount_colormap(rollout_n + 1)
    specs = []
    for k in range(rollout_n + 1):
        y = F.arr(f"v7__train__proposer__pass_count_hist__k{k}")
        specs.append((y, f"{k}/{rollout_n}", cmap(k / max(1, rollout_n))))
    drew = stacked(ax, R, specs, all_in_legend=True)
    if drew:
        panel_title(ax, n, "Proposer correctness pass-count dist.", "prompt groups", loc="upper left", ncol=3)
        ax.text(0.02, -0.22,
                "grouped by explicit uid using correctness_judge_score. Distinguishes 'can produce a "
                "correctly-proved proposition' from 'is the proposition useful' "
                "(see the proposer impact distribution panel).",
                transform=ax.transAxes, fontsize=6.0, color="gray", va="top")
    else:
        na(ax, n, "Proposer correctness pass-count dist.", "requires explicit uid in train rollout dump")


def _panel_proposer_impact_score(ax, n, R, F):
    """Mean proposer impact-judge rubric restricted to CORRECT conjectures
    (proposer rows with correctness_judge_score == 1) -- "among the conjectures
    that are correct, how influential are they judged to be?" Raw judge score,
    independent of the reward formula (the level breakdown and the reward-applied
    impact live in the impact distribution panel). N/A for steps with no correct
    proposer rows -- the all-row impact mean is NOT substituted.

    Y-axis upper bound is data-driven: max(F.glob["impact_levels"]).
    The basic impact rubric is 0-3; the extended rubric adds level 4
    ("Impact: 4" = "almost equivalent to the seed problem"). Hardcoding
    ylim=3 here would clip the high-impact signal and hide exactly the tier
    the extended rubric is meant to surface."""
    imp = F.arr("v7__train__proposer__impact_judge_score_correct_mean")
    line(ax, R, imp, CO["impact2"], "impact among correct", marker=".", lw=1.6)
    if has(imp):
        impact_levels = F.glob.get("impact_levels", [0, 1, 2, 3])
        y_max = max(impact_levels) if impact_levels else 3
        ax.set_ylim(0, y_max)
        panel_title(ax, n, "Proposer impact among correct conjectures",
                    f"impact (0-{y_max})", loc="best")
        ax.text(0.02, -0.16,
                "mean impact-judge rubric over proposer rollouts with correct "
                "correctness_judge_score only. v1: 0=no help, 1=weak, 2=clear, "
                "3=strong. v2 adds 4=“almost equivalent to the seed problem”. "
                "Raw judge score; the all-row level breakdown is in the impact "
                "distribution panel. N/A when a step has no correct conjectures.",
                transform=ax.transAxes, fontsize=6.0, color="gray", va="top")
    else:
        na(ax, n, "Proposer impact among correct conjectures",
           "no correct proposer rows yet\n(impact_judge_score_correct_mean)")


def _panel_proposer_impact_dist(ax, n, R, F):
    """Per-step impact level fractions, stacked. Plus mean impact_applied."""
    specs = []
    # Drive the loop from fig_data["global"]["impact_levels"] so the
    # panel automatically picks up the impact-4 tier (and any future
    # widening) without an edit here. Falls back to the 0..3 tuple if the
    # field is absent (older fig_data on disk).
    impact_levels = F.glob.get("impact_levels", [0, 1, 2, 3])
    for level in impact_levels:
        y = F.arr(f"v7__train__proposer__impact_level_fraction__l{level}", 100)
        specs.append((y, f"Impact {level}", impact_color(level)))
    drew = stacked(ax, R, specs, all_in_legend=True)
    if drew:
        ax.set_ylim(0, 100)
    # impact_applied mean on a second axis (in reward units, not %).
    ia = F.arr("v7__train__proposer__impact_applied_mean")
    if has(ia):
        ax2 = ax.twinx()
        ax2.grid(False)
        ax2.plot(R, ia, color=CO["impact3"], marker="o", ms=3, lw=1.8, label="impact_applied mean")
        ax2.set_ylabel("impact_applied")
        ax2.legend(fontsize=6.3, loc="upper right", framealpha=0.85)
    if drew or has(ia):
        panel_title(ax, n, "Proposer impact distribution", "% of rollouts", loc="upper left", ncol=2)
        ax.text(0.02, -0.20,
                "Impact 0=no help, 1=weak, 2=clear help, 3=strong help. Persistent Impact-0 "
                "dominance means the proposer learns trivial-but-correct lemmas.",
                transform=ax.transAxes, fontsize=6.0, color="gray", va="top")
    else:
        na(ax, n, "Proposer impact distribution")


def _panel_generated_vs_trained(ax, n, R, F):
    """Per-mode: all generated vs trained-subset average reward."""
    # Proposer: prefer proposer_reward (final, post-clamp); fall back to score.
    p_gen = F.arr_first([
        "v7__train__proposer__proposer_reward_mean",
        "v7__train__proposer__score_mean_generated",
    ])
    p_train = F.arr("v7__train__proposer__score_mean_trained")
    # Prover: use binary prover_judge_score where available; the post-penalty
    # final reward comes from optimized_reward_mean (pre-KL post-length-penalty,
    # mode-agnostic between linear-overlong and exponential penalty).
    r_gen_binary = F.arr_first([
        "v7__train__prover__rubric_grade_mean",
        "v7__train__prover__prover_judge_score_mean",
        "v7__train__prover__score_mean_generated",
    ])
    r_train_binary = F.arr("v7__train__prover__score_mean_trained")
    r_gen_opt = F.arr("v7__train__prover__optimized_reward_mean")

    line(ax, R, p_gen, CO["proposer_light"], "proposer all-gen", marker=".", lw=1.1, ls="--")
    line(ax, R, p_train, CO["proposer"], "proposer trained", marker="o", ms=3, lw=1.8)
    line(ax, R, r_gen_binary, CO["prover_light"], "prover all-gen (binary)", marker=".", lw=1.1, ls="--")
    line(ax, R, r_train_binary, CO["prover"], "prover trained (binary)", marker="o", ms=3, lw=1.8)
    line(ax, R, r_gen_opt, CO["orange"], "prover final reward", marker=".", ls=":", lw=1.0)
    if ax.get_legend_handles_labels()[0]:
        panel_title(ax, n, "Generated vs trained rewards", "reward", loc="best", ncol=2)
    else:
        na(ax, n, "Generated vs trained rewards")


def _panel_format_length_health(ax, n, R, F, resp_cap):
    """Format rates + length percentiles, per mode."""
    drew_rates = False
    # proposer rates (left axis, %)
    yp_prop = F.arr("v7__train__proposer__proposition_tag_present_rate", 100)
    yp_proof = F.arr("v7__train__proposer__proof_tag_present_rate", 100)
    yp_both = F.arr("v7__train__proposer__both_tags_present_rate", 100)
    yr_nonempty = F.arr_first([
        "v7__train__prover__proof_nonempty_rate",
        "v7__train__prover__proof_tag_present_rate",
    ], 100)
    yr_missing = F.arr("v7__train__prover__proof_missing_rate", 100)
    yr_empty = F.arr("v7__train__prover__proof_empty_rate", 100)
    if has(yp_prop):
        ax.plot(R, yp_prop, color=CO["proposer_light"], label="prop tag (proposer)", lw=1.1, ls=":")
        drew_rates = True
    if has(yp_proof):
        ax.plot(R, yp_proof, color=CO["proposer_accent"], label="proof tag (proposer)", lw=1.1, ls="--")
        drew_rates = True
    if has(yp_both):
        ax.plot(R, yp_both, color=CO["proposer"], label="both (proposer)", marker=".", lw=1.4)
        drew_rates = True
    if has(yr_nonempty):
        ax.plot(R, yr_nonempty, color=CO["prover"], label="nonempty <proof> block (prover)", marker=".", lw=1.4)
        drew_rates = True
    if has(yr_missing):
        ax.plot(R, yr_missing, color=CO["red"], label="no <proof> tag (prover)", marker=".", ls=":", lw=1.1)
        drew_rates = True
    if has(yr_empty):
        ax.plot(R, yr_empty, color=CO["orange"], label="empty <proof> block (prover)", marker=".", ls="--", lw=1.1)
        drew_rates = True
    # length p50 on the right axis.
    ax2 = ax.twinx()
    ax2.grid(False)
    drew_len = False
    cp50 = F.arr("v7__train__proposer__candidate_len_p50")
    rp50 = F.arr("v7__train__prover__proof_len_p50")
    if has(cp50):
        ax2.plot(R, cp50, color=CO["proposer_dark"], ls=":", marker=".", lw=1.0,
                 label="proposer candidate p50")
        drew_len = True
    if has(rp50):
        ax2.plot(R, rp50, color=CO["prover_dark"], ls=":", marker=".", lw=1.0,
                 label="prover proof p50")
        drew_len = True
    if drew_len:
        ax2.set_ylabel("chars (p50)")
        ax2.legend(fontsize=6.0, loc="upper right", framealpha=0.85)
    if drew_rates or drew_len:
        if drew_rates:
            ax.set_ylim(0, 105)
        panel_title(ax, n, "Format & length health", "% present", loc="best", ncol=2)
    else:
        na(ax, n, "Format & length health")


def _panel_proposer_resp_len(ax, n, R, F, resp_cap):
    """Proposer actor rollout response length in TOKENS, p50/p90/p99.

    This is the actor rollout length, kept as a
    standalone panel from the format/length-health panel. Tokens only: the parser
    omits rows without `response_length` (char fallback disallowed to avoid
    char/token mislabeling), so a run/step with no response-token fields renders
    N/A. The character-length view lives in the format/length-health panel.
    """
    drew = DC.percentile_family(
        ax, R, F, "v7__train__proposer__resp_len", "resp", CO["proposer_dark"],
    )
    if drew:
        _cap_axhline(ax, resp_cap, "resp cap")
        ax.set_ylim(bottom=0)
        panel_title(ax, n, "Proposer rollout response length", "tokens", loc="upper left")
    else:
        na(ax, n, "Proposer rollout response length",
           "no actor response_length field in dump")


def _panel_prover_resp_len(ax, n, R, F, resp_cap):
    """Prover actor rollout response length in TOKENS, p50/p60/p70/p80/p90/p99.

    Tokens only: the parser omits rows without `response_length` (the char
    fallback is disallowed, since mixing chars and tokens on one axis mislabeled
    mixed runs). A run/step with no response-token fields renders N/A rather than
    a misleading char curve. Kept separate from the proposer panel so each mode's
    length pressure is visible even when one tail is much longer.

    Extended percentile set (p60/p70/p80 added) renders as a smooth alpha+lw
    gradient so the body of the distribution is legible — not just median and
    extreme tails. p99 keeps its solid + arrow-marker emphasis. Older fig_data
    that lacks p60/p70/p80 keys just drops those curves (still renderable).
    """
    # Prefer the SNIS-WEIGHTED percentile family (resp_len_wtd_p{q}: quantiles of the
    # uniform-target length distribution under difficulty sampling, matching the SNIS-weighted
    # mean line) and fall back to the raw-sample family for older fig_data without the keys.
    wtd = DC.percentile_family(
        ax, R, F, "v7__train__prover__resp_len_wtd", "resp", CO["prover_dark"],
        percentiles=(10, 20, 30, 40, 50, 60, 70, 80, 90, 99),
        color_per_percentile=True,
    )
    drew = wtd or DC.percentile_family(
        ax, R, F, "v7__train__prover__resp_len", "resp", CO["prover_dark"],
        percentiles=(10, 20, 30, 40, 50, 60, 70, 80, 90, 99),
        color_per_percentile=True,
    )
    # (The SNIS-weighted MEAN moved to its own panel 38 — this panel is the quantile fan only.)
    if drew:
        _cap_axhline(ax, resp_cap, "resp cap")
        ax.set_ylim(bottom=0)
        panel_title(ax, n,
                    "Prover rollout response length" + (" (SNIS-wtd)" if wtd else ""),
                    "tokens", loc="upper left")
    else:
        na(ax, n, "Prover rollout response length",
           "no actor response_length field in dump")


def _panel_gen_len_all(ax, n, R, F, resp_cap):
    """NEWLY GENERATED tokens per completion, pooled over every row (panel 5b).

    Deliberately NOT split by Q lane, and deliberately not response_length.

    response_length includes the injected replay prefix, which makes it a poor length
    signal: measured at step 211, the g-capped lane's median response was 22,312 tokens
    while the model had generated exactly 10,000 of them -- the rest was prefix we pasted
    in. Its p99 response of 62,964 (= 52,964 prefix + 10,000 generated) looked like it
    violated the 10k cap when nothing had. Nearly all the spread in that fan was prefix
    sampling, not policy behaviour.

    Generated tokens is the quantity the Q budget g actually bounds and the only one that
    says anything about the model, so the lane split stops being necessary: the g-capped
    population shows up on its own as a pile-up at g (83% of short rows sit exactly there),
    and the full-budget population as the long tail above it.
    """
    pre = "v7__train__prover__gen_len"
    drew = DC.percentile_family(
        ax, R, F, pre, "gen", CO["prover_dark"],
        percentiles=(10, 20, 30, 40, 50, 60, 70, 80, 90, 99),
        color_per_percentile=True)
    if not drew:
        na(ax, n, "Prover generated tokens", "no gen_len keys (re-parse to backfill)")
        return
    # g is not recorded in the manifest, so the parse observes it (max generated on a
    # short-routed row). Drawn only when observed, never guessed.
    _g = F.arr(f"{pre}_short_cap_observed")
    if has(_g):
        _gv = int(np.nanmax(_g))
        ax.axhline(_gv, color="black", ls=":", lw=1.0,
                   label=f"Q short-lane cap g ({_gv})")
    _cap_axhline(ax, resp_cap, "response budget")
    ax.set_ylim(bottom=0)
    panel_title(ax, n, "Prover generated tokens (all completions)", "tokens", loc="upper left")


def _panel_resp_len_stream(ax, n, R, F, resp_cap, stream):
    """Response-length quantile fan for ONE stream (panels 5a/5b).

    5a = scratch / from-scratch inflow rows (sp_prefix_len==0, always full budget);
    5b = replay-prefix rows, with the g-capped SHORT lane split from the full-budget
    lanes (full+audit) so the Q-routing effect on length is legible — pooling them
    (the old single panel 5) hides that ~45% of replay slots are capped at g.
    Renders N/A when the stream keys are absent (older fig_data).
    """
    pre = "v7__train__prover__resp_len"
    _QS = (10, 20, 30, 40, 50, 60, 70, 80, 90, 99)
    if stream == "scratch":
        drew = DC.percentile_family(
            ax, R, F, f"{pre}_scratch", "resp", CO["prover_dark"],
            percentiles=_QS, color_per_percentile=True)
        title = "Prover resp length: from-scratch stream"
    else:
        # replay: two lanes on one axis. Full decile fan per lane, distinguished by
        # COLOR (lane) and alpha/width ramp (quantile) — solid for full+audit, dashed
        # for the g-capped short lane. p50 and p99 carry emphasis + labels; the middle
        # deciles render unlabelled so the legend stays readable.
        drew = False
        for _sfx, _col, _lab, _ls in (("replay_full", CO["prover_dark"], "full+audit", "-"),
                                      ("replay_short", CO["orange"], "short (g-capped)", "--")):
            for _q in _QS:
                y = F.arr(f"{pre}_{_sfx}_p{_q}")
                if not has(y):
                    continue
                _emph = _q in (50, 99)
                _t = (_q - 10) / 89.0  # 0 at p10 -> 1 at p99
                line(ax, R, y, _col,
                     f"{_lab} p{_q}" if _emph else "_nolegend_",
                     ls=_ls, lw=(1.9 if _q == 50 else 1.4 if _q == 99 else 0.8),
                     alpha=(1.0 if _emph else 0.30 + 0.35 * _t),
                     marker="." if _q == 50 else None, ms=3)
                drew = True
        title = "Prover resp length: replay stream by lane"
    if drew:
        if stream == "scratch":
            # From-scratch rows carry no prefix, so the nominal budget IS the ceiling.
            _cap_axhline(ax, resp_cap, "resp cap")
        else:
            # Plot the cap each lane ACTUALLY faces, not the nominal budget. Two
            # corrections:
            #
            #  1. prefix. A replay row's response CONTAINS its injected prefix, so its
            #     real headroom is budget - prefix, and the prefix is a fractional cut in
            #     [0, 0.80] of a trajectory that is itself tens of thousands of tokens.
            #     So we draw budget-mean_prefix and budget-max_prefix.
            #  2. the g cap. The short lane is bounded by SP_Q_BUDGET_G (10k), which is
            #     nowhere near the 75k budget — drawing only the 75k line made that lane
            #     look comfortably clear of a ceiling it is in fact pinned against.
            _cap_axhline(ax, resp_cap, "budget")
            _pm = F.arr("v7__train__prover__prefix_tokens_replay_full_mean")
            _px = F.arr("v7__train__prover__prefix_tokens_replay_full_max")
            # numpy arithmetic, NOT a list comprehension: F.arr() returns an ndarray with
            # NaN for missing steps, and has()/line() call .size on their argument -- a
            # plain list raises AttributeError at render time. Subtracting from the array
            # propagates NaN for free.
            if resp_cap is not None and has(_pm):
                line(ax, R, resp_cap - _pm,
                     "dimgray", "eff. cap (budget - mean prefix)", ls=":", lw=1.3)
            if resp_cap is not None and has(_px):
                line(ax, R, resp_cap - _px,
                     "darkgray", "min eff. cap (budget - max prefix)", ls=":", lw=1.0)
            _g = F.cfg("sp_q_budget_g_default")
            if _g not in (None, "", "None"):
                _cap_axhline(ax, int(_g), "short-lane g cap")
        ax.set_ylim(bottom=0)
        panel_title(ax, n, title, "tokens", loc="upper left")
    else:
        na(ax, n, title, "no stream-split resp_len keys (older fig_data)")


def _panel_q_reference_bank(ax, n, R, F):
    """Generative-Q reference bank growth.

    The bank is add-once, uncovered-problems-only: a problem enters when it first
    produces a judged-passing row. Plots cumulative size (left) against the per-step
    additions and the no-reference skip count — the latter is the health signal
    (Q-training records dropped because their problem has no reference proof).
    """
    size = F.arr("q__bank_size")
    added = F.arr("q__bank_added")
    skipped = F.arr("q__sample_skipped_no_reference")
    drew = False
    if has(size):
        line(ax, R, size, CO["prover_dark"], "bank size (problems)", marker=".", lw=2.0)
        drew = True
    if drew:
        panel_title(ax, n, "Q reference bank", "problems covered", loc="upper left")
        ax2 = ax.twinx()
        if has(added):
            line(ax2, R, added, CO["orange"], "added/step", marker=".", ls="--", lw=1.1)
        if has(skipped):
            line(ax2, R, skipped, CO["red"], "records skipped (no ref)", marker="x",
                 ls=":", lw=1.1)
        ax2.set_ylabel("per step", fontsize=7)
        ax2.tick_params(labelsize=6)
        ax2.legend(loc="lower right", fontsize=5.5, framealpha=0.6)
        ax.text(0.02, -0.16,
                "add-once, uncovered-problems-only: a problem enters on its first "
                "judged-passing row. Skips = Q-train records without a reference proof.",
                transform=ax.transAxes, fontsize=6.0, color="gray", va="top")
    else:
        na(ax, n, "Q reference bank", "no q/bank_size in fig_data")


def _panel_prover_resp_len_snis_mean(ax, n, R, F, resp_cap):
    """SNIS importance-weighted MEAN prover response length, standalone (panel 38).

    resp_len_mean is emitted as Σ c·L / Σ c (self-normalized IS over the difficulty
    draw-weights c) on difficulty-sampling runs, and equals the plain mean on uniform
    runs — the importance-corrected / uniform-target expected length. Split out of the
    percentile-fan panel (5) so the headline length trend reads on its own axis.
    """
    snis_mean = F.arr("v7__train__prover__resp_len_mean")
    if has(snis_mean):
        line(ax, R, snis_mean, CO["orange"], "mean (SNIS-wtd)", marker="o", ms=3.5, lw=2.0)
        _cap_axhline(ax, resp_cap, "resp cap")
        ax.set_ylim(bottom=0)
        panel_title(ax, n, "Prover mean response length (SNIS-wtd)", "tokens", loc="upper left")
    else:
        na(ax, n, "Prover mean response length (SNIS-wtd)",
           "no resp_len_mean in fig_data")


def _panel_judge_health(ax, n, R, F):
    """Judge health as the mutually-exclusive precedence buckets, stacked as a
    share of generated rows (proposer+prover summed). Only the four UNHEALTHY
    buckets are stacked, so the stack height == the judge-unhealthy rate and
    clean is the implied complement (100% - stack). Unlike the raw http/parse/
    trunc rates these buckets don't overlap (truncation implies a parse failure),
    so the composition is exact: you can see whether the unhealthiness is
    truncation (judge token cap), no-proof (malformed model output), genuine
    parse failures, or http outages."""
    # bottom -> top: truncation first (the usual dominant cause), then the rare ones.
    cats = [
        ("truncated", "truncated (judge cap)", CO["orange"]),
        ("parse_failed", "parse failed (not trunc)", CO["red"]),
        ("http_error", "http error", CO["purple"]),
        ("no_proof", "no proof/proposition", CO["gray"]),
    ]
    gen = (present(F.arr("v7__train__proposer__rows_generated"))
           + present(F.arr("v7__train__prover__rows_generated")))
    specs = []
    for cat, label, col in cats:
        cnt = (present(F.arr(f"v7__train__proposer__judge_cat__{cat}"))
               + present(F.arr(f"v7__train__prover__judge_cat__{cat}")))
        pct = np.where(gen > 0, 100.0 * cnt / np.maximum(gen, 1e-9), 0.0)
        specs.append((pct, label, col))
    # Percentage stack — lock the y-axis to [0, 100] so sparse-unhealthy
    # runs (mostly-clean judge) don't get inflated by auto-scaling.
    drew = stacked(ax, R, specs, all_in_legend=True, y_max=100)
    if drew:
        panel_title(ax, n, "Judge health buckets (% of generated)", "% unhealthy",
                    loc="upper left", ncol=2)
        ax.text(0.02, -0.16,
                "mutually-exclusive precedence buckets, proposer+prover summed. "
                "Stack height = judge-unhealthy rate; clean = 100% - stack.",
                transform=ax.transAxes, fontsize=6.0, color="gray", va="top")
    else:
        na(ax, n, "Judge health buckets",
           "judge_cat__* buckets absent — re-parse the run to populate")


def _panel_proposer_judge_token_budget(ax, n, R, F, judge_max_tokens):
    """Proposer judge prompt + completion p50/p90/p99, plus the configured cap."""
    drew = False
    drew |= DC.percentile_family(
        ax, R, F, "v7__train__proposer__judge_prompt_tokens",
        "prompt", CO["proposer_accent"],
    )
    drew |= DC.percentile_family(
        ax, R, F, "v7__train__proposer__judge_completion_tokens",
        "compl", CO["proposer_dark"],
    )
    if drew:
        _cap_axhline(ax, judge_max_tokens, "max tok")
        ax.set_ylim(bottom=0)
        panel_title(ax, n, "Proposer judge token budget", "tokens",
                    loc="upper left", ncol=2)
        ax.text(0.02, -0.16,
                "proposer makes 2 judge calls per generated theorem "
                "(correctness+impact); aggregated numbers are SUMS across both calls.",
                transform=ax.transAxes, fontsize=6.0, color="gray", va="top")
    else:
        na(ax, n, "Proposer judge token budget")


def _panel_prover_judge_token_budget(ax, n, R, F, judge_max_tokens):
    """Prover judge COMPLETION p50/p90/p99 + the configured cap.

    Prompt-token percentile curves were dropped: the prover judge prompt is
    a fixed template + theorem text, so its size is bounded and doesn't tell
    you much about the budget — the interesting signal is completion (the
    judge's actual reasoning output). The cap line uses
    ``manifest.judge_max_tokens`` (the real launched cap), not the cfg
    default."""
    drew = DC.percentile_family(
        ax, R, F, "v7__train__prover__judge_completion_tokens",
        "compl", CO["prover_dark"],
    )
    if drew:
        _cap_axhline(ax, judge_max_tokens, "max tok")
        ax.set_ylim(bottom=0)
        panel_title(ax, n, "Prover judge token budget", "tokens",
                    loc="upper left", ncol=2)
        ax.text(0.02, -0.16,
                "prover makes 1 judge call per generated theorem (correctness only). "
                "Completion tokens only; cap = manifest.judge_max_tokens.",
                transform=ax.transAxes, fontsize=6.0, color="gray", va="top")
    else:
        na(ax, n, "Prover judge token budget")


def _panel_judge_rollout_length(ax, n, R, F, judge_max_tokens):
    """Judge ROLLOUT length — completion (output) tokens the judge generates per
    grade: p50/p90/p99/max vs the configured cap. The p99/max tail riding the cap
    line is the main driver of reward/judge wall-time, since each step blocks on
    the slowest of ~(batch*n) judge calls. Uses the ``_p100`` (= max) family key."""
    drew = DC.percentile_family(
        ax, R, F, "v7__train__prover__judge_completion_tokens",
        "compl", CO["prover_dark"],
        percentiles=(50, 90, 99, 100), color_per_percentile=True,
    )
    if drew:
        _cap_axhline(ax, judge_max_tokens, "max tok")
        ax.set_ylim(bottom=0)
        panel_title(ax, n, "Judge rollout length (completion)", "tokens",
                    loc="upper left", ncol=2)
        ax.text(0.02, -0.16,
                "judge output tokens/grade (p50/p90/p99/max); cap = manifest.judge_max_tokens. "
                "p99/max hugging the cap => judge decodes to budget, inflating reward time.",
                transform=ax.transAxes, fontsize=6.0, color="gray", va="top")
    else:
        na(ax, n, "Judge rollout length (completion)")


def _panel_judge_input_length(ax, n, R, F):
    """Judge INPUT length — prompt (input) tokens per grade: p50/p90/p99/max.
    Tracks how large the candidate proofs the judge must read are (template +
    theorem + proof); long inputs raise judge prefill cost, the other half of
    reward wall-time. Uses the ``_p100`` (= max) family key."""
    drew = DC.percentile_family(
        ax, R, F, "v7__train__prover__judge_prompt_tokens",
        "prompt", CO["prover"],
        percentiles=(50, 90, 99, 100), color_per_percentile=True,
    )
    if drew:
        ax.set_ylim(bottom=0)
        panel_title(ax, n, "Judge input length (prompt)", "tokens",
                    loc="upper left", ncol=2)
        ax.text(0.02, -0.16,
                "judge input tokens/grade (p50/p90/p99/max) = template + theorem + candidate "
                "proof. Longer proofs raise judge prefill cost.",
                transform=ax.transAxes, fontsize=6.0, color="gray", va="top")
    else:
        na(ax, n, "Judge input length (prompt)")


def _panel_throughput(ax, n, R, F):
    """Per-node token throughput by phase.

    - **training tok/s/node** = ``perf__total_num_tokens / timing_s__update_actor / nnodes``
    - **rollout tok/s/node** = ``perf__tokens_per_sec__pi1_rollout / nnodes``
    - **judge tok/s/node** = ``Σ_mode (judge_completion_tokens_sum) / timing_s__reward / nnodes``
      (judge completion tokens summed across proposer + prover; the parser
      emits the sum per mode alongside the percentile family. Older fig_data
      that lacks the sum keys renders judge tok/s as N/A.)

    Per-node normalization (using ``manifest.nnodes`` or ``cfg.nnodes``) keeps
    runs at different node counts directly comparable on this panel."""
    try:
        node_count = float(
            (F.d.get("manifest") or {}).get("nnodes")
            or F.cfg("nnodes") or 1
        )
    except (TypeError, ValueError):
        node_count = 1.0
    node_count = max(node_count, 1.0)

    def per_node_rate(tokens, seconds, min_seconds=30.0):
        # min_seconds guard: a step that satisfied a phase from the step-cache (graft /
        # deliberate restart) reports ~seconds for that phase; tokens/epsilon explodes to
        # ~1e10 and flattens every real line. Sub-floor timings are cache hits, not
        # throughput — mask them.
        return np.divide(
            tokens, seconds * node_count,
            out=np.full(np.asarray(tokens).shape, np.nan, dtype=float),
            where=np.asarray(seconds) >= min_seconds,
        )

    train_tokens = F.arr("perf__total_num_tokens")
    train_time = F.arr("timing_s__update_actor")
    train_tps = per_node_rate(train_tokens, train_time)

    rollout_tps_total = F.arr("perf__tokens_per_sec__pi1_rollout")
    rollout_tps = rollout_tps_total / node_count if has(rollout_tps_total) else rollout_tps_total
    # The vendored verl does not emit perf/tokens_per_sec/pi1_rollout — derive rollout
    # throughput from the parser's generated-token sums (response minus replay
    # prefix; prefix tokens are injected, not generated) over timing_s/gen.
    if not has(rollout_tps):
        gen_tokens = (
            present(F.arr("v7__train__prover__generated_tokens_sum"))
            + present(F.arr("v7__train__proposer__generated_tokens_sum"))
        )
        gen_time = F.arr("timing_s__gen")
        if has(gen_tokens) and has(gen_time):
            rollout_tps = per_node_rate(gen_tokens, gen_time)

    judge_tokens = (
        present(F.arr("v7__train__proposer__judge_completion_tokens_sum"))
        + present(F.arr("v7__train__prover__judge_completion_tokens_sum"))
    )
    judge_time = F.arr("timing_s__reward")
    judge_tps = per_node_rate(judge_tokens, judge_time)

    line(ax, R, train_tps, CO["red"], "training tok/s/node", marker=".")
    line(ax, R, rollout_tps, CO["blue"], "rollout tok/s/node", marker=".")
    line(ax, R, judge_tps, CO["green"], "judge tok/s/node", marker=".")
    if has(train_tps) or has(rollout_tps) or has(judge_tps):
        ax.set_ylim(bottom=0)
        panel_title(ax, n, "Throughput", "tokens/sec/node", loc="upper left")
        ax.text(0.02, -0.16,
                f"normalized per node ({int(node_count)} nodes). "
                "judge sums proposer+prover judge_completion_tokens_sum.",
                transform=ax.transAxes, fontsize=6.0, color="gray", va="top")
    else:
        na(ax, n, "Throughput")


def _panel_actor_lr(ax, n, R, F):
    lr = F.arr_first(["actor__lr"])
    line(ax, R, lr, CO["proposer_dark"], "actor/lr", marker=".", lw=1.6)
    if has(lr):
        panel_title(ax, n, "Actor LR", "lr", loc="best")
        ax.text(0.02, -0.16,
                "standalone so schedule changes and resume-time HOT_FIX_LR are visible.",
                transform=ax.transAxes, fontsize=6.0, color="gray", va="top")
    else:
        na(ax, n, "Actor LR")


def _panel_grad_norm(ax, n, R, F):
    gn = F.arr("actor__grad_norm")
    line(ax, R, gn, CO["red"], "grad norm", marker=".", lw=1.4)
    # Read the actual clip threshold from the launch manifest
    # (`actor_grad_clip` — set per-run, e.g. 0.3) rather than assuming 1.0, which
    # would be wrong for any run with a non-default clip.
    manifest = F.d.get("manifest") or {}
    clip = manifest.get("actor_grad_clip")
    try:
        clip = float(clip) if clip is not None else None
    except (TypeError, ValueError):
        clip = None
    if clip is not None:
        ax.axhline(clip, color=CO["gray"], ls=":", lw=0.8,
                   label=f"grad clip {clip:g}")
    if has(gn):
        panel_title(ax, n, "Gradient norm", "norm", loc="best")
        ax.text(0.02, -0.16,
                "global shared-actor metric; not split by mode. "
                "Clip threshold from manifest.actor_grad_clip.",
                transform=ax.transAxes, fontsize=6.0, color="gray", va="top")
    else:
        na(ax, n, "Gradient norm")


def _panel_policy_loss(ax, n, R, F):
    loss = F.arr_first(["actor__loss", "actor__pg_loss"])
    line(ax, R, loss, CO["blue"], "loss", marker=".", lw=1.4)
    if has(loss):
        ax.axhline(0, color="black", lw=0.6)
        panel_title(ax, n, "Policy loss", "loss", loc="best")
    else:
        na(ax, n, "Policy loss")


def _panel_policy_kl(ax, n, R, F):
    kl_ppo = F.arr("actor__ppo_kl")
    kl_loss = F.arr("actor__kl_loss")
    line(ax, R, kl_ppo, CO["purple"], "ppo_kl", marker=".", lw=1.4)
    line(ax, R, kl_loss, CO["pink"], "kl_loss", marker=".", lw=1.2, ls="--")
    if has(kl_ppo) or has(kl_loss):
        panel_title(ax, n, "Policy KL", "KL", loc="best")
    else:
        na(ax, n, "Policy KL")


def _panel_ppo_clip(ax, n, R, F):
    upper = F.arr("actor__pg_clipfrac", 100)
    lower = F.arr("actor__pg_clipfrac_lower", 100)
    line(ax, R, upper, CO["orange"], "upper clip %", marker=".", lw=1.4)
    line(ax, R, lower, CO["teal"], "lower clip %", marker=".", lw=1.2, ls="--")
    if has(upper) or has(lower):
        ax.set_ylim(bottom=0)
        panel_title(ax, n, "PPO clipping", "%", loc="best")
    else:
        na(ax, n, "PPO clipping")


# ---------------------------------------------------------------------------
# Additional main-page panels and their helpers.
# ---------------------------------------------------------------------------


def _ema_alpha(y, alpha: float):
    """EMA over present (non-NaN) points only; NaN elsewhere so validation
    gaps are preserved rather than interpolated. ema_t = a*x_t + (1-a)*ema_{t-1}."""
    out = np.full(len(y), np.nan)
    prev = None
    for i, v in enumerate(y):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        prev = v if prev is None else alpha * v + (1.0 - alpha) * prev
        out[i] = prev
    return out


def _inflow_trained_lane_series(F):
    """Per-step (mask, mean%, best%, strict7%) of the TRAINED replay lane on an
    inflow-only run (SP_SCRATCH_INFLOW_ONLY) plus a per-step mask of where a
    BIASED non-HT estimate (gw or naive) is what is drawn, or (None,)*5 when the run never logged an sp_inflow
    key (such runs keep panel 1's
    default sources untouched).

    Sources, coalesced per step: the audit-reweighted Horvitz-Thompson estimators
    (parser: replay_stream_*_full_suffix_est) where present — the naive keys are
    BIASED LOW on Q-routed steps because ready groups' short rollouts hit the
    g-cap and are recorded rp=0/pjs=0 — else the trainer's naive replay-lane keys,
    which are unbiased exactly while no group is Q-routed.

    THE MASK IS DELIBERATELY NOT "steps that logged sp_inflow/trained_rows". Older
    dumps emit that metric only on steps that actually DROPPED inflow rows at
    the reward site, and on burst-inflow runs (the stale-replay-buffer ablation) that is one
    step in ten: panel 1 would then silently fall back to the route-blind
    rubric_grade_mean on the other nine, averaging every censored short row in as a
    zero — a sawtooth collapse artifact.
    Instead, the key appearing ANYWHERE marks the run inflow-only, and a step then
    belongs to the lane if it logged the key itself, OR if it carries a
    trained-lane estimate but NO statement stream at all (a zero-inflow burst-run
    step from before the trainer emitted the marker unconditionally). The
    no-statement-stream condition is what keeps grafted histories intact: a
    parent step from before inflow-only training also lacks the sp_inflow key and
    also has replay estimates, but it HAS a statement stream and must keep its
    orig-stream rendering. On every-step-inflow runs the mask reduces to "steps
    that logged the key", so their rendered panels are unchanged."""
    seen = F.arr("sp_inflow__trained_rows")
    if not has(seen):
        return None, None, None, None, None
    pm = F.arr("v7__train__prover__replay_stream_pass_full_suffix_est", 100)
    bn = F.arr("v7__train__prover__replay_stream_best_full_suffix_est", 100)
    s7 = F.arr("v7__train__prover__replay_stream_strict7_full_suffix_est", 100)
    # then the parser's GROUP-WEIGHTED dump-derived estimates (dynamic group size: the
    # trainer's naive per-row keys weigh a 32-group four times an 8-group; on fixed-n
    # runs these equal the naive keys)
    pm = np.where(np.isfinite(pm), pm, F.arr("v7__train__prover__replay_stream_pass_mean_gw", 100))
    bn = np.where(np.isfinite(bn), bn, F.arr("v7__train__prover__replay_stream_best_at_n_gw", 100))
    s7 = np.where(np.isfinite(s7), s7, F.arr("v7__train__prover__replay_stream_strict7_rate_gw", 100))
    # The HT estimator is the ONLY unbiased one here. Both fallbacks -- the parser's
    # group-weighted _gw estimate and the trainer's naive per-row key -- put every Q-routed
    # short-lane row into the denominator with pass=0 (the text is g-capped and structurally
    # unjudgeable), so they are biased LOW by exactly the censored share. On a run with no
    # audit lane (AUDIT_DEN=0) the HT key does not exist on ANY step, so the fallback is not
    # a rare gap-filler but the whole line, and its downward drift as the ready fraction
    # grows is mechanical, not learning. Report per step whether the HT estimate is what is
    # actually being drawn, so the caller can label the line truthfully (on a no-audit
    # run the gw fallback is drawn on every step, so the note must not claim an HT estimate).
    biased = ~np.isfinite(F.arr("v7__train__prover__replay_stream_pass_full_suffix_est"))
    pm = np.where(np.isfinite(pm), pm, F.arr("replay__pass_rate_replay", 100))
    bn = np.where(np.isfinite(bn), bn, F.arr("replay__best_at_n_replay", 100))
    s7 = np.where(np.isfinite(s7), s7, F.arr("replay__strict7_frac_replay", 100))
    biased = biased & np.isfinite(pm)
    # statement stream present? trainer key, else the parser's dump-derived key
    has_orig = (np.isfinite(F.arr("replay__pass_rate_orig"))
                | np.isfinite(F.arr("v7__train__prover__orig_stream_pass_mean")))
    mask = np.isfinite(seen) | ((np.isfinite(pm) | np.isfinite(bn)) & ~has_orig)
    return mask, pm, bn, s7, biased


def _mode_train_correctness(ax, n, R, F, rollout_n, mode, title, mean_key, col,
                            col_light, note=None):
    """One mode's train correctness: mean@n + EMA + best@n (>=1 of n).

    mean@n = average binary success across generated completions; best@n =
    whether any completion in the prompt group succeeded (= (groups - all-zero
    groups)/groups)."""
    mean_raw = F.arr(mean_key, 100)
    if mode == "prover":
        # rubric-judge runs: plot the NON-binary rubric grade (points/7) as the reward magnitude,
        # not the strict-7 prover_judge_score. Falls back to the binary key then score on old runs.
        _rg = F.arr("v7__train__prover__rubric_grade_mean", 100)
        _base = mean_raw if has(mean_raw) else F.arr("v7__train__prover__score_mean_generated", 100)
        if has(_rg):
            # Per-step COALESCE, not wholesale replace: across a parent chain the parent (a
            # different judge) may populate only the base prover_judge_score key while the
            # child populates rubric_grade_mean. Taking rubric wholesale would blank every
            # step the child didn't cover (e.g. the inherited <=boundary history). Prefer
            # the rubric grade where present, else the base score.
            mean_raw = np.where(np.isfinite(_rg), _rg, _base)
        else:
            mean_raw = _base
    groups = F.arr(f"v7__train__{mode}__groups_generated")
    allzero = F.arr(f"v7__train__{mode}__group_allzero")
    passn = np.where(groups > 0, 100.0 * (groups - allzero) / np.maximum(groups, 1e-9), np.nan)
    _censored_steps = None                  # set on no-audit runs; see the inflow block
    _mean_label = f"mean@{rollout_n}"
    # Replay-prefix runs: keep this panel backward-comparable across the parent chain
    # by showing ONLY the from-scratch (statement-row) stream on replay steps — the
    # dump-derived aggregates mix in the 192 prefix-conditioned rows, whose elevated pass
    # rate is a different quantity (panel 39 carries the split). Orig-stream keys are
    # logged by the sp_replay hook; replay steps recorded before those keys existed are
    # MASKED rather than plotted as inflated mixed values.
    _replay_step = None
    _inflow_step = None
    if mode == "prover":
        _rp = F.arr("replay__pass_rate_orig")
        if has(_rp):
            _replay_step = np.isfinite(_rp)
            # Orig-stream sources, coalesced PER STEP (trainer metrics from the sp_replay
            # hook win; the parser's dump-derived aggregates fill steps from before the
            # trainer logged them — dumps tagged by the trainer or the tagging scan).
            # Steps with neither source are masked rather than plotted as mixed values.
            def _coalesce(trainer_key, parser_key):
                a = F.arr(trainer_key, 100)
                b = F.arr(parser_key, 100)
                return np.where(np.isfinite(a), a, b)

            # mean@16 on replay steps = the from-scratch PASS RATE (points>=6), the same
            # quantity as panel 39's from-scratch line. Under a
            # smooth 0-7 judge the grade mean diverges upward (partial credit), so plotting
            # it here would break comparability with the parent's pass-based history; the
            # grade signal remains visible via the reward panels.
            _pm_o = _coalesce("replay__pass_rate_orig",
                              "v7__train__prover__orig_stream_pass_mean")
            mean_raw = np.where(_replay_step,
                                np.where(np.isfinite(_pm_o), _pm_o, np.nan), mean_raw)
            _bn_o = _coalesce("replay__best_at_n_orig",
                              "v7__train__prover__orig_stream_best_at_n")
            passn = np.where(_replay_step,
                             np.where(np.isfinite(_bn_o), _bn_o, np.nan), passn)
            note = ((note + " ") if note else "") + \
                "replay steps: from-scratch PASS RATE >=6 (matches panel 39; see it for the stream split)."
        # Inflow-only runs (SP_SCRATCH_INFLOW_ONLY): the TRAINED population IS the
        # replay lane. The naive replay/pass_rate + best_at_n are BIASED LOW on Q-routed
        # steps — ready groups' short rollouts hit the g-cap and are recorded pjs=0 (a
        # censored quarter+ of the batch). Use the UNBIASED Horvitz-Thompson / audit-
        # reweighted estimators instead (parser: replay_stream_pass_full_suffix_est =
        # mean@n, replay_stream_best_full_suffix_est = best@n), which represent the ready
        # lane by the full-budget audit sample weighted n_ready/n_audit. Detection + per-step coalescing live in
        # _inflow_trained_lane_series; the mask covers EVERY step with a trained-lane
        # estimate, not just the steps that logged sp_inflow keys (the burst-inflow
        # sawtooth artifact — see its docstring).
        _inflow_step, _pm_r, _bn_r, _s7_r, _biased_r = _inflow_trained_lane_series(F)
        if _inflow_step is not None and _inflow_step.any():
            mean_raw = np.where(_inflow_step,
                                np.where(np.isfinite(_pm_r), _pm_r, np.nan), mean_raw)
            passn = np.where(_inflow_step,
                             np.where(np.isfinite(_bn_r), _bn_r, np.nan), passn)
            # Which estimator is actually on the page? The HT estimate needs the audit lane;
            # a run with AUDIT_DEN=0 never has it, so every inflow step falls back to a
            # dump- or trainer-derived pass rate that records each Q-scored (g-capped) cut
            # row as 0.
            # As the ready fraction grows that censoring grows with it, and the line drifts
            # DOWN for a purely mechanical reason: a no-audit run read 38 -> 28
            # while its pass rate over the rows the judge actually saw stayed ~38, and its
            # validation rose, so that line must not be labelled "UNBIASED". Quantify the
            # censoring from counts the parser already has and say so on the panel.
            _naive_steps = _inflow_step & _biased_r
            _jr = F.arr("v7__train__prover__judge_rows")
            _rg = F.arr("v7__train__prover__rows_generated")
            _cens = np.where((_rg > 0) & np.isfinite(_jr), 1.0 - _jr / np.maximum(_rg, 1e-9), np.nan)
            _cens_on_naive = np.where(_naive_steps, _cens, np.nan)
            if _naive_steps.any() and np.nanmax(_cens_on_naive) > 0.02:
                _n_naive = int(_naive_steps.sum())
                _cmax = 100.0 * float(np.nanmax(_cens_on_naive))
                _clast = _cens_on_naive[np.where(_naive_steps)[0][-1]]
                # A biased line is not made honest by a caption -- it still READS as a
                # decline. So on these
                # steps the censored judge-only series (mean@n, its EMA, best@n, 7/7) are
                # NOT drawn at all. The mean line becomes score_mean_generated: judge score
                # where judged, Q's score where cut -- the full-population quantity the
                # policy is actually trained on. Not "correctness" in the strict judge sense
                # (it leans on Q's accuracy), and the label says so; validation remains the
                # clean signal. best@n and 7/7 have no Q-scored analogue, so they are simply
                # absent here rather than shown biased.
                _sg = F.arr("v7__train__prover__score_mean_generated", 100)
                mean_raw = np.where(_naive_steps, _sg if has(_sg) else np.nan, mean_raw)
                passn = np.where(_naive_steps, np.nan, passn)
                _censored_steps = _naive_steps
                _mean_label = f"mean@{rollout_n} (incl. Q-scored cuts)"
                # PREPENDED, not appended: the panel note is a single clipped line and the
                # trailing text is the part that gets clipped.
                note = (
                    "NO AUDIT LANE (%d steps): judge-only mean@n / best@n / 7-of-7 are censored "
                    "(%.0f%% of rows are Q-scored g-capped cuts counted as 0; peak %.0f%%) and are "
                    "NOT drawn. mean@n shown = score_mean_generated (judge where judged, Q where "
                    "cut). Use validation for strict correctness."
                    % (_n_naive, 100.0 * float(_clast), _cmax)
                ) + ((" " + note) if note else "")
            else:
                note = ((note + " ") if note else "") + \
                    "inflow-only steps: UNBIASED mean@n/best@n (audit-reweighted HT; ready lane " \
                    "represented by full-budget audit sample, not the g-capped short rollouts)."
    drew = False
    if has(mean_raw):
        # ±1σ band on the reported mean reward: per-step standard error of the (SNIS-corrected)
        # mean, from the prompt-group draws (parser: train_correctness_se). Narrow on the uniform
        # pre-graft steps, wider once difficulty weighting inflates the mean-estimator variance.
        se = F.arr(f"v7__train__{mode}__train_correctness_se", 100)
        if _replay_step is not None:
            se = np.where(_replay_step, np.nan, se)  # mixed-stream SE: mask on replay steps
        if _inflow_step is not None:
            # the SE describes the route-blind dump mean, not the HT estimate drawn on
            # these steps — a band from one estimator around another is noise, mask it
            se = np.where(_inflow_step, np.nan, se)
        if has(se):
            _r = np.asarray(R, dtype=float)
            _ok = np.isfinite(mean_raw) & np.isfinite(se)
            if _ok.any():
                ax.fill_between(_r[_ok], (mean_raw - se)[_ok], (mean_raw + se)[_ok],
                                color=col, alpha=0.15, lw=0, zorder=0, label="±1σ (mean SE)")
        line(ax, R, mean_raw, col, _mean_label, marker=".", lw=1.3, alpha=0.8)
        # EMA with NEW-point weight 0.95 (ema_t = 0.95*x_t + 0.05*ema_{t-1}).
        # (The generic ema() helper uses EMA_ALPHA=0.8 as the
        # PREVIOUS-EMA weight — the opposite — so it must not be used here.)
        line(ax, R, _ema_alpha(mean_raw, 0.95), col, "mean EMA (a=0.95)", lw=2.0)
        drew = True
    if has(passn):
        line(ax, R, passn, col_light, f"best@{rollout_n} (>=1 of {rollout_n})", marker=".", ls="--", lw=1.1)
        drew = True
    # Rubric-judge runs: the grade mean above is 0-7/7; also show the strict 7/7 portion (fraction
    # of rollouts graded full marks) so "correctness in the binary sense" stays visible alongside it.
    s7 = None
    if mode == "prover":
        # Only where the parser emitted the explicit strict7 key (rubric-judge steps);
        # no backfill onto parent steps — there the mean line already IS the strict rate.
        s7 = F.arr("v7__train__prover__prover_strict7_rate", 100)
        if _replay_step is not None:
            _s7_t = F.arr("replay__strict7_frac_orig", 100)
            _s7_p = F.arr("v7__train__prover__orig_stream_strict7_rate", 100)
            _s7_o = np.where(np.isfinite(_s7_t), _s7_t, _s7_p)
            s7 = np.where(_replay_step,
                          np.where(np.isfinite(_s7_o), _s7_o, np.nan), s7)
        if _inflow_step is not None and _inflow_step.any():
            # inflow-only steps: strict-7 follows the same trained lane as mean/best, and
            # gets the SAME unbiased audit-reweighted HT treatment (naive strict7 counts
            # the g-capped consumed rows as not-7, biasing it low). _s7_r comes coalesced
            # from _inflow_trained_lane_series (HT est, else naive replay strict7).
            s7 = np.where(_inflow_step,
                          np.where(np.isfinite(_s7_r), _s7_r, np.nan), s7)
            if _censored_steps is not None:
                s7 = np.where(_censored_steps, np.nan, s7)   # censored like mean@n; not drawn
        if has(s7):
            line(ax, R, s7, "#8e44ad", "7/7 rate (full marks)", marker="x", ls=":", lw=1.5)
            drew = True
        else:
            s7 = None
    if drew:
        # Include s7 in the bottom anchor: the strict rate sits BELOW the grade mean,
        # and a bottom set from mean/best alone clips it out of the axis entirely
        # (e.g. a strict rate of 26.4 under a bottom of 30).
        _floor10_bottom(ax, mean_raw, passn, s7)
        panel_title(ax, n, title, "% correct", loc="upper left")
        if note:
            ax.text(0.02, -0.16, note, transform=ax.transAxes, fontsize=6.0,
                    color="gray", va="top")
    else:
        na(ax, n, title)


def _panel_prover_train_correctness(ax, n, R, F, rollout_n):
    _mode_train_correctness(
        ax, n, R, F, rollout_n, "prover", "Prover training correctness",
        "v7__train__prover__prover_judge_score_mean", CO["prover"], CO["prover_light"],
        note="direct-proof correctness (prover_judge_score). mean@n over generated "
             "completions; best@n = any of n passed.")


def _panel_conjecture_train_correctness(ax, n, R, F, rollout_n):
    _mode_train_correctness(
        ax, n, R, F, rollout_n, "proposer", "Conjecture training correctness",
        "v7__train__proposer__correctness_judge_score_mean", CO["proposer"], CO["proposer_light"],
        note="binary correctness of the proposed proposition+proof "
             "(correctness_judge_score), independent of its impact "
             "(see the proposer impact distribution panel).")


def _panel_proposer_judge_tokens(ax, n, R, F, judge_max_tokens):
    """Proposer judge completion-token p50/p90/p99 (aggregate over the
    correctness + impact judge calls until per-call split fields are logged)."""
    drew = DC.percentile_family(
        ax, R, F, "v7__train__proposer__judge_completion_tokens",
        "proposer compl (aggr)", CO["proposer_dark"],
    )
    if drew:
        _cap_axhline(ax, judge_max_tokens, "max tok")
        ax.set_ylim(bottom=0)
        panel_title(ax, n, "Proposer judge response tokens", "tokens", loc="upper left")
        ax.text(0.02, -0.16,
                "aggregate over correctness+impact judge calls (a SUM across the two) "
                "until per-call token fields are split.",
                transform=ax.transAxes, fontsize=6.0, color="gray", va="top")
    else:
        na(ax, n, "Proposer judge response tokens")


def _panel_prover_judge_count(ax, n, R, F):
    """Prover judge counts (not rates): per-step totals plus per-prompt-group
    count percentiles, so a real correctness change is distinguishable from a
    judge-availability one. Totals: generated (denominator/context),
    attempts/success/missing/failed. Per-group p50/p90/p99 of judge attempts and
    of successful results overlay as a subordinate band (missing/failed per-group
    percentiles live in diagnostics)."""
    gen = F.arr("v7__train__prover__rows_generated")
    att = F.arr("v7__train__prover__judge_attempts")
    suc = F.arr("v7__train__prover__judge_success")
    mis = F.arr("v7__train__prover__judge_missing")
    fail = F.arr("v7__train__prover__judge_failed")
    # Totals (rows/step) on the left axis; generated is the denominator/context.
    line(ax, R, gen, CO["light_gray"], "generated (total)", marker=".", lw=1.6)
    line(ax, R, att, CO["gray"], "attempts (total)", marker=".", lw=1.4)
    line(ax, R, suc, CO["prover"], "success (total)", marker=".", lw=1.7)
    line(ax, R, mis, CO["orange"], "missing (total)", marker=".", ls="--", lw=1.1)
    line(ax, R, fail, CO["red"], "failed (total)", marker=".", ls=":", lw=1.1)
    drew_tot = any(has(y) for y in (gen, att, suc, mis, fail))
    # Per-prompt-group count percentiles on a SECONDARY axis (counts/group ~ 0..n
    # are dwarfed by the per-step totals ~1000s, so they get their own scale).
    ax2 = ax.twinx()
    ax2.grid(False)
    drew_att = DC.percentile_family(
        ax2, R, F, "v7__train__prover__judge_attempts_per_group", "attempts/grp", CO["navy"])
    drew_suc = DC.percentile_family(
        ax2, R, F, "v7__train__prover__judge_success_per_group", "success/grp", CO["prover_dark"])
    if drew_att or drew_suc:
        ax2.set_ylim(bottom=0)
        ax2.set_ylabel("count / group", fontsize=7)
        ax2.legend(fontsize=5.6, loc="upper right", framealpha=0.85, ncol=2)
    else:
        ax2.set_yticks([])
    if drew_tot or drew_att or drew_suc:
        ax.set_ylim(bottom=0)
        panel_title(ax, n, "Prover judge count", "rows / step", loc="upper left", ncol=2)
        ax.text(0.02, -0.16,
                "left=per-step totals (generated=denominator); right axis=per-group "
                "p50/p90/p99. attempts=proof present & judged; success=score==1; "
                "missing=no proof tag; failed=http/parse/trunc. missing/failed per-group: diagnostics.",
                transform=ax.transAxes, fontsize=6.0, color="gray", va="top")
    else:
        na(ax, n, "Prover judge count")


def _arr_coalesce(F, keys, scale=1.0):
    """Per-step first-finite coalesce across keys (unlike ``arr_first`` which picks
    ONE key wholesale). Across a parent chain a key may cover only part of the step axis
    (e.g. rubric_grade_mean exists only on the DS4-judge steps while
    prover_judge_score_mean covers the inherited history); arr_first would then blank
    every step the first-present key doesn't cover. Coalescing keeps the full curve."""
    out = None
    for k in keys:
        a = F.arr(k, scale)
        out = a.copy() if out is None else np.where(np.isfinite(out), out, a)
    return out if out is not None else np.array([])


def _panel_prover_reward_components(ax, n, R, F):
    """Prover step-wise reward components: final reward, correctness, length penalty.

    Correctness uses prover_judge_score (fallback score); length penalty uses
    the parser's length_penalty_mean (mode-agnostic: linear-overlong or
    exponential); final reward is the pre-KL post-penalty reward emitted by
    the DAPO manager (optimized_reward_mean).
    """
    correctness = _arr_coalesce(F, [
        "v7__train__prover__rubric_grade_mean",
        "v7__train__prover__prover_judge_score_mean",
        "v7__train__prover__score_mean_generated",
    ])
    lp = F.arr_first([
        "v7__train__prover__length_penalty_mean",
        "v7__train__prover__overlong_reward_mean",
    ])
    opt = F.arr("v7__train__prover__optimized_reward_mean")
    line(ax, R, correctness, CO["prover"], "correctness", marker=".", lw=1.2)
    line(ax, R, lp, CO["red"], "length penalty", marker=".", ls="--", lw=1.1)
    line(ax, R, opt, CO["prover_dark"], "final reward", marker="o", ms=4, lw=2.2)
    if any(has(y) for y in (correctness, lp, opt)):
        ax.axhline(0, color="black", lw=0.6)
        panel_title(ax, n, "Prover reward components", "reward", loc="best")
        ax.text(0.02, -0.16,
                "final reward = pre-KL post-penalty reward from the DAPO manager; "
                "length-penalty mode (linear / exponential) is shown on the config strip.",
                transform=ax.transAxes, fontsize=6.0, color="gray", va="top")
    else:
        na(ax, n, "Prover reward components")


def _mode_rollout_entropy(ax, n, R, F, mode, col):
    """Mode-specific rollout entropy: prefer rollout/entropy_<mode>, fall back to
    actor/entropy_<mode> (update-batch) with a clear label, else N/A.

    Prover-only fallback: with no proposer, every rollout is a prover rollout, so
    the global entropy IS the prover entropy (same population). When no mode-split
    key is logged we therefore fall back to the global rollout/entropy (or
    actor/entropy) instead of N/A — gated on ``is_prover_only(F)`` because this
    identity does NOT hold in the mixed proposer+prover setup (there global entropy
    must not be substituted into a per-mode panel)."""
    roll = F.arr(f"rollout__entropy_{mode}")
    actor = F.arr(f"actor__entropy_{mode}")
    title = f"{mode.capitalize()} rollout entropy"
    if has(roll):
        line(ax, R, roll, col, f"rollout/entropy_{mode}", marker=".", lw=1.6)
        panel_title(ax, n, title, "entropy", loc="best")
    elif has(actor):
        line(ax, R, actor, col, f"actor/entropy_{mode} (update-batch)", marker=".", lw=1.6)
        panel_title(ax, n, title, "entropy", loc="best")
        ax.text(0.02, -0.16,
                "update-batch fallback: actor/entropy_* is over KEPT training rows, not "
                "generated rollouts. Log rollout/entropy_* for the true signal.",
                transform=ax.transAxes, fontsize=6.0, color="gray", va="top")
    elif mode == "prover" and DC.is_prover_only(F):
        g_roll = F.arr("rollout__entropy")
        g_actor = F.arr("actor__entropy")
        if has(g_roll):
            line(ax, R, g_roll, col, "rollout/entropy (prover-only ⇒ global)",
                 marker=".", lw=1.6)
            panel_title(ax, n, title, "entropy", loc="best")
            ax.text(0.02, -0.16,
                    "prover-only run: no proposer, so global rollout/entropy IS the "
                    "prover entropy (same population).",
                    transform=ax.transAxes, fontsize=6.0, color="gray", va="top")
        elif has(g_actor):
            line(ax, R, g_actor, col,
                 "actor/entropy (prover-only ⇒ global, update-batch)", marker=".", lw=1.6)
            panel_title(ax, n, title, "entropy", loc="best")
            ax.text(0.02, -0.16,
                    "prover-only run: global actor/entropy IS the prover entropy (one "
                    "mode). Update-batch proxy — log rollout/entropy_prover for the true "
                    "rollout signal.",
                    transform=ax.transAxes, fontsize=6.0, color="gray", va="top")
        else:
            na(ax, n, title,
               f"no rollout/entropy_{mode}, actor/entropy_{mode}, or global entropy")
    else:
        na(ax, n, title,
           f"no rollout/entropy_{mode} or actor/entropy_{mode}\n(see global entropy panel)")


def _panel_prover_rollout_entropy(ax, n, R, F):
    _mode_rollout_entropy(ax, n, R, F, "prover", CO["prover"])


def _panel_proposer_rollout_entropy(ax, n, R, F):
    _mode_rollout_entropy(ax, n, R, F, "proposer", CO["proposer"])


def _panel_global_rollout_entropy(ax, n, R, F):
    """Overall rollout entropy as context for the mode-specific panels. Prefer
    rollout/entropy; fall back to global actor/entropy with a clear label."""
    roll = F.arr("rollout__entropy")
    actor = F.arr("actor__entropy")
    if has(roll):
        line(ax, R, roll, CO["gray"], "rollout/entropy", marker=".", lw=1.4)
        panel_title(ax, n, "Global rollout entropy", "entropy", loc="best")
    elif has(actor):
        line(ax, R, actor, CO["gray"], "actor/entropy (update-batch)", marker=".", lw=1.4)
        panel_title(ax, n, "Global rollout entropy", "entropy", loc="best")
        ax.text(0.02, -0.16,
                "fallback: global actor/entropy over the kept update batch; not "
                "rollout-specific and not split by mode.",
                transform=ax.transAxes, fontsize=6.0, color="gray", va="top")
    else:
        na(ax, n, "Global rollout entropy")


# ---------------------------------------------------------------------------
# Generative-Q (critic) readiness panels. Pinned numbers 41-50; rendered only when the
# run logged q/* keys (q__readiness_mae5 is the sentinel). Invalid signals are
# first-class (free decoding). The gate threshold, FIFO capacity and size-control rule
# are READ FROM THE RUN'S CONFIG (e.g. gate 0.18, FIFO 3,840 and a per-step rho cap of
# 0.5 under the movement cap; gate 0.13, FIFO 1,920 under the LR ladder), so the panels
# never assert a threshold the run did not use. Default values remain only as fallbacks
# for runs whose config predates the keys.
# ---------------------------------------------------------------------------

def _as_num(v):
    """cfg values arrive as strings from the manifest; None/""/"None" mean absent."""
    if v in (None, "", "None"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _q_mask_neg(y):
    """-1 sentinel (incomplete MAE window) -> NaN so the gate line starts clean."""
    if y is None:
        return None
    a = np.asarray(y, dtype=float)
    return np.where(a < 0, np.nan, a)


def _sp_q_panel_specs(R, F):
    """(pinned_num, render_fn) list for the Q mechanism; [] when q/* not logged."""
    if not has(F.arr("q__readiness_mae5")):
        return []
    # TWO thresholds: the GLOBAL gate is what pooled MAE5 is compared against, the
    # PROBLEM gate is what a single problem's fresh error is compared against. Each falls back to
    # the run's single knob, then to 0.18 -- so a run without split thresholds draws exactly one
    # line, and a split run draws both.
    _base = _as_num(F.cfg("sp_q_ready_thresh_default"))
    _base = 0.18 if _base is None else _base
    thresh = _as_num(F.cfg("sp_q_ready_thresh_global_default"))
    thresh = _base if thresh is None else thresh
    thresh_problem = _as_num(F.cfg("sp_q_ready_thresh_problem_default"))
    thresh_problem = _base if thresh_problem is None else thresh_problem
    return [
        (41, lambda ax, n: _panel_q_readiness_gate(ax, n, R, F, thresh, thresh_problem)),
        (42, lambda ax, n: _panel_q_ready_population(ax, n, R, F)),
        (43, lambda ax, n: _panel_q_fifo(ax, n, R, F)),
        (44, lambda ax, n: _panel_q_loss(ax, n, R, F, thresh)),
        (45, lambda ax, n: _panel_q_provenance(ax, n, R, F)),
        (46, lambda ax, n: _panel_q_displacement(ax, n, R, F)),
        (47, lambda ax, n: _panel_q_consumption(ax, n, R, F)),
        (48, lambda ax, n: _panel_q_calibration(ax, n, F)),
        (49, lambda ax, n: _panel_q_reference_bank(ax, n, R, F)),
        (50, lambda ax, n: _panel_q_lr_ladder(ax, n, R, F)),
    ]


def _panel_q_readiness_gate(ax, n, R, F, thresh, thresh_problem=None):
    """Pooled 5-step MAE vs the GLOBAL gate + fresh probe MAE vs the PER-PROBLEM gate.

    The two series are compared against two different thresholds when a run splits them, so
    drawing one line for both would misstate at least one of the gates."""
    mae5 = _q_mask_neg(F.arr("q__readiness_mae5"))
    fresh = F.arr("q__probe_mae_fresh")
    obs = F.arr("q__readiness_obs")
    inv = F.arr("q__readiness_obs_invalid")
    frac_inv = None
    if has(obs) and has(inv):
        o = np.asarray(obs, dtype=float)
        frac_inv = np.where(o > 0, np.asarray(inv, dtype=float) / np.maximum(o, 1e-9), np.nan)
    line(ax, R, mae5, CO["purple"], f"pooled 5-step MAE (global gate < {thresh:g})",
         marker=".", lw=1.5)
    _pl = (f"fresh probe MAE |Q - z| (per-problem gate < {thresh_problem:g})"
           if thresh_problem is not None else "fresh probe MAE |Q - z|")
    line(ax, R, fresh, CO["blue"], _pl, marker=".", lw=1.1)
    line(ax, R, frac_inv, CO["red"], "invalid-probe fraction", marker=".", lw=1.0, ls="--")
    # The audit-lane error is the same quantity measured against GROUND TRUTH:
    # if it drifts above the gate while the probe MAE stays under it, readiness is being
    # granted on self-referential evidence.
    line(ax, R, F.arr("q__audit_cut_mae"), CO["gold"], "audit-lane MAE |Q@cut - reward|",
         marker=".", lw=1.3)
    if has(mae5) or has(fresh):
        ax.axhline(thresh, color="black", lw=0.8, ls=":")
        if thresh_problem is not None and abs(thresh_problem - thresh) > 1e-9:
            # split thresholds: a second line, dashed, so which gate is which is unambiguous
            ax.axhline(thresh_problem, color=CO["blue"], lw=0.8, ls="--")
            _title = (f"Q readiness gate (global {thresh:g} / per-problem {thresh_problem:g})")
        else:
            _title = "Q readiness gate (MAE5 + fresh + invalid)"
        ax.set_ylim(0, 1.05)
        panel_title(ax, n, _title, "MAE / fraction", loc="best")
    else:
        na(ax, n, "Q readiness gate", "no q/readiness_mae5 data")


def _panel_q_ready_population(ax, n, R, F):
    """Monotone ready count, transitions/step, ready sampled, audit lane."""
    ready = F.arr("q__ready_problems")
    trans = F.arr("q__readiness_transitions")
    sampled = F.arr("q__ready_sampled_prefixes")
    audit = F.arr("q__audit_prefixes")
    line(ax, R, ready, CO["green"], "ready problems (monotone)", marker=".", lw=1.5)
    line(ax, R, trans, CO["orange"], "transitions/step", marker=".", lw=1.0, ls="--")
    line(ax, R, sampled, CO["blue"], "ready sampled prefixes", marker=".", lw=1.0)
    line(ax, R, audit, CO["gray"], "audit lane (ceil(/4))", marker=".", lw=1.0, ls=":")
    if has(ready):
        ax.set_ylim(bottom=0)
        panel_title(ax, n, "Q ready population + audit lane", "count", loc="best")
    else:
        na(ax, n, "Q ready population", "no q/ready_problems data")


def _panel_q_fifo(ax, n, R, F):
    """FIFO size vs its CONSTANT capacity (q/capacity; no shrink schedule)."""
    size = F.arr("q__fifo_size")
    cap = F.arr("q__capacity")
    adm = F.arr("q__admitted")
    ev = F.arr("q__evicted")
    below = F.arr("q__target_below_min_valid")
    line(ax, R, size, CO["blue"], "FIFO size", marker=".", lw=1.5)
    line(ax, R, cap, CO["gray"], "capacity C^Q (constant)", marker=".", lw=1.0, ls=":")
    line(ax, R, adm, CO["green"], "admitted/step", marker=".", lw=1.0)
    line(ax, R, ev, CO["orange"], "evicted/step", marker=".", lw=1.0, ls="--")
    line(ax, R, below, CO["red"], "groups < 8 valid", marker=".", lw=1.0, ls="--")
    if has(size):
        ax.set_yscale("log")
        panel_title(ax, n, "Q supervision FIFO (log)", "records", loc="best")
    else:
        na(ax, n, "Q supervision FIFO", "no q/fifo_size data")


def _panel_q_loss(ax, n, R, F, thresh):
    """L_Q + variant parts, with the MAE curves on the shared [0,1]-ish axis."""
    ql = F.arr("q__loss")
    qn = F.arr("q__loss_noref")
    qr = F.arr("q__loss_ref")
    fresh = F.arr("q__probe_mae_fresh")
    mae5 = _q_mask_neg(F.arr("q__readiness_mae5"))
    line(ax, R, ql, CO["red"], "L_Q (mean/record)", marker=".", lw=1.5)
    line(ax, R, qn, CO["orange"], "L_Q noref part", marker=".", lw=1.0, ls="--")
    line(ax, R, qr, CO["teal"], "L_Q ref part", marker=".", lw=1.0, ls="--")
    line(ax, R, fresh, CO["purple"], "fresh probe MAE", marker=".", lw=1.1)
    line(ax, R, mae5, CO["green"], f"MAE5 (gate < {thresh:g})", marker=".", lw=1.1, ls="--")
    if has(ql) or has(fresh):
        ax.axhline(thresh, color="black", lw=0.6, ls=":")
        ax.set_ylim(bottom=0)
        panel_title(ax, n, "Q training loss + target-vs-prediction MAE", "loss / MAE",
                    loc="best")
    else:
        na(ax, n, "Q training loss", "no q/loss data")


def _panel_q_provenance(ax, n, R, F):
    """FIFO provenance mix (%) + packing overflows."""
    any_data = False
    for key, label, color, ls in (
        ("q__fifo_frac_terminal_group", "% terminal_group (FIFO)", CO["green"], "-"),
        ("q__fifo_frac_q_bootstrap", "% q_bootstrap (FIFO)", CO["red"], "-"),
        ("q__fifo_frac_mixed_final_q", "% mixed_final_q (FIFO)", CO["orange"], "--"),
    ):
        y = F.arr(key, 100)
        any_data = any_data or has(y)
        line(ax, R, y, color, label, marker=".", lw=1.2, ls=ls)
    ov = F.arr("q__sample_overflowed")
    line(ax, R, ov, CO["gray"], "packing overflows (count)", marker=".", lw=1.0, ls=":")
    if any_data:
        ax.set_ylim(bottom=0)
        panel_title(ax, n, "Q target provenance mix + overflows", "% / count", loc="best")
    else:
        na(ax, n, "Q provenance mix", "no q/fifo_frac_* data")


def _panel_q_displacement(ax, n, R, F):
    """Applied displacement norms + whichever Q size-control rule the run used.

    TWO regimes, mutually exclusive, and the panel must not assert the other one's line:

    * Movement cap -- Delta_Q captured at theta_0 and applied scaled by
      s_t = min(1, rho_cap/rho_t), rho against ||Delta_PPO_net||. Plots `q__s` and the
      rho_cap line (0.5 by default, but READ from the run's config).
    * LR ladder (SP_Q_LR_LADDER=1) -- the Q step is applied in full
      between the PPO minibatches and priced in the NEXT step's LR. There is no s_t; the
      reference line is ratio_max (1.0), rho is against the SUM of the two PPO step norms,
      and the decision-critical series are the LR and the breach streak.
    """
    ladder = _as_num(F.cfg("sp_q_lr_ladder_default")) or 0
    if ladder:
        cap_line = _as_num(F.cfg("sp_q_lr_ratio_max_default"))
        cap_line = 1.0 if cap_line is None else cap_line
        cap_label = f"ratio_max {cap_line:g}"
        rho_label = "rho = dQ/(dPPO1+dPPO2)"
    else:
        cap_line = _as_num(F.cfg("sp_q_rho_cap_default"))
        cap_line = 0.5 if cap_line is None else cap_line
        cap_label = f"rho cap {cap_line:g}"
        rho_label = "rho = dQ/dPPO_net"
    specs = [
        ("q__delta_q", "||Delta_Q|| (raw)", CO["red"], "-", 1.4),
        ("q__delta_q_applied", "||Delta_Q|| applied", CO["salmon"], "--", 1.1),
        ("q__delta_ppo1", "||Delta_PPO1||", CO["blue"], "-", 1.0),
        ("q__delta_ppo2", "||Delta_PPO2||", CO["teal"], "--", 1.0),
        ("q__delta_ppo_net", "||Delta_PPO_net||", CO["navy"], "-", 1.4),
        ("q__rho", rho_label, CO["purple"], "-", 1.2),
    ]
    # regime-specific series: absent keys simply do not draw, so a run under either regime
    # renders only what it actually logged.
    specs.append(("q__s", "cap scale s_t", CO["gray"], ":", 1.0) if not ladder
                 else ("q__lr_current", "Q lr (ladder)", CO["gray"], ":", 1.2))
    any_data = False
    for key, label, color, ls, lw in specs:
        y = F.arr(key)
        any_data = any_data or has(y)
        line(ax, R, y, color, label, marker=".", lw=lw, ls=ls)
    if any_data:
        ax.set_yscale("log")
        ax.axhline(cap_line, color="black", lw=0.8, ls=":")
        panel_title(ax, n, f"Q vs PPO displacement + {cap_label} (log)",
                    "L2 / ratio", loc="best")
    else:
        na(ax, n, "Q displacement", "no q/delta_* data")


def _panel_q_lr_ladder(ax, n, R, F):
    """Q LR ladder state: the LR itself, the breach streak, and the alert flags.

    These are decision-critical -- a run whose rho sits above ratio_max walks
    the LR to its floor in ~6 steps and then learns Q very slowly, which is invisible on the
    displacement panel alone. Rendered only for ladder runs."""
    lr = F.arr("q__lr_current")
    if not has(lr):
        na(ax, n, "Q LR ladder", "no q/lr_current (not a ladder run)")
        return
    floor = _as_num(F.cfg("sp_q_lr_floor_default"))
    line(ax, R, lr, CO["purple"], "Q lr (this step)", marker=".", lw=1.6)
    if floor is not None:
        ax.axhline(floor, color="black", lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.set_ylabel("Q learning rate")
    ax2 = ax.twinx()
    for key, label, color, ls in (
        ("q__lr_breach_streak", "breach streak", CO["orange"], "-"),
        ("q__lr_reduced", "halved (1/0)", CO["red"], "--"),
        ("q__lr_floor_alert", "FLOOR alert (1/0)", CO["navy"], ":"),
        ("q__loss_non_finite", "non-finite loss", CO["gold"], "--"),
        ("q__grad_non_finite", "non-finite grad", CO["teal"], ":"),
    ):
        y = F.arr(key)
        if has(y):
            line(ax2, R, y, color, label, marker=".", lw=1.0, ls=ls)
    ax2.set_ylabel("streak / flags")
    ax2.set_ylim(bottom=0)
    panel_title(ax, n, f"Q LR ladder (floor {floor:g})" if floor is not None
                else "Q LR ladder", "learning rate (log)", loc="best")


def _panel_q_consumption(ax, n, R, F):
    """Consumed-Q calls, invalid rate (first-class: free decoding), drops,
    records trained, fallback-bank growth."""
    calls = F.arr("q__consumed_calls")
    inv = F.arr("q__consumed_invalid")
    drop = F.arr("q__dropped_rollouts")
    trained = F.arr("q__q_records_trained")
    line(ax, R, calls, CO["green"], "consumed-Q calls", marker=".", lw=1.4)
    line(ax, R, inv, CO["red"], "invalid consumed-Q", marker=".", lw=1.2)
    line(ax, R, drop, CO["orange"], "dropped rollouts", marker=".", lw=1.0, ls="--")
    line(ax, R, trained, CO["blue"], "Q records trained", marker=".", lw=1.0, ls=":")
    bank = F.arr("q__bank_size")
    if has(bank):
        ax2 = ax.twinx()
        ax2.grid(False)
        ax2.plot(R, bank, color=CO["teal"], label="fallback bank size", marker=".", lw=1.1)
        ax2.set_ylabel("bank problems")
        ax2.legend(fontsize=6.0, loc="lower right", framealpha=0.85)
    if has(calls) or has(trained):
        ax.set_ylim(bottom=0)
        panel_title(ax, n, "Q consumption + training rows", "count", loc="upper left")
    else:
        na(ax, n, "Q consumption", "no q/consumed_calls data")


def _panel_q_calibration(ax, n, F):
    """THE calibration plot, audit lane: the Q value that WOULD have been consumed —
    measured at the state where the short lane would have been cut — against the realized
    terminal judge reward of that same audit rollout. That pairing is the ground-truth
    check that Q consumption is not rewarding failures.

    Readiness probes are drawn behind it as faint context, and ONLY those whose target came
    from a real terminal group: a probe scored against a Q-bootstrapped target is Q graded
    by its own output and cannot validate Q. Probes also sit at the prefix state rather
    than the consumption state, so they are context, not the measurement. Invalid
    generations are excluded here — their rate lives on the consumption panel."""
    qc = ((F.d.get("global") or {}).get("q_calibration") or {}).get("by_step") or {}
    audit, probes = [], []
    for s, blk in qc.items():
        try:
            s_i = int(s)
        except (TypeError, ValueError):
            continue
        for q, r in zip(blk.get("audit_q") or [], blk.get("audit_r") or []):
            if q is not None and r is not None:
                audit.append((s_i, float(q), float(r)))
        tags = blk.get("tag") or []
        for j, (p, z) in enumerate(zip(blk.get("pred") or [], blk.get("z") or [])):
            if p is None or z is None:
                continue
            tag = tags[j] if j < len(tags) else None   # older fig_data: provenance unknown
            if tag is not None and tag != "terminal_group":
                continue
            probes.append((s_i, float(p), float(z)))
    if not audit and not probes:
        na(ax, n, "Q calibration (audit lane)", "no q_calibration data in fig_data")
        return
    ax.plot([0, 1], [0, 1], color="black", lw=0.8, ls=":")
    if probes:
        pz = np.asarray([p[2] for p in probes], dtype=float)
        pp = np.asarray([p[1] for p in probes], dtype=float)
        jit = (np.random.RandomState(0).rand(len(probes), 2) - 0.5) * 0.03
        ax.scatter(pz + jit[:, 0], pp + jit[:, 1], c=CO["light_gray"], s=5, alpha=0.35,
                   linewidths=0, label=f"terminal-target probes @prefix (n={len(probes)})")
    if audit:
        steps = np.asarray([a[0] for a in audit], dtype=float)
        qv = np.asarray([a[1] for a in audit], dtype=float)
        rv = np.asarray([a[2] for a in audit], dtype=float)
        jit = (np.random.RandomState(1).rand(len(audit), 2) - 0.5) * 0.03
        sc = ax.scatter(rv + jit[:, 0], qv + jit[:, 1], c=steps, cmap="viridis", s=14,
                        alpha=0.75, linewidths=0, label=f"audit lane @cut (n={len(audit)})")
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.03).set_label("step", fontsize=6.5)
        # per-reward-bin mean Q over the last 5 steps (the "current calibration" curve)
        recent = steps >= (steps.max() - 4)
        if recent.any():
            rb = np.round(rv[recent], 1)
            xs = sorted(set(rb.tolist()))
            ax.plot(xs, [float(qv[recent][rb == g].mean()) for g in xs], color=CO["red"],
                    marker="o", lw=1.4, ms=3.5, label="mean Q | reward (last 5 steps)")
    else:
        ax.text(0.03, 0.96, "no audit-lane pairs yet\n(needs ready problems)", va="top",
                fontsize=6.5, color=CO["gray"], transform=ax.transAxes)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    panel_title(ax, n, "Q calibration: audit lane vs terminal reward", "Q value",
                loc="lower right")
    # after panel_title: it stamps the shared "step" x-label, which is wrong here (this
    # panel's x axis is a reward, and the step is on the colour axis instead)
    ax.set_xlabel("realized terminal reward (audit) / group target z (probes)")


def _panel_aec_k(ax, n, R, F):
    """Adaptive entropy control (MAI-Thinking-1): the clip constant k that relaxes ONLY the PPO
    upper bound to 1 + clip_ratio_high + k. k is computed ONCE on the driver from the global
    actor/entropy and broadcast identically to every worker; integral update
    k <- clip(k + delta * sign(H* - H), 0, k_max), init 0. k>0 only floors entropy at H*
    (k=0 == the un-grafted run), so a flat-zero line means H has stayed at/above target."""
    k = F.arr("actor__aec_k")
    if not has(k):
        na(ax, n, "AEC clip constant k",
           "actor/aec_k not logged (SP_ADAPTIVE_ENTROPY off / pre-AEC steps)")
        return
    line(ax, R, k, CO["prover_dark"], "actor/aec_k", marker=".", lw=1.6)
    kmax = 0.08
    _km = F.cfg("aec_kmax_default")
    if _km not in (None, "", "None"):
        try:
            kmax = float(_km)
        except (TypeError, ValueError):
            pass
    ax.axhline(kmax, ls=":", lw=1.0, color=CO["gray"])
    ax.text(0.99, kmax, f" k_max={kmax:g}", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=6.0, color="gray")
    # y-limits: runs with k >= 0 keep k in [0, kmax] (bottom just under 0). Once k goes negative
    # (SP_AEC_KMIN<0 tightens the upper clip to suppress entropy), rescale symmetrically and
    # draw a solid zero line so the sign of k reads at a glance instead of hugging the frame.
    _kvals = [v for v in k if v is not None and v == v]  # drop None AND NaN (pre-AEC steps)
    _kfloor = min(_kvals) if _kvals else 0.0
    if _kfloor < 0:
        ax.axhline(0.0, lw=0.8, color=CO["gray"], alpha=0.7)
        _pad = 0.08 * (kmax - _kfloor)
        ax.set_ylim(bottom=_kfloor - _pad, top=kmax + _pad)
        ax.text(0.99, _kfloor, f" k_min seen={_kfloor:g}", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=6.0, color="gray")
    else:
        ax.set_ylim(bottom=-0.004)
    panel_title(ax, n, "AEC clip constant k", "k", loc="best")
    ax.text(0.02, -0.16,
            "shifts PPO upper clip to 1+clip_high+k (k<0 tightens, suppressing entropy); "
            "driver-synced integral controller drives global actor/entropy toward H*.",
            transform=ax.transAxes, fontsize=6.0, color="gray", va="top")


def _panel_replay_stream_pass(ax, n, R, F):
    """Replay-prefix run: rollout pass rate (prover_judge_score==1,
    i.e. points>=6) split by batch stream — 64 from-scratch statement rows vs 192 rows
    conditioned on a 30-80% prefix of a stored judged-correct proof. The replay curve should sit
    WELL ABOVE the from-scratch curve (the prefix must help — if the two coincide, the prefix
    is not reaching the engine). Dotted: fraction of zero-reward-variance GRPO groups per stream
    (no-gradient groups; expected higher on replay rows, trending down as the EMA weights adapt).
    On Q-routed steps the replay curve CONTINUES as the audit-reweighted FULL-SUFFIX
    estimate — non-ready rows as observed + audit rows (random ~1/4 of ready) x n_ready/n_audit —
    because the raw replay pass is censoring-biased there (Q-cut rows are structural pjs=0
    fails). One line, not two; pre-Q steps are uncensored raw."""
    po = F.arr("replay__pass_rate_orig", 100)
    pr = F.arr("replay__pass_rate_replay", 100)
    if not has(po) and not has(pr):
        na(ax, n, "replay: pass rate by stream", "replay/* not logged (SP_REPLAY_ENABLE off)")
        return
    # PREFER the dump-derived scratch-stream pass rate over the trainer metric, exactly
    # as the replay line below prefers its full-suffix estimate.
    #
    # replay/pass_rate_orig is computed in the trainer by splitting on sp_prefix_len==0.
    # That was an exact proxy for "statement row" only while every replay row carried a
    # non-empty prefix. With SP_REPLAY_CUT_GRAIN a k=0 cut is a legitimate draw, so ~27%
    # of REPLAY rows land in that split: in one run the line labelled "32 stmt rows" was
    # aggregating 448 rows and reading 25-38% while the true 32 statement rows ran 53-62%.
    #
    # The parser recomputes the same quantity from the rollout dumps every parse, keyed on
    # the inflow lane (parse_fig_data._is_scratch_row), so this repairs the curve for ALL
    # steps -- including those whose trainer-side value was computed with the older
    # sp_prefix_len split and cannot be rewritten. Where the parser key is absent (older caches) the trainer value
    # still shows through, and pre-grain steps agree with it anyway.
    _po_dump = F.arr("v7__train__prover__orig_stream_pass_mean", 100)
    if has(_po_dump):
        po = np.where(np.isfinite(_po_dump), _po_dump, po)
    # Parent-history backfill: parent steps were ALL from-scratch, so the dump-derived binary
    # pass mean IS the orig-stream pass rate there — extend the from-scratch curve back
    # through the inherited history for continuity across the graft boundary.
    _base = F.arr("v7__train__prover__prover_judge_score_mean", 100)
    po = np.where(np.isfinite(po), po, _base)
    # Row count is DERIVED, never hardcoded: the statement-row count is a per-experiment
    # batch-shape choice (e.g. 64 of 256, or 32 of 128 for inflow-only runs) and a stale
    # literal silently mislabels the panel across a parent chain. Prefer the per-step dropped-row
    # count (inflow-only runs: scratch rows dropped pre-loss) else the parsed scratch-stream
    # row count; fall back to an unqualified label when neither is available.
    _n_stmt = None
    for _k in ("sp_inflow__dropped_rows", "v7__train__prover__resp_len_scratch_n"):
        _v = F.arr(_k)
        if has(_v):
            _fin = _v[np.isfinite(_v)]
            if _fin.size:
                _n_stmt = int(_fin[-1])
                break
    _stmt_label = f"from-scratch ({_n_stmt} stmt rows)" if _n_stmt else "from-scratch (stmt rows)"
    line(ax, R, po, CO["prover_dark"], _stmt_label, marker=".", lw=1.6)
    pu = F.arr("v7__train__prover__replay_stream_pass_full_suffix_est", 100)
    # group-weighted dump-derived pass mean (dynamic group size): preferred over the
    # trainer's naive per-row key, which weighs a 32-group four times an 8-group; equal
    # to the naive key on fixed-n runs. Order: audit-lane HT est > group-weighted > naive.
    pg = F.arr("v7__train__prover__replay_stream_pass_mean_gw", 100)
    if has(pg):
        pr = np.where(np.isfinite(pg), pg, pr)
    if has(pu):
        pr = np.where(np.isfinite(pu), pu, pr)
        _pr_label = "prefix-replay (Q steps: full-suffix est, audit ~4x)"
    elif has(pg):
        _pr_label = "prefix-replay (group-weighted)"
    else:
        _pr_label = "prefix-replay (192 rows)"
    line(ax, R, pr, CO["proposer"], _pr_label, marker=".", lw=1.6)
    dg_o = F.arr("replay__degenerate_group_frac_orig", 100)
    dg_r = F.arr("replay__degenerate_group_frac_replay", 100)
    # The stmt zero-var series is meaningless once the scratch lane is inflow-only:
    # those rows carry rollout_n=1, so every "group" has a single member and
    # max-min == 0 BY CONSTRUCTION -> a tautological 100% that reads on the panel like
    # total mode collapse. It is informative only where scratch rows were grouped
    # (rollout_n=16, i.e. parent history before inflow-only training). Mask exactly the inflow-only steps,
    # keeping the earlier history where the number was real.
    #
    # This also hides a contamination artefact: the trainer computes the orig/replay
    # split on sp_prefix_len==0, so once a quantized cut made k=0 a legitimate draw the
    # 16-member replay groups leaked into this series and pulled it off 100% (63-87% in
    # one affected run). sp_replay now splits on the inflow lane, but values
    # already written to metrics.jsonl cannot be rewritten -- masking fixes them too.
    _inflow_only = F.arr("sp_inflow__dropped_rows")
    if has(dg_o) and has(_inflow_only):
        dg_o = np.where(np.isfinite(_inflow_only), np.nan, dg_o)
    if has(dg_o):
        line(ax, R, dg_o, CO["prover_dark"], "zero-var groups % (stmt)", ls=":", lw=1.0)
    if has(dg_r):
        line(ax, R, dg_r, CO["proposer"], "zero-var groups % (replay)", ls=":", lw=1.0)
    ax.set_ylim(-2, 102)
    panel_title(ax, n, "replay: pass rate by stream", "% correct", loc="best")


def _panel_replay_buffer(ax, n, R, F):
    """Replay-buffer state: covered problems + total stored trajectories (capacity 5/problem;
    left axis) and per-step admissions (judged-passing trajectories added). Right axis: mean
    success-EMA over covered problems (prior = bucket occupancy k/5; drives the sampling weight
    w = 3 - 2.5*ema, so a rising mean means the replay draw is concentrating on harder problems)."""
    cov = F.arr("replay__coverage")
    size = F.arr("replay__buffer_size")
    adm = F.arr("replay__admitted")
    if not has(cov):
        na(ax, n, "replay: buffer state", "replay/* not logged (SP_REPLAY_ENABLE off)")
        return
    line(ax, R, cov, CO["prover_dark"], "covered problems", marker=".", lw=1.6)
    line(ax, R, size, CO["proposer"], "stored trajectories", marker=".", lw=1.2)
    if has(adm):
        line(ax, R, adm, CO["gray"], "admitted this step", marker=".", lw=0.9)
    emam = F.arr("replay__ema_mean")
    if has(emam):
        ax2 = ax.twinx()
        ax2.plot(R, emam, color=CO["proposer_accent"], ls="--", lw=1.2,
                 label="mean success-EMA (right)")
        ax2.set_ylim(-0.02, 1.02)
        ax2.set_ylabel("mean EMA", fontsize=7)
        ax2.grid(False)
        panel_title(ax, n, "replay: buffer state", "count", legend=False)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=6.4,
                  framealpha=0.85, handletextpad=0.4)
    else:
        panel_title(ax, n, "replay: buffer state", "count", loc="best")


def _panel_replay_coverage(ax, n, R, F):
    """Replay COVERAGE only (companion to panel 40): number of DISTINCT problems with >=1 stored
    trajectory, on its own autoscaled axis so growth beyond the frozen seed is visible (panel 40's
    shared 'count' axis is dominated by total-trajectory size, ~30x larger, which flattens the
    coverage line). A problem enters coverage the first time a from-scratch (statement) rollout
    solves it, so a rising line = the policy is reaching previously-unsolved problems. Dotted line =
    frozen-seed baseline; right axis = net new problems covered since the seed."""
    cov = F.arr("replay__coverage")
    if not has(cov):
        na(ax, n, "replay: coverage", "replay/* not logged (SP_REPLAY_ENABLE off)")
        return
    line(ax, R, cov, CO["prover_dark"], "covered problems", marker=".", lw=1.8)
    finite = cov[np.isfinite(cov)]
    if len(finite):
        base = float(finite[0])
        ax.axhline(base, color=CO["gray"], ls=":", lw=1.0)
        ax.annotate(f"seed {int(base)}", xy=(R[0], base), fontsize=6.2,
                    color=CO["gray"], va="bottom")
        ax2 = ax.twinx(); ax2.grid(False)
        ax2.plot(R, cov - base, color=CO["proposer_accent"], ls="--", lw=1.0,
                 label="net new since seed (right)")
        ax2.set_ylabel("Δ vs seed", fontsize=7)
        panel_title(ax, n, "replay: coverage", "covered problems", legend=False)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=6.4,
                  framealpha=0.85, handletextpad=0.4)
    else:
        panel_title(ax, n, "replay: coverage", "covered problems", loc="best")


def _panel_opd_distill(ax, n, R, F):
    """OPD auxiliary distillation loss: per-token forward-KL toward the
    colocated frozen Instruct teacher over the thinking region strictly before ``</think>``. This
    is the RAW loss BEFORE the coef rescale -- actor/loss = pg_loss + distillation_loss_coef * this
    (coef from SP_OPD_COEF). Linear y autoscaled to the loss range; x starts at the first step the
    loss is defined; fraction of tokens whose KL hit the safety clip on the right axis."""
    mean = F.arr("actor__distillation__loss")
    if not has(mean):
        na(ax, n, "OPD distillation loss",
           "actor/distillation/loss not logged (SP_OPD_ENABLE off / pre-OPD steps)")
        return
    line(ax, R, mean, CO["prover_dark"], "loss (mean fwd-KL, pre-coef)", marker=".", lw=1.6)
    ax2 = ax.twinx(); ax2.grid(False)
    clip = F.arr("actor__opsd__kl_clipped_frac", 100)
    if has(clip):
        line(ax2, R, clip, CO["red"], "kl clipped %", marker=".", ls="--", lw=1.0)
        ax2.set_ylim(bottom=0)
        ax2.set_ylabel("% tokens clipped", fontsize=7)
        ax2.legend(fontsize=5.6, loc="upper right", framealpha=0.85)
    else:
        ax2.set_yticks([])
    # Start x where the OPD loss is defined (skip the empty pre-OPD range); keep y LINEAR and let
    # it autoscale to the loss range. Freeze x-autoscale so the shared vertical markers drawn later
    # (apply_vertical_markers -> axvline at pre-OPD steps) can't re-expand x back to step 1.
    defined = R[~np.isnan(mean)]
    if defined.size:
        lo, hi = float(np.nanmin(defined)), float(np.nanmax(R))
        pad = max(0.5, 0.02 * (hi - lo))
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_autoscalex_on(False)
    panel_title(ax, n, "OPD distillation loss", "fwd-KL (nats)", loc="upper left")
    ax.text(0.02, -0.16,
            "raw aux loss (pre-coef) over the thinking region before </think> only; "
            "actor/loss = pg_loss + coef*this (coef from SP_OPD_COEF).",
            transform=ax.transAxes, fontsize=6.0, color="gray", va="top")


def _panel_opd_region(ax, n, R, F):
    """OPD thinking-region coverage: which token span the distillation actually trains on.
    Fractions (left): responses containing ``</think>``, thinking share of the response, and the
    boolean-judge pass rate of the distilled batch. Token lengths (right): mean thinking-region
    and full-response lengths."""
    endthink = F.arr("opd__frac_with_endthink")
    tfrac = F.arr("opd__think_frac_of_resp")
    prate = F.arr("opd__pass_rate")
    if not (has(endthink) or has(tfrac) or has(prate)):
        na(ax, n, "OPD thinking-region coverage",
           "opd/* region stats not logged (SP_OPD_ENABLE off / pre-OPD steps)")
        return
    line(ax, R, endthink, CO["green"], "frac with </think>", marker=".", lw=1.6)
    line(ax, R, tfrac, CO["teal"], "think frac of resp", marker=".", lw=1.4)
    line(ax, R, prate, CO["purple"], "pass rate (batch)", marker=".", ls="--", lw=1.2)
    ax.set_ylim(-0.02, 1.02)
    ax2 = ax.twinx(); ax2.grid(False)
    tlen = F.arr("opd__think_len_mean")
    rlen = F.arr("opd__resp_len_mean")
    drew_len = False
    if has(tlen):
        line(ax2, R, tlen, CO["navy"], "think len (mean)", marker=".", ls=":", lw=1.0)
        drew_len = True
    if has(rlen):
        line(ax2, R, rlen, CO["gray"], "resp len (mean)", marker=".", ls=":", lw=1.0)
        drew_len = True
    if drew_len:
        ax2.set_ylim(bottom=0)
        ax2.set_ylabel("tokens", fontsize=7)
        ax2.legend(fontsize=5.6, loc="lower right", framealpha=0.85)
    else:
        ax2.set_yticks([])
    panel_title(ax, n, "OPD thinking-region coverage", "fraction", loc="center left")
    ax.text(0.02, -0.16,
            "distillation trains only tokens strictly before </think>; responses without the tag "
            "contribute their full length. pass_rate = boolean judge mean over the same batch.",
            transform=ax.transAxes, fontsize=6.0, color="gray", va="top")


# ---------------------------------------------------------------------------
# Difficulty-sampling raw page (page 3). When a run trains with SP_DIFF_SAMPLING=1,
# verl logs the SNIS-corrected (uniform-comparable) values under the STANDARD metric
# keys -- so every main-page panel above is already corrected -- and preserves the
# sampled-distribution values under difficulty_raw/<key>. This page shows those raw
# series overlaid on the corrected ones (bias made visible), plus sampler health.
# Self-gated: has_difficulty_raw(F) is False for uniform runs and the page is skipped.
# ---------------------------------------------------------------------------

# audited keys (IS-corrected metrics): standard key -> short panel title
_DIFF_RAW_KEYS = [
    ("critic/score/mean", "score mean"),
    ("critic/rewards/mean", "reward mean"),
    ("critic/advantages/mean", "advantages mean"),
    ("critic/returns/mean", "returns mean"),
    ("response_length/mean", "response len mean"),
    ("response_length/clip_ratio", "response len clip ratio"),
    ("response_length_non_aborted/mean", "response len mean (non-aborted)"),
    ("response/aborted_ratio", "aborted ratio"),
    ("prompt_length/mean", "prompt len mean"),
    ("num_turns/mean", "num turns mean"),
]


def _sk(key: str) -> str:
    return key.replace("/", "__")


def has_difficulty_raw(F: "DC.FigData") -> bool:
    return any(has(F.arr("difficulty_raw__" + _sk(k))) for k, _ in _DIFF_RAW_KEYS)


def _panel_raw_vs_corrected(ax, n, R, F, key, title):
    raw = F.arr("difficulty_raw__" + _sk(key))
    cor = F.arr(_sk(key))
    if not has(raw):
        na(ax, n, title, f"difficulty_raw/{key} not logged")
        return
    line(ax, R, cor, CO["prover_dark"], f"{key} (SNIS-corrected = main)", marker=".", lw=1.6)
    line(ax, R, raw, CO["gray"], "raw (sampled distribution)", marker=".", lw=1.2, ls="--")
    panel_title(ax, n, title, "value", loc="best")
    ax.text(0.02, -0.16,
            "solid = corrected (what page 1 and cross-run comparisons use); dashed = the biased "
            "batch value under difficulty-weighted sampling.",
            transform=ax.transAxes, fontsize=6.0, color="gray", va="top")


def _panel_diff_sampler_health(ax, n, R, F):
    ess = F.arr("difficulty__ess_frac")
    if not has(ess):
        na(ax, n, "difficulty sampler health")
        return
    line(ax, R, ess, CO["prover_dark"], "ESS fraction (batch)", marker=".", lw=1.6)
    fl = F.arr("difficulty__floored_frac")
    if has(fl):
        line(ax, R, fl, CO["gray"], "floored/demoted problem frac", marker=".", lw=1.2)
    ax.set_ylim(0.0, 1.05)
    panel_title(ax, n, "difficulty sampler health", "fraction", loc="best")


def _panel_diff_c_stats(ax, n, R, F):
    cm = F.arr("difficulty__c_mean")
    if not has(cm):
        na(ax, n, "IS correction c")
        return
    line(ax, R, cm, CO["prover_dark"], "c mean (E_q[c]=1 check)", marker=".", lw=1.6)
    cx = F.arr("difficulty__c_max")
    if has(cx):
        line(ax, R, cx, CO["gray"], "c max (floor bounds it)", marker=".", lw=1.2)
    ax.axhline(1.0, ls=":", lw=1.0, color=CO["gray"])
    panel_title(ax, n, "IS correction c", "c", loc="best")


def _panel_passcount_variant(ax, n, R, F, rollout_n, *, raw):
    """Prover pass-count distribution, either the RAW actual-batch histogram (`*_raw`) or the
    SNIS group-weighted one (the main key, mirroring page-1 panel 20). Same stacked rendering.
    The raw view is where difficulty sampling's compute-reallocation is visible: as easy problems
    get down-sampled, the actual batch has FEWER saturated (n/n) groups than the corrected
    (uniform-population) estimate."""
    suffix = "_raw" if raw else ""
    cmap = DC.passcount_colormap(rollout_n + 1)
    specs = []
    for k in range(rollout_n + 1):
        y = F.arr(f"v7__train__prover__pass_count_hist__k{k}{suffix}")
        specs.append((y, f"{k}/{rollout_n}", cmap(k / max(1, rollout_n))))
    label = "RAW / actual batch" if raw else "SNIS-corrected (= page 1)"
    if stacked(ax, R, specs, all_in_legend=True):
        panel_title(ax, n, f"Prover pass-count dist. [{label}]", "prompt groups",
                    loc="upper left", ncol=3)
        ax.text(0.02, -0.22,
                ("actual generated batch: difficulty sampling draws easy (n/n) groups less, so the "
                 "top band shrinks vs the corrected panel — this is the compute-reallocation win."
                 if raw else
                 "uniform-population estimate (group-weighted by draw-time c); comparable to a "
                 "uniform run. The raw actual-batch view is the adjacent panel."),
                transform=ax.transAxes, fontsize=6.0, color="gray", va="top")
    else:
        na(ax, n, f"Prover pass-count dist. [{label}]",
           "no difficulty pass-count data" if raw else "requires explicit uid")


def build_raw_page(F: "DC.FigData") -> "plt.Figure":
    """Page 3: raw (uncorrected) metric series + difficulty-sampler health + raw/corrected
    pass-count distributions (so the compute-reallocation is directly visible)."""
    DC.set_lineage_dash_boundary(_lineage_dash_step(F))
    try:
        return _build_raw_page(F)
    finally:
        DC.set_lineage_dash_boundary(None)


def _build_raw_page(F: "DC.FigData") -> "plt.Figure":
    R = F.steps
    rollout_n = int(F.cfg("train_rollout_n") or F.cfg("rollout_n") or 16)
    panels = [(k, t) for k, t in _DIFF_RAW_KEYS]
    n_active = len(panels) + 4  # + sampler health + c stats + raw passcount + corrected passcount
    ncols = 4
    nrows = (n_active + ncols - 1) // ncols
    fig = plt.figure(figsize=(24, 5.25 * nrows))
    top = DC.draw_header(fig, F, "difficulty sampling -- RAW (uncorrected) series; page-1 uses "
                                 "the SNIS-corrected values")
    gs = GridSpec(nrows, ncols, figure=fig, hspace=0.58, wspace=0.27, top=top)
    A = [fig.add_subplot(gs[i // ncols, i % ncols]) for i in range(n_active)]
    for slot, (key, title) in enumerate(panels):
        _panel_raw_vs_corrected(A[slot], f"R{slot + 1}", R, F, key, title)
    b = len(panels)
    _panel_diff_sampler_health(A[b], f"R{b + 1}", R, F)
    _panel_diff_c_stats(A[b + 1], f"R{b + 2}", R, F)
    _panel_passcount_variant(A[b + 2], f"R{b + 3}", R, F, rollout_n, raw=True)
    _panel_passcount_variant(A[b + 3], f"R{b + 4}", R, F, rollout_n, raw=False)
    fig.text(0.5, 0.005,
             "Dump-derived expectation panels (train correctness / pass-count, panels 1/2/20/21) are "
             "SNIS-corrected by the parser too; their raw counterparts "
             "live in fig_data under *_raw keys. Counts / percentiles / judge diagnostics remain "
             "actual-batch values by design.",
             ha="center", fontsize=8, color="dimgray", style="italic")
    fig.subplots_adjust(left=0.045, right=0.975, bottom=0.03)
    return fig


def _save(fig, out: str) -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


def main() -> None:
    fig_data = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("FIG_DATA", "fig_data.json")
    title = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("TITLE", "training run")
    out = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("OUT", "v7_dashboard_main.png")

    d = DC.load_fig_data(fig_data)
    F = DC.FigData(d, fig_data)
    main_fig = build_main_figure(F)

    if out.lower().endswith(".pdf"):
        import diagnostics_dashboard as DG
        from matplotlib.backends.backend_pdf import PdfPages

        diag_fig = DG.build_diag_figure(F)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(out) as pdf:
            pdf.savefig(main_fig)
            pdf.savefig(diag_fig)
        print(f"wrote {out} (2 pages)")
        plt.close(main_fig)
        plt.close(diag_fig)
    else:
        _save(main_fig, out)
        plt.close(main_fig)


if __name__ == "__main__":
    main()
