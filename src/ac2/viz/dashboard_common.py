#!/usr/bin/env python3
"""Shared rendering helpers for the training dashboards.

Both ``single_run_dashboard.py`` (main page) and ``diagnostics_dashboard.py``
build a matplotlib figure from a ``fig_data.json`` produced by
``parse_fig_data.py`` (the stdlib-only parser). This module holds
the small generic helpers (palette, ``arr()`` accessor, line/stack/EMA/
percentile drawing, headers) so the two page scripts stay focused on panel
content.

The fig_data format supports a two-mode (proposer + prover) setup; prover-only
runs drop the proposer-only panels. The palette deliberately
groups blues for proposer ("imagining a proposition") and greens for prover
("direct-proof correctness") so a reader can scan any panel and see which
mode it speaks to. Mixed/global metrics use neutral grays.

This module is rendering-only (matplotlib is imported), so it is never
imported by the stdlib-only parser. It is also intentionally **standalone**:
nothing here imports from any other viz package.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402


# Schema name the parser writes into fig_data.json (the dashboard fig_data
# format). The renderer refuses fig_data with any other schema (check below) so
# a stale path can't silently render the wrong dashboard.
SCHEMA = "riemann_v7_fig_data"
SCHEMA_VERSION = 1


# Palette. Two mode families + neutrals. Shared metrics (grad norm,
# throughput, judge) use the standard matplotlib tab10 colors. The
# proposer/prover families are picked so colorblind readers can still tell
# them apart by luminance order (proposer = lighter blue, prover = darker
# green).
CO = {
    # generic / shared
    "blue": "#1f77b4",
    "red": "#d62728",
    "green": "#2ca02c",
    "orange": "#ff7f0e",
    "purple": "#9467bd",
    "teal": "#17becf",
    "brown": "#8c564b",
    "pink": "#e377c2",
    "olive": "#7a9f22",
    "gray": "#7f7f7f",
    "gold": "#d59f00",
    "navy": "#393b79",
    "mint": "#8bd17c",
    "salmon": "#f28e8b",
    "light_gray": "#d0d0d0",
    # proposer family (blue): main, mean, best, ema, accent
    "proposer": "#1f77b4",
    "proposer_dark": "#10406b",
    "proposer_light": "#7eb1d4",
    "proposer_accent": "#3a8acc",
    # prover family (green): same role layering as proposer
    "prover": "#2ca02c",
    "prover_dark": "#155e15",
    "prover_light": "#8bd17c",
    "prover_accent": "#46c14a",
    # Replay family split: one proposer color, one prover_original color, one
    # prover_additional color. prover_original keeps the existing green family (so baseline-vs-replay rendering of
    # original-statement panels matches). prover_additional uses a violet/
    # purple family that's easy to tell apart from green at a glance and
    # stays distinguishable on monochrome / colorblind viewers.
    "prover_original": "#2ca02c",
    "prover_original_dark": "#155e15",
    "prover_original_light": "#8bd17c",
    "prover_original_accent": "#46c14a",
    "prover_additional": "#8a3aac",
    "prover_additional_dark": "#4a1f6a",
    "prover_additional_light": "#c39ce0",
    "prover_additional_accent": "#a85fcf",
    # impact rubric: 0 (no help) -> 4 (near-equivalent to seed), cool->warm.
    # impact4 ("almost equivalent to the seed problem") sits above impact3 in a
    # darker crimson so the stacked plots read 4 as the most-saturated warm bin.
    "impact0": "#393b79",
    "impact1": "#1f77b4",
    "impact2": "#ff7f0e",
    "impact3": "#d62728",
    "impact4": "#7a0e1a",
}

EMA_ALPHA = 0.8


# Dashed-history boundary for continuation runs. When a run continues a parent under
# a DIFFERENT reward regime, the inherited history (steps <= boundary) is drawn
# DASHED and the current run's own steps (>= boundary) stay SOLID, so a reader
# can see at a glance where the reward changed. The boundary is a step number
# (the last parent's ``end_step``) or None (root run / no dashing). It is a
# module-level switch set by the page builders around plotting and reset to None
# in a ``finally`` so it never leaks between renders.
_LINEAGE_DASH_BEFORE: float | None = None


def set_lineage_dash_boundary(step_or_none) -> None:
    """Set (or clear, with None) the dashed-history boundary step used by
    ``line()`` and ``single_run_dashboard._vline()``. Idempotent; pass None to
    restore the unmodified rendering behavior."""
    global _LINEAGE_DASH_BEFORE
    if step_or_none is None:
        _LINEAGE_DASH_BEFORE = None
        return
    try:
        _LINEAGE_DASH_BEFORE = float(step_or_none)
    except (TypeError, ValueError):
        _LINEAGE_DASH_BEFORE = None


def get_lineage_dash_boundary() -> float | None:
    """Return the current dashed-history boundary step, or None."""
    return _LINEAGE_DASH_BEFORE


def load_fig_data(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class FigData:
    """Thin accessor around a parsed fig_data dict.

    Centralizes the per_step ``arr()`` helper so panels read uniformly:
    ``F.arr("v7__train__prover__score_mean_generated")`` returns
    a NaN-padded float array aligned to ``F.steps``.
    """

    def __init__(self, d: dict, fig_path: str | None = None) -> None:
        schema = d.get("schema")
        if schema and schema != SCHEMA:
            # Loud, immediate failure if someone points the renderer at a
            # fig_data with a different schema. The renderer would otherwise silently miss
            # most keys and produce a sea of N/A panels.
            raise ValueError(
                f"fig_data schema {schema!r} does not match v7 renderer "
                f"({SCHEMA!r}). Re-parse with viz/v7/parse_fig_data.py."
            )
        self.d = d
        self.per_step = d.get("per_step") if isinstance(d.get("per_step"), dict) else {}
        self.config = d.get("config") if isinstance(d.get("config"), dict) else {}
        self.manifest = d.get("manifest") if isinstance(d.get("manifest"), dict) else {}
        self.glob = d.get("global") if isinstance(d.get("global"), dict) else {}
        self.train_jsonl = d.get("train_jsonl") if isinstance(d.get("train_jsonl"), dict) else {}
        self.val_jsonl = d.get("val_jsonl") if isinstance(d.get("val_jsonl"), dict) else {}
        self.dashboard_annotations = (
            d.get("dashboard_annotations")
            if isinstance(d.get("dashboard_annotations"), dict)
            else {}
        )
        self.warnings = d.get("warnings") if isinstance(d.get("warnings"), list) else []
        self.run_id = str(d.get("run_id") or (Path(fig_path).stem if fig_path else "v7-run"))
        self.steps = np.array(self.per_step.get("steps", []), dtype=float)

    # -- per_step accessors --------------------------------------------------
    def arr(self, key: str, scale: float = 1.0) -> np.ndarray:
        values = self.per_step.get(key)
        if not isinstance(values, list) or len(values) != len(self.steps):
            return np.full(len(self.steps), np.nan)
        return np.array([_to_float(v) * scale for v in values], dtype=float)

    def arr_first(self, keys: list[str], scale: float = 1.0) -> np.ndarray:
        for key in keys:
            y = self.arr(key, scale)
            if has(y):
                return y
        return np.full(len(self.steps), np.nan)

    def cfg(self, key: str, default: Any = None) -> Any:
        v = self.config.get(key)
        return default if v is None else v


def is_prover_only(F: "FigData") -> bool:
    """Detect a prover-only training run from ``fig_data``.

    A run is prover-only iff every parsed step generated 0 proposer rows. We
    check the per-step mode-mix counter the parser already emits
    (``v7__train__mode_mix__proposer_generated``); a single non-zero step means
    the run is mixed and we render the full proposer/prover layout. Falls back
    to checking ``mode_mix__proposer_generated_fraction`` when the absolute
    count is absent (older fig_data). Returns False if neither key is present
    (insufficient evidence — render the full layout, never silently hide
    panels).
    """
    arr = F.arr("v7__train__mode_mix__proposer_generated")
    if has(arr):
        return float(np.nanmax(arr)) == 0.0
    frac = F.arr("v7__train__mode_mix__proposer_generated_fraction")
    if has(frac):
        return float(np.nanmax(frac)) < 1e-6
    return False


def _to_float(x) -> float:
    try:
        if x is None:
            return math.nan
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def has(y: np.ndarray) -> bool:
    return bool(y.size and np.any(~np.isnan(y)))


def present(y: np.ndarray) -> np.ndarray:
    return np.nan_to_num(y, nan=0.0)


def ema(y: np.ndarray, alpha: float = EMA_ALPHA) -> np.ndarray:
    out = np.full(y.shape, np.nan)
    prev = math.nan
    for i, x in enumerate(y):
        if math.isnan(x):
            out[i] = prev
            continue
        prev = x if math.isnan(prev) else alpha * prev + (1.0 - alpha) * x
        out[i] = prev
    return out


def line(ax, x, y: np.ndarray, color: str, label: str, **kwargs) -> None:
    if not has(y):
        return
    b = _LINEAGE_DASH_BEFORE
    xa = np.asarray(x, dtype=float)
    # Only split when a boundary is set AND the x array actually straddles it
    # (some point <= boundary and some point strictly > boundary). Otherwise the
    # behavior is byte-identical to the historical single ax.plot.
    if b is None or not (xa.size and np.any(xa <= b) and np.any(xa > b)):
        ax.plot(x, y, color=color, label=label, **kwargs)
        return
    ya = np.asarray(y, dtype=float)
    before = xa <= b            # inherited (different-reward) history -> dashed
    after = xa >= b             # current run -> solid; boundary point in BOTH so
    #                             the dashed and solid segments visually connect.
    dash_kwargs = dict(kwargs)
    dash_kwargs.pop("ls", None)  # drop caller's ls= alias so it can't clash with linestyle
    dash_kwargs["linestyle"] = "--"
    dash_kwargs["alpha"] = 0.55
    # Label only on the solid segment to avoid a duplicate legend entry.
    ax.plot(xa[before], ya[before], color=color, label="_nolegend_", **dash_kwargs)
    ax.plot(xa[after], ya[after], color=color, label=label, **kwargs)


def stacked(ax, x, specs, width: float = 0.85, all_in_legend: bool = False,
            y_max: float | None = None) -> bool:
    """Stacked bar plot. ``specs`` is a sequence of ``(y, label, color)``.

    ``y_max`` — when given, lock the y-axis to ``[0, y_max]`` (and call
    ``set_autoscaley_on(False)`` so a later ``set_ylim(bottom=0)`` can't
    silently expand it). Use this for percentage-stack panels (e.g. panel 9
    judge health) so a sparse run with only a few small unhealthy buckets
    doesn't get visually inflated by matplotlib's auto-scaling — the reader
    always sees the bars against the same 0..100 ceiling. Count-based stacks
    (e.g. group-composition panels) leave it None and keep their auto-scaled
    range."""
    bottom = np.zeros(len(x))
    drew = False
    for y, label, color in specs:
        yy = present(y)
        if np.any(yy != 0):
            ax.bar(x, yy, bottom=bottom, label=label, color=color, width=width)
            bottom += yy
            drew = True
        elif all_in_legend:
            ax.bar(x, np.zeros(len(x)), bottom=bottom, label=label, color=color, width=width)
    if y_max is not None:
        ax.set_ylim(0, y_max)
        ax.set_autoscaley_on(False)
    return drew


def na(ax, n: int, name: str, msg: str = "not available") -> None:
    ax.text(
        0.5, 0.5, msg, ha="center", va="center", color="gray",
        transform=ax.transAxes, fontsize=8, wrap=True,
    )
    ax.set_title(f"({n}) {name}", fontweight="bold", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def panel_title(
    ax, n: int, name: str, ylabel: str = "", legend: bool = True,
    loc: str = "best", ncol: int = 1,
) -> None:
    ax.set_title(f"({n}) {name}", fontweight="bold", fontsize=10)
    ax.set_xlabel("step")
    if ylabel:
        ax.set_ylabel(ylabel)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(
            fontsize=6.3, framealpha=0.85, loc=loc, ncol=ncol,
            columnspacing=0.9, handletextpad=0.4,
        )


def _short_run_id(run_id: str, *, keep: int = 18) -> str:
    """Compress a verbose run id for inline annotation use. Strips a known
    ``fineproofs-...`` run-id prefix when present and truncates to ``keep`` chars."""
    s = (run_id or "").strip()
    for prefix in ("fineproofs-v7-", "fineproofs-"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    if len(s) > keep:
        s = s[: keep - 1] + "…"
    return s


def lineage_boundaries(F: "FigData") -> list[dict[str, Any]]:
    """Parent-chain boundaries: every transition between consecutive parents (and from
    the last parent to the current run) gets one vertical boundary, placed
    BETWEEN ``parents[i].end_step`` and the next segment's start step
    (i.e. at ``end_step + 0.5``). Returns a list of marker dicts compatible
    with ``apply_vertical_markers``: ``[{x, label, color, linestyle,
    linewidth, alpha}, ...]``. Empty for root runs (no parents).

    The parent chain is required to be contiguous (see
    ``merge_fig_data._validate_parents_contiguity``), so each transition is a
    single well-defined point. The label names the parent that just ENDED."""
    parents = F.d.get("parents") if hasattr(F, "d") else None
    if not isinstance(parents, list):
        return []
    out: list[dict[str, Any]] = []
    for p in parents:
        if not isinstance(p, dict):
            continue
        try:
            x = float(p.get("end_step")) + 0.5
        except (TypeError, ValueError):
            continue
        # Optional display override: a parent entry may carry an explicit ``label``
        # (e.g. name the REGIME the boundary starts — "replay buffer on" — instead of
        # the parent's run id). Falls back to the shortened run id.
        label = str(p.get("label") or "").strip() or _short_run_id(str(p.get("run_id") or ""))
        out.append({
            "x": x,
            "label": label or "parent",
            "color": "#444",
            # apply_vertical_markers `str()`-coerces the value (so JSON-loaded
            # markers stay safe); use a plain matplotlib string linestyle.
            "linestyle": "--",
            "linewidth": 1.0,
            "alpha": 0.55,
        })
    return out


def apply_vertical_markers(axes, F: "FigData", *, label_first_n: int = 4) -> None:
    markers: list[dict[str, Any]] = []
    explicit = F.dashboard_annotations.get("vertical_markers")
    if isinstance(explicit, list):
        markers.extend(m for m in explicit if isinstance(m, dict))
    # Parent-chain boundaries always join the explicit markers.
    markers.extend(lineage_boundaries(F))
    if not markers or not len(F.steps):
        return
    step_min = float(np.nanmin(F.steps))
    step_max = float(np.nanmax(F.steps))
    labeled = 0
    for ax in axes:
        if not ax.has_data():
            continue
        for marker in markers:
            if not isinstance(marker, dict):
                continue
            try:
                x = float(marker.get("x"))
            except (TypeError, ValueError):
                continue
            if x < step_min - 1.0 or x > step_max + 1.0:
                continue
            color = str(marker.get("color") or "#b03060")
            linestyle = str(marker.get("linestyle") or ":")
            try:
                linewidth = float(marker.get("linewidth", 1.1))
            except (TypeError, ValueError):
                linewidth = 1.1
            try:
                alpha = float(marker.get("alpha", 0.82))
            except (TypeError, ValueError):
                alpha = 0.82
            ax.axvline(
                x,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha,
                zorder=1.2,
            )
            label = str(marker.get("label") or "")
            # `label_all` labels the marker on EVERY panel (else only the first label_first_n).
            if label and (marker.get("label_all") or labeled < label_first_n):
                ax.text(
                    x,
                    0.98,
                    label,
                    rotation=90,
                    va="top",
                    ha="right",
                    color=color,
                    fontsize=6.4,
                    transform=ax.get_xaxis_transform(),
                    bbox={
                        "boxstyle": "round,pad=0.12",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.72,
                    },
                    zorder=4.0,
                )
        labeled += 1


_DEFAULT_PERCENTILE_STYLES = {
    50: {"ls": ":", "lw": 1.2, "marker": "."},
    90: {"ls": "--", "lw": 1.4, "marker": "o", "ms": 3.0},
    99: {"ls": "-", "lw": 2.0, "marker": "^", "ms": 3.2},
}


# High-contrast palette for the per-percentile color mode in `percentile_family`.
# Hand-picked over a sequential colormap (viridis) so adjacent percentiles are
# always easy to distinguish at a glance — viridis's purple→teal transition was
# too smooth for the body of a distribution. Indexed by ordinal position in the
# percentile list, NOT by the percentile value itself, so any percentile subset
# (3, 5, 6 entries) walks the same cool→warm→purple ramp.
_HIGH_CONTRAST_PERCENTILE_COLORS = (
    "#1f77b4",  # blue       — lowest percentile (e.g. p50, the body's median)
    "#17becf",  # cyan
    "#2ca02c",  # green
    "#ff7f0e",  # orange
    "#d62728",  # red
    "#9467bd",  # purple     — highest percentile (e.g. p99, the extreme tail)
)


def percentile_family(
    ax, x, F: "FigData", prefix: str, label: str, color: str,
    *,
    percentiles: tuple[int, ...] | list[int] = (50, 90, 99),
    color_per_percentile: bool = False,
) -> bool:
    """Plot percentile curves for a ``<prefix>_p{q}`` family.

    Three style modes:

    - **Default** (`percentiles=(50, 90, 99)`, `color_per_percentile=False`):
      historical look — one ``color`` for all 3 curves with rank-distinct
      ls/lw/marker via ``_DEFAULT_PERCENTILE_STYLES``. Byte-identical to the
      pre-extension behavior; preserves all panels that didn't opt in.

    - **Extended gradient** (any non-default percentile set,
      `color_per_percentile=False`): all curves on one ``color`` with a smooth
      rank-driven alpha+linewidth+linestyle gradient. Keeps 6 lines legible
      without a style-explosion legend.

    - **Per-percentile color** (`color_per_percentile=True`): ``color`` is
      IGNORED; each percentile gets its own hue from a hand-picked
      high-contrast palette (``_HIGH_CONTRAST_PERCENTILE_COLORS``): blue,
      cyan, green, orange, red, purple. The mapping is by ORDINAL POSITION
      in the percentile list (rank 0 → blue, last rank → purple), so any
      percentile subset (3, 5, 6 entries) still walks a deterministic
      cool-to-warm-to-purple ramp with maximum adjacent contrast. Uniform
      ls/lw — the COLOR alone carries the percentile. Use when the panel
      shows ONE percentile family (e.g. panel 5 prover resp_len) so the
      6-color palette doesn't collide with another family.

    Missing ``<prefix>_p{q}`` keys are skipped cleanly (older fig_data stays
    renderable). Markers are point-style throughout so dense step axes don't
    get visually noisy."""
    pct_list = list(percentiles)
    use_default = tuple(pct_list) == (50, 90, 99) and not color_per_percentile
    drew = False
    n = len(pct_list)
    for i, q in enumerate(pct_list):
        y = F.arr(f"{prefix}_p{q}")
        if not has(y):
            continue
        rank = i / max(1, n - 1)
        if color_per_percentile:
            # Hand-picked high-contrast palette indexed by rank position.
            # For n <= len(palette) the lines walk a saturated cool->warm
            # ramp; for n > len(palette) we just cycle (rare here).
            line_color = _HIGH_CONTRAST_PERCENTILE_COLORS[i % len(_HIGH_CONTRAST_PERCENTILE_COLORS)]
            ax.plot(x, y, color=line_color, label=f"{label} p{q}",
                    ls="-", lw=1.5, marker=".", ms=3.0, alpha=0.95)
        elif use_default:
            style = _DEFAULT_PERCENTILE_STYLES[q]
            ax.plot(x, y, color=color, label=f"{label} p{q}", **style)
        else:
            lw = 0.9 + 1.1 * rank          # 0.9 → 2.0
            alpha = 0.45 + 0.55 * rank     # 0.45 → 1.0
            ls = ":" if rank < 0.5 else ("--" if rank < 0.95 else "-")
            marker = ("^" if rank > 0.95
                      else ("o" if rank > 0.85 else "."))
            ms = 3.2 if rank > 0.95 else 2.6
            ax.plot(x, y, color=color, label=f"{label} p{q}",
                    ls=ls, lw=lw, alpha=alpha, marker=marker, ms=ms)
        drew = True
    return drew


def passcount_colormap(_n_bins: int):
    """Fixed pass-count palette: low-success
    **red/orange** through **neutral gray** to high-success **green**.

    The pass-count panels color bin ``k`` via ``cmap(k / n)``, so ``0/n`` groups
    (the actor failing) are red, ``n/n`` saturated groups are green, and the
    neutral-advantage middle is gray.
    """
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "passcount_rgg",
        [
            (0.0, "#c62828"),   # 0/n — low success: red
            (0.2, "#f57c00"),   # orange
            (0.5, "#b0b0b0"),   # neutral gray (mid)
            (1.0, "#2e7d32"),   # n/n — high success: green
        ],
    )


def impact_color(level: int) -> str:
    """Color for impact rubric level 0..4 (cool -> warm), clamped to 0..4
    (level 4 = the impact judge's "Impact: 4" bucket)."""
    return CO[f"impact{max(0, min(4, int(level)))}"]


def header_lines(F: "FigData", page_label: str) -> tuple[str, list[str]]:
    """Return (suptitle, [info lines]) for a page header."""
    cfg = F.config
    steps = F.steps
    span = f"{int(steps[0])}-{int(steps[-1])}" if steps.size else "no steps"
    suptitle = f"{page_label} | {F.run_id}"

    def g(key, default="?"):
        v = cfg.get(key)
        return default if v in (None, "") else v

    actor = str(g("actor_model")).split("/")[-1]
    judge = str(g("judge_model")).split("/")[-1]
    rollout_n = g("rollout_n", "?")
    val_n = g("val_n_default", "?")
    train_n = int(F.train_jsonl.get("n", 0) or 0)
    nnodes = g("nnodes")
    lr = g("actor_lr")
    klc = g("actor_kl_loss_coef")

    # Surface both the user-facing PROBLEM batch sizes and the
    # raw prompt-row batch sizes verl sees (which are 2x for the duplicated
    # rows). This makes it obvious at a glance that 64 problems => 128 rows.
    train_problem = g("train_problem_batch_size", "?")
    gen_problem = g("gen_problem_batch_size", "?")
    train_rows = g("train_batch_size")
    gen_rows = g("gen_batch_size")

    line1 = (
        f"{len(steps)} steps ({span}) | {train_n} train rows | "
        f"actor {actor} | judge {judge} | n={rollout_n} (val n={val_n}) | "
        f"{nnodes} node(s)"
    )
    # Overlong RESPONSE penalty: only shown when actually configured; a missing
    # buffer length means no penalty is applied (not a default 10000x0.9).
    _olb = cfg.get("overlong_buffer_len_default")
    _olp = cfg.get("overlong_penalty_factor_default")
    overlong = (
        f"overlong buf {_olb}x{_olp}"
        if _olb not in (None, "", "None")
        else "overlong off"
    )
    line2 = (
        f"problem batch t/g {train_problem}/{gen_problem} "
        f"(rows {train_rows}/{gen_rows}, mini {g('ppo_mini_batch_size')}) | "
        f"LR {lr} | KL {klc} | reward workers {g('reward_num_workers')} | "
        f"resp cap {g('max_response_length_default')} | "
        f"judge max tok {g('judge_max_tokens_default')} | "
        f"{overlong}"
    )
    wandb = cfg.get("wandb_url")
    heads = f"verl {str(g('verl_head'))[:10]} | code {str(g('riemann_head'))[:10]}"
    line3 = (
        f"exp {g('experiment')} | started {g('started_at')} | {heads} | "
        f"W&B {wandb if wandb else 'N/A'}"
    )
    return suptitle, [line1, line2, line3]


def draw_header(fig, F: "FigData", page_label: str) -> float:
    """Draw the page header and return the GridSpec ``top`` coordinate."""
    suptitle, lines = header_lines(F, page_label)
    fig.suptitle(suptitle, fontsize=15, fontweight="bold", y=0.997)
    step = 0.011
    for i, txt in enumerate(lines):
        fig.text(0.5, 0.982 - i * step, txt, ha="center", fontsize=8.5, style="italic")
    return max(0.86, 0.982 - len(lines) * step - 0.010)
