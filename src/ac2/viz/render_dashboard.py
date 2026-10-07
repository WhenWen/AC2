#!/usr/bin/env python3
"""Render the training dashboard from the canonical fig_data file.

This is the single renderer entry point: it reads the experiment's canonical
``analysis/<experiment_folder>_fig_data.json`` and writes the combined 2-page
dashboard PDF. The panel builders live in ``single_run_dashboard.py`` (main
page) and ``diagnostics_dashboard.py`` (diagnostics page); this module wires
them to the required output filenames so callers do not depend on the builder
modules' CLIs.

Naming (asymmetric):
  - fig_data:      ``<experiment_folder>_fig_data.json``        (NO infix)
  - rendered PDF:  ``<experiment_folder>_dashboard.pdf``        (WITH ``_dashboard`` suffix)

The asymmetry exists because the fig_data file is the canonical history (more
than just the dashboard — also diagnostics + samples), while the rendered PDF
IS the dashboard.

PNG output (per-page byproducts) is **off by default**: the PDF is the
canonical rendered artifact. Pass ``--emit-pngs`` to also write the per-page
PNGs for debugging.

Usage:
  python -m ac2.viz.render_dashboard FIG_DATA ANALYSIS_DIR [options]

Options:
  --experiment-folder NAME   defaults to ANALYSIS_DIR's parent dir name
  --figure-desc STR          appended to the basename; default "dashboard"
                              (use "" to drop the suffix; the canonical default
                              keeps "_dashboard")
  --emit-pngs                also write _main.png and _diagnostics.png (off by
                              default; PDF is the canonical artifact)
  --visual-results-dir DIR   override the mirror destination for the PDF
                              (default: the experiment directory, i.e. ANALYSIS_DIR's parent)

Writes (by default):
  <ANALYSIS_DIR>/<EXPERIMENT_FOLDER>_dashboard.pdf
  <EXPERIMENT_DIR>/<EXPERIMENT_FOLDER>_dashboard.pdf   (mirror, alongside run_data/)

With --emit-pngs, also writes:
  <ANALYSIS_DIR>/<EXPERIMENT_FOLDER>_dashboard_main.png
  <ANALYSIS_DIR>/<EXPERIMENT_FOLDER>_dashboard_diagnostics.png
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import dashboard_common as DC  # noqa: E402
from single_run_dashboard import build_main_figure, build_raw_page, has_difficulty_raw  # noqa: E402
from diagnostics_dashboard import build_diag_figure  # noqa: E402

# Repo root = nearest ancestor with a pyproject.toml (robust to where the viz
# package lives in the tree).
def _find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return start.parents[2]


_REPO_ROOT = _find_repo_root(Path(__file__).resolve())


def render(
    fig_data: str,
    analysis_dir: str,
    experiment_folder: str | None = None,
    figure_desc: str = "dashboard",
    visual_results_dir: str | None = None,
    emit_pngs: bool = False,
    prover_only: bool | None = None,
    hintgap_sidecar: str | None = None,
) -> dict[str, str]:
    """``prover_only`` — when True, drop proposer-only panels (auto-detected
    from fig_data via ``DC.is_prover_only(F)`` if left None).
    ``hintgap_sidecar`` — path to a hintgap_dashboard sidecar JSON; when given,
    a hint-gap conjecturer page is appended to the PDF."""
    d = DC.load_fig_data(fig_data)
    F = DC.FigData(d, fig_data)
    if prover_only is None:
        prover_only = DC.is_prover_only(F)
    main_fig = build_main_figure(F, prover_only=prover_only)
    diag_fig = build_diag_figure(F, prover_only=prover_only)
    hintgap_fig = None
    if hintgap_sidecar:
        import json as _json
        from hintgap_dashboard import build_hintgap_figure  # noqa: E402 (same dir on sys.path)
        with open(hintgap_sidecar, encoding="utf-8") as fh:
            hintgap_fig = build_hintgap_figure(F, _json.load(fh))
    analysis = Path(analysis_dir)
    analysis.mkdir(parents=True, exist_ok=True)
    folder = experiment_folder or analysis.resolve().parent.name
    base = f"{folder}_{figure_desc}" if figure_desc else folder
    pdf_path = analysis / f"{base}.pdf"
    paths: dict[str, str] = {}
    if emit_pngs:
        main_png = analysis / f"{base}_main.png"
        diag_png = analysis / f"{base}_diagnostics.png"
        main_fig.savefig(main_png, dpi=130, bbox_inches="tight")
        diag_fig.savefig(diag_png, dpi=130, bbox_inches="tight")
        paths["main"] = str(main_png)
        paths["diagnostics"] = str(diag_png)
    # Page 3 (difficulty-sampling runs only): raw sampled-distribution series vs the
    # SNIS-corrected values that pages 1-2 display under the standard keys.
    raw_fig = build_raw_page(F) if has_difficulty_raw(F) else None
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(main_fig)
        pdf.savefig(diag_fig)
        if raw_fig is not None:
            pdf.savefig(raw_fig)
        if hintgap_fig is not None:
            pdf.savefig(hintgap_fig)
    plt.close(main_fig)
    plt.close(diag_fig)
    if raw_fig is not None:
        plt.close(raw_fig)
    if hintgap_fig is not None:
        plt.close(hintgap_fig)
    paths["pdf"] = str(pdf_path)
    # The PDF is mirrored one level up, into the experiment directory itself (the parent of
    # analysis/), so each run's dashboard lives alongside its own run_data instead of in a shared
    # top-level folder. Override with --visual-results-dir if a different destination is wanted.
    vr = Path(visual_results_dir) if visual_results_dir else analysis.resolve().parent
    vr.mkdir(parents=True, exist_ok=True)
    vr_pdf = vr / f"{base}.pdf"
    if vr_pdf.resolve() != pdf_path.resolve():
        shutil.copy2(pdf_path, vr_pdf)
    paths["visual_results_pdf"] = str(vr_pdf)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("fig_data", help="canonical fig_data JSON path")
    ap.add_argument("analysis_dir", help="<experiment>/analysis dir to write into")
    ap.add_argument("--experiment-folder", default=None,
                    help="override the experiment folder name (defaults to "
                         "analysis_dir's parent name)")
    ap.add_argument("--figure-desc", default="dashboard",
                    help="suffix appended to the basename; default 'dashboard' "
                         "matches the default output naming. Use '' to "
                         "drop the suffix entirely.")
    ap.add_argument("--emit-pngs", action="store_true",
                    help="also write _main.png and _diagnostics.png (off by "
                         "default; PDF is the canonical artifact)")
    ap.add_argument("--visual-results-dir", default=None,
                    help="override the PDF mirror destination "
                         "(default: the experiment directory, ANALYSIS_DIR's parent)")
    ap.add_argument("--prover-only", action="store_true", default=None,
                    help="force prover-only mode (drop proposer-only panels). "
                         "Default: auto-detect from fig_data via "
                         "DC.is_prover_only(F) — set to True iff the run "
                         "generated 0 proposer rows throughout. Use this flag "
                         "to override the auto-detect.")
    ap.add_argument("--hintgap-sidecar", default=None,
                    help="hintgap_dashboard sidecar JSON; appends the hint-gap "
                         "conjecturer page to the PDF")
    # Legacy positional compatibility: prior callers passed EXPERIMENT_FOLDER
    # as argv[3] and FIGURE_DESC as argv[4]. We accept up to 2 trailing
    # positionals if the new --experiment-folder/--figure-desc weren't used.
    ap.add_argument("legacy_positionals", nargs="*",
                    help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.legacy_positionals:
        if args.experiment_folder is None and len(args.legacy_positionals) >= 1:
            args.experiment_folder = args.legacy_positionals[0]
        if len(args.legacy_positionals) >= 2:
            args.figure_desc = args.legacy_positionals[1]
    paths = render(
        args.fig_data, args.analysis_dir,
        experiment_folder=args.experiment_folder,
        figure_desc=args.figure_desc,
        visual_results_dir=args.visual_results_dir,
        emit_pngs=args.emit_pngs,
        prover_only=args.prover_only,
        hintgap_sidecar=args.hintgap_sidecar,
    )
    for label in ("main", "diagnostics", "pdf", "visual_results_pdf"):
        if label in paths:
            print(f"wrote {paths[label]}")


if __name__ == "__main__":
    main()
