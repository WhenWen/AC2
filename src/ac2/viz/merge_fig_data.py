#!/usr/bin/env python3
"""Merge a cluster scratch parse-output into the ONE canonical fig_data file.

The refresh lifecycle is:

    cluster parser (start at step N, inclusive) -> scratch parse-output
      -> copied back to the checkout -> merge into the canonical analysis/*_fig_data.json
      -> render

This tool is the merge step. It is stdlib-only. The refresh asks the cluster
parser to START AT the latest train step ``N`` already in the canonical file
(inclusive), so the parse-output re-includes step ``N`` as an intentional
one-step overlap. This merger:

1. **Strictly verifies the overlap.** Every ``(step, metric_key)`` present in
   BOTH the canonical file and the parse-output must match (numeric values within
   a tiny float tolerance), and every rollout-summary identity
   ``(split, step, source_file, source_line)`` present in both must match. On ANY
   conflict it exits non-zero and leaves the canonical file UNCHANGED.
2. **Appends.** Once the overlap checks out, it keeps the canonical values on the
   overlap, adds the parse-output's new metrics/steps, dedup-appends rollout
   summaries, unions warnings, merges the ``refresh`` provenance block, validates,
   and atomically replaces the canonical file.

With no existing ``--canonical`` (a fresh experiment / full re-parse), the
parse-output becomes the initial canonical file. A ``--seed`` (a copied parent
run's canonical history, for continuation runs) is used as the starting canonical
when the child has none yet.

Usage:
  python -m ac2.viz.merge_fig_data --parse-output PARSE.json --canonical CANON.json [--seed SEED.json]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

SCHEMA = "riemann_v7_fig_data"

# Top-level fields taken from the newest parse-output, with canonical fallback.
# "refresh" is merged specially; per_step / train_jsonl / val_jsonl / warnings
# have their own merge logic. "parents" is merged with canonical-wins semantics
# (see _merge_parents) so the parent chain set at seed time is immutable
# across routine refreshes.
_SCALAR_TOPLEVEL = (
    "schema", "schema_version", "run_id", "source", "manifest", "config",
    "global", "artifacts", "samples_full", "dashboard_annotations",
)


# ---------------------------------------------------------------------------
# Parent chain: seed promotion + canonical-wins merge of `parents`.
# ---------------------------------------------------------------------------

def _validate_parents_contiguity(parents: list[dict[str, Any]], *, context: str) -> None:
    """The parent chain must be contiguous: for every consecutive pair
    ``parents[i]``, ``parents[i+1]`` we require ``parents[i].end_step + 1 ==
    parents[i+1].start_step``. A gap or an overlap means the recorded chain
    cannot reconstruct a single continuous step axis; we refuse rather than
    render a misleading boundary. Raises SystemExit with a clear message that
    names the offending pair and the violation kind."""
    if not isinstance(parents, list) or len(parents) < 2:
        return
    for i in range(len(parents) - 1):
        a, b = parents[i], parents[i + 1]
        try:
            a_end = int(a.get("end_step"))
            b_start = int(b.get("start_step"))
        except (TypeError, ValueError):
            sys.exit(
                f"ERROR: malformed parents entry in {context}: "
                f"parents[{i}]={a!r} parents[{i+1}]={b!r} (end_step / "
                f"start_step must be integers)"
            )
        if b_start == a_end + 1:
            continue
        if b_start <= a_end:
            kind = (f"OVERLAP — parents[{i+1}].start_step={b_start} "
                    f"intersects parents[{i}] (end_step={a_end})")
        else:
            kind = (f"GAP — parents[{i+1}].start_step={b_start} but "
                    f"parents[{i}].end_step={a_end} (expected start_step="
                    f"{a_end + 1})")
        sys.exit(
            f"ERROR: non-contiguous lineage in {context}: {kind}. "
            f"Every transition between consecutive parents must satisfy "
            f"`parents[i+1].start_step == parents[i].end_step + 1`. "
            f"a=parents[{i}].run_id={a.get('run_id')!r}; "
            f"b=parents[{i+1}].run_id={b.get('run_id')!r}."
        )


def _seed_own_range(doc: dict[str, Any]) -> tuple[int, int] | None:
    """The step range a seed file contributed BEYOND its own ancestors.

    If the seed has no parents, that's just (min, max) of its per_step.steps.
    If it does, the seed's own contribution starts at the step AFTER its last
    parent's end_step (since `parents` is in chronological order and each entry's
    range is inclusive). Returns None if the seed has no steps."""
    ps = doc.get("per_step") or {}
    steps = ps.get("steps") or []
    if not steps:
        return None
    parents = doc.get("parents") or []
    if isinstance(parents, list) and parents:
        last = parents[-1]
        try:
            start = int(last.get("end_step")) + 1
        except (TypeError, ValueError):
            start = int(min(steps))
    else:
        start = int(min(steps))
    end = int(max(steps))
    return start, end


def _promote_seed_into_parents(doc: dict[str, Any]) -> None:
    """Mutate ``doc`` (a freshly-loaded seed) so it becomes the child's INITIAL
    canonical: append the seed's own ``run_id`` (+ its own step range) to the
    parents list, then clear ``run_id`` so the first parse-output sets the
    child's id. No-op if the seed has no run_id or no steps.

    Validates contiguity of the seed's existing parents chain BEFORE promoting
    (so a malformed seed fails loud rather than silently extending the chain),
    and of the new chain AFTER promoting (defense in depth)."""
    seed_run_id = doc.get("run_id") or ""
    rng = _seed_own_range(doc)
    if not seed_run_id or rng is None:
        return
    # (1) Pre-validate: the seed's existing chain must be contiguous.
    _validate_parents_contiguity(
        doc.get("parents") or [], context="seed file's existing `parents`")
    start, end = rng
    parents = list(doc.get("parents") or [])
    parents.append({
        "run_id": seed_run_id,
        "start_step": start,
        "end_step": end,
    })
    doc["parents"] = parents
    # (2) Post-validate: the newly extended chain must still be contiguous.
    _validate_parents_contiguity(
        parents, context=f"promoted seed `{seed_run_id}`")
    # Clear run_id — the next parse-output's run_id will set the child's id
    # via the normal _SCALAR_TOPLEVEL merge.
    doc["run_id"] = ""


def _merge_parents(c_parents, p_parents) -> list[dict[str, Any]]:
    """Canonical-wins. Once `parents` is set at seed time it is immutable
    across routine refreshes — strict-overlap merges never extend the chain.
    Returns [] when neither side carries a parents list. Validates contiguity
    on the returned list so any external corruption (hand-edit, future bug)
    fails loud at the next refresh rather than rendering a misleading
    boundary."""
    if isinstance(c_parents, list):
        out = c_parents
    elif isinstance(p_parents, list):
        out = p_parents
    else:
        out = []
    _validate_parents_contiguity(out, context="merge step")
    return out


def _load(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _per_step_to_rows(per_step: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Column-oriented per_step -> {step: {key: value}} (skip None)."""
    steps = per_step.get("steps") or []
    rows: dict[int, dict[str, Any]] = {}
    for i, s in enumerate(steps):
        si = int(s)
        cur = rows.setdefault(si, {})
        for key, col in per_step.items():
            if key == "steps":
                continue
            if i < len(col) and col[i] is not None:
                cur[key] = col[i]
    return rows


def _rows_to_per_step(rows: dict[int, dict[str, Any]]) -> dict[str, Any]:
    steps = sorted(rows)
    keys = sorted({k for r in rows.values() for k in r})
    out: dict[str, Any] = {"steps": steps}
    for k in keys:
        out[k] = [rows[s].get(k) for s in steps]
    return out


def _train_rollout_steps(doc: dict[str, Any]) -> set[int]:
    """Steps that carry TRAIN rollout data (a non-None ``v7__train__*`` metric).

    This is the level the refresh watermark + the intended one-step overlap live
    at (refresh_dashboard.sh picks the max such step as the parse start). Metric-
    only steps from ``metrics.jsonl`` are excluded, since those are always parsed
    in full and would mask a rollout-boundary gap.
    """
    ps = doc.get("per_step") or {}
    steps = ps.get("steps") or []
    out: set[int] = set()
    for k, col in ps.items():
        if isinstance(k, str) and k.startswith("v7__train__") and isinstance(col, list):
            for i, x in enumerate(col):
                if x is not None and i < len(steps):
                    out.add(int(steps[i]))
    return out


def _section_totals(rows: dict[int, dict[str, Any]], section: str) -> tuple[int, int]:
    """``(total_rows, n_step_files)`` for a {train,val} section, summed from the
    MERGED per_step (the source of truth) — so an incremental refresh reports the
    full history, not just the latest parse delta. ``total_generated`` is the
    per-step rollout-row count the parser also sums into ``{section}_jsonl.n``;
    one rollout file per step, so the step count is ``n_step_files``."""
    gen_key = f"v7__{section}__mode_mix__total_generated"
    total = files = 0
    for row in rows.values():
        v = row.get(gen_key)
        if v is not None:
            total += int(round(float(v)))
            files += 1
    return total, files


def _values_match(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def _identity_map(jsonl: dict[str, Any] | None, split: str) -> dict[tuple, dict[str, Any]]:
    """All example + pooled rollout-summary rows keyed by stable identity."""
    out: dict[tuple, dict[str, Any]] = {}
    jsonl = jsonl or {}
    rows: list[dict[str, Any]] = []
    for cat_rows in (jsonl.get("examples") or {}).values():
        rows.extend(cat_rows or [])
    rows.extend(jsonl.get("pooled") or [])
    for row in rows:
        ident = (split, row.get("step"), row.get("source_file"), row.get("source_line"))
        out[ident] = row
    return out


def overlap_conflicts(canonical: dict[str, Any], parse_output: dict[str, Any]) -> list[str]:
    """Every (step, metric_key) and rollout-summary identity present in BOTH must
    match. Returns the list of conflicts (empty == clean overlap)."""
    conflicts: list[str] = []
    c_rows = _per_step_to_rows(canonical.get("per_step") or {})
    p_rows = _per_step_to_rows(parse_output.get("per_step") or {})
    for step in sorted(set(c_rows) & set(p_rows)):
        crow, prow = c_rows[step], p_rows[step]
        for k in sorted(set(crow) & set(prow)):
            if not _values_match(crow[k], prow[k]):
                conflicts.append(
                    f"per_step step {step} metric {k!r}: canonical={crow[k]!r} parse={prow[k]!r}")
    for split in ("train", "val"):
        cmap = _identity_map(canonical.get(f"{split}_jsonl"), split)
        pmap = _identity_map(parse_output.get(f"{split}_jsonl"), split)
        for ident in sorted(set(cmap) & set(pmap), key=lambda t: tuple(str(x) for x in t)):
            if cmap[ident] != pmap[ident]:
                conflicts.append(f"{split} rollout summary {ident} differs at the overlap")
    return conflicts


def _merge_jsonl(canon: dict[str, Any] | None, parse: dict[str, Any] | None, split: str) -> dict[str, Any]:
    canon = canon or {}
    parse = parse or {}

    def key(row: dict[str, Any]) -> tuple:
        return (split, row.get("step"), row.get("source_file"), row.get("source_line"))

    out: dict[str, Any] = {
        "n": (parse.get("n") or 0) or (canon.get("n") or 0),
        "n_step_files": parse.get("n_step_files", canon.get("n_step_files")),
    }
    out_ex: dict[str, list] = {}
    c_ex = canon.get("examples") or {}
    p_ex = parse.get("examples") or {}
    for cat in sorted(set(c_ex) | set(p_ex)):
        seen, merged = set(), []
        for row in (c_ex.get(cat) or []) + (p_ex.get(cat) or []):
            k = key(row)
            if k in seen:
                continue
            seen.add(k)
            merged.append(row)
        out_ex[cat] = merged
    out["examples"] = out_ex
    seen, pooled = set(), []
    for row in (canon.get("pooled") or []) + (parse.get("pooled") or []):
        k = key(row)
        if k in seen:
            continue
        seen.add(k)
        pooled.append(row)
    out["pooled"] = pooled
    return out


def _merge_refresh(c_ref: dict[str, Any] | None, p_ref: dict[str, Any] | None) -> dict[str, Any] | None:
    if not p_ref:
        return c_ref
    out = dict(p_ref)  # newest refresh metadata
    files = {f["step"]: f for f in (c_ref or {}).get("parsed_train_files", []) or []}
    files.update({f["step"]: f for f in p_ref.get("parsed_train_files", []) or []})
    out["parsed_train_files"] = [files[s] for s in sorted(files)]
    lasts = [x for x in ((c_ref or {}).get("last_train_rollout_step"),
                         p_ref.get("last_train_rollout_step")) if x is not None]
    out["last_train_rollout_step"] = max(lasts) if lasts else None
    return out


def _merge_warnings(*docs: dict[str, Any] | None) -> list[str]:
    seen, out = set(), []
    for d in docs:
        for w in (d or {}).get("warnings", []) or []:
            if w in seen:
                continue
            seen.add(w)
            out.append(w)
    return out


def _merge_global_step_contrib(merged: dict[str, Any], canonical: dict[str, Any],
                               parse_output: dict[str, Any]) -> None:
    """Rebuild the CUMULATIVE global histograms from per-step contributions.

    The parse output's ``global`` covers only the steps it parsed, so on an
    incremental refresh the newest-wins scalar merge would leave the cumulative
    panels (score-by-length, overlong hist) computed over the parse window only.
    When BOTH sides carry ``global.global_step_contrib`` (contrib-aware parser),
    union the contributions -- train: canonical wins on the overlap step, parse
    adds new steps; val: parse wins outright (val files are re-read in full every
    refresh) -- and recompute the cumulative fields as exact column-sums. This
    makes incremental refreshes exact and removes the need for the periodic FULL
    re-parse. Old-format inputs (no contrib block on either side) fall through to
    the existing newest-wins behaviour untouched.
    """
    cg = (canonical.get("global") or {}).get("global_step_contrib") or {}
    pg = (parse_output.get("global") or {}).get("global_step_contrib") or {}
    if not cg or not pg:
        return
    train: dict[str, Any] = dict(cg.get("train") or {})
    for k, v in (pg.get("train") or {}).items():
        if k not in train:  # canonical wins on the verified-overlap step
            train[k] = v
    val: dict[str, Any] = dict(pg.get("val") or {}) or dict(cg.get("val") or {})

    contribs = list(train.values()) + list(val.values())
    if not contribs:
        return
    n_len = max(len(c["len_bins_proposer"]) for c in contribs)
    n_ov = max(len(c["overlong_hist_proposer"]) for c in contribs)
    tot = {
        "len_bins_proposer": [[0.0, 0.0] for _ in range(n_len)],
        "len_bins_prover": [[0.0, 0.0] for _ in range(n_len)],
        "overlong_hist_proposer": [0] * n_ov,
        "overlong_hist_prover": [0] * n_ov,
        "n_overlong_proposer": 0,
        "n_overlong_prover": 0,
    }
    for c in contribs:
        for key in ("len_bins_proposer", "len_bins_prover"):
            for i, (cnt, ssum) in enumerate(c.get(key) or []):
                tot[key][i][0] += cnt
                tot[key][i][1] += ssum
        for key in ("overlong_hist_proposer", "overlong_hist_prover"):
            for i, cnt in enumerate(c.get(key) or []):
                tot[key][i] += cnt
        for key in ("n_overlong_proposer", "n_overlong_prover"):
            tot[key] += int(c.get(key) or 0)

    g = dict(merged.get("global") or {})
    g["score_by_len_proposer"] = [
        {"count": int(cnt), "score_sum": ssum} for cnt, ssum in tot["len_bins_proposer"]]
    g["score_by_len_prover"] = [
        {"count": int(cnt), "score_sum": ssum} for cnt, ssum in tot["len_bins_prover"]]
    g["overlong_hist_proposer"] = list(tot["overlong_hist_proposer"])
    g["overlong_hist_prover"] = list(tot["overlong_hist_prover"])
    g["n_overlong_proposer"] = int(tot["n_overlong_proposer"])
    g["n_overlong_prover"] = int(tot["n_overlong_prover"])
    g["global_step_contrib"] = {"train": train, "val": val}
    merged["global"] = g


def append_merge(canonical: dict[str, Any], parse_output: dict[str, Any]) -> dict[str, Any]:
    """Overlap already verified clean. Keep canonical values on the overlap, add
    the parse-output's new metrics/steps, and append rollout summaries."""
    c_rows = _per_step_to_rows(canonical.get("per_step") or {})
    p_rows = _per_step_to_rows(parse_output.get("per_step") or {})
    for step, prow in p_rows.items():
        crow = c_rows.setdefault(step, {})
        for k, v in prow.items():
            if k not in crow:  # overlap verified equal -> keep canonical
                crow[k] = v
    merged: dict[str, Any] = {}
    for k in _SCALAR_TOPLEVEL:
        # `config` is merged FIELD-WISE (prefer the parse-output's non-null value,
        # else keep canonical's). Wholesale replacement would clobber parent-carried
        # fields a continuation run cannot yet derive: e.g. val_data_source /
        # val_best_at_n are parsed from OBSERVED val metrics, so before the child's
        # first val step its config has them None and the parent's imoproofbench/16
        # would be lost -> panel 3 (val best@n) would read best@None and render
        # empty. Field-wise keeps the parent's until the child derives its own.
        if k == "config":
            c_cfg = canonical.get("config") or {}
            p_cfg = parse_output.get("config") or {}
            if c_cfg or p_cfg:
                m_cfg = dict(c_cfg)
                for ck, cv in p_cfg.items():
                    if cv not in (None, "", {}):
                        m_cfg[ck] = cv
                merged["config"] = m_cfg
            continue
        if k in parse_output and parse_output[k] not in (None, {}, ""):
            merged[k] = parse_output[k]
        elif k in canonical:
            merged[k] = canonical[k]
    merged["refresh"] = _merge_refresh(canonical.get("refresh"), parse_output.get("refresh"))
    merged["parents"] = _merge_parents(canonical.get("parents"), parse_output.get("parents"))
    merged["per_step"] = _rows_to_per_step(c_rows)
    merged["train_jsonl"] = _merge_jsonl(canonical.get("train_jsonl"), parse_output.get("train_jsonl"), "train")
    merged["val_jsonl"] = _merge_jsonl(canonical.get("val_jsonl"), parse_output.get("val_jsonl"), "val")
    # Row / step-file counts reflect the MERGED history (recomputed from per_step),
    # not just the latest parse delta — otherwise the header + diagnostics
    # underreport rows/files after an incremental refresh.
    for section, jkey in (("train", "train_jsonl"), ("val", "val_jsonl")):
        n_rows, n_files = _section_totals(c_rows, section)
        merged[jkey]["n"] = n_rows
        merged[jkey]["n_step_files"] = n_files
    merged["warnings"] = _merge_warnings(canonical, parse_output)
    _merge_global_step_contrib(merged, canonical, parse_output)
    _merge_q_calibration(merged, canonical, parse_output)
    return merged


def _merge_q_calibration(merged: dict[str, Any], canonical: dict[str, Any],
                         parse_output: dict[str, Any]) -> None:
    """Union the generative-Q calibration contributions by step: canonical wins
    on the overlap step, the parse delta adds new steps. Without this the newest-wins
    scalar merge of ``global`` would drop all pre-window probe history each refresh."""
    cq = ((canonical.get("global") or {}).get("q_calibration") or {}).get("by_step") or {}
    pq = ((parse_output.get("global") or {}).get("q_calibration") or {}).get("by_step") or {}
    if not cq and not pq:
        return
    by_step = dict(cq)
    for k, v in pq.items():
        if k not in by_step:
            by_step[k] = v
    g = dict(merged.get("global") or {})
    g["q_calibration"] = {"by_step": by_step}
    merged["global"] = g


def _validate(doc: dict[str, Any]) -> list[str]:
    errs = []
    if doc.get("schema") != SCHEMA:
        errs.append(f"schema is {doc.get('schema')!r}, expected {SCHEMA!r}")
    ps = doc.get("per_step") or {}
    steps = ps.get("steps")
    if not isinstance(steps, list):
        errs.append("per_step.steps missing or not a list")
    else:
        if steps != sorted(steps) or len(set(steps)) != len(steps):
            errs.append("per_step.steps not strictly increasing / has duplicates")
        for key, col in ps.items():
            if key == "steps":
                continue
            if not isinstance(col, list) or len(col) != len(steps):
                errs.append(f"per_step[{key!r}] length "
                            f"{len(col) if isinstance(col, list) else 'n/a'} != steps {len(steps)}")
                break
    return errs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parse-output", required=True, help="cluster scratch parse-output to merge in")
    ap.add_argument("--canonical", required=True, help="the one canonical fig_data file (read+write)")
    ap.add_argument("--seed", default=None, help="parent-run canonical history to seed a continuation run")
    ap.add_argument("--seed-adjacent", action="store_true",
                    help="with --seed: allow a continuation whose train steps are ADJACENT to "
                         "(start exactly one after) the parent's last train step, sharing NO "
                         "overlap step. Verifies max(parent_train)+1 == min(child_train) so a real "
                         "gap is still refused. Use when the child re-baselines under a new reward "
                         "regime and therefore shares no train step with the parent.")
    args = ap.parse_args()

    parse_output = _load(args.parse_output)
    if parse_output is None:
        sys.exit(f"ERROR: parse-output not found: {args.parse_output}")

    canonical = _load(args.canonical)
    _seeded = False
    if canonical is None and args.seed:
        canonical = _load(args.seed)
        # Parent promotion: the seed becomes the child's initial
        # canonical with the seed's OWN run_id appended into parents. This is
        # the only place `parents` is mutated; routine refreshes preserve it.
        if canonical is not None:
            _promote_seed_into_parents(canonical)
            _seeded = True

    if canonical is None:
        # Fresh experiment / full re-parse: the parse-output IS the canonical.
        # A root run has no parents — emit an explicit empty
        # list so the field is always present in the canonical schema.
        merged = dict(parse_output)
        merged.setdefault("parents", [])
        merged["warnings"] = _merge_warnings(parse_output)
    else:
        conflicts = overlap_conflicts(canonical, parse_output)
        if conflicts:
            shown = conflicts[:20]
            extra = f"\n  ... and {len(conflicts) - 20} more" if len(conflicts) > 20 else ""
            ov = (parse_output.get("refresh") or {}).get("overlap_step")
            sys.exit(f"ERROR: overlap conflict at step {ov} — the cluster parser and the "
                     f"canonical history disagree at the boundary. Canonical file LEFT "
                     f"UNCHANGED.\n  - " + "\n  - ".join(shown) + extra)
        # Require the intended one-step overlap to ACTUALLY exist. The conflict
        # check above is vacuously clean when the two files share no train-rollout
        # step, so without this guard a disjoint parse-output would be appended —
        # leaving the boundary unverified and risking a silent gap.
        c_train = _train_rollout_steps(canonical)
        p_train = _train_rollout_steps(parse_output)
        if c_train and p_train:
            shared = c_train & p_train
            ov = (parse_output.get("refresh") or {}).get("overlap_step")
            if not shared:
                if args.seed_adjacent and _seeded and max(c_train) + 1 == min(p_train):
                    # Adjacent contiguous chain: parent trains ..N, child trains N+1..,
                    # sharing no step to verify — but max(parent)+1 == min(child) means there
                    # is no gap. The child re-baselines under a new reward regime, so no shared
                    # train step exists by construction. Explicitly allowed via --seed-adjacent.
                    pass
                else:
                    sys.exit(
                        f"ERROR: no train-rollout overlap — canonical train steps end at "
                        f"{max(c_train)} but the parse-output's start at {min(p_train)} "
                        f"(overlap_step={ov}). The boundary cannot be verified and a gap may be "
                        f"introduced; re-run the parser starting at the canonical watermark "
                        f"(inclusive), or pass --seed-adjacent for an intentional no-overlap "
                        f"lineage. Canonical file LEFT UNCHANGED.")
            if ov is not None and int(ov) not in shared:
                sys.exit(
                    f"ERROR: declared overlap_step {ov} is not a train-rollout step present in "
                    f"BOTH files (shared train steps e.g. {sorted(shared)[:5]}). Canonical file "
                    f"LEFT UNCHANGED.")
        merged = append_merge(canonical, parse_output)

    errs = _validate(merged)
    if errs:
        sys.exit("ERROR: merged fig_data failed validation (canonical left unchanged):\n  - "
                 + "\n  - ".join(errs))

    # Atomic replace: write a sibling tmp, then os.replace over the canonical so a
    # failure never corrupts the canonical file.
    out_path = Path(args.canonical)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(merged, f, separators=(",", ":"), sort_keys=True, default=str)
        f.write("\n")
    os.replace(tmp, out_path)

    steps = merged["per_step"].get("steps") or []
    rng = f"{min(steps)}-to-{max(steps)}" if steps else "none"
    print(f"wrote {args.canonical}: {len(steps)} steps ({rng}), "
          f"train_rows={(merged.get('train_jsonl') or {}).get('n')}, "
          f"{len(merged.get('warnings', []))} warning(s)")


if __name__ == "__main__":
    main()
