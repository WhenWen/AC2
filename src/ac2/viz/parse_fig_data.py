#!/usr/bin/env python3
"""Parse a training run directory into dashboard-ready fig_data.

Reads a run's manifest, ``metrics.jsonl`` and per-step rollout dumps and writes a
compact ``fig_data.json`` (the dashboard fig_data format) consumed by
``single_run_dashboard.py`` and ``diagnostics_dashboard.py`` via
``render_dashboard.py``. Two run-directory layouts are auto-detected (see
``_default_paths``):

    experiments/<exp>/                     (standard layout)
      manifest/config.yaml
      run_data/metrics.jsonl
      run_data/rollouts/<step>.jsonl
      run_data/val_rollouts/<step>.jsonl

    <run_dir>/                             (older durable layout)
      manifests/launch_env.txt
      summaries/metrics.jsonl
      rollouts/train/<step>.jsonl
      rollouts/val/<step>.jsonl

This parser is stdlib-only so it can run on the cluster login/head node next to
the raw rollouts; the renderer (matplotlib) runs wherever the fig_data is copied.

The format supports a two-mode (proposer + prover) setup in which each source
theorem yields a proposer and a prover prompt row (``extra_info.mode``);
prover-only runs simply have no proposer rows. The reward path emits:

- proposer rows: ``correctness_judge_score`` (0/1), ``impact_judge_score``
  (0-3), derived ``impact_applied`` (-0.25 / 0 / 0.25 / 0.5),
  ``proposer_pre_length_score = correctness + impact_applied``, and
  ``proposer_reward = max(0, pre_length - length_penalty)``.
- prover rows: ``prover_judge_score`` (0/1, binary proof semantics), with the
  configured length penalty applied.

Key contract decisions:

* The parser dispatches every training rollout into ``proposer``,
  ``prover``, or ``unknown`` mode using ``mode_is_proposer`` /
  ``mode_is_prover`` from the reward. There is no prompt-pattern
  fallback — the reward fails loud if mode is absent, so an unknown row is a
  real parser error.
* Validation rows are treated as prover/direct-proof unless the validation
  dump explicitly carries mode fields (a forward-compat hedge).
* ``score`` in the rollout dump is the *base* reward returned by
  ``compute_score(...)``. For proposer rows that's the pre-length scalar
  in ``[-0.25, 1.5]``; for prover rows it's binary 0/1. Optimized rewards
  are read from ``pre_kl_post_penalty_reward`` — the post-length-penalty,
  pre-KL scalar emitted by the reward manager (see ``dapo.py``). The field
  is gated to exponential-penalty mode; runs with the linear overlong buffer
  do not carry it.
* Prompt-group metrics require explicit ``uid`` in the rollout dump. When
  ``uid`` is absent, group-level panels render N/A rather than falling back
  to repeated prompt text.
* Raw rollout text is huge; the parser streams line-by-line, aggregates
  per step, and keeps only slim excerpts. Full text never lands in
  ``fig_data.json``.

The parser is intentionally **standalone**: nothing here imports from any other
viz package. Metric keys carry a fixed ``v7__`` prefix that is part of the format.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import statistics
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Iterator

SCHEMA = "riemann_v7_fig_data"
SCHEMA_VERSION = 1

# Cap on slim example rows kept per diagnostic category (latest-wins ring).
MAX_EXAMPLES_PER_CATEGORY = 8
# Cap on slim rows kept for the optional HTML sample viewer, per step.
MAX_POOLED_PER_STEP = 6
MAX_POOLED_TOTAL = 600
# Excerpt sizes for slim samples (chars).
INPUT_EXCERPT_CHARS = 600
OUTPUT_HEAD_CHARS = 400
PROOF_EXCERPT_CHARS = 600
PROPOSITION_EXCERPT_CHARS = 400

# Candidate-length bins (chars) for the diagnostics "score by length" panel.
# Edges are right-open; the final bin is open-ended. Fixed edges so cross-run
# comparisons are meaningful.
CANDIDATE_LEN_BIN_EDGES = (1, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000)

# overlong_reward histogram edges (buffer emits values in [-penalty, 0]).
OVERLONG_BIN_EDGES = (-0.9, -0.6, -0.3, -0.1, 0.0)

# Impact rubric levels: a discrete distribution per step. 0-4 because the
# extended impact rubric has "Impact: 4" ("almost equivalent to the seed
# problem"). Runs using the basic 0-3 rubric simply emit 0 for the
# impact_level_*__l4 keys.
IMPACT_LEVELS = (0, 1, 2, 3, 4)

_PROOF_RE = re.compile(r"<proof>(.*?)</proof>", re.DOTALL)
_PROPOSITION_RE = re.compile(r"<proposition>(.*?)</proposition>", re.DOTALL)
_THINK_CLOSE = "</think>"

# --- Theorem identity (proposer<->prover join key) --------------------------
# A rollout `input` wraps the dataset problem as
#   ...Consider the following mathematical problem:\n<THEOREM>\n\n<mode instruction>
# The instruction sentinel differs by mode; cutting at it isolates the shared
# <THEOREM> text. After whitespace normalization this is byte-identical across a
# theorem's proposer and prover rows and matches the dataset's
# extra_info.theorem exactly (verified 100% against train.parquet on a live
# step file). It is the join key for the per-theorem direct-proof statistic:
# "of a conjectured theorem's 8 direct (prover) proofs, how many were correct".
THEOREM_PROBLEM_PREFIX = "Consider the following mathematical problem:"
THEOREM_INSTRUCTION_SENTINELS = (
    "Think briefly about how to solve the problem",   # proposer template
    "Solve the problem. Write a complete solution",    # prover template
)


def _theorem_key(input_str: str | None) -> str | None:
    """Whitespace-normalized dataset theorem text from a rollout ``input``.

    Returns ``None`` when the shared problem prefix is absent (e.g. a prompt
    from a different template), in which case the row simply does not participate in the
    direct-proof join.
    """
    if not input_str:
        return None
    i = input_str.find(THEOREM_PROBLEM_PREFIX)
    if i < 0:
        return None
    s = input_str[i + len(THEOREM_PROBLEM_PREFIX):]
    cut = len(s)
    for sentinel in THEOREM_INSTRUCTION_SENTINELS:
        j = s.find(sentinel)
        if j >= 0:
            cut = min(cut, j)
    norm = " ".join(s[:cut].split())
    return norm or None


def _theorem_hash(key: str) -> str:
    """Stable short id for a theorem key (process-independent, unlike hash())."""
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Small typed-input helpers.
# ---------------------------------------------------------------------------


def _sanitize_key(key: str) -> str:
    return str(key).replace("/", "__").replace(" ", "_")


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _as_int(value: Any) -> int | None:
    out = _as_float(value)
    return int(out) if out is not None else None


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "t", "yes", "y"}:
            return True
        if v in {"0", "false", "f", "no", "n"}:
            return False
    return None


def _percentile_from_sorted(clean: list[float], q: float) -> float | None:
    if not clean:
        return None
    if len(clean) == 1:
        return float(clean[0])
    k = (len(clean) - 1) * (q / 100.0)
    lo = int(math.floor(k))
    hi = min(lo + 1, len(clean) - 1)
    frac = k - lo
    return float(clean[lo] + frac * (clean[hi] - clean[lo]))


def _weighted_percentile_from_sorted(vals: list[float], wts: list[float], q: float) -> float | None:
    """SNIS/importance-WEIGHTED percentile: with vals sorted ascending and aligned draw-weights
    c_i, the q-th percentile is vals[k*] for the MINIMAL k* with c_1+...+c_k* > (q/100)·Σc — the
    right-continuous inverse of the weighted CDF. Under difficulty sampling this estimates the
    UNIFORM-distribution quantile from the biased sample (with c ≡ 1 it matches the plain
    empirical quantile up to the raw variant's interpolation)."""
    if not vals:
        return None
    if len(wts) != len(vals):
        return _percentile_from_sorted(vals, q)
    total = float(sum(wts))
    if total <= 0.0:
        return _percentile_from_sorted(vals, q)
    thresh = (q / 100.0) * total
    acc = 0.0
    for v, w in zip(vals, wts):
        acc += w
        if acc > thresh:
            return float(v)
    return float(vals[-1])


def _candidate_len_bin(n: int) -> int:
    for i, edge in enumerate(CANDIDATE_LEN_BIN_EDGES):
        if n < edge:
            return i
    return len(CANDIDATE_LEN_BIN_EDGES)


def _overlong_bin(value: float) -> int:
    if value >= 0:
        return len(OVERLONG_BIN_EDGES)
    for i, edge in enumerate(OVERLONG_BIN_EDGES):
        if value <= edge:
            return i
    return len(OVERLONG_BIN_EDGES) - 1


class _Warnings:
    """Ordered, de-duplicated warning collector surfaced in fig_data."""

    def __init__(self) -> None:
        self._seen: set[str] = set()
        self.items: list[str] = []

    def add(self, msg: str) -> None:
        if msg not in self._seen:
            self._seen.add(msg)
            self.items.append(msg)

    def emit_stderr(self) -> None:
        for msg in self.items:
            print(f"WARN: {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Manifest + metrics loading (the file logger format is verl-wide), kept here
# so this module stays standalone.
# ---------------------------------------------------------------------------


def _load_manifest(path: Path | None, warnings: _Warnings) -> dict[str, str]:
    if path is None or not path.exists():
        if path is not None:
            warnings.add(f"manifest not found: {path}")
        return {}
    # The standard layout uses a resolved Hydra config.yaml instead of the older
    # launch_env.txt (k=v). Flatten the interesting fields to the same flat key
    # names _derive_config reads; the header is cosmetic, so this is best-effort.
    if path.suffix in (".yaml", ".yml"):
        return _load_config_yaml(path, warnings)
    out: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            out[key.strip()] = value.strip()
    except OSError as exc:
        warnings.add(f"manifest read failed: {exc}")
    return out


def _load_config_yaml(path: Path, warnings: _Warnings) -> dict[str, str]:
    """Flatten a verl/Hydra ``config.yaml`` (standard-layout manifest) into the flat manifest
    keys ``_derive_config`` expects. pyyaml-optional; degrades to an empty header."""
    try:
        import yaml  # optional; only needed for the config.yaml header
    except Exception:
        warnings.add(f"pyyaml unavailable; skipping config.yaml header ({path})")
        return {}
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # noqa: BLE001 - header is best-effort
        warnings.add(f"config.yaml read failed: {exc}")
        return {}

    def g(*keys: str) -> Any:
        cur: Any = cfg
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        return cur

    flat = {
        "run_id": g("trainer", "experiment_name"),
        "experiment": g("trainer", "experiment_name"),
        "nnodes": g("trainer", "nnodes"),
        "total_steps": g("trainer", "total_training_steps"),
        "save_freq": g("trainer", "save_freq"),
        "train_batch_size": g("data", "train_batch_size"),
        "gen_batch_size": g("data", "gen_batch_size"),
        "ppo_mini_batch_size": g("actor_rollout_ref", "actor", "ppo_mini_batch_size"),
        "actor_lr": g("actor_rollout_ref", "actor", "optim", "lr"),
        "actor_kl_loss_coef": g("actor_rollout_ref", "actor", "kl_loss_coef"),
        "pi1_init": g("actor_rollout_ref", "model", "path"),
        "reward": g("reward", "custom_reward_function", "name"),
        # Colocated generative judge (reward_model) + real reward/judge knobs.
        "reward_model": g("reward", "reward_model", "model_path"),
        "reward_num_workers": g("reward", "num_workers"),
        "max_response_length": g("data", "max_response_length"),
        "judge_max_tokens": g(
            "reward", "custom_reward_function", "reward_kwargs", "judge_max_tokens"
        ),
        # Overlong RESPONSE penalty knobs live in the custom reward's kwargs.
        # Absent => no overlong penalty is applied (do NOT invent a default; a
        # nonzero default here makes the dashboard falsely report a linear penalty).
        "overlong_buffer_length": g(
            "reward", "custom_reward_function", "reward_kwargs", "overlong_buffer_length"
        ),
        "overlong_penalty_factor": g(
            "reward", "custom_reward_function", "reward_kwargs", "overlong_penalty_factor"
        ),
        "test_freq": g("trainer", "test_freq"),
        "val_n": g("actor_rollout_ref", "rollout", "val_kwargs", "n"),
        # ---- generative-Q knobs -----------------------------------------------------------
        # The runner passes these as `+data.sp_q_*`, so they land under data/ in the resolved
        # config. _derive_config below asks for them by flattened name; without these entries
        # every one of those lookups would return None and the panels would silently fall
        # back to default constants (gate 0.18, movement-cap regime) for a run that pinned
        # neither.
        "sp_q_ready_thresh": g("data", "sp_q_ready_thresh"),
        # Readiness can have TWO thresholds (global pooled-MAE5 gate vs per-problem
        # fresh-error gate). Absent for runs where the single knob covers both.
        "sp_q_ready_thresh_global": g("data", "sp_q_ready_thresh_global"),
        "sp_q_ready_thresh_problem": g("data", "sp_q_ready_thresh_problem"),
        "sp_q_budget_g": g("data", "sp_q_budget_g"),
        "sp_q_fifo_cap": g("data", "sp_q_fifo_cap"),
        "sp_q_train_n": g("data", "sp_q_train_n"),
        "sp_q_min_valid": g("data", "sp_q_min_valid"),
        "sp_q_audit_den": g("data", "sp_q_audit_den"),
        "sp_q_train_noref": g("data", "sp_q_train_noref"),
        "sp_q_prompt_variant": g("data", "sp_q_prompt_variant"),
        "sp_q_ref_require_pass": g("data", "sp_q_ref_require_pass"),
        "sp_q_rho_cap": g("data", "sp_q_rho_cap"),
        # Q LR ladder / interleave. These are ENV-gated at runtime (SP_Q_LR_LADDER etc.), so
        # runners that use them also mirror them into `+data.sp_q_lr_*` purely so they land in
        # config.yaml and the dashboard can read the real pins instead of guessing. When
        # absent, the metrics-based inference below decides the regime.
        "sp_q_lr_ladder": g("data", "sp_q_lr_ladder"),
        "sp_q_lr_ratio_max": g("data", "sp_q_lr_ratio_max"),
        "sp_q_lr_initial": g("data", "sp_q_lr_initial"),
        "sp_q_lr_floor": g("data", "sp_q_lr_floor"),
        "sp_q_interleave": g("data", "sp_q_interleave"),
        # The Q LR ladder + interleave are ENV-gated (SP_Q_LR_LADDER / SP_Q_INTERLEAVE), so they
        # are not in the hydra config. Infer the regime from what the run actually logged: the
        # ladder emits q/lr_current every step and the movement cap emits q/s. Set by
        # _derive_config's caller via the metrics, not from config -- see _infer_q_regime.
    }

    # Sibling manifest files live next to config.yaml (manifest/git_commit.txt).
    # vendored verl shares the repo, so both heads are the same snapshot commit.
    manifest_dir = path.parent
    try:
        commit = (manifest_dir / "git_commit.txt").read_text(encoding="utf-8").strip()
        if commit:
            flat.setdefault("riemann_head", commit)
            flat.setdefault("verl_head", commit)
    except OSError:
        pass
    # "started" ~ when the manifest was dumped (run launch time).
    try:
        import datetime as _dt

        flat["started_at"] = _dt.datetime.fromtimestamp(path.stat().st_mtime).strftime(
            "%Y-%m-%d %H:%M"
        )
    except OSError:
        pass

    # W&B run URL: entity is nested in the ray runtime-env vars; project + run id
    # are top-level trainer fields. Build the canonical URL when all are present.
    def _find(node: Any, target: str) -> Any:
        if isinstance(node, dict):
            for k, v in node.items():
                if k == target and isinstance(v, str) and v:
                    return v
                found = _find(v, target)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for item in node:
                found = _find(item, target)
                if found is not None:
                    return found
        return None

    _entity = _find(cfg, "WANDB_ENTITY")
    _project = g("trainer", "project_name")
    _run_id = g("trainer", "experiment_name")
    if _entity and _project and _run_id:
        flat["wandb_url"] = f"https://wandb.ai/{_entity}/{_project}/runs/{_run_id}"

    return {k: str(v) for k, v in flat.items() if v is not None}


def _load_metrics(path: Path | None, warnings: _Warnings) -> dict[int, dict[str, Any]]:
    """MERGE all logged rows per ``training/global_step`` (union of keys, later row wins on overlap).

    A single step can be logged as MULTIPLE rows: a validation step writes a val-only row in
    addition to the train row (verl logs them separately), and auto-resume can re-append a step.
    Last-row-wins would let the val-only row CLOBBER the train row's keys (difficulty, policy loss,
    KL, grad-norm, difficulty_raw ...), blanking every train-side panel at each val step. Merging
    keeps the union; on genuine overlap (auto-resume) the later value wins.
    """
    if path is None or not path.exists():
        if path is not None:
            warnings.add(f"metrics.jsonl not found: {path}")
        return {}
    rows: list[tuple[int, dict[str, Any]]] = []
    malformed = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            data = obj.get("data") if isinstance(obj.get("data"), dict) else None
            if not data:
                continue
            step = _as_int(data.get("training/global_step"))
            if step is None:
                step = _as_int(obj.get("step"))
            if step is None:
                step = line_no
            rows.append((step, data))
    if malformed:
        warnings.add(f"metrics.jsonl: dropped {malformed} malformed JSON line(s)")
    latest: dict[int, dict[str, Any]] = {}
    for step, data in rows:
        latest.setdefault(step, {}).update(data)
    return latest


def _detect_metric_data_source(metrics: dict[int, dict[str, Any]], section: str) -> str | None:
    pat = re.compile(rf"^{re.escape(section)}-core/([^/]+)/")
    for data in metrics.values():
        for key in data:
            m = pat.match(key)
            if m:
                return m.group(1)
    return None


def _detect_best_at_n(metrics: dict[int, dict[str, Any]], section: str, metric: str) -> int | None:
    pat = re.compile(rf"^{re.escape(section)}-core/[^/]+/{re.escape(metric)}/(?:mean|best)@(\d+)")
    best: int | None = None
    for data in metrics.values():
        for key in data:
            m = pat.match(key)
            if m:
                n = int(m.group(1))
                best = n if best is None or n > best else best
    return best


# ---------------------------------------------------------------------------
# Rollout streaming + per-step aggregation.
# ---------------------------------------------------------------------------


def _iter_step_files(patterns: list[str]) -> list[tuple[int, Path]]:
    """Resolve glob patterns to ``(step, path)`` sorted by step."""
    paths: list[str] = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern))
    seen: set[str] = set()
    numbered: list[tuple[int, Path]] = []
    unnumbered: list[tuple[float, Path]] = []
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        path = Path(p)
        try:
            step = int(path.stem)
            numbered.append((step, path))
        except ValueError:
            try:
                mtime = path.stat().st_mtime
            except OSError:
                mtime = 0.0
            unnumbered.append((mtime, path))
    numbered.sort(key=lambda t: t[0])
    base = numbered[-1][0] + 1 if numbered else 0
    unnumbered.sort(key=lambda t: t[0])
    numbered.extend((base + i, path) for i, (_, path) in enumerate(unnumbered))
    return numbered


def _extract_post_think(output: str) -> str:
    return output.rsplit(_THINK_CLOSE, 1)[-1] if _THINK_CLOSE in output else output


def _extract_proof_excerpt(post_think: str) -> str:
    m = _PROOF_RE.search(post_think)
    if not m:
        return ""
    return m.group(1).strip()[:PROOF_EXCERPT_CHARS]


def _extract_proposition_excerpt(post_think: str) -> str:
    m = _PROPOSITION_RE.search(post_think)
    if not m:
        return ""
    return m.group(1).strip()[:PROPOSITION_EXCERPT_CHARS]


# Capture one [content]\n...  block, stopping at the next "\n\n[" outer
# header (next [content], [reasoning], [correctness], [impact]) or EOF.
_CONTENT_BLOCK_RE = re.compile(r"\[content\]\n(.*?)(?=\n\n\[|\Z)", re.DOTALL)

JUDGE_RESPONSE_SLIM_CHARS = 300


def _slim_judge_response(response: str, max_chars: int = JUDGE_RESPONSE_SLIM_CHARS) -> str:
    """Build a content-aware preview of a ``judge_response``.

    The full reply written by ``ac2.rewards.prover_judge._format_judge_response``
    places ``[reasoning]`` before ``[content]``; a naive ``[:max_chars]`` cut
    on high-effort reasoning runs hides the parsed score line
    (``Score: 0/1`` / ``Impact: 0..4`` live in ``[content]``) and can even
    truncate away the ``[impact]`` half of a proposer reply entirely.

    This helper:

    - For a proposer reply (outer ``[correctness]`` / ``[impact]`` headers):
      keeps the tail of each call's ``[content]`` block, budgeted evenly,
      and preserves the outer headers so a downstream reader can still split
      the two calls cleanly.
    - For a prover reply (single judge call): returns the tail of the
      single ``[content]`` block.
    - For legacy excerpts (no ``[content]`` markers at all) and parse-failure
      replies (no ``[content]`` block emitted): falls back to a plain tail
      so whatever is there still shows.
    """
    if not response:
        return ""
    if "[content]" not in response:
        return response[-max_chars:]
    contents = _CONTENT_BLOCK_RE.findall(response)
    is_proposer = (
        response.startswith("[correctness]\n") and "\n\n[impact]\n" in response
    )
    if is_proposer and len(contents) >= 2:
        # Reserve enough budget for both headers + the "\n\n" separator,
        # then split the remainder evenly between the two content tails.
        header_bytes = len("[correctness]\n") + len("\n\n[impact]\n")
        per = max(40, (max_chars - header_bytes) // 2)
        c_tail = contents[0].strip()[-per:]
        i_tail = contents[1].strip()[-per:]
        return f"[correctness]\n{c_tail}\n\n[impact]\n{i_tail}"
    if contents:
        return contents[0].strip()[-max_chars:]
    return response[-max_chars:]


def _resolve_mode(row: dict[str, Any]) -> str:
    """proposer / prover / unknown.

    Reads ``mode_is_proposer`` / ``mode_is_prover`` from reward_extra. The prover
    proof-judge reward sets ``mode_is_prover=1``; the (unused-for-now) proposer
    path would set ``mode_is_proposer=1``.

    GENERAL-mode fallback: a run whose rollout rows carry NEITHER flag (e.g. the
    DeepScaleR single-reward task, where ``score`` is itself the correctness
    signal) is treated as a single "prover"/solver mode so its correctness /
    length / optimizer panels populate and only the judge/proposer-specific
    panels render N/A. A row with an ambiguous flag combination (both set, or
    both explicitly 0) stays ``unknown`` so a genuine contract violation is
    still surfaced by the parser warning.
    """
    raw_p = row.get("mode_is_proposer")
    raw_q = row.get("mode_is_prover")
    if raw_p is None and raw_q is None:
        return "prover"
    p = _as_int(raw_p) or 0
    q = _as_int(raw_q) or 0
    if p == 1 and q == 0:
        return "proposer"
    if q == 1 and p == 0:
        return "prover"
    return "unknown"


# ---------------------------------------------------------------------------
# Row family classification (replay vs original).
#
# For replay runs the train batch concatenates the original
# prover/proposer parquet with one or more "additional" prover-only parquets,
# and the dashboards want a family-aware split into:
#   - proposer            (every proposer row)
#   - prover_original     (prover row from the original fineproofs-rl dataset)
#   - prover_additional   (prover row from a replay/synthetic dataset)
#
# Ideally the family would come from extra_info.data_source on the row, but
# verl's rollout dumper strips extra_info, so we RECOVER family identity via a
# parquet-hash SIDECAR built once per run (not included in this repository; no
# experiment here uses it). The sidecar maps
# theorem_hash -> data_source for every theorem in the additional parquet(s).
# Any prover row whose theorem hash hits the sidecar is `prover_additional`
# with the sidecar's data_source; anything else (including theorems we can't
# hash) is `prover_original` with data_source = ORIGINAL_DATA_SOURCE.
# ---------------------------------------------------------------------------

ORIGINAL_DATA_SOURCE = "fineproofs-rl"
ADDITIONAL_DATA_SIDECAR_BASENAME = "additional_data_sidecar.json"


def _load_additional_data_sidecar(
    path: Path | None, warnings: "_Warnings",
) -> dict[str, dict[str, Any]]:
    """Load ``{theorem_hash: {data_source, **replay_provenance}}`` from a JSON
    sidecar. Returns ``{}`` if no sidecar (or the file is missing / unreadable);
    never raises.

    Two on-disk shapes are accepted, both normalised to the dict shape:

    - **legacy** ``{hash: "<data_source>"}``  (older sidecars without
      replay provenance)
    - **current** ``{hash: {"data_source": "<source>",
      "proposition_uid": ..., "seed_statement_uid": ..., "source_run_id": ...,
      "source_step": ..., "source_completion_index": ..., ...}}``  (the
      current format, which lifts replay provenance off the replay parquet so
      the parser can attach it to prover_additional rows without any dump
      writer change).
    """
    if path is None or not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        warnings.add(
            f"additional-data sidecar at {path} could not be loaded ({e}); "
            f"family split will degrade to 'all prover -> prover_original'."
        )
        return {}
    out: dict[str, dict[str, Any]] = {}
    for k, v in raw.items():
        if isinstance(v, str):
            out[k] = {"data_source": v}
        elif isinstance(v, dict):
            ds = v.get("data_source")
            if isinstance(ds, str):
                out[k] = dict(v)
    return out


def _resolve_data_source(
    row: dict[str, Any],
    theorem_hash: str | None,
    sidecar: dict[str, dict[str, Any]],
) -> str | None:
    """Decide a row's data_source. Precedence:
      1. explicit top-level ``data_source`` field (forward-compat dump path),
      2. ``extra_info.data_source`` (nested),
      3. flat ``extra_info_data_source`` (transitional),
      4. sidecar lookup by ``theorem_hash`` (recovery for current dumps),
      5. ``None`` (unknown).
    Returns the data_source string or None."""
    v = row.get("data_source")
    if isinstance(v, str) and v:
        return v
    ei = row.get("extra_info")
    if isinstance(ei, dict):
        v = ei.get("data_source")
        if isinstance(v, str) and v:
            return v
    v = row.get("extra_info_data_source")
    if isinstance(v, str) and v:
        return v
    if theorem_hash is not None and theorem_hash in sidecar:
        return sidecar[theorem_hash].get("data_source")
    return None


def _sidecar_provenance(
    theorem_hash: str | None,
    sidecar: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Return the replay-provenance fields the sidecar lifted off the
    additional parquet's ``extra_info`` for this theorem, or ``{}``.

    This is the recovery path the dashboards use for the replay-provenance
    breakdown / seed drilldown / additional-statement examples panels: every
    provenance field expected on the dump row (proposition_uid,
    seed_statement_uid, source_run_id, source_step, source_completion_index,
    source_correctness/impact_judge_score) is recoverable here as long as the
    sidecar was built in the current format against the additional parquet.
    """
    if theorem_hash is None:
        return {}
    entry = sidecar.get(theorem_hash)
    if not isinstance(entry, dict):
        return {}
    # Strip the data_source key — provenance is the rest.
    return {k: v for k, v in entry.items() if k != "data_source"}


def _resolve_family(mode: str, data_source: str | None) -> str:
    """proposer / prover_original / prover_additional / unknown.

    Family is the dashboard's split dimension for replay runs:
      - mode == "proposer"  -> "proposer"  (proposer family covers every proposer row;
                                            data_source is currently always the original
                                            parquet because the replay parquet is
                                            prover-only by design)
      - mode == "prover" + data_source == ORIGINAL_DATA_SOURCE or None
                            -> "prover_original"
      - mode == "prover" + data_source != ORIGINAL_DATA_SOURCE
                            -> "prover_additional"
      - mode == "unknown"   -> "unknown" (caller should not classify these)
    """
    if mode == "proposer":
        return "proposer"
    if mode == "prover":
        if data_source is None or data_source == ORIGINAL_DATA_SOURCE:
            return "prover_original"
        return "prover_additional"
    return "unknown"


# Replay-provenance keys that may live on a prover_additional row, dumped
# either flat at top level or nested under ``extra_info``.
_REPLAY_PROVENANCE_KEYS = (
    "proposition_uid",
    "seed_statement_uid",
    "seed_data_source",
    "seed_split",
    "seed_index",
    "source_run_id",
    "source_step",
    "source_completion_index",
    "source_correctness_judge_score",
    "source_impact_judge_score",
)


def _extract_replay_provenance(
    row: dict[str, Any],
    sidecar_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pull replay-provenance fields off a rollout row + sidecar fallback.

    Precedence (highest first):

    1. ``extra_info.<key>`` (nested)  — the preferred dump-writer target
    2. ``extra_info_<key>`` (flat)     — a transitional alternative
    3. top-level ``<key>``             — direct row field
    4. ``sidecar_provenance[<key>]``   — the sidecar fallback the parser uses
       when the dump strips ``extra_info`` (which it currently does)

    Returns a dict of only the present keys.
    """
    out: dict[str, Any] = {}
    ei = row.get("extra_info") if isinstance(row.get("extra_info"), dict) else None
    sp = sidecar_provenance or {}
    for k in _REPLAY_PROVENANCE_KEYS:
        v = None
        if ei is not None and k in ei:
            v = ei[k]
        if v is None:
            v = row.get(f"extra_info_{k}")
        if v is None:
            v = row.get(k)
        if v is None and k in sp:
            v = sp[k]
        if v is not None:
            out[k] = v
    return out


def _sample_id(split: str, source_file: str, line_no: int) -> str:
    """Stable id linking a slim sample row to its full record in the lazy
    full-sample JSONL (``v7_samples_full_<RUN_ID>.jsonl``)."""
    return f"{split}:{source_file}:{line_no}"


def _slim_example(
    row: dict[str, Any],
    step: int,
    source_file: str,
    line_no: int,
    *,
    mode: str,
    post_think: str,
    theorem_hash: str | None = None,
    split: str = "train",
    family: str | None = None,
    data_source: str | None = None,
    replay_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a slim sample row carried in fig_data['{train,val}_jsonl']['examples']
    or 'pooled'. Mode-aware: only the proposer slot carries proposition.

    ``theorem_hash`` is the join key used post-parse to attach each proposer
    row's direct-proof statistic (and optional canonical theorem index).

    ``family`` / ``data_source`` / ``replay_provenance`` flow through to the
    HTML sample viewer so additional-statement examples can show
    source_run_id, source_step, proposition_uid, seed_statement_uid, etc.
    They are absent (omitted from the slim dict) for non-replay runs so the
    fig_data stays compact when no replay data is present.
    """
    output = row.get("output") or ""
    slim = {
        "id": _sample_id(split, source_file, line_no),
        "split": split,
        "step": step,
        "source_file": source_file,
        "source_line": line_no,
        "mode": mode,
        "theorem_hash": theorem_hash,
        "score": _as_float(row.get("score")),
        "de_filtered": _as_bool(row.get("de_filtered")),
        "candidate_tag_present": _as_int(row.get("candidate_tag_present")),
        "proposition_tag_present": _as_int(row.get("proposition_tag_present")),
        "proof_tag_present": _as_int(row.get("proof_tag_present")),
        "candidate_len_chars": _as_int(row.get("candidate_len_chars")),
        "proof_len_chars": _as_int(row.get("proof_len_chars")),
        "correctness_judge_score": _as_int(row.get("correctness_judge_score")),
        "impact_judge_score": _as_int(row.get("impact_judge_score")),
        "impact_applied": _as_float(row.get("impact_applied")),
        "proposer_pre_length_score": _as_float(row.get("proposer_pre_length_score")),
        "proposer_reward": _as_float(row.get("proposer_reward")),
        "length_penalty": _as_float(row.get("length_penalty")),
        "prover_judge_score": _as_int(row.get("prover_judge_score")),
        "overlong": _as_bool(row.get("overlong")),
        "overlong_reward": _as_float(row.get("overlong_reward")),
        "judge_http_error": _as_int(row.get("judge_http_error")),
        "judge_parse_failed": _as_int(row.get("judge_parse_failed")),
        "judge_truncated": _as_int(row.get("judge_truncated")),
        "judge_prompt_tokens": _as_int(row.get("judge_prompt_tokens")),
        "judge_completion_tokens": _as_int(row.get("judge_completion_tokens")),
        "input_excerpt": (row.get("input") or "")[:INPUT_EXCERPT_CHARS],
        "output_head": output[:OUTPUT_HEAD_CHARS],
        "proposition_excerpt": (
            _extract_proposition_excerpt(post_think) if mode == "proposer" else ""
        ),
        "proof_excerpt": _extract_proof_excerpt(post_think),
        "judge_response": _slim_judge_response(
            row.get("judge_response") or row.get("judge_response_excerpt") or ""
        ),
    }
    if family is not None:
        slim["family"] = family
    if data_source is not None:
        slim["data_source"] = data_source
    if replay_provenance:
        slim["replay_provenance"] = replay_provenance
    return slim


def _full_record(
    row: dict[str, Any], split: str, source_file: str, line_no: int, mode: str,
    post_think: str,
) -> dict[str, Any]:
    """Full (untruncated) text for one sampled row, written to the lazy
    full-sample JSONL and loaded on demand by the HTML viewer. Keyed by the same
    ``id`` as the slim row so the viewer can join them."""
    prop = ""
    if mode == "proposer":
        m = _PROPOSITION_RE.search(post_think or "")
        prop = m.group(1).strip() if m else ""
    m = _PROOF_RE.search(post_think or "")
    proof = m.group(1).strip() if m else ""
    # Note: the raw `output` (full response incl. <think>) is intentionally NOT
    # stored — it dominates size. Prompt + extracted proposition/proof + judge
    # text are what the viewer lazy-loads; recover the raw
    # response from source_file:line if ever needed.
    return {
        "id": _sample_id(split, source_file, line_no),
        "split": split,
        "step": _as_int(row.get("step")),
        "mode": mode,
        "source_file": source_file,
        "source_line": line_no,
        "input": row.get("input") or "",
        "proposition": prop,
        "proof": proof,
        "judge_response": (
            row.get("judge_response") or row.get("judge_response_excerpt") or ""
        ),
    }


# ---------------------------------------------------------------------------
# Per-mode accumulator. One instance per (mode, step) — flushed at end of step.
# ---------------------------------------------------------------------------


class _ModeAccum:
    """Per-mode per-step counters. Kept flat to keep aggregation cheap and
    the code obviously correct on small inputs."""

    __slots__ = (
        "n", "n_trained", "n_filtered", "n_de_unknown",
        "score_sum_generated", "score_sum_trained", "score_sum_filtered",
        "opt_reward_sum", "opt_reward_n",
        "n_pre_kl_post_penalty_missing",
        "overlong_reward_sum", "overlong_reward_n", "n_overlong",
        "exp_penalty_factor_sum", "exp_penalty_factor_n",
        "exp_penalty_exceed_tokens_sum", "exp_penalty_exceed_tokens_n",
        "n_candidate_tag", "n_proposition_tag", "n_proof_tag", "n_proof_empty",
        "n_judged", "judge_cat_counts",
        "n_http", "n_parse", "n_trunc",
        # proposer-only specific
        "correctness_sum", "correctness_n", "n_correct_1",
        "impact_sum", "impact_n", "impact_correct_sum", "impact_correct_n",
        "impact_applied_sum", "impact_applied_n",
        "pre_length_sum", "pre_length_n",
        "proposer_reward_sum", "proposer_reward_n",
        "length_penalty_sum", "length_penalty_n",
        "impact_level_counts",  # Counter
        # prover-only specific
        "prover_score_sum", "prover_score_n", "n_prover_1",
        "rubric_points_sum", "n_rubric", "n_rubric_7",
        # Replay-prefix stream split: from-scratch-only aggregates over rows the
        # dumps tag with sp_prefix_len==0 (the trainer stamps the tag; older files can be
        # backfilled by a tagging scan). Emitted only when the tag is present, so
        # every pre-replay run's keyset is byte-identical.
        "orig_grade_sum", "orig_grade_n", "orig_pass_sum", "orig_pass_n",
        "orig_strict7_n", "orig_groups", "n_prefix_tagged",
        # Q-routed replay lanes: route-split pass counters over sp_prefix_len>0
        # rows for the audit-reweighted full-suffix pass estimate (see emission).
        "repl_route_full_pass", "repl_route_full_n",
        "repl_route_audit_pass", "repl_route_audit_n", "repl_route_short_n",
        # strict-7 (full marks, rp==7) route counts for the audit-reweighted 7/7 estimate
        "repl_route_full_strict7", "repl_route_audit_strict7",
        # Group-level route map for the audit-reweighted best@n estimate (uid ->
        # {"rt": route, "any": any-pass-yet}). best@n is a GROUP indicator (>=1 of n
        # passed), so it needs per-uid aggregation, unlike the rollout-level pass sums.
        "repl_group_route",
        "generated_tokens_sum",
        # length and judge-token vectors (per-step percentiles)
        "candidate_len_vals", "proof_len_vals", "resp_len_vals", "resp_len_wts", "tot_tok_vals",
        # Stream/route-split response lengths: scratch (from-scratch inflow rows,
        # sp_prefix_len==0) vs replay-prefix rows, and within replay the g-capped short lane
        # vs the full-budget lanes. Panels 5a/5b read these; empty on pre-split runs.
        "resp_len_scratch", "resp_len_replay", "resp_len_replay_short", "resp_len_replay_full",
        # injected-prefix stats per replay lane -> panel 5b effective cap
        "prefix_len_replay", "prefix_len_replay_short", "prefix_len_replay_full",
        # NEWLY GENERATED tokens per row (response minus injected prefix) -> panel 5b
        "gen_len_vals", "gen_len_short_max",
        "jp_tokens", "jc_tokens",
        # prompt-row groups: uid -> [score_bin or correctness or prover_score].
        # The lookup key depends on mode; we store the per-group sequences and
        # compute composition + pass-counts at flush.
        "group_scores", "group_de", "n_group_uid_missing",
        # PROVER-only per-group judge-count sequences (panel 11 per-group
        # percentiles): uid -> [1 per completion that was a judge attempt /
        # success / missing / failed]. Summed per group at flush.
        "group_attempt", "group_success", "group_missing", "group_failed",
        # Difficulty-sampling SNIS side-accumulator:
        # w["<metric>@n"] = Σ c·v and w["<metric>@d"] = Σ c for every per-problem
        # EXPECTATION emitted by _emit_mode_block. Raw counters stay untouched; with
        # c ≡ 1 (uniform / pre-difficulty dumps) weighted == raw EXACTLY, so main keys
        # are always computed from here. group_c = each prompt-group's draw-time c
        # (constant within a group); diff_weighted flips when any row has c != 1,
        # gating the extra <key>_raw emissions so old fig_data keysets are unchanged.
        "w", "w_impact_levels", "group_c", "diff_weighted",
        # Group-level weighting (dynamic group size): per-row factor
        # grp_ref/|group| so every multi-row prompt group carries equal total weight in
        # every expectation and length distribution. grp_ref = mean multi-row group size
        # of the step, so on a fixed-n run every factor is exactly 1 and nothing changes.
        "grp_ref", "resp_len_scratch_w", "resp_len_replay_w", "resp_len_replay_short_w",
        "resp_len_replay_full_w", "gen_len_wts", "group_weighted",
    )

    def __init__(self) -> None:
        self.n = 0
        self.n_trained = 0
        self.n_filtered = 0
        self.n_de_unknown = 0
        self.score_sum_generated = 0.0
        self.score_sum_trained = 0.0
        self.score_sum_filtered = 0.0
        self.opt_reward_sum = 0.0
        self.opt_reward_n = 0
        # Rows that should have carried pre_kl_post_penalty_reward (the
        # exponential-penalty contract) but did not. Surfaced via a one-time
        # warning + a sentinel flag in fig_data["config"]. Linear-overlong runs
        # never set this counter (no rows carry the field), so their
        # optimized_reward_mean path is omitted rather than flagged.
        self.n_pre_kl_post_penalty_missing = 0
        self.overlong_reward_sum = 0.0
        self.overlong_reward_n = 0
        self.n_overlong = 0
        # Exponential-penalty bookkeeping. Each row in an exponential-
        # penalty run emits exponential_penalty_factor (multiplicative shaping
        # factor in [0, 1]) and exponential_penalty_exceed_tokens (tokens
        # above the free budget). Aggregated as a simple mean across rollouts,
        # following the same per-family pattern as length_penalty_mean.
        self.exp_penalty_factor_sum = 0.0
        self.exp_penalty_factor_n = 0
        self.exp_penalty_exceed_tokens_sum = 0.0
        self.exp_penalty_exceed_tokens_n = 0
        self.n_candidate_tag = 0
        self.n_proposition_tag = 0
        self.n_proof_tag = 0
        self.n_proof_empty = 0
        self.n_judged = 0
        # Mutually-exclusive judge-health bucket per row (precedence-assigned):
        # clean / no_proof / http_error / truncated / parse_failed. Sums to n.
        self.judge_cat_counts: Counter[str] = Counter()
        self.n_http = 0
        self.n_parse = 0
        self.n_trunc = 0
        self.correctness_sum = 0
        self.correctness_n = 0
        self.n_correct_1 = 0
        self.impact_sum = 0
        self.impact_n = 0
        # impact_judge_score accumulated ONLY over correct conjectures
        # (correctness_judge_score == 1) -- the "impact among correct" panel.
        self.impact_correct_sum = 0
        self.impact_correct_n = 0
        self.impact_applied_sum = 0.0
        self.impact_applied_n = 0
        self.pre_length_sum = 0.0
        self.pre_length_n = 0
        self.proposer_reward_sum = 0.0
        self.proposer_reward_n = 0
        self.length_penalty_sum = 0.0
        self.length_penalty_n = 0
        self.impact_level_counts: Counter[int] = Counter()
        self.prover_score_sum = 0
        self.prover_score_n = 0
        # QED-Nano rubric judge: mean 0-7 rubric grade (the NON-binary reward). Kept separate from
        # prover_score_sum (which stays the strict 7/7 correctness that difficulty sampling uses).
        self.rubric_points_sum = 0.0
        self.n_rubric = 0
        self.n_prover_1 = 0
        self.n_rubric_7 = 0   # rows with rubric_points==7 (TRUE full marks, for the 7/7-rate panel line)
        # replay-prefix stream split; see __slots__ comment
        self.orig_grade_sum = 0.0
        self.orig_grade_n = 0
        self.orig_pass_sum = 0
        self.orig_pass_n = 0
        self.orig_strict7_n = 0
        self.orig_groups: dict = {}
        self.n_prefix_tagged = 0
        # Q-routed replay lanes; see __slots__ comment
        self.repl_route_full_pass = 0
        self.repl_route_full_n = 0
        self.repl_route_audit_pass = 0
        self.repl_route_audit_n = 0
        self.repl_route_short_n = 0
        self.repl_route_full_strict7 = 0
        self.repl_route_audit_strict7 = 0
        self.repl_group_route = {}
        self.generated_tokens_sum = 0.0
        self.candidate_len_vals: list[float] = []
        self.proof_len_vals: list[float] = []
        # Actor rollout response length (length panels), TOKENS ONLY from the
        # row's `response_length` field. NEVER fall back to `total_tokens` —
        # total_tokens = prompt_length + response_length, so the substitution
        # would silently inflate response-length panels with prompt-inclusive
        # numbers. Rows without `response_length` contribute nothing.
        self.resp_len_vals: list[float] = []
        self.resp_len_scratch: list[float] = []
        self.resp_len_replay: list[float] = []
        self.resp_len_replay_short: list[float] = []
        self.resp_len_replay_full: list[float] = []
        # Injected prefix tokens per replay row, per lane. Feed the effective-cap lines
        # on panel 5b (budget - prefix).
        self.prefix_len_replay: list[float] = []
        self.prefix_len_replay_short: list[float] = []
        self.prefix_len_replay_full: list[float] = []
        # Tokens the policy ACTUALLY generated this row. response_length includes the
        # injected replay prefix, so it conflates "what the model wrote" with "what we
        # pasted in front of it" -- on the g-capped lane at step 211 the median row
        # generated exactly 10000 tokens while its response_length read 22312.
        self.gen_len_vals: list[float] = []
        self.gen_len_short_max: float = 0.0
        # Aligned per-row difficulty draw-weights (c_w; 1.0 on uniform/pre-difficulty rows) for
        # the SNIS-weighted resp-len percentiles. Appended at the SAME site as resp_len_vals so
        # the two lists are index-aligned by construction.
        self.resp_len_wts: list[float] = []
        # Actor rollout total tokens (prompt+response), secondary length context.
        # Prefer `total_tokens`, else compute as `prompt_length + response_length`.
        # Empty when neither is dumped.
        self.tot_tok_vals: list[float] = []
        self.jp_tokens: list[float] = []
        self.jc_tokens: list[float] = []
        self.group_scores: dict[str, list[int]] = defaultdict(list)
        self.group_de: dict[str, list[int]] = defaultdict(list)
        self.n_group_uid_missing = 0
        self.group_attempt: dict[str, list[int]] = defaultdict(list)
        self.group_success: dict[str, list[int]] = defaultdict(list)
        self.group_missing: dict[str, list[int]] = defaultdict(list)
        self.group_failed: dict[str, list[int]] = defaultdict(list)
        self.w: defaultdict[str, float] = defaultdict(float)
        self.w_impact_levels: Counter[int] = Counter()
        self.group_c: dict[str, float] = {}
        self.diff_weighted = False
        self.grp_ref: float = 1.0
        self.group_weighted = False   # any row factor != 1 this step
        self.resp_len_scratch_w: list[float] = []
        self.resp_len_replay_w: list[float] = []
        self.resp_len_replay_short_w: list[float] = []
        self.resp_len_replay_full_w: list[float] = []
        self.gen_len_wts: list[float] = []


class _GlobalAccum:
    """Cross-step diagnostics aggregates, split by mode where it matters."""

    def __init__(self) -> None:
        # candidate-length score histograms, one per mode. Key: bin_idx ->
        # [count, score_sum] over judged rows.
        self.len_bins_proposer = [[0.0, 0.0] for _ in range(len(CANDIDATE_LEN_BIN_EDGES) + 1)]
        self.len_bins_prover = [[0.0, 0.0] for _ in range(len(CANDIDATE_LEN_BIN_EDGES) + 1)]
        # overlong_reward histogram, by mode.
        self.overlong_hist_proposer = [0] * (len(OVERLONG_BIN_EDGES) + 1)
        self.overlong_hist_prover = [0] * (len(OVERLONG_BIN_EDGES) + 1)
        # overlong rollups (cross-mode)
        self.n_overlong_proposer = 0
        self.n_overlong_prover = 0
        # Per-theorem direct-proof tally: theorem_hash -> count of prover
        # (direct-proof) rollouts and how many were judged correct, summed over
        # all parsed TRAIN steps. Attached to proposer sample rows post-parse so
        # each conjecture shows its theorem's k/n direct-proof correctness.
        self.theorem_prover_total: Counter = Counter()
        self.theorem_prover_correct: Counter = Counter()
        # Train rollout rows with vs without actor token fields. Token rows feed
        # the length panels; "char" rows (no token field) are OMITTED and only
        # drive the "rows omitted from length panels" warning.
        self.resp_len_token_rows = 0
        self.resp_len_char_rows = 0
        # Replay-provenance breakdown: prover_additional rows aggregated
        # by (source_run_id, source_step) and by seed_statement_uid, over the
        # whole train run. Emitted as top-level fig_data blocks (not per_step)
        # so panels can render a single table per run. Each entry is a small
        # dict of running sums; the final aggregates are emitted in parse_run.
        # source_run_id × source_step -> {n, correct, overlong, resp_len_sum,
        # resp_len_n, tot_tok_sum, tot_tok_n}
        self.replay_by_source: dict[tuple[str, int], dict[str, float]] = {}
        # seed_statement_uid -> {n, correct, overlong, examples: [first few rows]}
        self.replay_by_seed: dict[str, dict[str, Any]] = {}
        # Per-step contributions to the fixed-bin cumulative histograms above
        # (score-by-length + overlong dist), so an incremental refresh can union
        # per-step deltas and rebuild the cumulative panels EXACTLY without a full
        # re-parse (see merge_fig_data._merge_global_step_contrib). Keyed by parsed
        # step, train and val kept separate. Populated in _aggregate_step by diffing
        # the cumulative fields across the step. Each value is a dict with the same
        # shape the merger sums: len_bins_{proposer,prover} ([[count, score_sum]]),
        # overlong_hist_{proposer,prover} ([int]), n_overlong_{proposer,prover} (int).
        self.step_contrib_train: dict[int, dict] = {}
        self.step_contrib_val: dict[int, dict] = {}


# ---------------------------------------------------------------------------
# Legacy chunk-based uid backfill.
#
# Some older runs wrote train rollout dumps WITHOUT per-row
# `uid`. Without uid, every group-keyed metric silently disappears
# (pass-count distribution, group composition, prover judge_attempts_per_group
# percentiles, etc.). We refuse (input)-only grouping because the same input
# string can appear in two distinct prompt rows in one step file, which would
# silently merge them into a fake size-2N group.
#
# Safe recovery: verl writes all N rollouts of one prompt CONTIGUOUSLY in file
# order. Chunk the file into runs of `rollout_n` rows; validate each chunk has
# homogeneous input, de_filtered, AND mode (proposer/prover); assign synthetic
# uid `legacy_<step>_<chunk_idx>` to passing chunks. Mixed-mode chunks indicate
# a broken assumption — those chunks are dropped, not silently merged.

def _is_scratch_row(row, uid_counts) -> bool:
    """True for a FROM-SCRATCH (statement inflow) row.

    NOT `sp_prefix_len == 0`. That was an exact proxy for "statement row" only while
    every replay row carried a non-empty prefix. Once SP_REPLAY_CUT_GRAIN made a k=0
    cut a legitimate draw, ~27% of REPLAY rows had prefix_len == 0 and were silently
    folded into the from-scratch stream: in one run the stream labelled
    "32 stmt rows" actually held 448 rows -- 32 real ones plus 416 replay k=0 cuts.

    Prefer sp_source_type when the dump carries it. Older dumps do not, so fall back to
    the rollout-count signature, which is exact here: scratch rows are inflow-only
    (SP_SCRATCH_INFLOW_ONLY -> rollout_n=1) so their uid appears EXACTLY ONCE in a step,
    while every replay prompt contributes rollout_n (16) rows under one uid.

    Degrades safely: a run with no prefix tag at all, or with uids missing, classifies
    everything as scratch exactly as the pre-replay parser did.
    """
    st = row.get("sp_source_type")
    if st is not None:
        # "cold_scratch" (cold bootstrap) is a fresh-problem row standing in for a
        # replay slot while the buffer is still empty: it generates from scratch, so it
        # belongs in the from-scratch stream. Unlike "statement" it is PPO-trained and
        # carries the full rollout count, so the rollout-count fallback below would
        # misclassify it -- which is why the tag is checked first.
        return str(st) in ("statement", "cold_scratch")
    pl = row.get("sp_prefix_len")
    if pl is None:
        return True          # pre-replay run: one undifferentiated stream
    try:
        if int(pl) != 0:
            return False     # a real prefix -> unambiguously replay
    except (TypeError, ValueError):
        return True
    uid = row.get("uid")
    if not uid or not uid_counts:
        return True          # cannot disambiguate -> preserve legacy behaviour
    return uid_counts.get(uid, 0) <= 1


def _detect_rollout_n_from_input(rows, default: int = 8) -> int:

    """Heuristic: the leading run-length of identical ``input`` strings is the
    implicit ``rollout_n`` (verl writes all N rollouts of one prompt contiguously).
    Sanity: must be in [4, 64] and divide the row count evenly; else fall back
    to ``default``. Val files (prover-only, rollout_n=16) auto-detect 16
    naturally because every prover prompt has 16 contiguous rollouts."""
    if len(rows) < 2:
        return default
    first_input = rows[0].get("input")
    n = 1
    for r in rows[1:]:
        if r.get("input") == first_input:
            n += 1
        else:
            break
    if 4 <= n <= 64 and len(rows) % n == 0:
        return n
    return default


def _backfill_legacy_uids(rows: list[dict], *, step: int, rollout_n: int) -> dict:
    """Walk ``rows`` in file order; assign synthetic uids to contiguous chunks
    of ``rollout_n`` that pass validation (homogeneous ``input``, ``de_filtered``,
    and ``mode``). Mutates ``rows`` in place: passing chunks get
    ``row["uid"] = f"legacy_{step}_{chunk_idx}"``. Failed chunks leave rows
    ungrouped. Returns provenance counters."""
    chunks_total = (len(rows) + rollout_n - 1) // rollout_n
    chunks_recovered = 0
    drop_size = drop_input = drop_de = drop_mode = 0
    legacy_uid_rows = 0
    chunk_idx = 0
    for start in range(0, len(rows), rollout_n):
        chunk = rows[start : start + rollout_n]
        chunk_idx += 1
        if len(chunk) != rollout_n:
            drop_size += 1
            continue
        if len({r.get("input") for r in chunk}) != 1:
            drop_input += 1
            continue
        des = {(None if r.get("de_filtered") is None else bool(r.get("de_filtered")))
               for r in chunk}
        if len(des) != 1:
            drop_de += 1
            continue
        modes = {_resolve_mode(r) for r in chunk}
        if len(modes) != 1 or "unknown" in modes:
            drop_mode += 1
            continue
        syn = f"legacy_{step}_{chunk_idx}"
        for r in chunk:
            r["uid"] = syn
        chunks_recovered += 1
        legacy_uid_rows += len(chunk)
    return {
        "rollout_n_used": rollout_n,
        "chunks_total": chunks_total,
        "chunks_recovered": chunks_recovered,
        "chunks_dropped_size": drop_size,
        "chunks_dropped_input": drop_input,
        "chunks_dropped_de": drop_de,
        "chunks_dropped_mode": drop_mode,
        "rows_with_legacy_uid": legacy_uid_rows,
        "rows_unrecoverable": len(rows) - legacy_uid_rows,
    }


def _aggregate_step(
    rows_iter: Iterator[tuple[int, dict[str, Any]]],
    *,
    step: int,
    source_file: str,
    examples: dict[str, deque],
    pooled: list[dict[str, Any]],
    global_accum: _GlobalAccum,
    is_val: bool,
    max_group_size_seen: list[int],
    warnings: _Warnings,
    additional_data_sidecar: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Stream one step's rows and return the step's aggregate dict.

    Splits each row into ``proposer`` / ``prover`` / ``unknown`` based on
    ``mode_is_*`` and into ``proposer`` / ``prover_original`` /
    ``prover_additional`` / ``unknown`` for the family-aware replay panels.
    The output dict carries:
      - ``v7__{train,val}__{proposer,prover}__<key>``  (per-mode keys; kept as
         aggregate context for replay runs)
      - ``v7__{train,val}__family__{proposer,prover_original,prover_additional}__<key>``
         (family keys; ``prover_original`` ≡ ``prover`` for non-replay runs)
      - ``v7__{train,val}__source__<sanitized_data_source>__prover__<key>``
         (per-source drilldown for prover_additional rows)
      - ``v7__{train,val}__mode_mix__*``  (the existing 2-mode mix)
      - ``v7__{train,val}__family_mix__*`` (the new 3-family mix for the replay
         "row family mix" panel)
    """
    section = "val" if is_val else "train"
    prefix_root = f"v7__{section}"
    sidecar = additional_data_sidecar or {}

    # Baseline snapshot of the cumulative global histograms so we can record THIS
    # step's contribution (the delta it adds) for exact incremental merges. The
    # only mutation sites for these fields are in the per-row loop below, so the
    # end-of-step delta captures the step's contribution precisely.
    _base_len_p = [list(x) for x in global_accum.len_bins_proposer]
    _base_len_v = [list(x) for x in global_accum.len_bins_prover]
    _base_ov_p = list(global_accum.overlong_hist_proposer)
    _base_ov_v = list(global_accum.overlong_hist_prover)
    _base_nov_p = global_accum.n_overlong_proposer
    _base_nov_v = global_accum.n_overlong_prover

    accums: dict[str, _ModeAccum] = {"proposer": _ModeAccum(), "prover": _ModeAccum()}
    family_accums: dict[str, _ModeAccum] = {
        "proposer": _ModeAccum(),
        "prover_original": _ModeAccum(),
        "prover_additional": _ModeAccum(),
    }
    source_accums: dict[str, _ModeAccum] = {}  # data_source -> accum (for prover_additional drilldown)
    n_unknown = 0
    n_family_unknown = 0
    n_total = 0
    n_replay_provenance = 0

    pooled_this_step = 0

    # Materialize the per-step iterator once so we can (a) detect dumps where
    # every row is missing `uid` (older dumps) and (b) run
    # the chunk-based legacy uid backfill BEFORE the main loop. The backfill
    # mutates row dicts in place; the main loop then sees recovered synthetic
    # uids in `row["uid"]`. Mode-aware: mixed-mode chunks are dropped, not
    # silently merged.
    _materialized = list(rows_iter)
    _row_dicts = [r for _, r in _materialized]
    # Rollout-count signature for the stream split (see _is_scratch_row): scratch rows
    # are inflow-only so their uid appears once; replay uids appear rollout_n times.
    _uid_counts = Counter(r.get("uid") for r in _row_dicts if r.get("uid"))
    # Group-level weighting (dynamic group size): a prompt group that ran to 32
    # completions must not weigh four times a group that stopped at 8. Every row of a
    # multi-row group gets factor grp_ref/|group| (grp_ref = this step's mean multi-row
    # group size), singleton inflow rows keep 1. Fixed-n runs: every factor == 1 exactly,
    # so their fig_data is byte-identical. The factor multiplies the SNIS draw weight c in
    # every per-problem EXPECTATION and in the length-distribution weights; it does NOT
    # touch group_c (the per-group draw weight) because the group-level estimators that
    # use it already weigh groups, not rows.
    _multi_sizes = [c for c in _uid_counts.values() if c > 1]
    _grp_ref = (sum(_multi_sizes) / len(_multi_sizes)) if _multi_sizes else 1.0

    def _group_w(uid) -> float:
        c = _uid_counts.get(uid, 0) if uid else 0
        return (_grp_ref / c) if c > 1 else 1.0
    _legacy_provenance: dict | None = None
    if _row_dicts and all(r.get("uid") is None for r in _row_dicts):
        _legacy_provenance = _backfill_legacy_uids(
            _row_dicts,
            step=step,
            rollout_n=_detect_rollout_n_from_input(_row_dicts, default=8),
        )
    rows_iter = iter(_materialized)

    for line_no, row in rows_iter:
        n_total += 1
        mode = _resolve_mode(row)
        # Validation runs are direct-proof (prover) only; the launcher
        # does not emit proposer validation prompts. If a val row lacks mode
        # fields, treat it as prover so the val dashboard is never empty
        # just because forward-compat fields aren't there.
        if is_val and mode == "unknown":
            mode = "prover"
        if mode == "unknown":
            n_unknown += 1
            continue

        # Theorem identity for the proposer<->prover direct-proof join AND for
        # the additional-data sidecar lookup that classifies prover rows into
        # original / additional families.
        tkey = _theorem_key(row.get("input"))
        thash = _theorem_hash(tkey) if tkey is not None else None

        # Family + data_source classification (replay-aware split).
        data_source = _resolve_data_source(row, thash, sidecar)
        family = _resolve_family(mode, data_source)
        if mode == "proposer" and data_source is None:
            # Proposer side is currently always the original parquet by
            # construction (the replay parquet is prover-only), so default the
            # proposer data_source to ORIGINAL_DATA_SOURCE for tidy
            # source/family accounting.
            data_source = ORIGINAL_DATA_SOURCE
        # Replay provenance only when this row is plausibly an additional one.
        if family == "prover_additional":
            rprov = _extract_replay_provenance(
                row, sidecar_provenance=_sidecar_provenance(thash, sidecar))
        else:
            rprov = {}

        # Replay provenance + seed drilldown aggregates (cross-step).
        # Populated only when this row is prover_additional AND has the replay
        # provenance fields (via the sidecar fallback or the dump). Aggregated
        # by (source_run_id, source_step) and by seed_statement_uid over the
        # whole train run; emitted as top-level fig_data blocks in parse_run.
        if rprov and not is_val:
            src_key = (
                str(rprov["source_run_id"]) if "source_run_id" in rprov else None,
                int(rprov["source_step"]) if rprov.get("source_step") is not None else None,
            )
            if src_key[0] is not None and src_key[1] is not None:
                slot = global_accum.replay_by_source.get(src_key)
                if slot is None:
                    slot = {"n": 0, "correct": 0, "overlong": 0,
                            "resp_len_sum": 0.0, "resp_len_n": 0,
                            "tot_tok_sum": 0.0, "tot_tok_n": 0}
                    global_accum.replay_by_source[src_key] = slot
            else:
                slot = None
            seed_uid = rprov.get("seed_statement_uid")
            if seed_uid:
                sslot = global_accum.replay_by_seed.get(seed_uid)
                if sslot is None:
                    sslot = {"n": 0, "correct": 0, "overlong": 0,
                             "data_source": data_source or "",
                             "source_run_id": rprov.get("source_run_id"),
                             "source_steps": Counter(),
                             "source_completion_indices": Counter()}
                    global_accum.replay_by_seed[seed_uid] = sslot
            else:
                sslot = None
            # The mutations below are postponed to AFTER the core per-row
            # accumulator block so the row's `group_bin` (correctness) and
            # `is_overlong` are computed; this keeps the data flow single-pass.
        if rprov:
            n_replay_provenance += 1

        # Targets to accumulate into.
        targets: list[_ModeAccum] = [accums[mode]]
        if family in family_accums:
            targets.append(family_accums[family])
        else:
            n_family_unknown += 1
        if family == "prover_additional" and data_source:
            sacc = source_accums.get(data_source)
            if sacc is None:
                sacc = _ModeAccum()
                source_accums[data_source] = sacc
            targets.append(sacc)

        # ---- Per-row computations (once per row, then broadcast to targets) ----
        score = _as_float(row.get("score"))
        if is_val:
            # VALIDATION panels report the UNPENALIZED judge fraction: with SP_LENPEN_ENABLE the judge writes score = acc*(1-penalty),
            # and a length-shaped VAL metric would conflate capability with the shaping
            # term. `acc` is the raw points/7 (or binary) fraction; on every run without
            # the penalty acc == score, so all historical val curves are unchanged.
            # TRAIN panels intentionally keep the shaped `score` — that is the reward
            # the optimizer sees. Val context length is untouched.
            _acc = _as_float(row.get("acc"))
            if _acc is not None:
                score = _acc
        if score is None:
            score = 0.0
        # Difficulty-sampling IS correction: the row's draw-time c (uniform / pre-difficulty
        # dumps have no field -> c = 1 and every weighted aggregate equals its raw value).
        c_w = _as_float(row.get("diff_c_raw"))
        if c_w is None or c_w <= 0.0:
            c_w = 1.0
        de = _as_bool(row.get("de_filtered"))
        uid = row.get("uid")
        group_key = str(uid) if uid is not None else ""
        # effective per-row weight = draw weight c x group-level factor (see _group_w)
        g_w = _group_w(uid)
        cw = c_w * g_w
        if g_w != 1.0:
            acc_flag_group_weighted = True
        else:
            acc_flag_group_weighted = False
        # Choose the per-mode group "score bin" we'll histogram for the
        # pass-count panel: proposer uses correctness 0/1; prover uses
        # the binary prover judge. base `score` is NOT a clean 0/1 for
        # proposer (it's pre-length real), so we histogram the binary
        # signal that has a single intelligible interpretation per mode.
        if mode == "proposer":
            corr = _as_int(row.get("correctness_judge_score"))
            if corr is None:
                corr = 0
            group_bin = 1 if corr >= 1 else 0
        else:
            pjs0 = _as_int(row.get("prover_judge_score"))
            if pjs0 is None:
                pjs0 = 1 if score >= 0.5 else 0
            group_bin = 1 if pjs0 >= 1 else 0

        # Tokens-only response length. The per-row response_length field is the
        # ONLY source — never fall back to total_tokens. Reason:
        # total_tokens = prompt_length + response_length, so substituting it
        # would silently plot CONTEXT length on a response-length panel
        # (response panels measure actor response only; total_tokens
        # is secondary context that lives on different panels). Rows without
        # response_length contribute nothing to response-length panels.
        rlen_tok = _as_int(row.get("response_length"))
        # Total tokens: prefer the row field; reconstruct from
        # prompt_length + response_length only when total_tokens is absent
        # (mathematically equivalent — both sides describe the same context
        # measurement).
        tt = _as_int(row.get("total_tokens"))
        if tt is None:
            pl = _as_int(row.get("prompt_length"))
            rl2 = _as_int(row.get("response_length"))
            if pl is not None and rl2 is not None:
                tt = pl + rl2

        ctag = _as_int(row.get("candidate_tag_present")) or 0
        ppt = _as_int(row.get("proposition_tag_present")) or 0
        ptag = _as_int(row.get("proof_tag_present")) or 0
        plen = _as_int(row.get("proof_len_chars")) or 0
        clen = _as_int(row.get("candidate_len_chars")) or 0
        # A row is "judged" iff the reward path actually ran the judge.
        judged = (
            (mode == "proposer" and ppt == 1) or
            (mode == "prover" and ptag == 1 and plen > 0)
        )
        jp_v = _as_int(row.get("judge_prompt_tokens")) if judged else None
        jc_v = _as_int(row.get("judge_completion_tokens")) if judged else None

        is_http = bool(_as_bool(row.get("judge_http_error")))
        is_parse = bool(_as_bool(row.get("judge_parse_failed")))
        is_trunc = bool(_as_bool(row.get("judge_truncated")))

        ov_reward = _as_float(row.get("overlong_reward"))
        # optimized_reward_mean is computed from the pre_kl_post_penalty_reward
        # field emitted by dapo.py when present. No fallback to score+overlong or
        # proposer_reward — those reconstructions are incorrect once the exponential
        # penalty path is active. (The ONE exception, applied at the accumulation site
        # below and guarded to the linear path, is prover_judge_score*(1-length_penalty)
        # for runs that apply a linear correct-only penalty in the reward fn rather than
        # dapo.py, where that product is exactly the post-penalty reward.) Linear-overlong
        # runs never emit this field, in which case opt_val stays None and
        # optimized_reward_mean is simply omitted for the step. Rows where
        # the field is missing but SHOULD be present (i.e. the row carries
        # score and a length_penalty signal so the reward path ran the
        # full shaping pipeline) are counted as missing and surfaced via a
        # one-time warning + a sentinel flag in fig_data["config"].
        opt_val = _as_float(row.get("pre_kl_post_penalty_reward"))
        rp = _as_float(row.get("rubric_points"))   # QED-Nano rubric judge 0-7 grade (None on binary runs)
        is_overlong = bool(_as_bool(row.get("overlong")))
        # Exponential-penalty per-rollout values (absent on linear-penalty
        # runs and rows, in which case the means are
        # naturally omitted).
        exp_pen_factor = _as_float(row.get("exponential_penalty_factor"))
        exp_pen_exceed = _as_float(row.get("exponential_penalty_exceed_tokens"))

        # Mode-specific reward decomposition values (proposer / prover).
        # `length_penalty` is per-rollout in both modes (prover-family
        # rows also carry it whenever the exponential penalty path runs),
        # so it is read at the row level and broadcast to whichever mode the
        # row belongs to. The emit site below gates on n>0 per family.
        lp_v = _as_float(row.get("length_penalty"))
        if mode == "proposer":
            corr_v = _as_int(row.get("correctness_judge_score"))
            imp_v = _as_int(row.get("impact_judge_score"))
            ia_v = _as_float(row.get("impact_applied"))
            pre_v = _as_float(row.get("proposer_pre_length_score"))
            pr_v = _as_float(row.get("proposer_reward"))
            prover_judge_v = None
        else:
            corr_v = None
            imp_v = None
            ia_v = None
            pre_v = None
            pr_v = None
            prover_judge_v = _as_int(row.get("prover_judge_score"))

        # ---- Broadcast to every target accumulator ----
        for acc in targets:
            acc.n += 1
            acc.grp_ref = _grp_ref
            if acc_flag_group_weighted:
                acc.group_weighted = True
            acc.score_sum_generated += score
            if de is True:
                acc.n_filtered += 1
                acc.score_sum_filtered += score
            elif de is False:
                acc.n_trained += 1
                acc.score_sum_trained += score
            else:
                acc.n_de_unknown += 1
            if group_key:
                acc.group_scores[group_key].append(group_bin)
                acc.group_de[group_key].append(1 if de is True else 0)
                acc.group_c[group_key] = c_w  # constant within a group (one problem draw)
            else:
                acc.n_group_uid_missing += 1
            if rlen_tok is not None:
                acc.resp_len_vals.append(float(rlen_tok))
                acc.resp_len_wts.append(cw)
                # stream/route split: panels 5a/5b. sp_prefix_len is 0/absent on
                # non-replay runs, so pre-split runs put everything in "scratch" and the
                # replay lists stay empty -> those keys are simply not emitted.
                _pl_split = row.get("sp_prefix_len")
                # Replay lane INCLUDES k=0 cuts -- they are ordinary cuts drawn from the
                # quantized grid, not statement rows. Split on the real predicate.
                if _pl_split is not None and not _is_scratch_row(row, _uid_counts):
                    acc.resp_len_replay.append(float(rlen_tok))
                    acc.resp_len_replay_w.append(cw)
                    # Prefix tokens ride INSIDE the response but were injected, not
                    # generated, so a replay row's real headroom is budget - prefix.
                    # Panel 5b plots that effective cap; drawing only the flat nominal
                    # budget would overstate the ceiling these rows work against.
                    acc.prefix_len_replay.append(float(_pl_split))
                    if row.get("sp_q_route") == "short":
                        acc.resp_len_replay_short.append(float(rlen_tok))
                        acc.resp_len_replay_short_w.append(cw)
                        acc.prefix_len_replay_short.append(float(_pl_split))
                    else:
                        acc.resp_len_replay_full.append(float(rlen_tok))
                        acc.resp_len_replay_full_w.append(cw)
                        acc.prefix_len_replay_full.append(float(_pl_split))
                else:
                    acc.resp_len_scratch.append(float(rlen_tok))
                    acc.resp_len_scratch_w.append(cw)
                # GENERATED tokens this step (rollout-throughput numerator): the
                # response minus any replay prefix (prefix tokens ride in the
                # response but were injected, not generated — sp_prefix_len is 0
                # or absent on non-replay runs, so this equals response_length).
                _spl_tok = _as_int(row.get("sp_prefix_len")) or 0
                _gen = max(float(rlen_tok) - _spl_tok, 0.0)
                acc.generated_tokens_sum += _gen
                acc.gen_len_vals.append(_gen)
                acc.gen_len_wts.append(cw)
                # The Q short lane bounds GENERATED tokens at g. We cannot read g from the
                # manifest (it is not recorded there), so observe it: the largest generated
                # count on a short-routed row IS g, since 83% of them sit exactly on it.
                if row.get("sp_q_route") == "short" and _gen > acc.gen_len_short_max:
                    acc.gen_len_short_max = _gen
            if tt is not None:
                acc.tot_tok_vals.append(float(tt))
            if ctag:
                acc.n_candidate_tag += 1
            if ppt:
                acc.n_proposition_tag += 1
            if ptag:
                acc.n_proof_tag += 1
                if plen == 0:
                    acc.n_proof_empty += 1
            if judged:
                acc.n_judged += 1
                if clen > 0:
                    acc.candidate_len_vals.append(float(clen))
                if plen > 0:
                    acc.proof_len_vals.append(float(plen))
                if jp_v is not None:
                    acc.jp_tokens.append(float(jp_v))
                if jc_v is not None:
                    acc.jc_tokens.append(float(jc_v))
            if is_http:
                acc.n_http += 1
            if is_parse:
                acc.n_parse += 1
            if is_trunc:
                acc.n_trunc += 1
            # MECE judge-health bucket (precedence: first match wins).
            if not judged:
                acc.judge_cat_counts["no_proof"] += 1
            elif is_http:
                acc.judge_cat_counts["http_error"] += 1
            elif is_trunc:
                acc.judge_cat_counts["truncated"] += 1
            elif is_parse:
                acc.judge_cat_counts["parse_failed"] += 1
            else:
                acc.judge_cat_counts["clean"] += 1
            # Prover-only per-group judge-count sequences (panel 11 per-group
            # percentiles + diagnostics panel 17).
            if mode == "prover" and group_key:
                acc.group_attempt[group_key].append(1 if judged else 0)
                acc.group_success[group_key].append(group_bin)
                acc.group_missing[group_key].append(0 if judged else 1)
                acc.group_failed[group_key].append(
                    1 if (judged and (is_http or is_parse or is_trunc)) else 0
                )
            # Mode-specific reward decomposition.
            if mode == "proposer":
                if corr_v is not None:
                    acc.correctness_sum += corr_v
                    acc.correctness_n += 1
                    if corr_v == 1:
                        acc.n_correct_1 += 1
                if imp_v is not None:
                    acc.impact_sum += imp_v
                    acc.impact_n += 1
                    acc.impact_level_counts[imp_v] += 1
                    if corr_v == 1:
                        acc.impact_correct_sum += imp_v
                        acc.impact_correct_n += 1
                if ia_v is not None:
                    acc.impact_applied_sum += ia_v
                    acc.impact_applied_n += 1
                if pre_v is not None:
                    acc.pre_length_sum += pre_v
                    acc.pre_length_n += 1
                if pr_v is not None:
                    acc.proposer_reward_sum += pr_v
                    acc.proposer_reward_n += 1
            else:
                if prover_judge_v is not None:
                    acc.prover_score_sum += prover_judge_v
                    acc.prover_score_n += 1
                    if prover_judge_v == 1:
                        acc.n_prover_1 += 1
                if rp is not None:
                    acc.rubric_points_sum += rp
                    acc.n_rubric += 1
                    if rp == 7:
                        acc.n_rubric_7 += 1
                # Replay-prefix stream split: from-scratch-only aggregates over
                # sp_prefix_len==0 rows so panel 1 stays backward-comparable on replay runs.
                _spl = _as_int(row.get("sp_prefix_len"))
                if _spl is not None:
                    acc.n_prefix_tagged += 1
                    if _is_scratch_row(row, _uid_counts):
                        if rp is not None:
                            acc.orig_grade_sum += rp / 7.0
                            acc.orig_grade_n += 1
                            if rp == 7:
                                acc.orig_strict7_n += 1
                        if prover_judge_v is not None:
                            acc.orig_pass_sum += prover_judge_v
                            acc.orig_pass_n += 1
                        if group_key:
                            acc.orig_groups[group_key] = (
                                acc.orig_groups.get(group_key, False) or prover_judge_v == 1
                            )
                    else:
                        # Q-routing lanes: replay rows carry sp_q_route
                        # (full = non-ready, judged to the end; audit = random ~1/4 of
                        # ready prefixes run to the full budget; short = ready, capped
                        # at g — cut rows are structurally pjs=0). Missing judge counts
                        # as fail (matches sp_replay._pass). Rows from runs without Q
                        # routing have no route tag, so these counters stay 0 and the estimate is omitted.
                        _rt = row.get("sp_q_route")
                        _is7 = rp is not None and rp == 7
                        if _rt == "full":
                            acc.repl_route_full_n += 1
                            if prover_judge_v == 1:
                                acc.repl_route_full_pass += 1
                            if _is7:
                                acc.repl_route_full_strict7 += 1
                        elif _rt == "audit":
                            acc.repl_route_audit_n += 1
                            if prover_judge_v == 1:
                                acc.repl_route_audit_pass += 1
                            if _is7:
                                acc.repl_route_audit_strict7 += 1
                        elif _rt == "short":
                            acc.repl_route_short_n += 1
                        # Group-weighted TRAINED-lane estimates over ALL route-tagged replay
                        # rows (dynamic group size): the trainer's naive
                        # replay/pass_rate_replay weighs a 32-group four times an 8-group.
                        # Same quantity as that naive key (pjs over replay rows, cut short
                        # rows count as fail), weighted by cw = c x grp_ref/|group|; on a
                        # fixed-n run it equals the naive value.
                        if _rt in ("full", "audit", "short"):
                            acc.w["repl@d"] += cw
                            if prover_judge_v == 1:
                                acc.w["repl_pass@n"] += cw
                            if _is7:
                                acc.w["repl_s7@n"] += cw
                        # group-level any-pass by route (for best@n HT estimate)
                        if _rt in ("full", "audit", "short"):
                            _guid = row.get("uid")
                            if _guid is not None:
                                _gg = acc.repl_group_route.get(_guid)
                                if _gg is None:
                                    _gg = {"rt": _rt, "any": False}
                                    acc.repl_group_route[_guid] = _gg
                                if prover_judge_v == 1:
                                    _gg["any"] = True
            # length_penalty is broadcast for both modes (exponential-penalty
            # prover-family rows also carry it). Linear-overlong prover rows have no
            # length_penalty field, so length_penalty_n stays 0 and the mean
            # is omitted.
            if lp_v is not None:
                acc.length_penalty_sum += lp_v
                acc.length_penalty_n += 1
            if ov_reward is not None:
                acc.overlong_reward_sum += ov_reward
                acc.overlong_reward_n += 1
            # pre_kl_post_penalty_reward is the canonical post-penalty
            # pre-KL scalar. Accumulated when present; otherwise, if the row
            # ran the shaping pipeline (lp_v signals an exponential or linear
            # penalty was computed), count the row as a missing-field row so
            # the parser can surface a hard signal in fig_data.
            # Linear correct-only penalty exception: when the penalty is applied in the
            # reward fn (prover_judge.py) rather than dapo.py, pre_kl_post_penalty_reward is
            # absent, but the post-penalty pre-KL reward is EXACTLY prover_judge_score*(1-lp)
            # for the linear path -- reconstruct it so panel 7's final-reward line renders.
            # Gated on: no canonical field, NOT exponential (exp_pen_factor None -- the
            # reconstruction is only valid for the linear path), and a prover row carrying
            # raw correctness + a length_penalty signal. (score field gives the same value.)
            if opt_val is None and exp_pen_factor is None and lp_v is not None:
                if rp is not None:
                    # Rubric run: the post-penalty pre-KL reward is the graded `score`
                    # (points/7 * (1-lp)), NOT the strict-7 prover_judge_score.
                    opt_val = float(score)
                elif prover_judge_v is not None:
                    opt_val = float(prover_judge_v) * (1.0 - lp_v)
            if opt_val is not None:
                acc.opt_reward_sum += opt_val
                acc.opt_reward_n += 1
            elif lp_v is not None:
                acc.n_pre_kl_post_penalty_missing += 1
                # One-time warning per parse covering all such rows.
                warnings.add(
                    "pre_kl_post_penalty_reward missing on row(s) that "
                    "carry length_penalty; optimized_reward_mean omitted "
                    "and pre_kl_post_penalty_reward_missing counter raised"
                )
            # Exponential-penalty per-rollout aggregation. Mirrors the
            # length_penalty pattern: per-family means via simple sum/n.
            if exp_pen_factor is not None:
                acc.exp_penalty_factor_sum += exp_pen_factor
                acc.exp_penalty_factor_n += 1
            if exp_pen_exceed is not None:
                acc.exp_penalty_exceed_tokens_sum += exp_pen_exceed
                acc.exp_penalty_exceed_tokens_n += 1
            if is_overlong:
                acc.n_overlong += 1

            # ---- Difficulty-sampling weighted mirrors (SNIS numerators/denominators) ----
            # One "@n"/"@d" pair per per-problem EXPECTATION; conditions mirror the raw
            # accumulations above exactly. Counts / percentiles / judge diagnostics are
            # deliberately NOT mirrored: they describe the actual sampled batch (compute
            # actually spent), not the uniform-target population.
            if cw != 1.0:
                acc.diff_weighted = True
            w = acc.w
            w["all@d"] += cw
            w["score_gen@n"] += cw * score
            # Full-marks indicator weighted per problem draw. For the rubric judge that's
            # points==7 (the RAW grade, pre-length-penalty); prover_judge_score is only a
            # points>=pass_min PASS flag, not full marks. For the binary judge (no rubric_points)
            # it's prover_judge_score==1. Alongside, accumulate the SNIS-weighted raw grade
            # (points/7, NO length penalty) so rubric_grade_mean is CORRECTNESS, not the penalized
            # reward — keeps panel-1 mean@n on the same (all@d) denominator as the 7/7 rate.
            if rp is not None:
                w["strict7@n"] += cw * (1.0 if rp == 7 else 0.0)
                w["grade@n"] += cw * (rp / 7.0)
            elif prover_judge_v is not None:
                w["strict7@n"] += cw * (1.0 if prover_judge_v == 1 else 0.0)
            if de is True:
                w["score_fil@n"] += cw * score
                w["score_fil@d"] += cw
            elif de is False:
                w["score_tr@n"] += cw * score
                w["score_tr@d"] += cw
            if is_overlong:
                w["overlong@n"] += cw
            if ov_reward is not None:
                w["ovr@n"] += cw * ov_reward
                w["ovr@d"] += cw
            if opt_val is not None:
                w["opt@n"] += cw * opt_val
                w["opt@d"] += cw
            if exp_pen_factor is not None:
                w["expf@n"] += cw * exp_pen_factor
                w["expf@d"] += cw
            if exp_pen_exceed is not None:
                w["expe@n"] += cw * exp_pen_exceed
                w["expe@d"] += cw
            if lp_v is not None:
                w["lp@n"] += cw * lp_v
                w["lp@d"] += cw
            if ctag:
                w["ctag@n"] += cw
            if ppt:
                w["ppt@n"] += cw
            if ptag:
                w["ptag@n"] += cw
                if plen == 0:
                    w["pempty@n"] += cw
            if rlen_tok is not None:
                w["rlen@n"] += cw * float(rlen_tok)
                w["rlen@d"] += cw
            if tt is not None:
                w["ttok@n"] += cw * float(tt)
                w["ttok@d"] += cw
            if judged:
                if clen > 0:
                    w["clen@n"] += cw * float(clen)
                    w["clen@d"] += cw
                if plen > 0:
                    w["plen@n"] += cw * float(plen)
                    w["plen@d"] += cw
            if mode == "proposer":
                if corr_v is not None:
                    w["corr@n"] += cw * corr_v
                    w["corr@d"] += cw
                    if corr_v == 1:
                        w["corr1@n"] += cw
                if imp_v is not None:
                    w["imp@n"] += cw * imp_v
                    w["imp@d"] += cw
                    acc.w_impact_levels[imp_v] += cw
                    if corr_v == 1:
                        w["impc@n"] += cw * imp_v
                        w["impc@d"] += cw
                if ia_v is not None:
                    w["ia@n"] += cw * ia_v
                    w["ia@d"] += cw
                if pre_v is not None:
                    w["pre@n"] += cw * pre_v
                    w["pre@d"] += cw
                if pr_v is not None:
                    w["pr@n"] += cw * pr_v
                    w["pr@d"] += cw
            else:
                if prover_judge_v is not None:
                    w["pjs@n"] += cw * prover_judge_v
                    w["pjs@d"] += cw
                    if prover_judge_v == 1:
                        w["pjs1@n"] += cw

        # ---- Replay drilldown aggregates ----
        # Populated only for prover_additional rows with provenance (slot/sslot
        # were created above when the provenance is present). group_bin is the
        # row's binary correctness; is_overlong is set above.
        if family == "prover_additional" and not is_val:
            if slot is not None:
                slot["n"] += 1
                slot["correct"] += group_bin
                if is_overlong:
                    slot["overlong"] += 1
                if rlen_tok is not None:
                    slot["resp_len_sum"] += float(rlen_tok)
                    slot["resp_len_n"] += 1
                if tt is not None:
                    slot["tot_tok_sum"] += float(tt)
                    slot["tot_tok_n"] += 1
            if sslot is not None:
                sslot["n"] += 1
                sslot["correct"] += group_bin
                if is_overlong:
                    sslot["overlong"] += 1
                src_step = rprov.get("source_step")
                src_ci = rprov.get("source_completion_index")
                if src_step is not None:
                    sslot["source_steps"][int(src_step)] += 1
                if src_ci is not None:
                    sslot["source_completion_indices"][int(src_ci)] += 1

        # ---- Global side effects (once per row, not per-acc) ----
        # Direct-proof tally: each TRAIN prover rollout is one direct proof of
        # its theorem; `group_bin` is its binary correctness. Validation prover
        # rows are a different (IMOProofBench) problem set, so they are excluded.
        if mode == "prover" and not is_val and thash is not None:
            global_accum.theorem_prover_total[thash] += 1
            global_accum.theorem_prover_correct[thash] += group_bin
        # Tokens-only resp_len omission counters.
        if rlen_tok is not None:
            if not is_val:
                global_accum.resp_len_token_rows += 1
        elif not is_val:
            global_accum.resp_len_char_rows += 1
        # Score-by-length histogram (judged rows only).
        if judged:
            len_for_bin = clen if mode == "proposer" else plen
            if len_for_bin > 0:
                bins = (
                    global_accum.len_bins_proposer if mode == "proposer"
                    else global_accum.len_bins_prover
                )
                b = _candidate_len_bin(len_for_bin)
                bins[b][0] += 1.0
                bins[b][1] += score
        # Overlong reward histogram + cross-mode overlong rollups.
        if ov_reward is not None:
            hist = (
                global_accum.overlong_hist_proposer if mode == "proposer"
                else global_accum.overlong_hist_prover
            )
            hist[_overlong_bin(ov_reward)] += 1
        if is_overlong:
            if mode == "proposer":
                global_accum.n_overlong_proposer += 1
            else:
                global_accum.n_overlong_prover += 1

        # ---- Slim sampling: mode-aware diagnostic categories + pooled viewer ----
        # Full text rides on the slim row as a transient ``_full`` field so it is
        # bounded by the (small) deque/pooled caps; parse_run harvests it into
        # the lazy full-sample JSONL and strips it.
        def _build_slim() -> dict[str, Any]:
            post_think = _extract_post_think(row.get("output") or "")
            slim = _slim_example(
                row, step, source_file, line_no,
                mode=mode, post_think=post_think,
                theorem_hash=thash, split=section,
                family=family if family in family_accums else None,
                data_source=data_source,
                replay_provenance=rprov or None,
            )
            slim["_full"] = _full_record(
                row, section, source_file, line_no, mode, post_think)
            return slim

        category = _pick_example_category(row, mode, score)
        if category is not None:
            cat_key = f"{mode}__{category}"
            if cat_key in examples:
                examples[cat_key].append(_build_slim())
        if pooled_this_step < MAX_POOLED_PER_STEP and len(pooled) < MAX_POOLED_TOTAL:
            pooled.append(_build_slim())
            pooled_this_step += 1

    if n_unknown:
        warnings.add(
            f"{section} step {step}: {n_unknown} row(s) had no mode_is_proposer / "
            f"mode_is_prover flag — counted as 'unknown' and excluded from mode-split "
            f"panels. Likely a non-v7 reward path or a logging gap."
        )
    if n_family_unknown:
        warnings.add(
            f"{section} step {step}: {n_family_unknown} row(s) had a known mode "
            f"but no resolved family — they contribute to the per-mode block but "
            f"not to the per-family block. Should not happen unless _resolve_family "
            f"changes."
        )

    if n_total == 0:
        return {}

    out: dict[str, Any] = {}

    def put(key: str, value: Any) -> None:
        out[f"{prefix_root}__{key}"] = value

    # Legacy uid backfill provenance (per-step) — emitted only when the
    # chunk-fallback ran (uniformly missing per-row uid in the dump). Makes
    # recovered runs honest: dashboards can see which steps used the fallback
    # and how much was recovered vs dropped.
    if _legacy_provenance is not None:
        lp = _legacy_provenance
        put("legacy_chunk_uid_used", 1.0)
        put("legacy_chunk_rollout_n", float(lp["rollout_n_used"]))
        put("legacy_chunks_total", float(lp["chunks_total"]))
        put("legacy_chunks_recovered", float(lp["chunks_recovered"]))
        if lp["chunks_dropped_size"]:
            put("legacy_chunks_dropped_size", float(lp["chunks_dropped_size"]))
        if lp["chunks_dropped_input"]:
            put("legacy_chunks_dropped_input", float(lp["chunks_dropped_input"]))
        if lp["chunks_dropped_de"]:
            put("legacy_chunks_dropped_de", float(lp["chunks_dropped_de"]))
        if lp["chunks_dropped_mode"]:
            put("legacy_chunks_dropped_mode", float(lp["chunks_dropped_mode"]))
        put("legacy_rows_with_uid", float(lp["rows_with_legacy_uid"]))
        if lp["rows_unrecoverable"]:
            put("legacy_rows_unrecoverable", float(lp["rows_unrecoverable"]))

    # 2-mode mix block (existing).
    n_prop = accums["proposer"].n
    n_prov = accums["prover"].n
    n_known = n_prop + n_prov
    put("mode_mix__total_generated", float(n_total))
    put("mode_mix__proposer_generated", float(n_prop))
    put("mode_mix__prover_generated", float(n_prov))
    put("mode_mix__unknown_generated", float(n_unknown))
    if n_known > 0:
        put("mode_mix__proposer_generated_fraction", n_prop / n_known)
        put("mode_mix__prover_generated_fraction", n_prov / n_known)
    nt_prop = accums["proposer"].n_trained
    nt_prov = accums["prover"].n_trained
    nt_known = nt_prop + nt_prov
    put("mode_mix__proposer_trained", float(nt_prop))
    put("mode_mix__prover_trained", float(nt_prov))
    if nt_known > 0:
        put("mode_mix__proposer_trained_fraction", nt_prop / nt_known)
        put("mode_mix__prover_trained_fraction", nt_prov / nt_known)

    # 3-family mix block (new; replay "row family mix" panel input).
    fam_counts = {f: facc.n for f, facc in family_accums.items()}
    fam_total = sum(fam_counts.values())
    put("family_mix__total_generated", float(fam_total))
    for fam, c in fam_counts.items():
        put(f"family_mix__{fam}__generated", float(c))
        if fam_total > 0:
            put(f"family_mix__{fam}__generated_fraction", c / fam_total)
    fam_trained = {f: facc.n_trained for f, facc in family_accums.items()}
    fam_trained_total = sum(fam_trained.values())
    for fam, c in fam_trained.items():
        put(f"family_mix__{fam}__trained", float(c))
        if fam_trained_total > 0:
            put(f"family_mix__{fam}__trained_fraction", c / fam_trained_total)

    # Per-mode aggregates (existing namespace, kept for backwards compat AND
    # as the all-prover aggregate context for replay runs).
    for mode_name, acc in accums.items():
        if acc.n == 0:
            continue
        _emit_mode_block(put, mode_name, acc, max_group_size_seen=max_group_size_seen)

    # Per-family aggregates (new namespace; primary rendering for replay runs).
    family_to_emit_mode = {
        "proposer": "proposer",
        "prover_original": "prover",
        "prover_additional": "prover",
    }
    for fam, facc in family_accums.items():
        if facc.n == 0:
            continue
        _emit_mode_block(
            put, family_to_emit_mode[fam], facc,
            max_group_size_seen=max_group_size_seen,
            prefix=f"family__{fam}",
        )

    # Per-source aggregates for prover_additional rows (replay drilldown).
    for ds, sacc in source_accums.items():
        if sacc.n == 0:
            continue
        _emit_mode_block(
            put, "prover", sacc,
            max_group_size_seen=max_group_size_seen,
            prefix=f"source__{_sanitize_key(ds)}__prover",
        )

    # Record this step's contribution to the cumulative global histograms (delta
    # vs the baseline snapshot). merge_fig_data sums these across steps to rebuild
    # the cumulative panels exactly on an incremental refresh. Lists are fixed-size
    # (never resized), so zip pairs baseline and current element-for-element.
    _contrib = {
        "len_bins_proposer": [[cur[0] - base[0], cur[1] - base[1]]
                              for cur, base in zip(global_accum.len_bins_proposer, _base_len_p)],
        "len_bins_prover": [[cur[0] - base[0], cur[1] - base[1]]
                            for cur, base in zip(global_accum.len_bins_prover, _base_len_v)],
        "overlong_hist_proposer": [cur - base
                                   for cur, base in zip(global_accum.overlong_hist_proposer, _base_ov_p)],
        "overlong_hist_prover": [cur - base
                                 for cur, base in zip(global_accum.overlong_hist_prover, _base_ov_v)],
        "n_overlong_proposer": global_accum.n_overlong_proposer - _base_nov_p,
        "n_overlong_prover": global_accum.n_overlong_prover - _base_nov_v,
    }
    _store = global_accum.step_contrib_val if is_val else global_accum.step_contrib_train
    _prev = _store.get(step)
    if _prev is None:
        _store[step] = _contrib
    else:
        # Same step split across multiple files: accumulate into the existing entry.
        for k in ("len_bins_proposer", "len_bins_prover"):
            for i, pair in enumerate(_contrib[k]):
                _prev[k][i][0] += pair[0]
                _prev[k][i][1] += pair[1]
        for k in ("overlong_hist_proposer", "overlong_hist_prover"):
            for i, v in enumerate(_contrib[k]):
                _prev[k][i] += v
        _prev["n_overlong_proposer"] += _contrib["n_overlong_proposer"]
        _prev["n_overlong_prover"] += _contrib["n_overlong_prover"]

    return out


def _pick_example_category(
    row: dict[str, Any], mode: str, score: float,
) -> str | None:
    """Return the diagnostic example category for one row, or None.

    Categories are mode-aware: proposer rows can be classified by
    correctness+impact (correct_high_impact, correct_low_impact,
    incorrect_wellformed, malformed_no_proposition), and prover rows by
    binary outcome (correct, incorrect_wellformed, missing_proof). Judge
    failures are tagged the same way for both.
    """
    if _as_bool(row.get("judge_http_error")):
        return "judge_http_error"
    if _as_bool(row.get("judge_truncated")):
        return "judge_truncated"
    if _as_bool(row.get("judge_parse_failed")):
        return "judge_parse_failed"
    if mode == "proposer":
        ppt = _as_int(row.get("proposition_tag_present")) or 0
        if ppt == 0:
            return "missing_proposition"
        corr = _as_int(row.get("correctness_judge_score")) or 0
        imp = _as_int(row.get("impact_judge_score")) or 0
        if corr == 1 and imp >= 2:
            return "correct_high_impact"
        if corr == 1 and imp < 2:
            return "correct_low_impact"
        if corr == 0:
            return "incorrect_wellformed"
        return None
    # prover
    ptag = _as_int(row.get("proof_tag_present")) or 0
    if ptag == 0:
        return "missing_proof"
    pjs = _as_int(row.get("prover_judge_score"))
    if pjs is None:
        pjs = 1 if score >= 0.5 else 0
    if pjs == 1:
        return "correct"
    return "incorrect_wellformed"


def _emit_mode_block(
    put, mode: str, acc: _ModeAccum, *, max_group_size_seen: list[int],
    prefix: str | None = None,
) -> None:
    """Flush one mode's per-step counters into the ``prefix``__<metric> namespace.

    ``mode`` controls the *logic* branches (proposer-only vs prover-only fields).
    ``prefix`` controls the emitted *key prefix*; defaults to ``mode`` so the
    classic per-mode block stays ``proposer__*`` / ``prover__*``. Pass an
    explicit prefix to emit family / source blocks (e.g.
    ``family__prover_original``, ``source__<data_source>__prover``) using the same
    metric layout.
    """
    pre = prefix if prefix is not None else mode

    def m(k: str, v: Any) -> None:
        put(f"{pre}__{k}", v)

    # Difficulty-sampling emission rule: per-problem
    # EXPECTATIONS are emitted from the SNIS side-accumulator (with c ≡ 1 these equal
    # the raw values exactly, so uniform/pre-difficulty runs are bit-identical); on
    # difficulty runs (any c != 1 seen) the raw value is ALSO emitted as <key>_raw.
    w = acc.w

    def wm(key: str, num: str, den: str, raw: float) -> None:
        d_ = w.get(den, 0.0)
        m(key, (w[num] / d_) if d_ > 0 else raw)
        if acc.diff_weighted:
            m(f"{key}_raw", raw)

    m("rows_generated", float(acc.n))
    if acc.group_weighted:
        # dynamic group size: every multi-row group was re-weighted to this many rows
        m("group_weight_ref", float(acc.grp_ref))
    m("rows_trained", float(acc.n_trained))
    m("rows_filtered", float(acc.n_filtered))
    if acc.n_de_unknown:
        m("rows_de_unknown", float(acc.n_de_unknown))

    wm("score_mean_generated", "score_gen@n", "all@d", acc.score_sum_generated / acc.n)
    if acc.n_trained:
        wm("score_mean_trained", "score_tr@n", "score_tr@d", acc.score_sum_trained / acc.n_trained)
    if acc.n_filtered:
        wm("score_mean_filtered", "score_fil@n", "score_fil@d", acc.score_sum_filtered / acc.n_filtered)
    if (acc.n_trained + acc.n_filtered) > 0:
        _defil_d = w.get("score_tr@d", 0.0) + w.get("score_fil@d", 0.0)
        m("de_filtered_fraction",
          (w.get("score_fil@d", 0.0) / _defil_d) if _defil_d > 0
          else acc.n_filtered / (acc.n_trained + acc.n_filtered))
        if acc.diff_weighted:
            m("de_filtered_fraction_raw", acc.n_filtered / (acc.n_trained + acc.n_filtered))

    if acc.overlong_reward_n:
        wm("overlong_reward_mean", "ovr@n", "ovr@d", acc.overlong_reward_sum / acc.overlong_reward_n)
    wm("overlong_rate", "overlong@n", "all@d", acc.n_overlong / acc.n)
    if acc.opt_reward_n:
        wm("optimized_reward_mean", "opt@n", "opt@d", acc.opt_reward_sum / acc.opt_reward_n)
    # Count of rows that SHOULD have carried pre_kl_post_penalty_reward
    # (i.e. they ran the shaping pipeline) but did not. Surfaced only when
    # nonzero so linear-overlong runs stay clean.
    if acc.n_pre_kl_post_penalty_missing:
        m("pre_kl_post_penalty_reward_missing", float(acc.n_pre_kl_post_penalty_missing))
    # Exponential-penalty per-family means. Emitted only when at least
    # one row in the family carries the field (so linear-penalty
    # runs naturally omit these).
    if acc.exp_penalty_factor_n:
        wm("exponential_penalty_factor_mean", "expf@n", "expf@d",
           acc.exp_penalty_factor_sum / acc.exp_penalty_factor_n)
    if acc.exp_penalty_exceed_tokens_n:
        wm("exponential_penalty_exceed_tokens_mean", "expe@n", "expe@d",
           acc.exp_penalty_exceed_tokens_sum / acc.exp_penalty_exceed_tokens_n)

    _alld = w.get("all@d", 0.0)
    wm("candidate_tag_present_rate", "ctag@n", "all@d", acc.n_candidate_tag / acc.n)
    wm("proposition_tag_present_rate", "ppt@n", "all@d", acc.n_proposition_tag / acc.n)
    wm("proof_tag_present_rate", "ptag@n", "all@d", acc.n_proof_tag / acc.n)
    m("proof_nonempty_rate",
      ((w["ptag@n"] - w["pempty@n"]) / _alld) if _alld > 0
      else (acc.n_proof_tag - acc.n_proof_empty) / acc.n)
    m("proof_missing_rate",
      ((_alld - w["ptag@n"]) / _alld) if _alld > 0 else (acc.n - acc.n_proof_tag) / acc.n)
    wm("proof_empty_rate", "pempty@n", "all@d", acc.n_proof_empty / acc.n)
    if acc.diff_weighted:
        m("proof_nonempty_rate_raw", (acc.n_proof_tag - acc.n_proof_empty) / acc.n)
        m("proof_missing_rate_raw", (acc.n - acc.n_proof_tag) / acc.n)
    if mode == "proposer":
        # Also surface "both tags present" rate, which is the more useful
        # health signal for proposer (a proof without a proposition is
        # judged as 0 correctness by short-circuit).
        wm("both_tags_present_rate", "ctag@n", "all@d",
           acc.n_candidate_tag / acc.n)  # candidate_tag == prop+proof

    m("judge_rows", float(acc.n_judged))
    if acc.n_judged:
        m("judge_http_error_rate", acc.n_http / acc.n_judged)
        m("judge_parse_failed_rate", acc.n_parse / acc.n_judged)
        m("judge_truncated_rate", acc.n_trunc / acc.n_judged)
    m("judge_http_error_count", float(acc.n_http))
    m("judge_parse_failed_count", float(acc.n_parse))
    m("judge_truncated_count", float(acc.n_trunc))
    # Mutually-exclusive judge-health buckets (precedence-assigned); these five
    # partition every generated row, so sum == rows_generated. Use these for a
    # dedup'd unhealthy rate: 1 - clean/n  (the raw *_count fields above overlap).
    for _cat in ("clean", "no_proof", "http_error", "truncated", "parse_failed"):
        m(f"judge_cat__{_cat}", float(acc.judge_cat_counts.get(_cat, 0)))

    if acc.candidate_len_vals:
        acc.candidate_len_vals.sort()
        wm("candidate_len_mean", "clen@n", "clen@d", statistics.fmean(acc.candidate_len_vals))
        for q in (50, 90, 99):
            m(f"candidate_len_p{q}", _percentile_from_sorted(acc.candidate_len_vals, q))
    if acc.proof_len_vals:
        acc.proof_len_vals.sort()
        wm("proof_len_mean", "plen@n", "plen@d", statistics.fmean(acc.proof_len_vals))
        # Extended percentile set (50/60/70/80/90/99) — same rationale as resp_len.
        # Panel 5 + diagnostics panel 10 read the body of the distribution.
        for q in (50, 60, 70, 80, 90, 99):
            m(f"proof_len_p{q}", _percentile_from_sorted(acc.proof_len_vals, q))
    if acc.resp_len_vals:
        # sort values TOGETHER with their draw-weights so the weighted percentiles stay aligned
        _rl_pairs = sorted(zip(acc.resp_len_vals, acc.resp_len_wts)) if (
            len(acc.resp_len_wts) == len(acc.resp_len_vals)
        ) else [(v, 1.0) for v in sorted(acc.resp_len_vals)]
        acc.resp_len_vals = [p[0] for p in _rl_pairs]
        _rl_wts = [p[1] for p in _rl_pairs]
        wm("resp_len_mean", "rlen@n", "rlen@d", statistics.fmean(acc.resp_len_vals))
        # Full decile-ish percentile set (10/20/30/40/50/60/70/80/90/99): the lower
        # deciles (p10-p40) added so panel 5 shows the LOWER body of the distribution
        # (short-response mass), not just the median and upper tail. Older fig_data
        # lacks some keys; the renderer skips missing ones cleanly via `has()`, so this
        # stays back-compatible (a FULL re-parse backfills the new percentiles).
        for q in (10, 20, 30, 40, 50, 60, 70, 80, 90, 99):
            m(f"resp_len_p{q}", _percentile_from_sorted(acc.resp_len_vals, q))
            # SNIS-weighted counterpart: quantile of the UNIFORM-target length distribution
            # (difficulty-sampling draw-weights c as masses; == raw on uniform runs). Panel 5
            # prefers this family when present, matching the SNIS-weighted mean line.
            m(f"resp_len_wtd_p{q}", _weighted_percentile_from_sorted(acc.resp_len_vals, _rl_wts, q))
    # Stream/route-split response lengths (panels 5a/5b). Same percentile family
    # per stream; keys omitted entirely when a stream has no rows (pre-split runs emit
    # only the scratch family, which duplicates the pooled one — harmless and unread).
    for _sfx, _vals, _wts in (("scratch", acc.resp_len_scratch, acc.resp_len_scratch_w),
                              ("replay", acc.resp_len_replay, acc.resp_len_replay_w),
                              ("replay_short", acc.resp_len_replay_short, acc.resp_len_replay_short_w),
                              ("replay_full", acc.resp_len_replay_full, acc.resp_len_replay_full_w)):
        if not _vals:
            continue
        # Group-level (and draw-) weighted: sort values WITH their weights. On a fixed-n
        # run every weight is 1 and these equal the plain mean / percentiles exactly.
        _pairs = sorted(zip(_vals, _wts)) if len(_wts) == len(_vals) else [(v, 1.0) for v in sorted(_vals)]
        _vals[:] = [p[0] for p in _pairs]
        _w = [p[1] for p in _pairs]
        _sw = sum(_w)
        m(f"resp_len_{_sfx}_mean", (sum(v * w_ for v, w_ in _pairs) / _sw) if _sw > 0 else statistics.fmean(_vals))
        m(f"resp_len_{_sfx}_n", float(len(_vals)))
        # full decile set + p99, matching the pooled panel-5 family so 5a/5b show the
        # body of each stream's distribution, not just median/tail.
        for q in (10, 20, 30, 40, 50, 60, 70, 80, 90, 99):
            m(f"resp_len_{_sfx}_p{q}", _weighted_percentile_from_sorted(_vals, _w, q))
    # Injected-prefix stats per replay lane. Panel 5b turns these into the EFFECTIVE cap
    # (budget - prefix): a replay row's response contains the prefix, so the ceiling it can
    # actually reach is lower than the nominal budget, and by a step-varying amount.
    # Generated-token distribution over ALL completions (panel 5b). This is the quantity
    # the Q budget g actually governs and the only length signal that reflects the policy
    # rather than our prefix sampling.
    if acc.gen_len_vals:
        # group-level weighted (see _group_w); identical to the plain stats on fixed-n runs
        _gp = sorted(zip(acc.gen_len_vals, acc.gen_len_wts)) if len(acc.gen_len_wts) == len(acc.gen_len_vals) \
            else [(v, 1.0) for v in sorted(acc.gen_len_vals)]
        acc.gen_len_vals = [p[0] for p in _gp]
        _gw = [p[1] for p in _gp]
        _gsw = sum(_gw)
        m("gen_len_mean", (sum(v * w_ for v, w_ in _gp) / _gsw) if _gsw > 0 else statistics.fmean(acc.gen_len_vals))
        m("gen_len_n", float(len(acc.gen_len_vals)))
        for q in (10, 20, 30, 40, 50, 60, 70, 80, 90, 99):
            m(f"gen_len_p{q}", _weighted_percentile_from_sorted(acc.gen_len_vals, _gw, q))
        if acc.gen_len_short_max > 0:
            m("gen_len_short_cap_observed", acc.gen_len_short_max)
    for _sfx, _vals in (("replay", acc.prefix_len_replay),
                        ("replay_short", acc.prefix_len_replay_short),
                        ("replay_full", acc.prefix_len_replay_full)):
        if not _vals:
            continue
        m(f"prefix_tokens_{_sfx}_mean", statistics.fmean(_vals))
        m(f"prefix_tokens_{_sfx}_max", max(_vals))
    if acc.tot_tok_vals:
        acc.tot_tok_vals.sort()
        wm("tot_tok_mean", "ttok@n", "ttok@d", statistics.fmean(acc.tot_tok_vals))
        for q in (50, 90, 99):
            m(f"tot_tok_p{q}", _percentile_from_sorted(acc.tot_tok_vals, q))
    if acc.jp_tokens:
        acc.jp_tokens.sort()
        for q in (50, 90, 99, 100):  # p100 == max (judge input-length tail)
            m(f"judge_prompt_tokens_p{q}", _percentile_from_sorted(acc.jp_tokens, q))
    if acc.jc_tokens:
        acc.jc_tokens.sort()
        for q in (50, 90, 99, 100):  # p100 == max (judge rollout-length tail vs cap)
            m(f"judge_completion_tokens_p{q}", _percentile_from_sorted(acc.jc_tokens, q))
        # Per-step total judge completion tokens (used by the throughput panel
        # to compute judge tokens/sec without re-iterating raw rollout rows).
        m("judge_completion_tokens_sum", float(sum(acc.jc_tokens)))

    # Proposer reward decomposition.
    if mode == "proposer":
        if acc.correctness_n:
            wm("correctness_judge_score_mean", "corr@n", "corr@d",
               acc.correctness_sum / acc.correctness_n)
            wm("correctness_judge_pass_rate", "corr1@n", "corr@d",
               acc.n_correct_1 / acc.correctness_n)
        if acc.impact_n:
            wm("impact_judge_score_mean", "imp@n", "imp@d", acc.impact_sum / acc.impact_n)
            _impd = w.get("imp@d", 0.0)
            for level in IMPACT_LEVELS:
                # counts stay raw (actual work); fractions are population expectations
                m(f"impact_level_count__l{level}", float(acc.impact_level_counts.get(level, 0)))
                _rawfrac = acc.impact_level_counts.get(level, 0) / acc.impact_n
                m(f"impact_level_fraction__l{level}",
                  (acc.w_impact_levels.get(level, 0.0) / _impd) if _impd > 0 else _rawfrac)
                if acc.diff_weighted:
                    m(f"impact_level_fraction__l{level}_raw", _rawfrac)
        # Mean impact among CORRECT conjectures only. Emitted only when there is
        # at least one correct proposer row this step, so the panel renders N/A
        # (rather than substituting the all-row impact mean) for empty steps.
        if acc.impact_correct_n:
            wm("impact_judge_score_correct_mean", "impc@n", "impc@d",
               acc.impact_correct_sum / acc.impact_correct_n)
        if acc.impact_applied_n:
            wm("impact_applied_mean", "ia@n", "ia@d", acc.impact_applied_sum / acc.impact_applied_n)
        if acc.pre_length_n:
            wm("proposer_pre_length_score_mean", "pre@n", "pre@d",
               acc.pre_length_sum / acc.pre_length_n)
        if acc.proposer_reward_n:
            wm("proposer_reward_mean", "pr@n", "pr@d",
               acc.proposer_reward_sum / acc.proposer_reward_n)
    else:
        if acc.prover_score_n:
            wm("prover_judge_score_mean", "pjs@n", "pjs@d",
               acc.prover_score_sum / acc.prover_score_n)
            wm("prover_judge_pass_rate", "pjs1@n", "pjs@d",
               acc.n_prover_1 / acc.prover_score_n)
        # QED-Nano rubric judge: the NON-binary reward the dashboard should plot. rubric_grade_mean
        # (points/7 in [0,1]) is prepended to the reward/correctness key lists in the renderer so it
        # wins over the strict-7 prover_judge_score_mean; rubric_points_mean is the raw 0-7 grade;
        # prover_strict7_rate preserves the "fraction fully correct" signal for its own panel.
        if acc.n_rubric:
            _raw_grade = (acc.rubric_points_sum / acc.n_rubric) / 7.0
            m("rubric_points_mean", acc.rubric_points_sum / acc.n_rubric)   # raw mean 0-7 (reference)
            # SNIS-CORRECTED grade for the page-1 reward panels, so `correctness` matches the
            # (also-corrected) optimized_reward_mean instead of deviating once difficulty sampling
            # reweights the batch. grade == score == points/7 (length penalty off), so the corrected
            # grade IS the corrected score mean (w[score_gen@n]/w[all@d]). Raw grade -> _raw (page 3).
            _d = w.get("all@d", 0.0)
            # rubric_grade_mean = CORRECTNESS (points/7, NO length penalty), SNIS-corrected —
            # NOT the penalized score. The penalized final reward is optimized_reward_mean /
            # score_mean_generated (shown in the reward-components panel). Because panel-1's mean@n
            # and its 7/7 rate now share the all@d denominator and the same (unpenalized) grade,
            # mean@n >= 7/7 rate holds again.
            m("rubric_grade_mean", (w.get("grade@n", 0.0) / _d) if _d > 0 else _raw_grade)
            if acc.diff_weighted:
                m("rubric_grade_mean_raw", _raw_grade)
            if acc.prover_score_n:
                # SNIS-corrected full-marks rate (Σ c·1[7] / Σ c), matching rubric_grade_mean's
                # correction; raw batch fraction -> prover_strict7_rate_raw (page 3) on diff runs.
                wm("prover_strict7_rate", "strict7@n", "all@d",
                   (acc.n_rubric_7 / acc.n_rubric) if acc.n_rubric else (acc.n_prover_1 / acc.prover_score_n))
        # Replay-prefix stream split: from-scratch-only correctness in panel-1
        # units. Emitted ONLY when this step's dump rows carry the sp_prefix_len tag
        # (trainer-stamped, or backfilled by the tagging scan) — pre-replay runs and
        # untagged steps emit nothing, keeping their keysets byte-identical. These are
        # plain (unweighted) means: difficulty sampling is off on replay runs by design.
        if acc.n_prefix_tagged:
            if acc.orig_grade_n:
                m("orig_stream_grade_mean", acc.orig_grade_sum / acc.orig_grade_n)
                m("orig_stream_strict7_rate", acc.orig_strict7_n / acc.orig_grade_n)
            if acc.orig_pass_n:
                m("orig_stream_pass_mean", acc.orig_pass_sum / acc.orig_pass_n)
            if acc.orig_groups:
                m("orig_stream_best_at_n",
                  sum(1 for v in acc.orig_groups.values() if v) / len(acc.orig_groups))
            # Audit-reweighted full-suffix pass estimate (Q routing): non-ready
            # (route=full) replay rows keep their judged outcome; ready rows are
            # represented by the audit lane alone (the random ~1/4 of ready prefixes run
            # to the full budget), each carrying weight n_ready/n_audit (~4x, Horvitz-
            # Thompson). Cut short-lane rows (structurally pjs=0 on truncated text) and
            # naturally-finished short rows (a biased, early-terminating subset) are both
            # excluded -> an unbiased estimate of the replay-stream pass rate had every
            # row run its full suffix. Key omitted when there is no audit lane.
            # Group-weighted trained-lane estimates (see accumulation): the renderer's
            # panel-1 trained-lane series prefers the audit-lane HT estimators, then these,
            # then the trainer's naive per-row keys.
            _repl_d = acc.w.get("repl@d", 0.0)
            if _repl_d > 0:
                m("replay_stream_pass_mean_gw", acc.w.get("repl_pass@n", 0.0) / _repl_d)
                m("replay_stream_strict7_rate_gw", acc.w.get("repl_s7@n", 0.0) / _repl_d)
                if acc.repl_group_route:
                    # best@n is a per-GROUP indicator, so it is group-level by construction
                    m("replay_stream_best_at_n_gw",
                      sum(1 for g in acc.repl_group_route.values() if g["any"]) / len(acc.repl_group_route))
            if acc.repl_route_audit_n:
                _n_ready = acc.repl_route_short_n + acc.repl_route_audit_n
                _w_aud = _n_ready / acc.repl_route_audit_n
                _den = acc.repl_route_full_n + _n_ready
                if _den:
                    m("replay_stream_pass_full_suffix_est",
                      (acc.repl_route_full_pass + _w_aud * acc.repl_route_audit_pass) / _den)
                    # 7/7 full-marks (rp==7): same audit-reweighted HT, indicator rp==7.
                    m("replay_stream_strict7_full_suffix_est",
                      (acc.repl_route_full_strict7 + _w_aud * acc.repl_route_audit_strict7) / _den)
                # Audit-reweighted best@n (GROUP >=1-of-n pass) full-suffix estimate:
                # same Horvitz-Thompson post-stratification, but the per-group any-pass
                # indicator. Unready groups keep their judged best; ready groups are
                # represented by the audit lane's best rate, weighted by the true ready
                # group count. Unbiased for best@n had every ready group run full-suffix.
                _full_g = [g for g in acc.repl_group_route.values() if g["rt"] == "full"]
                _aud_g = [g for g in acc.repl_group_route.values() if g["rt"] == "audit"]
                _n_sh_g = sum(1 for g in acc.repl_group_route.values() if g["rt"] == "short")
                if _aud_g:
                    _n_un_g = len(_full_g)
                    _n_rd_g = len(_aud_g) + _n_sh_g
                    _N_g = _n_un_g + _n_rd_g
                    _best_full = sum(1 for g in _full_g if g["any"])
                    _best_aud_rate = sum(1 for g in _aud_g if g["any"]) / len(_aud_g)
                    if _N_g:
                        m("replay_stream_best_full_suffix_est",
                          (_best_full + _n_rd_g * _best_aud_rate) / _N_g)
        # Prover judge COUNTS (not rates) — main panel "prover judge count".
        # Derived from the MECE judge-health buckets so they partition every
        # generated row: attempts (judge ran) + missing (could not run) ==
        # rows_generated, and failed (judge ran but errored) is a subset of
        # attempts. `judged` requires a NON-EMPTY proof, so empty-proof rows are
        # "missing" (raw n_proof_tag counts them as present); the raw
        # http/parse/trunc flags overlap, so sum via the precedence buckets.
        m("judge_attempts", float(acc.n_judged))
        m("judge_success", float(acc.n_prover_1))
        m("judge_missing", float(acc.judge_cat_counts.get("no_proof", 0)))
        m("judge_failed", float(
            acc.judge_cat_counts.get("http_error", 0)
            + acc.judge_cat_counts.get("truncated", 0)
            + acc.judge_cat_counts.get("parse_failed", 0)))
        # Per-prompt-group judge-count percentiles (panel 11 per-group band):
        # each group's count is the sum of its per-completion flags; emit
        # p50/p90/p99 of those counts across this step's groups.
        if acc.group_attempt:
            for label, gdict in (
                ("judge_attempts_per_group", acc.group_attempt),
                ("judge_success_per_group", acc.group_success),
                ("judge_missing_per_group", acc.group_missing),
                ("judge_failed_per_group", acc.group_failed),
            ):
                per_group = sorted(sum(v) for v in gdict.values())
                for q in (50, 90, 99):
                    pv = _percentile_from_sorted(per_group, q)
                    if pv is not None:
                        m(f"{label}_p{q}", pv)

    # length_penalty_mean is emitted for BOTH proposer and prover
    # (prover-family rows also carry length_penalty under the exponential
    # penalty path). Gated on length_penalty_n>0 so linear-overlong prover families
    # naturally omit it.
    if acc.length_penalty_n:
        wm("length_penalty_mean", "lp@n", "lp@d", acc.length_penalty_sum / acc.length_penalty_n)

    # GENERATED tokens per step (rollout-throughput numerator; both modes): sum of
    # response_length minus any replay prefix (see the accumulation comment). The
    # throughput panel divides by timing_s/gen and the node count.
    if acc.resp_len_vals:
        m("generated_tokens_sum", float(acc.generated_tokens_sum))

    if acc.n_group_uid_missing:
        m("group_uid_missing_rows", float(acc.n_group_uid_missing))

    # Group-level stats are exact only with explicit uid. Do not fall back to
    # repeated prompt text: duplicate theorem text can merge distinct prompt
    # groups and produce fake group sizes.
    if acc.group_scores:
        group_sizes = Counter(len(v) for v in acc.group_scores.values())
        if group_sizes:
            mode_size = group_sizes.most_common(1)[0][0]
            max_group_size_seen[0] = max(max_group_size_seen[0], mode_size)
        n_groups = len(acc.group_scores)
        pass_counts = Counter(sum(v) for v in acc.group_scores.values())
        g_allzero = sum(1 for v in acc.group_scores.values() if sum(v) == 0)
        g_allone = sum(1 for v in acc.group_scores.values() if sum(v) == len(v))
        g_mixed = n_groups - g_allzero - g_allone
        g_fully_filtered = sum(1 for v in acc.group_de.values() if v and sum(v) == len(v))
        g_fully_kept = sum(1 for v in acc.group_de.values() if sum(v) == 0)
        g_mixed_de = n_groups - g_fully_filtered - g_fully_kept

        # Actual pipeline accounting (DAPO-filter composition of THIS batch): always raw.
        m("groups_generated", float(n_groups))
        m("groups_trained", float(g_fully_kept))
        m("groups_filtered", float(g_fully_filtered))
        m("groups_mixed_de", float(g_mixed_de))

        # Difficulty-sampling GROUP weighting: each prompt
        # group is ONE problem draw, so the pass-count histogram / allzero-mixed-allone
        # composition is weighted once per group by its draw-time c (NOT 16x per row) and
        # rescaled so the bars still sum to the actual group count. With c ≡ 1 the scale
        # is exactly 1 and every value equals the raw count.
        _gc_sum = sum(acc.group_c.get(g, 1.0) for g in acc.group_scores)
        _gscale = (n_groups / _gc_sum) if _gc_sum > 0 else 1.0

        def _gw(pred) -> float:
            return _gscale * sum(
                acc.group_c.get(g, 1.0) for g, v in acc.group_scores.items() if pred(v)
            )

        m("group_allzero", _gw(lambda v: sum(v) == 0))
        m("group_mixed", _gw(lambda v: 0 < sum(v) < len(v)))
        m("group_allone", _gw(lambda v: sum(v) == len(v)))

        # Per-step standard error of the (SNIS-corrected) mean reward, from the prompt-group draws:
        # each group g is one problem draw with weight c_g and value rbar_g = its pass rate. The
        # reported mean is mu = Σ c_g rbar_g / Σ c_g; its self-normalized-IS SE is
        # sqrt(Σ c_g^2 (rbar_g - mu)^2) / Σ c_g. With c_g ≡ 1 (no difficulty weighting) this reduces
        # to the ordinary std(rbar)/sqrt(#groups). Computed from the rollout dumps for EVERY step, so
        # the reward curve backfills a ±1σ band -- wider on difficulty-weighted steps (the IS
        # variance cost) than on the uniform pre-graft history.
        _rc = [(sum(v) / len(v), acc.group_c.get(g, 1.0)) for g, v in acc.group_scores.items() if v]
        _C = sum(cg for _, cg in _rc)
        if _C > 0 and len(_rc) > 1:
            _mu = sum(cg * r for r, cg in _rc) / _C
            _se = math.sqrt(sum((cg * cg) * (r - _mu) ** 2 for r, cg in _rc)) / _C
            m("train_correctness_se", _se)
        if acc.diff_weighted:
            m("group_allzero_raw", float(g_allzero))
            m("group_mixed_raw", float(g_mixed))
            m("group_allone_raw", float(g_allone))

        # Pass-count histogram k=0..16 (fixed range; renderer trims to rollout_n).
        pass_counts_w: dict[int, float] = {}
        for g, v in acc.group_scores.items():
            k = sum(v)
            pass_counts_w[k] = pass_counts_w.get(k, 0.0) + acc.group_c.get(g, 1.0)
        for k in range(17):
            m(f"pass_count_hist__k{k}", _gscale * pass_counts_w.get(k, 0.0))
            if acc.diff_weighted:
                m(f"pass_count_hist__k{k}_raw", float(pass_counts.get(k, 0)))


def _build_per_step(
    metrics: dict[int, dict[str, Any]],
    step_stats: dict[int, dict[str, Any]],
) -> dict[str, list[Any]]:
    steps = sorted(set(metrics) | set(step_stats))
    per_step: dict[str, list[Any]] = {"steps": steps}
    metric_keys = sorted({k for data in metrics.values() for k in data})
    for key in metric_keys:
        skey = _sanitize_key(key)
        per_step.setdefault(skey, [None] * len(steps))
        col = per_step[skey]
        for i, s in enumerate(steps):
            col[i] = metrics.get(s, {}).get(key)
    computed_keys = sorted({k for stats in step_stats.values() for k in stats})
    for key in computed_keys:
        col = per_step.setdefault(key, [None] * len(steps))
        for i, s in enumerate(steps):
            v = step_stats.get(s, {}).get(key)
            if v is not None:
                col[i] = v
    return per_step


def _load_theorem_index(
    manifest: dict[str, str],
    explicit_path: str | None,
    warnings: _Warnings,
) -> dict[str, int] | None:
    """Optional theorem_hash -> dataset index map.

    The canonical index (``extra_info.index``) lives in the training parquet,
    which the stdlib parser cannot read. An optional, separately built
    ``theorem_index.json`` sidecar next to the parquet supplies it. We
    auto-discover it from the manifest's ``train_file`` (or take an explicit
    path) and use it only to surface a stable theorem number on cards. The
    direct-proof statistic itself does NOT need it — it joins on theorem text.
    """
    path: Path | None = None
    if explicit_path:
        path = Path(explicit_path)
    else:
        train_file = (manifest or {}).get("train_file")
        if train_file:
            path = Path(train_file).parent / "theorem_index.json"
    if not path or not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        warnings.add(f"theorem index sidecar present but unreadable ({path}): {exc}")
        return None
    mapping = doc.get("by_theorem_hash")
    if not isinstance(mapping, dict):
        warnings.add(f"theorem index sidecar {path} missing 'by_theorem_hash' map")
        return None
    out: dict[str, int] = {}
    for k, v in mapping.items():
        iv = _as_int(v)
        if iv is not None:
            out[k] = iv
    return out or None


def _q_calibration_block(source, start_step):
    """Generative-Q readiness: the calibration data from
    ``run_data/q_state_deltas.jsonl``, in two distinct series.

      * ``audit`` — THE audit calibration: for one member of each audit group, the Q value
        that would have been consumed (measured at the cut state) vs that same rollout's
        realized terminal judge reward. Ground truth; no self-reference.
      * ``pred``/``z`` — readiness probes over every qualifying group, with the group's
        ``route`` and ``source_tag``. A probe whose target came from consumed Q
        (source_tag q_bootstrap/mixed_final_q) is Q scored against itself, so the panel
        must be able to exclude it — hence the provenance travels with the pair.

    Contribution-style, keyed by GLOBAL step (= dataset_step + 1) so merge_fig_data can
    union incrementally. Lines with dataset_step < start_step - 1 are skipped via a cheap
    regex (the delta lines also carry ~15MB/step of admitted prefix ids — never parse what
    we don't need). Returns None when the run has no q delta log. Torn tail (live-run copy)
    tolerated. Blocks cached by an older parser simply lack the new keys.
    """
    run_dir = (source or {}).get("durable_run_dir") or ""
    path = os.path.join(run_dir, "run_data", "q_state_deltas.jsonl") if run_dir else ""
    if not path or not os.path.exists(path):
        return None
    step_re = re.compile(r'"dataset_step":\s*(\d+)')
    min_ds = None if start_step is None else max(int(start_step) - 1, 0)
    by_step: dict[str, dict] = {}
    torn = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = step_re.search(line[:200])
            if m is not None and min_ds is not None and int(m.group(1)) < min_ds:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                torn += 1
                continue
            gstep = str(int(d["dataset_step"]) + 1)
            blk = by_step.setdefault(
                gstep, {"pred": [], "z": [], "route": [], "tag": [], "audit_q": [], "audit_r": []}
            )
            for p in d.get("probes") or []:
                blk["pred"].append(p.get("prediction"))
                blk["z"].append(p.get("z"))
                blk["route"].append(p.get("route"))
                blk["tag"].append(p.get("source_tag"))
            for a in d.get("audit_pairs") or []:
                if a.get("q_at_cut") is None:
                    continue  # invalid generation: counted in q/audit_cut_invalid, not plotted
                blk["audit_q"].append(a.get("q_at_cut"))
                blk["audit_r"].append(a.get("terminal_reward"))
    if not by_step:
        return None
    if torn > 1:
        print(f"[q_calibration] WARNING: {torn} unparseable delta lines", file=sys.stderr)
    return {"by_step": by_step}


def _difficulty_phat_block(source, train_patterns, start_step=None):
    """Snapshot of the difficulty sampler's per-problem p-hat (its EMA of the binary-judge pass
    rate) as a histogram, for the difficulty p-hat dashboard panel. GROUND TRUTH is the latest
    checkpoint's ``difficulty_state.json`` ({"stats": {qid: {"p": p_hat, ...}}}) — exactly what the
    live sampler holds. When none exists yet (e.g. between a graft and the first post-graft
    checkpoint) it RECONSTRUCTS p-hat from the rollout dumps via difficulty.py's own warm-start
    (same EMA/qid/row_pass), so the panel is populated from step 1. Returns None when there is no
    difficulty signal at all. Stdlib-only (difficulty.py imports no torch).

    The reconstruction re-reads EVERY rollout dump in the run (``json.loads`` per line — tens of
    GB on a long run) and, unlike the rest of the parser, cannot be windowed: p-hat is an EMA over
    the full history. On an INCREMENTAL refresh (``start_step is not None``) that full scan would
    dominate the cycle, so the reconstructed block is CACHED at ``<run_dir>/.dash_difficulty_phat
    .json`` and reused; it is recomputed only on a cold parse (``start_step is None``). The
    checkpoint path stays live on every refresh — it is one small file read, and for runs that
    actually enable the sampler it is the path that fires, so caching costs them nothing."""
    run_dir = (source or {}).get("durable_run_dir") or ""
    cache_path = os.path.join(run_dir, ".dash_difficulty_phat.json") if run_dir else ""
    phats = None
    src_label = None
    # 1) latest checkpoint difficulty_state.json (the real p-hat)
    try:
        ckpt_root = os.path.join(run_dir, "run_data", "checkpoints") if run_dir else ""
        if ckpt_root and os.path.isdir(ckpt_root):
            found = []
            for d in glob.glob(os.path.join(ckpt_root, "global_step_*")):
                stem = os.path.basename(d).rsplit("_", 1)[-1]
                sfile = os.path.join(d, "difficulty_state.json")
                if stem.isdigit() and os.path.exists(sfile):
                    found.append((int(stem), sfile))
            if found:
                found.sort()
                with open(found[-1][1]) as f:
                    sd = json.load(f)
                stats = sd.get("stats") or {}
                phats = [float(v["p"]) for v in stats.values()
                         if isinstance(v, dict) and v.get("p") is not None]
                src_label = f"checkpoint global_step_{found[-1][0]}"
    except Exception:
        phats = None
    # 2a) incremental refresh: reuse the cached reconstruction instead of re-reading every dump
    if not phats and start_step is not None and cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            if isinstance(cached, dict) and cached.get("n"):
                return cached
        except Exception:
            pass  # unreadable cache -> fall through and rebuild it
    # 2b) reconstruct from rollouts via difficulty.py's warm-start (same EMA the driver ran)
    if not phats:
        try:
            roll_dir = None
            cand = os.path.join(run_dir, "run_data", "rollouts") if run_dir else ""
            if cand and os.path.isdir(cand):
                roll_dir = cand
            elif train_patterns:
                roll_dir = os.path.dirname(train_patterns[0])
            if roll_dir and os.path.isdir(roll_dir):
                # Match the live run's difficulty config so the reconstructed EMA is faithful.
                os.environ.setdefault("SP_DIFF_SAMPLING", "1")
                os.environ.setdefault("SP_DIFF_EMA_ALPHA", "0.9")
                os.environ.setdefault("SP_DIFF_P_PRIOR", "0.5")
                os.environ.setdefault("SP_DIFF_MIN_OBS", "0")
                os.environ.setdefault("SP_DIFF_WARMSTART", "1")
                import importlib.util
                dpath = os.path.abspath(os.path.join(
                    os.path.dirname(__file__), "..", "..", "verl", "verl",
                    "trainer", "ppo", "difficulty.py"))
                spec = importlib.util.spec_from_file_location("difficulty_viz", dpath)
                Dm = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(Dm)
                Dm._init()
                Dm.warmstart_from_rollouts(roll_dir)
                phats = [float(st["p"]) for st in Dm._S["stats"].values()]
                src_label = "reconstructed from rollouts (EMA a=0.9)"
        except Exception:
            phats = None
    if not phats:
        return None
    N = len(phats)
    phats.sort()
    B = 16  # 16 bins over [0,1] — one per k/16, matching the pass-count granularity
    counts = [0] * B
    for p in phats:
        counts[min(B - 1, max(0, int(p * B)))] += 1
    floor = 1.0 / 32.0
    at_floor = sum(1 for p in phats if math.sqrt(max(p * (1 - p), 0.0)) <= floor + 1e-9)
    mid = N // 2
    median = phats[mid] if N % 2 else 0.5 * (phats[mid - 1] + phats[mid])
    block = {
        "n": N,
        "n_bins": B,
        "counts": counts,
        "mean": sum(phats) / N,
        "median": median,
        "frac_le_1_16": sum(1 for p in phats if p <= 1 / 16 + 1e-9) / N,
        "frac_ge_15_16": sum(1 for p in phats if p >= 15 / 16 - 1e-9) / N,
        "frac_at_w_floor": at_floor / N,
        "source": src_label,
    }
    # Persist ONLY the expensive rollout reconstruction; the checkpoint path is cheap and stays
    # live, so caching it would freeze a panel that costs nothing to keep fresh.
    if cache_path and (src_label or "").startswith("reconstructed"):
        try:
            tmp = cache_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(block, f)
            os.replace(tmp, cache_path)
        except Exception:
            pass
    return block


def _load_dashboard_annotations(source):
    """Optional per-experiment dashboard annotations from <run_dir>/dashboard_annotations.json,
    surfaced as fig_data['dashboard_annotations'] (e.g. {"vertical_markers": [{"x":56.5,
    "label":"...", "label_all":true}]}) -- the renderer draws them on every step panel. Returns {}
    when absent/unreadable so runs without the file are unaffected."""
    run_dir = (source or {}).get("durable_run_dir") or ""
    if not run_dir:
        return {}
    try:
        with open(os.path.join(run_dir, "dashboard_annotations.json")) as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def parse_run(
    *,
    run_id: str,
    manifest_path: Path | None,
    metrics_path: Path | None,
    train_patterns: list[str],
    val_patterns: list[str],
    source: dict[str, str],
    start_step: int | None,
    previous_max_train_step: int | None = None,
    theorem_index_path: str | None = None,
    samples_full_path: str | None = None,
    additional_data_sidecar_path: Path | None = None,
) -> dict[str, Any]:
    warnings = _Warnings()
    manifest = _load_manifest(manifest_path, warnings)
    metrics = _load_metrics(metrics_path, warnings)

    # Additional-data sidecar: maps theorem_hash -> data_source so the parser
    # can classify each prover row into prover_original / prover_additional
    # families even when the rollout dump strips extra_info.data_source. Path
    # precedence: explicit CLI flag, then auto-discovery next to the train
    # parquet (manifest["train_file"]'s directory), then the run-dir manifests/
    # folder. An empty/missing sidecar simply leaves every prover row as
    # prover_original (back-compat for non-replay runs).
    sidecar_dict: dict[str, str] = {}
    sidecar_resolved: Path | None = None
    if additional_data_sidecar_path is not None:
        sidecar_resolved = additional_data_sidecar_path
    elif manifest.get("train_file"):
        cand = Path(manifest["train_file"]).parent / ADDITIONAL_DATA_SIDECAR_BASENAME
        if cand.exists():
            sidecar_resolved = cand
    if sidecar_resolved is None and manifest_path is not None:
        cand = manifest_path.parent / ADDITIONAL_DATA_SIDECAR_BASENAME
        if cand.exists():
            sidecar_resolved = cand
    if sidecar_resolved is not None:
        sidecar_dict = _load_additional_data_sidecar(sidecar_resolved, warnings)
        if sidecar_dict:
            warnings.add(
                f"using additional-data sidecar {sidecar_resolved} "
                f"({len(sidecar_dict)} theorem-hash entries) to classify "
                f"prover rows into prover_original / prover_additional families."
            )
        # Record the sidecar's path + SHA-256 in fig_data["source"] so a future
        # reader knows EXACTLY which sidecar version classified the families
        # for this fig_data. The sidecar is a replay-run artifact, not an
        # opaque fallback — its identity must be reproducible.
        try:
            sc_bytes = sidecar_resolved.read_bytes()
            source["additional_data_sidecar"] = str(sidecar_resolved)
            source["additional_data_sidecar_sha256"] = hashlib.sha256(sc_bytes).hexdigest()
            source["additional_data_sidecar_entries"] = len(sidecar_dict)
        except OSError as e:
            warnings.add(
                f"could not read additional-data sidecar {sidecar_resolved} for "
                f"SHA-256 provenance: {e}"
            )

    if manifest and not manifest.get("wandb_url"):
        warnings.add(
            "manifest has no wandb_url (logged after W&B init; PDF header W&B "
            "link will read N/A)"
        )

    global_accum = _GlobalAccum()
    # Mode-keyed example categories. Keys are "<mode>__<category>".
    train_example_keys = [
        # judge / format failures — both modes
        "proposer__judge_http_error", "proposer__judge_truncated",
        "proposer__judge_parse_failed", "proposer__missing_proposition",
        "prover__judge_http_error", "prover__judge_truncated",
        "prover__judge_parse_failed", "prover__missing_proof",
        # mode-specific outcomes
        "proposer__correct_high_impact", "proposer__correct_low_impact",
        "proposer__incorrect_wellformed",
        "prover__correct", "prover__incorrect_wellformed",
    ]
    examples_train: dict[str, deque] = {
        k: deque(maxlen=MAX_EXAMPLES_PER_CATEGORY) for k in train_example_keys
    }
    # Validation is prover only.
    examples_val: dict[str, deque] = {
        k: deque(maxlen=MAX_EXAMPLES_PER_CATEGORY) for k in (
            "prover__correct", "prover__incorrect_wellformed",
            "prover__missing_proof", "prover__judge_parse_failed",
        )
    }
    pooled_train: list[dict[str, Any]] = []
    pooled_val: list[dict[str, Any]] = []
    full_samples: dict[str, dict[str, Any]] = {}
    train_max_group_size = [0]
    val_max_group_size = [0]

    def _process(patterns: list[str], is_val: bool,
                 from_step: int | None = None
                 ) -> tuple[dict[int, dict[str, Any]], int, int, list[dict[str, Any]]]:
        files = _iter_step_files(patterns)
        if from_step is not None:
            # INCLUSIVE start: parse step files with step >= from_step. The
            # boundary file (from_step) is re-parsed on purpose so the laptop
            # merger receives the intentional one-step overlap.
            kept = [(s, p) for (s, p) in files if s >= from_step]
            if len(kept) < len(files):
                warnings.add(
                    f"{'val' if is_val else 'train'} rollouts: --start-step={from_step} "
                    f"parsed {len(kept)} of {len(files)} step files (steps >= {from_step}); "
                    f"earlier steps come from the existing canonical history (overlap at "
                    f"step {from_step})"
                )
            files = kept
        stats: dict[int, dict[str, Any]] = {}
        total_rows = 0
        malformed = 0
        parsed_files: list[dict[str, Any]] = []
        examples = examples_val if is_val else examples_train
        pooled = pooled_val if is_val else pooled_train
        for step, path in files:
            try:
                f = path.open("r", encoding="utf-8")
            except OSError as exc:
                warnings.add(f"could not open rollout file {path}: {exc}")
                continue
            step_malformed = [0]

            def rows() -> Iterator[tuple[int, dict[str, Any]]]:
                with f:
                    for line_no, line in enumerate(f, start=1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            yield line_no, json.loads(line)
                        except json.JSONDecodeError:
                            step_malformed[0] += 1

            agg = _aggregate_step(
                rows(),
                step=step,
                source_file=path.name,
                examples=examples,
                pooled=pooled,
                global_accum=global_accum,
                is_val=is_val,
                max_group_size_seen=val_max_group_size if is_val else train_max_group_size,
                warnings=warnings,
                additional_data_sidecar=sidecar_dict,
            )
            malformed += step_malformed[0]
            if agg:
                stats[step] = agg
                gen_key = (
                    "v7__val__mode_mix__total_generated" if is_val
                    else "v7__train__mode_mix__total_generated"
                )
                nrows = int(agg.get(gen_key, 0))
                total_rows += nrows
                parsed_files.append({"step": step, "file": path.name, "rows": nrows})
        if malformed:
            warnings.add(
                f"{'val' if is_val else 'train'} rollouts: dropped {malformed} malformed "
                f"JSONL line(s) (often a partially-written final line on a live run)"
            )
        return stats, total_rows, len(files), parsed_files

    # --start-step applies to TRAIN only (incremental from the canonical history
    # watermark, INCLUSIVE so the boundary step overlaps). Validation is sparse +
    # small, so it is always fully parsed and dedup'd at merge time — this
    # guarantees val never develops a coverage gap.
    train_stats, train_rows, n_train_files, parsed_train_files = _process(
        train_patterns, is_val=False, from_step=start_step)
    val_stats, val_rows, n_val_files, _ = _process(val_patterns, is_val=True)

    # Direct-proof statistic: attach, to each proposer conjecture sample, how
    # many of its theorem's TRAIN prover (direct-proof) rollouts were judged
    # correct — joined by theorem text over all parsed steps. n is typically the
    # rollout_n (8) seen once per epoch; it is whatever prover rollouts for that
    # theorem fell inside the parsed window (so --start-step lowers per-delta
    # coverage; the merged history accumulates it across refreshes).
    theorem_index = _load_theorem_index(manifest, theorem_index_path, warnings)
    n_with_dp = 0
    n_proposer_rows = 0

    def _attach_direct_proof(rows: Iterable[dict[str, Any]]) -> None:
        nonlocal n_with_dp, n_proposer_rows
        for r in rows:
            if r.get("mode") != "proposer":
                continue
            n_proposer_rows += 1
            th = r.get("theorem_hash")
            if not th:
                continue
            total = global_accum.theorem_prover_total.get(th, 0)
            if total:
                r["direct_proof_total"] = int(total)
                r["direct_proof_correct"] = int(global_accum.theorem_prover_correct.get(th, 0))
                n_with_dp += 1
            if theorem_index is not None and th in theorem_index:
                r["theorem_index"] = theorem_index[th]

    for _bucket in examples_train.values():
        _attach_direct_proof(_bucket)
    _attach_direct_proof(pooled_train)

    if train_stats:
        train_missing_uid = sum(
            st.get("v7__train__proposer__group_uid_missing_rows", 0)
            + st.get("v7__train__prover__group_uid_missing_rows", 0)
            for st in train_stats.values()
        )
        legacy_recovered = sum(
            st.get("v7__train__legacy_rows_with_uid", 0) for st in train_stats.values()
        )
        legacy_steps = sum(
            1 for st in train_stats.values()
            if st.get("v7__train__legacy_chunk_uid_used", 0)
        )
        legacy_unrecoverable = sum(
            st.get("v7__train__legacy_rows_unrecoverable", 0)
            for st in train_stats.values()
        )
        if legacy_recovered:
            warnings.add(
                "train rollouts had NO per-row `uid` (a v7 dump regression) on "
                f"{legacy_steps} step file(s); recovered {int(legacy_recovered)} "
                "row(s) via contiguous chunk-of-rollout_n grouping (validated "
                "homogeneous input + de_filtered + mode per chunk; mixed-mode "
                "chunks dropped, not silently merged; raw (input)-only grouping "
                "is REFUSED because duplicate prompts in one step can fake "
                f"size-2N groups). {int(legacy_unrecoverable)} row(s) failed "
                "validation and remain ungrouped. Fix at the dump writer (add "
                "`uid` to full_reward_extra in dapo_ray_trainer.py) so future "
                "runs need no recovery."
            )
        elif train_missing_uid:
            warnings.add(
                "train rollouts are missing explicit `uid` on "
                f"{int(train_missing_uid)} mode row(s); prompt-group pass-count, "
                "filter-pressure group counts, and per-group judge-count panels are "
                "disabled (chunk-fallback did not apply because some rows did carry "
                "uid — mixed-uid state is unexpected and refused)"
            )
        if n_proposer_rows:
            cov = f"{n_with_dp}/{n_proposer_rows}"
            idx_note = (
                "canonical theorem index attached from the parquet sidecar"
                if theorem_index is not None else
                "no theorem_index.json sidecar found (run scripts/v7/build_theorem_index.py "
                "for canonical theorem numbers; the k/n stat works without it)"
            )
            warnings.add(
                "proposer<->prover theorems are joined by the shared problem text "
                "extracted from `input` (matches dataset extra_info.theorem exactly); "
                f"{cov} sampled proposer cards have a direct-proof k/n stat "
                f"(the rest had no prover rollout for that theorem inside the parsed "
                f"steps). {idx_note}. Adding `extra_info.index` to the dump writer "
                "would make this an O(1) id join and enable the full paired-theorem panel."
            )
    if val_stats:
        val_missing_uid = sum(
            st.get("v7__val__proposer__group_uid_missing_rows", 0)
            + st.get("v7__val__prover__group_uid_missing_rows", 0)
            for st in val_stats.values()
        )
        if val_missing_uid:
            warnings.add(
                "validation rollouts are missing explicit `uid` on "
                f"{int(val_missing_uid)} mode row(s); validation group-size detection "
                "uses val-core metrics when available"
            )
    if train_stats and not train_max_group_size[0]:
        warnings.add(
            "train prompt-group metrics require explicit `uid`; no train group "
            "histogram was emitted for this refresh"
        )
    if n_val_files == 0:
        warnings.add(
            "no validation rollout files found (expected: sparse, first at test_freq; "
            "validation panels will render N/A until val runs)"
        )
    if train_stats and len(metrics) < len(train_stats):
        warnings.add(
            f"metrics.jsonl logged only {len(metrics)} step(s) but {len(train_stats)} "
            f"train rollout step(s) were dumped — metric-derived panels (timing, "
            f"optimizer, throughput) will be sparse. Likely a VERL_FILE_LOGGER_PATH "
            f"file-logger gap; raw-rollout panels are unaffected."
        )
    # Mode-specific rollout entropy: the renderer prefers rollout/entropy_*,
    # falls back to actor/entropy_* (update-batch) with a label, else N/A. Warn
    # about whichever sources are missing so the dashboard label is justified.
    def _has(metric: str) -> bool:
        return any(metric in data for data in metrics.values())

    has_rollout_mode_entropy = _has("rollout/entropy_proposer") or _has("rollout/entropy_prover")
    has_actor_mode_entropy = _has("actor/entropy_proposer") or _has("actor/entropy_prover")
    has_mode_entropy = has_rollout_mode_entropy or has_actor_mode_entropy
    if metrics and not has_rollout_mode_entropy:
        detail = (
            "actor/entropy_* present (update-batch fallback, labeled as such)"
            if has_actor_mode_entropy else
            "neither is logged; the prover/proposer rollout-entropy panels render N/A "
            "and only the global entropy panel has data"
        )
        warnings.add(
            "mode-specific ROLLOUT entropy (`rollout/entropy_proposer`, "
            f"`rollout/entropy_prover`) not in metrics.jsonl — {detail}. To enable, "
            "log per-mode rollout entropy over generated rollout rows."
        )
    if global_accum.resp_len_char_rows > 0:
        warnings.add(
            f"rollout response-length panels (4/5) are TOKENS ONLY: "
            f"{global_accum.resp_len_char_rows} train rollout rows had no actor "
            f"`response_length` field and are OMITTED (the char fallback "
            f"`len(output)` is disallowed to avoid char/token mislabeling; "
            f"`total_tokens` is NOT a valid substitute — it includes prompt "
            f"length and would silently inflate response panels); "
            f"{global_accum.resp_len_token_rows} rows had `response_length`. Steps/runs with "
            f"no token rows render the length panels as N/A. Add token fields to the "
            f"dump writer to populate them."
        )

    merged_stats: dict[int, dict[str, Any]] = defaultdict(dict)
    for step, st in train_stats.items():
        merged_stats[step].update(st)
    for step, st in val_stats.items():
        merged_stats[step].update(st)
    per_step = _build_per_step(metrics, dict(merged_stats))

    global_block: dict[str, Any] = {
        "candidate_len_bin_edges": list(CANDIDATE_LEN_BIN_EDGES),
        "score_by_len_proposer": [
            {"count": int(c), "score_sum": s} for c, s in global_accum.len_bins_proposer
        ],
        "score_by_len_prover": [
            {"count": int(c), "score_sum": s} for c, s in global_accum.len_bins_prover
        ],
        "overlong_bin_edges": list(OVERLONG_BIN_EDGES),
        "overlong_hist_proposer": list(global_accum.overlong_hist_proposer),
        "overlong_hist_prover": list(global_accum.overlong_hist_prover),
        "n_overlong_proposer": global_accum.n_overlong_proposer,
        "n_overlong_prover": global_accum.n_overlong_prover,
        # Per-step contributions to the cumulative histograms above (train keyed by
        # parsed step; val is re-read in full each refresh). merge_fig_data rebuilds
        # the cumulative fields from the union of these, so incremental refreshes
        # keep the global panels exact without a periodic FULL re-parse.
        "global_step_contrib": {
            "train": {str(k): v for k, v in sorted(global_accum.step_contrib_train.items())},
            "val": {str(k): v for k, v in sorted(global_accum.step_contrib_val.items())},
        },
        "impact_levels": list(IMPACT_LEVELS),
        # resp_len_* per-mode percentiles (main panels 4/5) are TOKENS ONLY now
        # (char fallback disallowed), so the unit is unconditionally tokens; a
        # run with no token fields simply has no resp_len data (N/A panels).
        "rollout_len_unit": "tokens",
    }

    # Difficulty sampler p-hat distribution (snapshot histogram) for its dashboard panel.
    _phat_block = _difficulty_phat_block(source, train_patterns, start_step)
    if _phat_block:
        global_block["difficulty_phat"] = _phat_block

    # Generative-Q readiness: per-step probe (prediction, z) pairs for the
    # calibration panel. Contribution-style; merge_fig_data unions by step.
    _qcal = _q_calibration_block(source, start_step)
    if _qcal:
        global_block["q_calibration"] = _qcal

    # Replay drilldown blocks (needed for replay runs that have
    # prover_additional rows). Emitted as a top-level "replay" dict so the
    # diagnostics dashboard can render the provenance/seed/source-step
    # breakdown panels. Absent or empty when
    # no prover_additional rows had provenance (sidecar missing, or row data
    # could not be classified) — diagnostics panels render N/A in that case.
    if global_accum.replay_by_source or global_accum.replay_by_seed:
        by_source_rows = []
        for (run_id_v, step_v), s in sorted(
            global_accum.replay_by_source.items(),
            key=lambda kv: (kv[0][0] or "", kv[0][1] if kv[0][1] is not None else -1),
        ):
            n = s.get("n", 0)
            by_source_rows.append({
                "source_run_id": run_id_v,
                "source_step": step_v,
                "n": int(n),
                "correct": int(s.get("correct", 0)),
                "overlong": int(s.get("overlong", 0)),
                "pass_at_1": (s["correct"] / n) if n else None,
                "overlong_rate": (s["overlong"] / n) if n else None,
                "resp_len_mean": (s["resp_len_sum"] / s["resp_len_n"]) if s.get("resp_len_n") else None,
                "tot_tok_mean": (s["tot_tok_sum"] / s["tot_tok_n"]) if s.get("tot_tok_n") else None,
            })
        # Seed drilldown: sort by failure rate (descending), then by overlong
        # rate, then by sample count. Cap to top-K to keep fig_data slim
        # (top-k tables are enough for examples and failure summaries); K is generous so the seed-drilldown panel can render either
        # "top failures" or "top overlong" without re-querying.
        K = 200
        seed_items = []
        for uid, s in global_accum.replay_by_seed.items():
            n = s.get("n", 0)
            if n == 0:
                continue
            fail_rate = 1.0 - (s["correct"] / n)
            overlong_rate = s["overlong"] / n
            seed_items.append({
                "seed_statement_uid": uid,
                "data_source": s.get("data_source", ""),
                "source_run_id": s.get("source_run_id"),
                "n": int(n),
                "correct": int(s["correct"]),
                "overlong": int(s["overlong"]),
                "fail_rate": fail_rate,
                "overlong_rate": overlong_rate,
                "n_source_steps": len(s.get("source_steps") or {}),
                "n_source_completion_indices": len(s.get("source_completion_indices") or {}),
            })
        # Two top-K orderings the panel will pick from.
        top_by_failure = sorted(seed_items, key=lambda r: (-r["fail_rate"], -r["n"]))[:K]
        top_by_overlong = sorted(seed_items, key=lambda r: (-r["overlong_rate"], -r["n"]))[:K]
        global_block["replay"] = {
            "by_source_run_step": by_source_rows,
            "by_seed_statement_uid": {
                "n_distinct": len(seed_items),
                "top_by_failure": top_by_failure,
                "top_by_overlong": top_by_overlong,
            },
        }

    config = _derive_config(
        manifest,
        metrics,
        train_rollout_n=train_max_group_size[0] or None,
        val_rollout_n=val_max_group_size[0] or None,
    )

    # Lazy full-sample JSONL: harvest the transient ``_full`` text off the FINAL
    # sample set (bounded by the deque/pooled caps), dedup by id, then strip it so
    # fig_data stays slim. This caps the JSONL at the ~sample count, not all rows.
    def _harvest(rows: Iterable[dict[str, Any]]) -> None:
        for r in rows:
            full = r.pop("_full", None)
            if full is not None and full.get("id") not in full_samples:
                full_samples[full["id"]] = full
    for _bucket in examples_train.values():
        _harvest(_bucket)
    for _bucket in examples_val.values():
        _harvest(_bucket)
    _harvest(pooled_train)
    _harvest(pooled_val)

    # Full (untruncated) text for every sampled row, for the HTML viewer to load
    # on demand. Slim rows + fig_data only carry excerpts.
    samples_full_meta: dict[str, Any] = {"n": len(full_samples), "file": None}
    if samples_full_path:
        sp = Path(samples_full_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        with sp.open("w", encoding="utf-8") as sf:
            for rec in full_samples.values():
                sf.write(json.dumps(rec, ensure_ascii=False, default=str))
                sf.write("\n")
        samples_full_meta["file"] = sp.name

    # Refresh provenance: describes THIS parse so the laptop merger can audit the
    # intentional inclusive-overlap step before appending.
    train_steps_parsed = [pf["step"] for pf in parsed_train_files]
    last_train_rollout_step = max(train_steps_parsed) if train_steps_parsed else None
    overlap_step = (
        start_step if (start_step is not None and start_step in train_steps_parsed)
        else None
    )
    refresh_block = {
        "previous_max_train_step": previous_max_train_step,
        "cluster_start_step": start_step,
        "overlap_step": overlap_step,
        "last_train_rollout_step": last_train_rollout_step,
        "parsed_train_files": parsed_train_files,
    }

    warnings.emit_stderr()
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "source": source,
        "refresh": refresh_block,
        "manifest": manifest,
        "config": config,
        "per_step": per_step,
        "dashboard_annotations": _load_dashboard_annotations(source),
        "train_jsonl": {
            "n": train_rows,
            "n_step_files": n_train_files,
            "examples": {k: list(v) for k, v in examples_train.items()},
            "pooled": pooled_train,
        },
        "val_jsonl": {
            "n": val_rows,
            "n_step_files": n_val_files,
            "examples": {k: list(v) for k, v in examples_val.items()},
            "pooled": pooled_val,
        },
        "samples_full": samples_full_meta,
        "global": global_block,
        "artifacts": {
            "n_metric_steps": len(metrics),
            "metric_last_step": (max(metrics) if metrics else None),
            "n_train_step_files": n_train_files,
            "n_val_step_files": n_val_files,
            "train_rows": train_rows,
            "val_rows": val_rows,
            "manifest_found": bool(manifest),
            "has_mode_entropy_metrics": has_mode_entropy,
        },
        "warnings": warnings.items,
    }


def _metrics_have_key(metrics: dict[int, dict[str, Any]], key: str) -> bool:
    """True when ANY step logged `key` with a usable value. Used to infer which Q size-control
    regime a run used when its config predates the explicit pins."""
    try:
        for row in (metrics or {}).values():
            v = (row or {}).get(key)
            if v is not None and v == v:      # not None, not NaN
                return True
    except AttributeError:
        return False
    return False


def _derive_config(
    manifest: dict[str, str],
    metrics: dict[int, dict[str, Any]],
    train_rollout_n: int | None,
    val_rollout_n: int | None,
) -> dict[str, Any]:
    """Header/identity values from the manifest, with fallbacks.

    Surfaces the problem-batch knobs alongside the prompt-row batch knobs so
    the header makes any two-mode 2x duplication obvious.
    """

    def m(key: str) -> str | None:
        v = manifest.get(key)
        return v if v else None

    train_ds = _detect_metric_data_source(metrics, "train")
    val_ds = _detect_metric_data_source(metrics, "val")
    train_best_n = _detect_best_at_n(metrics, "train", "score")
    val_best_n = _detect_best_at_n(metrics, "val", "acc")

    _train_bs = _as_int(m("train_batch_size")) or _as_int(m("data_train_batch_size"))
    _gen_bs = (
        _as_int(m("gen_batch_size")) or _as_int(m("data_gen_batch_size")) or _train_bs
    )

    return {
        "run_id": m("run_id"),
        "experiment": m("experiment"),
        "reward": m("reward"),
        "cohort": m("cohort"),
        "nnodes": _as_int(m("nnodes")),
        "total_steps": _as_int(m("total_steps")),
        # Problem-batch knobs are two-mode (proposer+prover) concepts;
        # prover-only runs have no duplication, so problems == rows.
        # gen defaults to the train batch when a run doesn't set it separately.
        "train_problem_batch_size": _as_int(m("train_problem_batch_size"))
        or _train_bs,
        "gen_problem_batch_size": _as_int(m("gen_problem_batch_size")) or _gen_bs,
        "train_batch_size": _train_bs,
        "gen_batch_size": _gen_bs,
        "ppo_mini_batch_size": _as_int(m("ppo_mini_batch_size")),
        "actor_lr": m("actor_lr"),
        "hot_fix_lr": m("hot_fix_lr"),
        "actor_kl_loss_coef": m("actor_kl_loss_coef"),
        "save_freq": _as_int(m("save_freq")),
        "reward_num_workers": _as_int(m("reward_num_workers")),
        "actor_model": m("pi1_init"),
        "judge_model": m("reward_model"),
        "verl_head": m("verl_head"),
        "riemann_head": m("riemann_head"),
        "started_at": m("started_at"),
        "host": m("host"),
        "wandb_url": m("wandb_url"),
        # DAPO filter + exponential-penalty manifest knobs (optional).
        # Type coercion matches existing patterns: bool for True/False flags,
        # int for token counts, float for shaping coefficients, str passthrough
        # for free-form labels.
        "no_drop_except_true_zero_std": _as_bool(m("no_drop_except_true_zero_std")),
        "dapo_filter_metric": m("dapo_filter_metric"),
        "exponential_penalty": _as_bool(m("exponential_penalty")),
        "exponential_penalty_base_free_tokens": _as_int(m("exponential_penalty_base_free_tokens")),
        "exponential_penalty_gamma": _as_float(m("exponential_penalty_gamma")),
        # Auto-detected.
        # Train and validation can have different multiplicities. Keep
        # `rollout_n` as the train-side alias for older renderers.
        "rollout_n": train_rollout_n or train_best_n,
        "train_rollout_n": train_rollout_n or train_best_n,
        "val_rollout_n": val_rollout_n or val_best_n,
        "train_data_source": train_ds,
        "val_data_source": val_ds,
        "train_best_at_n": train_best_n,
        "val_best_at_n": val_best_n,
        # NO fabricated defaults. Every field is read from the run's real
        # config; anything absent resolves to None and renders as N/A, rather
        # than a plausible-looking wrong number (a hardcoded overlong default,
        # for instance, made _derive_length_penalty_mode falsely report linear).
        "max_response_length_default": _as_int(m("max_response_length")),
        "judge_max_tokens_default": _as_int(m("judge_max_tokens")),
        "overlong_buffer_len_default": _as_int(m("overlong_buffer_length")),
        "overlong_penalty_factor_default": _as_float(m("overlong_penalty_factor")),
        "test_freq_default": _as_int(m("test_freq")),
        "val_n_default": _as_int(m("val_n")),
        # Q short-lane token budget. The g-capped lane is bounded by min(g, budget-prefix),
        # NOT by max_response_length, so a panel drawing only the 75k line misrepresents it.
        "sp_q_budget_g_default": _as_int(m("sp_q_budget_g")),
        # Q MECHANISM KNOBS. Read from the run's own config so the panels never draw a gate
        # line at a threshold the run did not use; the panels fall back to default constants
        # (gate 0.18, FIFO 3,840, rho cap 0.5) only when a key is genuinely absent.
        "sp_q_ready_thresh_default": _as_float(m("sp_q_ready_thresh")),
        # Each falls back to the single knob, then (in the panel) to 0.18.
        "sp_q_ready_thresh_global_default": (
            _as_float(m("sp_q_ready_thresh_global")) or _as_float(m("sp_q_ready_thresh"))
        ),
        "sp_q_ready_thresh_problem_default": (
            _as_float(m("sp_q_ready_thresh_problem")) or _as_float(m("sp_q_ready_thresh"))
        ),
        "sp_q_fifo_cap_default": _as_int(m("sp_q_fifo_cap")),
        "sp_q_train_n_default": _as_int(m("sp_q_train_n")),
        "sp_q_min_valid_default": _as_int(m("sp_q_min_valid")),
        "sp_q_audit_den_default": _as_int(m("sp_q_audit_den")),
        "sp_q_train_noref_default": _as_int(m("sp_q_train_noref")),
        "sp_q_prompt_variant_default": m("sp_q_prompt_variant"),
        # Which Q size-control rule the run used: the per-step movement cap
        # (sp_q_rho_cap, scaling the applied delta) or the halving LR ladder. They are
        # mutually exclusive and the displacement panel must not draw the other one's line.
        "sp_q_rho_cap_default": _as_float(m("sp_q_rho_cap")),
        # Regime: prefer the config pin, else INFER from what the run logged. q/lr_current is
        # emitted only by the ladder path and q/s only by the movement-cap path, so the metrics
        # themselves are authoritative for older runs that never wrote the config keys.
        "sp_q_lr_ladder_default": (
            _as_int(m("sp_q_lr_ladder"))
            if m("sp_q_lr_ladder") is not None
            else (1 if _metrics_have_key(metrics, "q/lr_current") else None)
        ),
        "sp_q_lr_ratio_max_default": _as_float(m("sp_q_lr_ratio_max")),
        "sp_q_lr_initial_default": _as_float(m("sp_q_lr_initial")),
        "sp_q_lr_floor_default": _as_float(m("sp_q_lr_floor")),
        "sp_q_interleave_default": _as_int(m("sp_q_interleave")),
    }


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _default_paths(run_dir: Path) -> dict[str, Any]:
    # Standard layout (experiments/<exp>/): manifest/config.yaml + run_data/.
    # Auto-detected by the presence of run_data/, so the same CLI works on both
    # the standard layout and the older durable run-dir layout.
    if (run_dir / "run_data").is_dir():
        return {
            "manifest": run_dir / "manifest" / "config.yaml",
            "metrics": run_dir / "run_data" / "metrics.jsonl",
            "train": [str(run_dir / "run_data" / "rollouts" / "*.jsonl")],
            "val": [str(run_dir / "run_data" / "val_rollouts" / "*.jsonl")],
        }
    # Older durable run-dir layout.
    return {
        "manifest": run_dir / "manifests" / "launch_env.txt",
        "metrics": run_dir / "summaries" / "metrics.jsonl",
        "train": [str(run_dir / "rollouts" / "train" / "*.jsonl")],
        "val": [str(run_dir / "rollouts" / "val" / "*.jsonl")],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, default=_env_path("V7_RUN_DIR"))
    parser.add_argument("--file-log", type=Path, default=_env_path("V7_FILE_LOG"))
    parser.add_argument("--train-rollouts", nargs="+", default=_env_list("V7_TRAIN_ROLLOUTS"))
    parser.add_argument("--val-rollouts", nargs="+", default=_env_list("V7_VAL_ROLLOUTS"))
    parser.add_argument("--manifest", type=Path, default=_env_path("V7_MANIFEST"))
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID", ""))
    parser.add_argument(
        "--out", type=Path,
        default=Path(os.environ.get("OUT", "/tmp/v7_fig_data.json")),
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=(int(os.environ["V7_START_STEP"]) if os.environ.get("V7_START_STEP") else None),
        help="INCLUSIVE start: only parse TRAIN rollout step files with step >= "
        "START_STEP. The refresh sets this to the latest train step already in the "
        "canonical history (N), so the boundary file is re-parsed and the laptop "
        "merger receives the intentional one-step overlap to verify. Every new step "
        "since the last refresh is parsed (no gaps), no matter how far the run "
        "advanced. Validation rollouts and metrics.jsonl are always read in full.",
    )
    parser.add_argument(
        "--previous-max-train-step",
        type=int,
        default=(int(os.environ["V7_PREVIOUS_MAX_TRAIN_STEP"])
                 if os.environ.get("V7_PREVIOUS_MAX_TRAIN_STEP") else None),
        help="The latest train step N already in the laptop canonical history, "
        "recorded verbatim into the fig_data 'refresh' provenance block. Set by the "
        "refresh; the cluster parser cannot see the laptop history itself.",
    )
    parser.add_argument(
        "--theorem-index",
        default=os.environ.get("V7_THEOREM_INDEX") or None,
        help="Optional theorem_index.json sidecar (theorem_hash -> dataset index) "
        "for canonical theorem numbers on proposer cards. Auto-discovered next to "
        "the manifest's train_file when omitted; the direct-proof k/n stat does "
        "not require it.",
    )
    parser.add_argument(
        "--base-fig-data",
        default=os.environ.get("V7_BASE_FIG_DATA") or None,
        help="Optional prior fig_data history. Recorded in source provenance; the "
        "actual append/merge into the reusable history is done by merge_fig_data.py "
        "(this parser emits the freshly-parsed delta).",
    )
    parser.add_argument(
        "--samples-full-out",
        default=os.environ.get("V7_SAMPLES_FULL_OUT") or None,
        help="Optional path for the lazy full-sample JSONL (full prompt/proposition/"
        "proof/judge text for every sampled row). The HTML viewer loads it on demand.",
    )
    parser.add_argument(
        "--additional-data-sidecar",
        type=Path,
        default=(Path(os.environ["V7_ADDITIONAL_DATA_SIDECAR"])
                 if os.environ.get("V7_ADDITIONAL_DATA_SIDECAR") else None),
        help="Optional JSON sidecar mapping theorem_hash -> data_source for "
        "replay-aware family classification (prover_original vs prover_additional). "
        "Built by scripts/v7/build_additional_data_sidecar.py from the additional "
        "training parquet(s). Auto-discovered next to the manifest's train_file "
        f"(filename '{ADDITIONAL_DATA_SIDECAR_BASENAME}') when omitted; absent => "
        "every prover row is classified as prover_original.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    defaults = _default_paths(run_dir) if run_dir else {}

    manifest_path = args.manifest or defaults.get("manifest")
    metrics_path = args.file_log or defaults.get("metrics")
    train_patterns = args.train_rollouts or defaults.get("train")
    val_patterns = args.val_rollouts or defaults.get("val")

    if not train_patterns:
        raise SystemExit("missing --train-rollouts or --run-dir")

    run_id = args.run_id or (
        run_dir.name if run_dir else (metrics_path.stem if metrics_path else "v7-run")
    )

    source = {
        "durable_run_dir": str(run_dir) if run_dir else "",
        "metrics_jsonl": str(metrics_path) if metrics_path else "",
        "manifest": str(manifest_path) if manifest_path else "",
        "train_rollout_glob": ";".join(train_patterns),
        "val_rollout_glob": ";".join(val_patterns or []),
        "base_fig_data": args.base_fig_data or "",
    }

    data = parse_run(
        run_id=run_id,
        manifest_path=manifest_path,
        metrics_path=metrics_path,
        train_patterns=train_patterns,
        val_patterns=val_patterns or [],
        source=source,
        start_step=args.start_step,
        previous_max_train_step=args.previous_max_train_step,
        theorem_index_path=args.theorem_index,
        samples_full_path=args.samples_full_out,
        additional_data_sidecar_path=args.additional_data_sidecar,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"), sort_keys=True, default=str)
        f.write("\n")

    steps = data["per_step"].get("steps", [])
    print(
        f"wrote {args.out}: {len(steps)} steps "
        f"(train rows={data['train_jsonl']['n']} from {data['train_jsonl']['n_step_files']} files, "
        f"val rows={data['val_jsonl']['n']} from {data['val_jsonl']['n_step_files']} files), "
        f"{len(data['warnings'])} warning(s)"
    )


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def _env_list(name: str) -> list[str] | None:
    value = os.environ.get(name)
    return value.split() if value else None


if __name__ == "__main__":
    main()
