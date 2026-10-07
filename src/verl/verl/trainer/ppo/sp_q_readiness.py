"""Generative-Q readiness -- driver-side state.

Bookkeeping for the generative critic Q (the policy prompted with the problem, the partial
attempt and, when available, a reference solution; it decodes a value on the grid
{0, 0.1, ..., 1}) and for the readiness gate that decides when the critic's value at the end
of an action chunk may stand in for a full rollout. Follows the sp_replay.py pattern: a
torch-free, env-gated module whose state lives ONLY in the driver process. Everything is a
no-op unless SP_Q_ENABLE=1.

What it owns:
  * the Q-supervision FIFO B^Q: constant capacity C^Q=3,840, seeded from historical dumps
    (q_seed shards + manifest), online admission of >=8-valid 16-rollout groups, oldest
    (smallest q_replay_seq) evicted;
  * the monotone readiness table + pooled 5-step MAE window (gate 0.18 on both the pooled
    and the fresh individual error);
  * the question-wise FALLBACK reference bank K (add-once); the PRIMARY trajectory-wise
    reference is resolved per record at creation from the sp_replay buffer entry;
  * per-step routing: ready/audit draw (|A_t| = ceil(|R_t|/4), pinned rng) + budget caps
    stamped into dataset rows;
  * Q-prompt construction at the TOKEN level: context = templated statement
    prompt + attempt ids + "</think>" turn close + new user turn (instruction, with/without
    reference) + forced '<think>\n\n</think>\n\nQ value: ' prefill; free generation
    max_tokens=4, permissive grid parse, NO constrained decoding;
  * the per-step reward-site update (admissions, eviction, readiness errors/transitions,
    fallback-bank add-once, delta append) and the Q-training sample with the C_Q
    packing check (fail-loud);
  * the audit-lane calibration pairs: for one member of each audit group, the Q
    value that WOULD have been consumed -- measured at the cut state (prefix + g), not at
    the prefix -- against that same rollout's realized terminal judge reward. Pure
    measurement: it never enters targets, readiness, rewards, or PPO.

Persistence (the sp_replay resume contract):
  * q_state.json in every checkpoint dir: dataset-step cursor, q_seq counter, readiness
    table, MAE window tail, seed manifest shas.
  * q_state_deltas.jsonl: one append-only record per step (admitted records VERBATIM,
    evicted seqs, bank additions, readiness errors/transitions, displacement norms). On
    resume, deltas with dataset_step >= cursor are dropped AND physically truncated
    (atomic rewrite) before state is rebuilt.
  * The reward-site placement guarantees "delta k durable before checkpoint k exists".

Determinism: every draw is keyed (rng_seed, purpose, dataset_step[, ...]) via
sp_replay.rng_for. Requires SP_REPLAY_ENABLE=1 (the Q mechanism rides on the replay
dataset's (step,slot) stream and buffer entries).
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import OrderedDict

import numpy as np

from verl.trainer.ppo import sp_replay as _sp_replay
from verl.trainer.ppo.sp_replay import (
    _env_step_set,
    rng_for,
    sha256_file,
    verify_manifest_shards,
)

Q_DELTA_LOG_NAME = "q_state_deltas.jsonl"
Q_STATE_FILE_NAME = "q_state.json"
Q_SEED_MANIFEST_NAME = "q_seed_manifest.json"
Q_SEED_SHARD_GLOB = "q_seed/shard_*.jsonl"
REF_BANK_MANIFEST_NAME = "reference_bank_manifest.json"
REF_BANK_SHARD_GLOB = "reference_bank/shard_*.jsonl"

GRID = [round(i / 10.0, 1) for i in range(11)]  # 0.0, 0.1, ..., 1.0
# z rendered as in the prompt's "one of 0, 0.1, ..., 1" (no trailing .0 on the endpoints).
GRID_STR = {0.0: "0", 1.0: "1", **{g: f"{g:.1f}" for g in GRID if 0.0 < g < 1.0}}

# Permissive parse: leading legal grid value, ignore trailing text. Accepts
# "0.7", ".7" (-> 0.7), "0.70", "1", "1.0", "0"; anything else (incl. off-grid) invalid.
_LEAD_NUM_RE = re.compile(r"^\s*(1(?:\.0+)?|0(?:\.\d+)?|\.\d+)")

_S = {
    "env_inited": False,
    "enable": False,
    "harness": None,
}


def _env_init():
    if _S["env_inited"]:
        return
    _S["enable"] = os.environ.get("SP_Q_ENABLE", "0") in ("1", "true", "True")
    _S["env_inited"] = True


def _is_statement_row(ei) -> bool:
    """True for the SCRATCH inflow lane (sp_source_type == "statement").

    Q owns the replay lane: it probes those rows, forms targets from them, and may be
    consumed for them. The scratch lane is inflow-only (rollout_n=1, dropped pre-loss)
    and is deliberately outside Q. Keyed on sp_source_type -- stamped on EVERY row by
    q_dataset._finalize_row -- and NOT on prefix_len, which is not equivalent: a
    quantized cut makes prefix_len == 0 a normal outcome for a replay row.
    """
    try:
        # "cold_scratch" (cold bootstrap) is also outside Q: it is a fresh problem with
        # no prefix and no parent trajectory, so it has no Q record to form and nothing to
        # cut short -- the same reason prefix-0 statement groups are excluded from the Q seed.
        # It differs from "statement" only in being PPO-trained.
        return str((ei or {}).get("sp_source_type", "")) in ("statement", "cold_scratch")
    except AttributeError:
        return False


def enabled():
    _env_init()
    return _S["enable"]


def installed():
    return _S["harness"] is not None


def harness() -> "QHarness":
    h = _S["harness"]
    assert h is not None, "sp_q_readiness used before install() (train dataset never initialized?)"
    return h


def round_to_grid(x: float) -> float:
    return min(GRID, key=lambda g: (abs(g - float(x)), g))


def parse_grid_value(text: str):
    """Permissive parse of a generated Q output -> grid float, or None if invalid."""
    if not text:
        return None
    m = _LEAD_NUM_RE.match(text)
    if m is None:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    g = round_to_grid(v)
    if abs(g - v) > 1e-6:
        return None  # off-grid value: invalid by design, never coerced
    return g


# The Q instruction texts. The wording IS the trained Q format: never edit it in place.
Q_INSTRUCTION_WITH_REF = (
    "Pause here and estimate the probability that continuing this attempt leads to a "
    "complete and correct proof within totally 50K tokens (including the existing "
    "thinking trace tokens). You should refer to a reference correct proof here:\n\n"
    "{reference_proof}\n\n"
    "Don't think; answer immediately with a single line and nothing else:\n\n"
    "Q value: z\n\n"
    "where z is one of 0, 0.1, ..., 1. The first characters of your response must be "
    "`Q value:`."
)
Q_INSTRUCTION_NO_REF = (
    "Pause here and estimate the probability that continuing this attempt leads to a "
    "complete and correct proof within totally 50K tokens (including the existing "
    "thinking trace tokens).\n\n"
    "Don't think; answer immediately with a single line and nothing else:\n\n"
    "Q value: z\n\n"
    "where z is one of 0, 0.1, ..., 1. The first characters of your response must be "
    "`Q value:`."
)

# ---- SP_Q_PROMPT_VARIANT=reward_horizon ---------------------------------------------
# The legacy wording above asks for "the probability that continuing this attempt leads to a
# complete and correct proof". The TARGET it is fitted against is not that probability: `z` is
# the grid-rounded MEAN of the group's judge rewards (`observe_and_update`), and under the DS4
# fine-grained judge each reward is a continuous `points/7` rubric fraction, not the binary
# `points >= 6` correctness label the difficulty sampler uses. A group that scores 4/7 twice
# and 6/7 twice yields z = 0.7, which is not any probability of correctness. So the legacy
# prompt asks for one quantity and grades the answer against another.
#
# This variant states the quantity the label actually is -- expected rubric credit -- and keeps
# a HORIZON, because the labels are censored: every member reward comes from a rollout that was
# cut off at the response budget, so "if it ran forever" is never what was measured. It says
# "remaining token budget" rather than a hardcoded 50K, so the wording cannot go stale the way
# the legacy clause does when the response budget changes (e.g. to 100k).
#
# A SEPARATE variant, not an edit of the constants above: the wording IS the trained Q format,
# existing checkpoints were fitted on the legacy version, and there is no prompt hash in
# q_state.json that would catch a silent swap under a resume.
# Default stays "legacy".
Q_INSTRUCTION_WITH_REF_REWARD_HORIZON = (
    "Pause here and estimate how much rubric credit this attempt will earn if you continue it "
    "to completion within its remaining token budget: 0 means no credit, 1 means full credit "
    "for a complete and correct proof, and values in between mean partial credit. You should "
    "refer to a reference correct proof here:\n\n"
    "{reference_proof}\n\n"
    "Don't think; answer immediately with a single line and nothing else:\n\n"
    "Q value: z\n\n"
    "where z is one of 0, 0.1, ..., 1. The first characters of your response must be "
    "`Q value:`."
)
Q_INSTRUCTION_NO_REF_REWARD_HORIZON = (
    "Pause here and estimate how much rubric credit this attempt will earn if you continue it "
    "to completion within its remaining token budget: 0 means no credit, 1 means full credit "
    "for a complete and correct proof, and values in between mean partial credit.\n\n"
    "Don't think; answer immediately with a single line and nothing else:\n\n"
    "Q value: z\n\n"
    "where z is one of 0, 0.1, ..., 1. The first characters of your response must be "
    "`Q value:`."
)

Q_PROMPT_VARIANTS = {
    "legacy": (Q_INSTRUCTION_WITH_REF, Q_INSTRUCTION_NO_REF),
    "reward_horizon": (Q_INSTRUCTION_WITH_REF_REWARD_HORIZON,
                       Q_INSTRUCTION_NO_REF_REWARD_HORIZON),
}
# Forced answer-turn prefill. The trailing space is part of the stem.
Q_ANSWER_PREFILL = "<think>\n\n</think>\n\nQ value: "
THINK_CLOSE = "</think>"


class QLrController:
    """Displacement-ratio Q learning-rate ladder (the critic's halving LR schedule).

      * lr starts at SP_Q_LR_INITIAL (2e-6) and is only ever REDUCED;
      * a completed, nonempty Q phase BREACHES when
        rho = ||Delta_Q|| / max(||Delta_PPO1|| + ||Delta_PPO2||, 1e-12) > ratio_max (1.0);
      * `breach_patience` (2) CONSECUTIVE breaches halve the next step's lr
        (reduction_factor 0.5), flooring at SP_Q_LR_FLOOR (5e-7) -- the pinned
        schedule is 2e-6 -> 1e-6 -> 5e-7 and nothing below it;
      * at the floor a further two-step breach only sets `floor_alert`;
      * a non-finite Q loss/gradient (the worker already skips the O_Q step) halves
        IMMEDIATELY, resetting the streak;
      * an empty Q sample (skipped phase) resets the streak and changes nothing.

    Note this is a per-STEP LR, not the per-step movement cap (SP_Q_RHO_CAP): the ladder
    applies the Q update in full and prices an over-large step in the NEXT step's LR,
    whereas the movement cap scales the applied delta by s_t = min(1, 0.5/rho_t) and leaves
    the LR alone. The two are mutually exclusive; SP_Q_LR_LADDER=1 selects the ladder.

    State is (lr, breach_streak) -- both persisted in q_state.json and replayable
    from the per-step delta log, so a resume continues the ladder exactly.
    """

    def __init__(self, *, lr=2e-6, floor=5e-7, ratio_max=1.0, breach_patience=2,
                 reduction_factor=0.5, breach_streak=0):
        self.lr = float(lr)
        self.floor = float(floor)
        self.ratio_max = float(ratio_max)
        self.breach_patience = int(breach_patience)
        self.reduction_factor = float(reduction_factor)
        self.breach_streak = int(breach_streak)

    @classmethod
    def from_env(cls) -> "QLrController":
        return cls(
            lr=float(os.environ.get("SP_Q_LR_INITIAL", os.environ.get("SP_Q_LR_BASE", "2e-6"))),
            floor=float(os.environ.get("SP_Q_LR_FLOOR", "5e-7")),
            ratio_max=float(os.environ.get("SP_Q_LR_RATIO_MAX", "1.0")),
            breach_patience=int(os.environ.get("SP_Q_LR_BREACH_PATIENCE", "2")),
            reduction_factor=float(os.environ.get("SP_Q_LR_REDUCTION_FACTOR", "0.5")),
        )

    def apply_step(self, *, delta_q, delta_ppo1, delta_ppo2, q_phase_skipped,
                   q_non_finite) -> dict:
        """Evaluate one completed step (after PPO minibatch 2). Returns the record that
        goes into the delta log; `self.lr` afterwards is the NEXT step's LR."""
        current_lr = self.lr
        rho = None
        breach = False
        reduced = False
        floor_alert = False

        if q_non_finite:
            self.breach_streak = 0
            if self.lr <= self.floor + 1e-20:
                floor_alert = True
            else:
                self.lr = max(self.floor, self.lr * self.reduction_factor)
                reduced = True
        elif q_phase_skipped:
            self.breach_streak = 0
        else:
            if delta_q is None:
                raise ValueError("completed Q phase without a displacement measurement")
            rho = float(delta_q) / max(float(delta_ppo1) + float(delta_ppo2), 1e-12)
            breach = rho > self.ratio_max
            self.breach_streak = self.breach_streak + 1 if breach else 0
            if self.breach_streak >= self.breach_patience:
                self.breach_streak = 0
                if self.lr <= self.floor + 1e-20:
                    floor_alert = True
                else:
                    self.lr = max(self.floor, self.lr * self.reduction_factor)
                    reduced = True
        return {
            "lr_current": current_lr,
            "lr_next": self.lr,
            "rho": rho,
            "delta_q": delta_q,
            "delta_ppo1": delta_ppo1,
            "delta_ppo2": delta_ppo2,
            "breach": breach,
            "breach_streak_after": self.breach_streak,
            "reduced": reduced,
            "floor_alert": floor_alert,
            "q_phase_skipped": bool(q_phase_skipped),
            "q_non_finite": bool(q_non_finite),
        }

    def replay_step(self, section: dict) -> None:
        """Restore from a delta-log record (resume path; must be exact)."""
        self.lr = float(section["lr_next"])
        self.breach_streak = int(section["breach_streak_after"])

    def state_dict(self) -> dict:
        return {
            "lr": self.lr,
            "breach_streak": self.breach_streak,
            "floor": self.floor,
            "ratio_max": self.ratio_max,
            "breach_patience": self.breach_patience,
            "reduction_factor": self.reduction_factor,
        }

    def load_state_dict(self, st: dict) -> None:
        """Restore lr/streak from q_state.json. The four CONFIG fields are NOT restored:
        they are launcher pins, so a deliberate re-pin must take effect on resume -- but a
        silent drift would be invisible, so it is logged loudly."""
        self.lr = float(st["lr"])
        self.breach_streak = int(st.get("breach_streak", 0))
        for name, live in (("floor", self.floor), ("ratio_max", self.ratio_max),
                           ("breach_patience", self.breach_patience),
                           ("reduction_factor", self.reduction_factor)):
            was = st.get(name)
            if was is not None and float(was) != float(live):
                print(f"[sp_q] Q-LR ladder {name} re-pinned: checkpoint {was} -> launcher "
                      f"{live} (launcher wins)", flush=True)


def _validate_q_record(r: dict, where: str) -> dict:
    for k in ("rec_id", "qid", "prefix_token_ids", "z"):
        if k not in r or r[k] in (None, ""):
            if not (k == "prefix_token_ids" and isinstance(r.get(k), list)):
                raise ValueError(f"q record missing/empty {k!r} ({where}): {str(r)[:200]}")
    if not isinstance(r["prefix_token_ids"], list):
        raise ValueError(f"q record prefix_token_ids not a list ({where})")
    # snap-within-epsilon, never exact float equality (0.1*3 != 0.3 in doubles);
    # normalize the stored z to the canonical grid double.
    z = float(r["z"])
    g = round_to_grid(z)
    if abs(g - z) > 1e-6:
        raise ValueError(f"q record target z={r['z']} not on the grid ({where})")
    r["z"] = g
    return r


class QHarness:
    """Q FIFO + readiness + fallback bank + routing + prompt building. Driver-only."""

    def __init__(self, cfg: dict, tokenizer, replay_harness: "_sp_replay.ReplayHarness",
                 raw_prompt_of_qid):
        # ---- science knobs (all pinned by the runner via +data.sp_q_*) ----
        self.seed_dir = str(cfg["sp_q_seed_dir"])
        self.bank_dir = str(cfg.get("sp_q_bank_dir", self.seed_dir))
        self.delta_dir = str(cfg["sp_replay_delta_dir"])  # same run_data dir as replay
        self.budget_g = int(cfg.get("sp_q_budget_g", 10000))
        # Readiness has TWO checks with separate thresholds. `ready_thresh` is the single
        # legacy knob and the DEFAULT for both, so a run that sets only it uses one threshold
        # for both:
        #   * GLOBAL  gate: pooled 5-step model-wide MAE < ready_thresh_global
        #   * PROBLEM gate: that problem's fresh probe error < ready_thresh_problem
        # They answer different questions -- "is Q calibrated overall?" vs "is Q right about
        # THIS problem?" -- so they need not share a threshold. A looser global gate with a tighter per-problem one lets readiness open
        # while still demanding local evidence before any single problem is trusted.
        self.ready_thresh = float(cfg.get("sp_q_ready_thresh", 0.18))
        self.ready_thresh_global = float(
            cfg.get("sp_q_ready_thresh_global", self.ready_thresh))
        self.ready_thresh_problem = float(
            cfg.get("sp_q_ready_thresh_problem", self.ready_thresh))
        # ---- readiness MODE ----
        #   * "problem" (default): a slot is routed off the full
        #     lane iff ITS problem has flipped ready -- the per-problem gate above, monotone.
        #   * "global": the per-problem clause is DROPPED. Every replay slot is ready iff the
        #     GLOBAL gate (pooled 5-step MAE < ready_thresh_global) is open; with
        #     sp_q_ready_global_latch=1 (default) the first opening is latched and every slot
        #     stays ready for the rest of the run -- the literal removal of the per-problem
        #     condition from a monotone flip. latch=0 makes it a LIVE gate that can close again.
        #   The per-problem table (`self.ready`) is still MAINTAINED in global mode so
        #   q/ready_problems keeps reporting what the per-problem gate WOULD have admitted; it
        #   just no longer decides routing. Persisted in q_state.json; a parent state without
        #   the fields derives gate_open from its saved mae_tail (so a branch taken while the
        #   parent's gate was closed starts closed even if the gate had opened earlier).
        self.ready_mode = str(cfg.get("sp_q_ready_mode", "problem"))
        if self.ready_mode not in ("problem", "global"):
            raise ValueError(f"sp_q_ready_mode must be 'problem' or 'global', got {self.ready_mode!r}")
        self.ready_global_latch = bool(int(cfg.get("sp_q_ready_global_latch", 1)))
        self.fifo_cap = int(cfg.get("sp_q_fifo_cap", 3840))
        self.min_valid = int(cfg.get("sp_q_min_valid", 8))
        self.train_n = int(cfg.get("sp_q_train_n", 768))
        self.ctx_limit = int(cfg.get("sp_q_ctx_limit", 53296))
        self.gen_reserve = 4  # max_tokens for the value + end-of-turn
        # Audit lane: |A_t| = ceil(|R_t| / den); den=4 (a quarter) is the
        # default. Raise it to thin the lane (16 -> ~6% of ready rows) or set <=0 to drop
        # it entirely. WARNING: the audit lane is the ONLY unbiased sample of ready-problem
        # performance -- the HT estimator's y_audit term (panel 1) and q/audit_cut_mae both
        # come from it, so den<=0 blinds both and leaves only the censored train reward.
        self.audit_frac_den = int(cfg.get("sp_q_audit_den", 4))
        self.mae_window = 5
        # Audit-cut calibration: one counterfactual Q call per audit group, at the state where
        # the short lane would have been cut. Pure measurement (never a reward/target);
        # set +data.sp_q_audit_cut=0 to drop its wave cost.
        self.audit_cut_probes = bool(int(cfg.get("sp_q_audit_cut", 1)))
        # ---- SEGMENTED Q ADVANTAGE ----------------------------------------------------
        # One INTERIOR Q read-out per training row, at an absolute offset K into the
        # GENERATED tokens, so the advantage can be split at that point:
        #     A1 = q_i - mean_j q_j   on [t, t+K)      A2 = r_i - q_i   on [t+K, end)
        # Default OFF: with seg_enable=0 no `seg` wave row is ever emitted, nothing is
        # stamped, and the `sp_segment` estimator is not selected -- runs without it are
        # byte-identical. K is an ABSOLUTE token offset, not a
        # fraction of the realised length, so segment boundaries are comparable across
        # siblings whose lengths differ by 10x.
        self.seg_enable = bool(int(cfg.get("sp_q_seg_enable", 0)))
        # K defaults to g/2.
        self.seg_k = int(cfg.get("sp_q_seg_k", 0)) or max(1, self.budget_g // 2)
        # n^seg_u minimum. This is ARITHMETIC ONLY (qbar needs >=1 value),
        # not a quality floor: there is no group-level fallback. A group with 2
        # valid seg rows is segmented on those 2; a group whose q all collapse onto one grid
        # value is segmented with A1 == 0. Kept as a knob so a stricter floor (e.g. 8) can
        # be reproduced; the default is 1.
        self.seg_min_valid = int(cfg.get("sp_q_seg_min_valid", 1))
        # Grid-collapse REPORTING threshold. It changes no behaviour -- if segment 1 has no
        # gradient to give, it gives zero, which is the honest answer. The threshold only
        # defines what q/seg_collapsed_frac counts. 0.05 = half the 0.1 grid step, so exact
        # ties and sub-grid jitter both register.
        self.seg_collapse_eps = float(cfg.get("sp_q_seg_collapse_eps", 0.05))
        # Soft grid-expectation read-out (a mitigation for grid collapse).
        # NOT IMPLEMENTED: it needs first-token logprobs, which sp_q_agent_loop currently
        # discards (response_logprobs=None). Reject the flag loudly rather than accept it
        # and silently keep scoring with the greedy argmax.
        if bool(int(cfg.get("sp_q_soft_value", 0))):
            raise NotImplementedError(
                "sp_q_soft_value is not implemented: sp_q_agent_loop returns "
                "response_logprobs=None, so there are no grid logprobs to renormalise. "
                "Measure the greedy-collapse fraction offline "
                "before wiring it into the engine path."
            )
        # ---- NO-GROUP TD (the "ungrouped" variant) -------------------------------------
        # For a READY (short-route) replay slot, the group is not run at all: the slot's
        # rollout budget is spread over td_siblings DISTINCT prefixes of the same stored
        # trajectory, one rollout each, and every such row's advantage is
        #     A_i = r_i - Q(p_i)   on the whole generated span,
        # with r_i the consumed Q(p_i + g continuation) (or the judge score if the row
        # finished before the cap) and Q(p_i) the row's own readiness probe. Implemented by
        # per-copy prefix rewrite in the trainer plus the probe value stamped into the
        # sp_q_seg_* fields at bound = prefix_len, consumed by the `sp_segment` estimator's
        # singleton-split path (a1 spans only mask-0 prefix tokens, a2 = r - q spans the
        # generated tokens). Default OFF: nothing is rewritten, nothing extra is stamped,
        # and runs without it are byte-identical.
        self.td_enable = bool(int(cfg.get("sp_q_td_enable", 0)))
        # How many single-rollout prefixes one ready slot expands into. MUST equal
        # rollout.n (the trainer repeat count is untouched by this mode; only the copies'
        # prefixes and uids change), so the trained row count and token budget match the
        # grouped configuration exactly. The runner asserts the equality.
        self.td_siblings = int(cfg.get("sp_q_td_siblings", 0))
        # Cut quantum for the TD prefix draws. 0 = inherit the replay lane's cut_grain.
        # The TD arm runs a FINER grain than the grouped lanes' 10k: at grain 10k a 30k
        # trajectory exposes ~3 grid points, so 16 "distinct" prefixes would collapse onto
        # ~3 states -- and the coarse grain's purpose (siblings sharing one round-offset
        # prefix for cache reuse) does not apply to single-rollout rows.
        self.td_cut_grain = int(cfg.get("sp_q_td_cut_grain", 0))
        # Bellman-backup volume control: max TD singleton records admitted to the Q FIFO
        # per (step, replay slot). 0 = admit every valid one. Each record embeds its full
        # prefix token ids in the append-only delta log, so unbounded admission at 16
        # singletons/slot writes tens of MB per step at a high ready fraction (see the
        # admission-loop comment).
        self.td_admit_per_slot = int(cfg.get("sp_q_td_admit_per_slot", 0))
        # Which replay lane the TD mechanism applies to. "short" (default):
        # READY slots -- each copy is cut at its own prefix + g and scored by Q, the
        # advantage is consumed Q minus the prefix probe. "full": NOT-READY slots -- each
        # copy runs to the full budget and earns a terminal judge reward, the advantage is
        # r_i minus the prefix probe, and the admitted records are terminal transitions
        # (ground truth) rather than Q-on-Q backups. The dataset reads this to decide which
        # slots to materialize as TD slots; every trainer hook downstream keys on the
        # sp_q_td_row stamp and is lane-agnostic.
        self.td_lane = str(cfg.get("sp_q_td_lane", "short"))
        if self.td_lane not in ("short", "full"):
            raise ValueError(f"sp_q_td_lane must be 'short' or 'full', got {self.td_lane!r}")
        if self.td_enable:
            if self.seg_enable:
                raise ValueError(
                    "sp_q_td_enable and sp_q_seg_enable are mutually exclusive: both write "
                    "the sp_q_seg_* stamps (TD at bound=prefix_len from the probe, seg at "
                    "bound=prefix_len+K from its own wave call), and a row carrying both "
                    "would silently score one mechanism under the other's name."
                )
            if self.td_siblings < 1:
                raise ValueError(
                    "sp_q_td_enable=1 requires sp_q_td_siblings >= 1 (set it to rollout.n; "
                    f"got {self.td_siblings})"
                )
        # Train the with-reference variant ONLY by default (the noref variant is never
        # consumed, so training it is pure regularization at ~2x cost).
        # sp_q_train_noref=1 restores both-variants training as an ablation.
        self.train_noref = bool(int(cfg.get("sp_q_train_noref", 0)))
        self.rng_seed = int(cfg.get("sp_q_rng_seed", 718001))
        self.response_length = int(cfg.get("max_response_length", 50000))
        # Q instruction wording. "legacy" carries the "within totally 50K tokens
        # (including the existing thinking trace tokens)" clause; "no_budget" drops it. This IS the trained Q format, so it is a
        # pinned launcher choice, checked against the checkpoint below.
        # Tier-1 references must be JUDGED-CORRECT trajectories (see trajectory_ref_proof).
        # Default ON: handing Q a failed attempt as a "reference correct proof" is a defect, not
        # a configuration choice. Set 0 only to reproduce the ungated behavior deliberately.
        self.ref_require_pass = bool(int(cfg.get("sp_q_ref_require_pass", 1)))
        # The per-step number is computed as a DELTA of the cumulative total, not as a separate
        # counter reset somewhere in the step. A reset is order-dependent: the Q WAVE
        # (pre-reward) resolves references and so does most of the rejecting, but a reset in
        # observe_and_update runs AFTER the wave -- and because trajectory_ref_proof caches per
        # entry_id, the later admission lookups hit the cache and never re-increment, so the
        # per-step metric would read ~0 while the total climbs. A delta cannot be wrong about where in the step the increments happened.
        self.ref_rejected_unpassed_total = 0      # cumulative; PERSISTED in q_state.json
        self._ref_rej_total_at_step_start = 0     # snapshot taken when the previous step closed
        self.prompt_variant = str(cfg.get("sp_q_prompt_variant", "legacy"))
        if self.prompt_variant not in Q_PROMPT_VARIANTS:
            raise ValueError(
                f"sp_q_prompt_variant {self.prompt_variant!r} not in "
                f"{sorted(Q_PROMPT_VARIANTS)}"
            )
        self.instruction_texts = Q_PROMPT_VARIANTS[self.prompt_variant]

        self.tokenizer = tokenizer
        self.replay = replay_harness
        self.raw_prompt_of_qid = raw_prompt_of_qid  # qid -> chat messages (lazy prompt ids)

        # ---- token pieces for the Q-prompt suffix, built once (token-exact, preflighted) ----
        self._tok_cache: dict = {}
        self._prompt_ids_cache: OrderedDict = OrderedDict()
        self._ref_proof_cache: dict = {}
        self._build_token_pieces()

        # ---- FIFO + bank: frozen seeds (deltas applied later by on_checkpoint_load) ----
        self.fifo: OrderedDict[int, dict] = OrderedDict()  # seq -> record, ascending
        self.q_seq_next = 0
        self.seed_manifest_sha = self._load_q_seed()
        self.bank: dict[str, dict] = {}
        self.bank_manifest_sha = self._load_ref_bank()

        # ---- readiness state ----
        self.ready: dict[str, bool] = {}
        self.mae_tail: list[tuple[int, list[float]]] = []  # [(dataset_step, [e...])] last 5
        # global-mode routing state (see sp_q_ready_mode above): the gate as of the LAST
        # observed step, and the monotone latch. Both persisted; both False until the first
        # complete window says otherwise.
        self.global_gate_open: bool = False
        self.global_ready_latched: bool = False
        # Per-problem monotone latch, True once a valid probe predicted Q̂ > 0.
        # Tracked ALWAYS (cheap; metric q/nonzero_seen_total); it CONDITIONS the ready flip
        # only when SP_Q_REQUIRE_NONZERO=1 (else readiness ignores it).
        self.q_nonzero_seen: dict[str, bool] = {}
        self.require_nonzero = os.environ.get("SP_Q_REQUIRE_NONZERO", "0") in ("1", "true", "True")
        # SP_Q_READY_REQUIRE_BANK: readiness additionally requires the problem to
        # be SOLVED -- i.e. present in the fallback reference bank, which is add-once and only
        # ever filled from a judged-PASSING row (prover_judge_score >= 1, see the bank block in
        # observe_and_update). Without it, readiness keys purely on Q's prediction ERROR, so a
        # problem the prover never solves is marked ready as soon as Q confidently predicts its
        # z of 0 -- 305 of 1,415 ready problems in ac2 were in exactly that
        # state. Those are the groups the short lane then trusts Q on, and the step-40 critic
        # probe measured Q over-predicting their reward by +0.26 (worse than a constant predictor).
        self.ready_require_bank = os.environ.get(
            "SP_Q_READY_REQUIRE_BANK", "0") in ("1", "true", "True")
        self._readiness_resets: list[dict] = []  # SP_Q_RESET_STEPS stamps
        self.resume_base = 0
        self.next_dataset_step = 0
        self._finalized = False

        # ---- cumulative applied displacement (persisted in q_state.json) ----
        self.cum_delta_q = 0.0
        self.cum_delta_ppo_net = 0.0

        # ---- Q-LR ladder (QLrController). OFF by default: without the flag the
        # movement-cap path (SP_Q_RHO_CAP) is unchanged and this object is never consulted. ----
        self.q_lr_ladder_enabled = os.environ.get("SP_Q_LR_LADDER", "0") in ("1", "true", "True")
        self.q_lr = QLrController.from_env() if self.q_lr_ladder_enabled else None

        # ---- per-step caches ----
        self._route_cache: tuple | None = None  # (step, {rslot: route})
        self._wave_result: dict | None = None   # parsed Q wave for the current step
        self._train_sample: dict | None = None  # materialized Q training rows
        self._pending_delta: dict | None = None  # built at the reward site, written post-update
        self._last_admitted_recs: list = []      # records admitted THIS step (separate-Q new-records draw)

    # ------------------------------------------------------------ token pieces
    def _build_token_pieces(self):
        tok = self.tokenizer

        def enc(s: str) -> list[int]:
            return tok.encode(s, add_special_tokens=False)

        self._ids_think_close = enc(THINK_CLOSE)
        self._ids_turn_close = enc("<|im_end|>\n")
        self._ids_user_open = enc("<|im_start|>user\n")
        self._ids_assistant_open = enc("<|im_start|>assistant\n")
        self._ids_prefill = enc(Q_ANSWER_PREFILL)
        self._ids_im_end = enc("<|im_end|>")

        # Stem/target boundary check: the teacher-forced target ids
        # must be the exact continuation of the forced stem. With the trailing space inside
        # the prefill, enc(prefill + z) must start with enc(prefill) for every grid value;
        # if BPE merges across the boundary, fall back to stem without the trailing space
        # (target then starts with " ") -- loudly, and recorded for the manifest.
        self._stem_trailing_space = True
        prefill = Q_ANSWER_PREFILL
        ok = all(
            enc(prefill + GRID_STR[g])[: len(self._ids_prefill)] == self._ids_prefill
            for g in GRID
        )
        if not ok:
            alt = prefill.rstrip(" ")
            alt_ids = enc(alt)
            ok_alt = all(enc(alt + " " + GRID_STR[g])[: len(alt_ids)] == alt_ids for g in GRID)
            assert ok_alt, "sp_q: neither stem boundary tokenizes cleanly; inspect tokenizer"
            self._stem_trailing_space = False
            self._ids_prefill = alt_ids
            prefill = alt
            print("[sp_q] WARNING: prefill trailing space merges across the stem boundary; "
                  "using stem without trailing space (targets start with ' ')", flush=True)
        self._prefill_text = prefill

        # target ids per grid value: continuation of the stem + <|im_end|>
        self.target_ids = {}
        for g in GRID:
            z = GRID_STR[g] if self._stem_trailing_space else " " + GRID_STR[g]
            cont = enc(prefill + z)[len(self._ids_prefill):]
            assert 1 <= len(cont) <= 3, (g, cont)
            self.target_ids[g] = cont + self._ids_im_end
        max_tgt = max(len(v) for v in self.target_ids.values())
        assert max_tgt <= self.gen_reserve, (max_tgt, self.gen_reserve)

    def prompt_ids_for_qid(self, qid: str) -> list[int]:
        """Templated statement prompt ids (add_generation_prompt) -- identical to the
        rollout row's prompt, so the Q wave hits the engine's prefix cache. Uses the
        SAME wrapper + normalizer pair as AgentLoopBase.apply_chat_template: the raw
        tokenizer call returns a BatchEncoding (not list[int]) on this transformers
        version, which would leak the string 'input_ids' into the wave context and crash
        the wave build."""
        ids = self._prompt_ids_cache.get(qid)
        if ids is None:
            from verl.utils.chat_template import apply_chat_template as _act
            from verl.utils.tokenizer import normalize_token_ids

            messages = self.raw_prompt_of_qid(qid)
            ids = normalize_token_ids(
                _act(self.tokenizer, list(messages), add_generation_prompt=True, tokenize=True)
            )
            assert ids and all(isinstance(t, int) for t in ids[:4]), (
                f"sp_q: prompt ids for {qid[:12]}... are not a flat int list: {ids[:4]!r}"
            )
            self._prompt_ids_cache[qid] = ids
            if len(self._prompt_ids_cache) > 8192:
                self._prompt_ids_cache.popitem(last=False)
        return ids

    def _instruction_ids(self, ref_proof) -> list[int]:
        key = ("instr", ref_proof if ref_proof is None else hash(ref_proof))
        ids = self._tok_cache.get(key)
        if ids is None:
            with_ref, no_ref = self.instruction_texts
            text = (with_ref.format(reference_proof=ref_proof)
                    if ref_proof is not None else no_ref)
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(self._tok_cache) > 4096:
                self._tok_cache.clear()
            self._tok_cache[key] = ids
        return ids

    def build_q_context_ids(self, qid: str, attempt_ids: list[int], ref_proof,
                            attempt_think_closed: bool | None = None) -> list[int]:
        """Full Q-prompt token ids: prompt + attempt + turn close + user turn +
        forced prefill. `attempt_think_closed`: whether the attempt already contains
        </think> (computed from ids if None) -- if so we do not append a second one."""
        prompt_ids = self.prompt_ids_for_qid(qid)
        if attempt_think_closed is None:
            # containment check on ids (</think> is a single special token for Qwen3;
            # `in` on a list is C-speed — never a python-level window scan over ~50K ids)
            tc = self._ids_think_close
            if len(tc) == 1:
                attempt_think_closed = tc[0] in attempt_ids
            else:
                attempt_think_closed = self._find_sub(attempt_ids, tc) >= 0
        parts = [prompt_ids, attempt_ids]
        if not attempt_think_closed:
            parts.append(self._ids_think_close)
        parts += [
            self._ids_turn_close,
            self._ids_user_open,
            self._instruction_ids(ref_proof),
            self._ids_turn_close,
            self._ids_assistant_open,
            self._ids_prefill,
        ]
        out = []
        for p in parts:
            out.extend(p)
        return out

    @staticmethod
    def _find_sub(hay: list[int], needle: list[int]) -> int:
        n = len(needle)
        for i in range(len(hay) - n, -1, -1):
            if hay[i:i + n] == needle:
                return i
        return -1

    def fits(self, ctx_ids: list[int]) -> bool:
        return len(ctx_ids) + self.gen_reserve <= self.ctx_limit

    # -------------------------------------------------------------- references
    def trajectory_ref_proof(self, entry_id: str):
        """Reference tier 1: the source buffer entry's own extracted proof (cached), or None.

        CORRECTNESS GATE (`sp_q_ref_require_pass`, default ON). Tier 1 is meant to be "the
        correct proof the prefix was truncated from", which holds by construction only when
        the buffer admits judged-correct trajectories. Under `sp_replay_admission=ungated`
        the buffer admits EVERY non-empty scratch response regardless of judge result, so an
        unchecked tier-1 lookup happily hands Q a FAILED attempt labelled to the model as a
        "reference correct proof" -- and because tier 1 takes priority over the
        correctness-gated bank in `reference_for`, it does so even when a real proof is
        available. `_admit_global` already records `meta.judge_pass`; this consults it.

        A missing `judge_pass` means the entry predates the field OR came from the
        `judged_correct` admission path / a frozen seed shard -- both correct by construction,
        so absent is treated as passing. Only an explicit 0 is rejected.
        """
        if not entry_id:
            return None
        if entry_id in self._ref_proof_cache:
            return self._ref_proof_cache[entry_id]
        proof = None
        for bucket in self.replay.entries.values():
            for e in bucket:
                if e["entry_id"] != entry_id:
                    continue
                if self.ref_require_pass and int((e.get("meta") or {}).get("judge_pass", 1)) == 0:
                    # Not a correct proof: refuse it as a reference and fall through to the
                    # question-wise bank (tier 2), or to no reference at all.
                    self.ref_rejected_unpassed_total += 1
                    break
                text = self.tokenizer.decode(e["response_token_ids"], skip_special_tokens=True)
                proof = _extract_proof_text(text)
                break
            if proof is not None:
                break
        self._ref_proof_cache[entry_id] = proof
        if len(self._ref_proof_cache) > 8192:
            self._ref_proof_cache.clear()
        return proof

    def prewarm_trajectory_refs(self, non_tensor_batch) -> dict:
        """Resolve (and cache) the tier-1 proof for every source entry in THIS batch, before
        the replay harness admits/evicts.

        Why this exists: on an uninterrupted step the Q WAVE resolves those proofs early, so by
        the time Q admission runs they are cached and the source entries' later eviction does not
        matter. A postwave/postreward step-cache resume rehydrates the wave RESULTS but not the
        proof cache -- and the replay update (admission + FIFO eviction, up to 32 entries) runs
        BEFORE Q admission. So a resumed step could find the source entry gone and silently fall
        back to the question-wise bank or to no reference, training on different Q contexts than
        the uninterrupted run would have. Pre-warming here makes the two paths equivalent,
        because it happens before anything can be evicted either way.

        Idempotent and cheap: bounded by the batch's DISTINCT sp_entry_ids (<= n_replay), and a
        no-op for ids already cached by the wave."""
        try:
            extra = non_tensor_batch["extra_info"]
        except (KeyError, TypeError):
            return {"q/ref_prewarmed": 0}
        seen, warmed, resolved = set(), 0, 0
        for ei in extra:
            eid = str((ei or {}).get("sp_entry_id", "") or "")
            if not eid or eid in seen:
                continue
            seen.add(eid)
            if eid in self._ref_proof_cache:
                continue
            warmed += 1
            if self.trajectory_ref_proof(eid) is not None:
                resolved += 1
        return {"q/ref_prewarmed": warmed, "q/ref_prewarmed_resolved": resolved}

    def reference_for(self, qid: str, ref_proof_stored):
        """Record-stored trajectory-wise proof first, else fallback bank, else None."""
        if ref_proof_stored:
            return ref_proof_stored
        b = self.bank.get(qid)
        return b["proof"] if b else None

    # ------------------------------------------------------------------- seeds
    def _load_q_seed(self) -> str:
        import glob as _glob

        manifest_path = os.path.join(self.seed_dir, Q_SEED_MANIFEST_NAME)
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"sp_q seed manifest missing: {manifest_path}")
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        shard_paths = sorted(_glob.glob(os.path.join(self.seed_dir, Q_SEED_SHARD_GLOB)))
        if not shard_paths:
            raise FileNotFoundError(f"no q seed shards under {self.seed_dir}")
        verify_manifest_shards(self.seed_dir, manifest["shards"], shard_paths, "q seed")
        n = 0
        for path in shard_paths:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    r = _validate_q_record(json.loads(line), where=path)
                    seq = int(r["seq"])
                    assert seq not in self.fifo, f"duplicate q seed seq {seq}"
                    self.fifo[seq] = r
                    n += 1
        self.fifo = OrderedDict(sorted(self.fifo.items()))
        assert len(self.fifo) <= self.fifo_cap, (
            f"q seed has {len(self.fifo)} records > capacity {self.fifo_cap}; the builder "
            "must emit only the newest C^Q records"
        )
        self.q_seq_next = (max(self.fifo) + 1) if self.fifo else 0
        print(f"[sp_q] q seed loaded: {n} records (cap {self.fifo_cap}), "
              f"q_seq_next={self.q_seq_next} from {self.seed_dir}", flush=True)
        return sha256_file(manifest_path)

    def _load_ref_bank(self) -> str:
        import glob as _glob

        manifest_path = os.path.join(self.bank_dir, REF_BANK_MANIFEST_NAME)
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"sp_q reference bank manifest missing: {manifest_path}")
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        shard_paths = sorted(_glob.glob(os.path.join(self.bank_dir, REF_BANK_SHARD_GLOB)))
        verify_manifest_shards(self.bank_dir, manifest["shards"], shard_paths, "reference bank")
        n = 0
        for path in shard_paths:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    b = json.loads(line)
                    assert b.get("qid") and b.get("proof"), f"bad bank record in {path}"
                    assert b["qid"] not in self.bank, f"duplicate bank qid {b['qid'][:12]}"
                    self.bank[b["qid"]] = b
                    n += 1
        print(f"[sp_q] reference bank loaded: {n} problems from {self.bank_dir}", flush=True)
        return sha256_file(manifest_path)

    # ------------------------------------------------------------------ resume
    def _delta_path(self) -> str:
        return os.path.join(self.delta_dir, Q_DELTA_LOG_NAME)

    def finalize_from_checkpoint(self, ckpt_dir):
        assert not self._finalized, "sp_q finalize_from_checkpoint called twice"
        # The Q cursor rides on the replay dataset's (step, slot) stream: the dataset labels
        # every row sp_dataset_step = replay.resume_base + local, and the reward hook asserts
        # the batch's step == this cursor. So the two cursors must be finalized in lockstep,
        # replay FIRST (ray_trainer._load_checkpoint calls it immediately above this).
        assert getattr(self.replay, "_finalized", False), (
            "sp_q finalize_from_checkpoint ran before sp_replay's -- the Q cursor is seeded "
            "from the replay cursor at a fresh-Q branch point (ordering contract)"
        )
        replay_base = int(self.replay.resume_base)
        state_path = os.path.join(ckpt_dir, Q_STATE_FILE_NAME) if ckpt_dir else None
        if state_path and os.path.exists(state_path):
            with open(state_path, encoding="utf-8") as f:
                st = json.load(f)
            for name, sha in (("seed_manifest_sha", self.seed_manifest_sha),
                              ("bank_manifest_sha", self.bank_manifest_sha)):
                rec = st.get(name)
                if rec and rec != sha:
                    raise ValueError(f"sp_q {name} drift: ckpt {rec[:12]} != loaded {sha[:12]}")
            # The instruction wording is the trained Q format: resuming a checkpoint under a
            # DIFFERENT variant silently asks the model a question it was never fitted on, and
            # nothing downstream would flag it. Checkpoints written before this field existed
            # carry None and are treated as "legacy" (which is what they trained on).
            # Readiness thresholds: log a change loudly but do NOT fail. Unlike the prompt
            # variant (which silently invalidates the trained Q format), moving a threshold is a
            # legitimate mid-run operation -- the flip is monotone, so problems already ready
            # stay ready and only FUTURE flips see the new value. Silence would still be wrong:
            # the readiness table would then mix two criteria with nothing recording it.
            for _name, _live in (("ready_thresh_global", self.ready_thresh_global),
                                 ("ready_thresh_problem", self.ready_thresh_problem)):
                _was = st.get(_name)
                if _was is not None and abs(float(_was) - _live) > 1e-9:
                    print(f"[sp_q] WARNING: {_name} CHANGED {_was} -> {_live} at this resume. "
                          f"Problems already flipped ready under the old value STAY ready "
                          f"(monotone); only future flips use the new one, so the readiness "
                          f"table now mixes two criteria.", flush=True)
            _rec_variant = st.get("prompt_variant") or "legacy"
            if _rec_variant != self.prompt_variant:
                raise ValueError(
                    f"sp_q prompt_variant drift: checkpoint trained on {_rec_variant!r} but "
                    f"this launch pins {self.prompt_variant!r}. The instruction wording IS the "
                    "trained Q format -- re-pin sp_q_prompt_variant to the checkpoint's value, "
                    "or start a new run."
                )
            self.ready = {str(k): bool(v) for k, v in st["ready"].items()}
            self.mae_tail = [(int(s), [float(x) for x in es]) for s, es in st["mae_tail"]]
            self.global_gate_open, self.global_ready_latched = self._global_flags_from_state(
                st, mae_tail=self.mae_tail, thresh=self.ready_thresh_global,
                window=self.mae_window)
            if self.ready_mode == "global":
                print(f"[sp_q] READY MODE = global (latch={int(self.ready_global_latch)}): "
                      f"gate_open={int(self.global_gate_open)} latched="
                      f"{int(self.global_ready_latched)} -> "
                      f"{'ALL' if self._global_all_ready() else 'NO'} replay slots route to Q "
                      f"next step; per-problem table ({sum(self.ready.values())} ready) is "
                      f"observability only", flush=True)
            self.q_nonzero_seen = {
                str(k): bool(v) for k, v in (st.get("q_nonzero_seen") or {}).items()
            }
            self._readiness_resets = list(st.get("readiness_resets", []) or [])
            self.resume_base = int(st["next_dataset_step"])
            if self.resume_base != replay_base:
                raise ValueError(
                    f"sp_q cursor {self.resume_base} != sp_replay cursor {replay_base} in "
                    f"{ckpt_dir}: both advance exactly once per dataset step, so a mismatch "
                    "means one of the two state files is stale (mixed/hand-edited checkpoint)"
                )
            self.q_seq_next = max(self.q_seq_next, int(st["q_seq_next"]))
            self.cum_delta_q = float(st.get("cum_delta_q", 0.0))
            self.cum_delta_ppo_net = float(st.get("cum_delta_ppo_net", 0.0))
            # Q-LR ladder: the LR is RUN STATE, not a constant -- a resume that reset it to
            # 2e-6 would silently undo every reduction the run had earned. Fail closed: with
            # the ladder on, a checkpoint written by a ladder run MUST carry its section.
            if self.q_lr is not None:
                _lad = st.get("q_lr_ladder")
                if _lad is not None:
                    self.q_lr.load_state_dict(_lad)
                    print(f"[sp_q] Q-LR ladder resumed: lr={self.q_lr.lr:.4g} "
                          f"breach_streak={self.q_lr.breach_streak}", flush=True)
                elif int(st.get("next_dataset_step", 0)) > int(
                    os.environ.get("SP_Q_LADDER_START_STEP", "0") or 0
                ):
                    raise ValueError(
                        f"SP_Q_LR_LADDER=1 but {state_path} has no q_lr_ladder section at "
                        f"cursor {st.get('next_dataset_step')}: this checkpoint was written "
                        "without the ladder, so its LR history is unknown. Set "
                        "SP_Q_LADDER_START_STEP to that cursor to declare a deliberate "
                        "ladder start at the initial LR."
                    )
                else:
                    print(f"[sp_q] Q-LR ladder starting fresh at lr={self.q_lr.lr:.4g} "
                          f"(declared start cursor {st.get('next_dataset_step')})", flush=True)
            # carry the warm-start provenance forward so later saves don't write null and
            # the resume-detection contract (warm-start applied exactly once) holds.
            self._readiness_warmstart = st.get("readiness_warmstart")
            self.ref_rejected_unpassed_total = int(st.get("ref_rejected_unpassed_total", 0) or 0)
            # a resume starts a fresh per-step window at the restored total
            self._ref_rej_total_at_step_start = self.ref_rejected_unpassed_total
            print(f"[sp_q] resumed cursor={self.resume_base}, ready={sum(self.ready.values())}, "
                  f"q_seq_next={self.q_seq_next} from {state_path}", flush=True)
            # ---- one-shot readiness COLD RESET at the resume steps listed in
            # SP_Q_RESET_STEPS, stamp-guarded (idempotent under crash-loops; auditable;
            # decoupled from SP_Q_BRANCH_STEP's init guard). Clears exactly the gate state:
            # per-problem monotone table, pooled MAE window, nonzero-Q latch. ----
            reset_steps = _env_step_set("SP_Q_RESET_STEPS")
            stamped = {int(r["at_step"]) for r in self._readiness_resets}
            if self.resume_base in reset_steps and self.resume_base not in stamped:
                stamp = {
                    "at_step": self.resume_base,
                    "prior_n_ready": sum(1 for v in self.ready.values() if v),
                    "prior_nonzero_seen": sum(1 for v in self.q_nonzero_seen.values() if v),
                }
                self.ready = {}
                self.mae_tail = []
                self.q_nonzero_seen = {}
                self.global_gate_open = False
                self.global_ready_latched = False
                self._readiness_resets.append(stamp)
                print(
                    f"[sp_q] READINESS RESET fired at cursor={self.resume_base}: "
                    f"ready cleared (was {stamp['prior_n_ready']}), mae_tail cleared, "
                    f"nonzero latch cleared (was {stamp['prior_nonzero_seen']}); global "
                    f"gate closed for >= {self.mae_window} steps by construction",
                    flush=True,
                )
        else:
            # Fresh Q at the BRANCH POINT: the parent checkpoint carries replay
            # state but no q_state.json. Readiness starts empty, the FIFO is the frozen seed
            # and O_Q is fresh -- but the cursor must CONTINUE the parent's dataset-step
            # stream. Starting at 0 here would label the first branch batch with the parent's
            # cursor (e.g. 50) while the Q hook expected 0, and the reward site would abort.
            self.resume_base = replay_base
            print(f"[sp_q] no persisted q_state; fresh Q state at the branch point (cursor="
                  f"{self.resume_base} adopted from the replay cursor, all unready, O_Q fresh)",
                  flush=True)
        self.next_dataset_step = self.resume_base
        self._apply_and_truncate_deltas(up_to_exclusive=self.resume_base)
        self._finalized = True

    def _apply_and_truncate_deltas(self, up_to_exclusive: int):
        path = self._delta_path()
        if not os.path.exists(path):
            return
        kept, applied, dropped = [], 0, 0
        last_step = None
        with open(path, encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                delta = json.loads(stripped)
                step = int(delta["dataset_step"])
                if last_step is not None and step < last_step:
                    raise ValueError(f"q delta log out of order: {step} after {last_step}")
                last_step = step
                if step >= up_to_exclusive:
                    dropped += 1
                    continue
                self._apply_delta(delta)
                applied += 1
                kept.append(stripped + "\n")
        if dropped > 0:
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.writelines(kept)
            os.replace(tmp, path)
        print(f"[sp_q] q delta log: applied {applied}, truncated {dropped}", flush=True)

    def _apply_delta(self, delta: dict):
        for r in delta.get("admitted", []):
            r = _validate_q_record(r, where="q delta")
            self.fifo[int(r["seq"])] = r
        for seq in delta.get("evicted_seqs", []):
            self.fifo.pop(int(seq), None)
        self.fifo = OrderedDict(sorted(self.fifo.items()))
        for b in delta.get("bank_added", []):
            self.bank.setdefault(b["qid"], b)
        # Q-LR ladder: replayable from the delta log so a checkpoint whose q_state.json
        # predates the tail still lands on the LR the tail earned (the log is fsynced
        # BEFORE checkpoint k exists, so the log is never behind the state file).
        if self.q_lr is not None:
            _lad = (delta.get("displacement") or {}).get("q_lr_ladder")
            if _lad:
                self.q_lr.replay_step(_lad)
        # readiness + counters are restored from q_state.json (checkpointed), not replayed;
        # the delta copies exist for audit/offline analysis.

    def save_state(self, ckpt_dir: str):
        path = os.path.join(ckpt_dir, Q_STATE_FILE_NAME)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "next_dataset_step": self.next_dataset_step,
                    "q_seq_next": self.q_seq_next,
                    "ready": self.ready,
                    "mae_tail": self.mae_tail[-self.mae_window:],
                    "seed_manifest_sha": self.seed_manifest_sha,
                    "bank_manifest_sha": self.bank_manifest_sha,
                    "fifo_size": len(self.fifo),
                    "stem_trailing_space": self._stem_trailing_space,
                    "cum_delta_q": self.cum_delta_q,
                    "cum_delta_ppo_net": self.cum_delta_ppo_net,
                    "readiness_warmstart": getattr(self, "_readiness_warmstart", None),
                    "q_nonzero_seen": self.q_nonzero_seen,
                    "readiness_resets": self._readiness_resets,
                    "q_lr_ladder": self.q_lr.state_dict() if self.q_lr is not None else None,
                    "prompt_variant": self.prompt_variant,
                    "ready_thresh_global": self.ready_thresh_global,
                    "ready_thresh_problem": self.ready_thresh_problem,
                    "ref_rejected_unpassed_total": self.ref_rejected_unpassed_total,
                    "ready_mode": self.ready_mode,
                    "global_gate_open": self.global_gate_open,
                    "global_ready_latched": self.global_ready_latched,
                },
                f,
            )
            f.flush()
            os.fsync(f.fileno())  # the checkpoint must not outlive the state it references
        os.replace(tmp, path)
        print(f"[sp_q] saved q_state (cursor={self.next_dataset_step}, "
              f"ready={sum(self.ready.values())}, fifo={len(self.fifo)}) -> {path}", flush=True)

    # ------------------------------------------------------------------- routing
    @staticmethod
    def _global_flags_from_state(st: dict, *, mae_tail, thresh: float, window: int):
        """(gate_open, latched) for a loaded q_state. States written before the fields
        existed derive gate_open from the saved window exactly as observe_and_update would
        have computed it at that step (complete window and pooled MAE < thresh), and take
        latched = gate_open: the flip is defined at the branch point, not back-dated to an
        earlier opening the parent may have had."""
        if "global_gate_open" in st:
            return bool(st["global_gate_open"]), bool(st.get("global_ready_latched", False))
        pooled = [e for _, es in mae_tail for e in es]
        complete = len(mae_tail) == window
        open_ = bool(pooled) and complete and (sum(pooled) / len(pooled)) < thresh
        return open_, open_

    def _global_all_ready(self) -> bool:
        """global mode: are ALL replay slots ready right now?"""
        return self.global_ready_latched if self.ready_global_latch else self.global_gate_open

    def is_ready(self, qid: str) -> bool:
        """The routing predicate. problem mode: this problem's monotone flag. global mode:
        the global gate (latched or live), the same answer for every qid."""
        if self.ready_mode == "global":
            return self._global_all_ready()
        return self.ready.get(qid, False)

    def route_plan_for_step(self, step: int):
        """Per-replay-slot route for this step: 'full' | 'short' | 'audit'.
        Statement rows are never routed. Cached; pure function of (state, step)."""
        if self._route_cache is not None and self._route_cache[0] == step:
            return self._route_cache[1]
        qids, _fill = self.replay.replay_plan_for_step(step)
        ready_slots = [i for i, q in enumerate(qids) if self.is_ready(q)]
        audit_slots: set = set()
        if ready_slots and self.audit_frac_den > 0:  # den<=0 -> no audit lane at all
            k = math.ceil(len(ready_slots) / self.audit_frac_den)
            rng = rng_for(self.rng_seed, "q_audit", step)
            pick = rng.choice(len(ready_slots), size=k, replace=False)
            audit_slots = {ready_slots[int(i)] for i in pick}
        routes = {}
        for i in range(len(qids)):
            if i in audit_slots:
                routes[i] = "audit"
            elif i in ready_slots:
                routes[i] = "short"
            else:
                routes[i] = "full"
        self._route_cache = (step, routes)
        return routes

    def budget_cap_for(self, route: str, prefix_len: int) -> int:
        """max NEW tokens for this row. The agent already caps at
        response_length - prefix_len; 'short' additionally caps at g."""
        full = self.response_length - prefix_len
        if route == "short":
            return min(self.budget_g, full)
        return full

    # ----------------------------------------------------------- Q wave assembly
    def build_wave_rows(self, *, step, extra, uids, resp_ids_of_row, gen_len_of_row,
                        decoded_of_row, probes: bool = True) -> list[dict]:
        """Assemble the post-generation Q wave: consumed-Q calls, readiness probes, and
        the audit-lane counterfactual calls.

        Inputs are per-row accessors over the TRAIN batch (driver side, at theta_0):
          extra[i] dict, uids[i], resp_ids_of_row(i)->valid response ids (prefix+gen),
          gen_len_of_row(i)->generated (non-prefix) token count,
          decoded_of_row(i)->decoded response text. (The statement prompt ids come from
          prompt_ids_for_qid — identical to the rollout's templated prompt.)

        Returns wave rows: {kind: consumed|probe|audit_cut, row (consumed/audit_cut) /
        group uid (probe), ctx_ids, qid, variant, overflow flag}. Overflowed consumed rows
        are still returned (flagged) so the caller can drop those rollouts; overflowed
        probes and audit_cut rows are returned flagged to record "no observation".

        probes=False (dynamic group size, sp_dyn_group): this call covers a LATER round of
        the same step, so the per-uid one-shot diagnostics -- the readiness probe and the
        audit_cut counterfactual -- are skipped; only consumed (and seg) rows are built.
        Round 1 passes probes=True and owns those diagnostics for the step.
        """
        rows = []
        n = len(extra)
        seen_probe_uid = set()
        audit_cand: dict[str, list[int]] = {}
        for i in range(n):
            ei = extra[i]
            route = ei.get("sp_q_route", "full")
            qid = ei.get("sp_qid")
            prefix_len = int(ei.get("sp_prefix_len", 0))
            # Exclude the STATEMENT (scratch inflow) lane only. `prefix_len <= 0` is NOT
            # an equivalent test: it is an exact proxy for "statement row" ONLY while
            # every replay row has a non-empty prefix. With SP_REPLAY_CUT_GRAIN the cut
            # is drawn from the multiples of the grain inside [cut_low*t, cut_high*t], so
            # k=0 is a legitimate draw and a sizeable share (~27%) of REPLAY rows have
            # prefix_len == 0. A cut at 0 is a cut like any other: the state it denotes
            # -- (problem, empty attempt) -- is exactly V(s_0), and Q must probe it, form
            # targets from it, and be consumable for it the same as for a 10k or 20k cut.
            # Test the real predicate, not the proxy.
            if _is_statement_row(ei):
                continue

            # --- readiness probe: one per sampled training PREFIX (uid group) ---
            uid = str(uids[i])
            if probes and uid not in seen_probe_uid:
                seen_probe_uid.add(uid)
                prefix_ids = resp_ids_of_row(i)[:prefix_len]
                ref = self.reference_for(qid, self.trajectory_ref_proof(ei.get("sp_entry_id", "")))
                ctx = self.build_q_context_ids(qid, prefix_ids, ref)
                rows.append({
                    "kind": "probe", "uid": uid, "qid": qid, "row": i,
                    "variant": "ref" if ref is not None else "noref",
                    "ctx_ids": ctx, "overflow": not self.fits(ctx),
                })

            # --- SEGMENTED-Q interior read-out: one call per training row at
            #     the ABSOLUTE offset seg_k into the GENERATED tokens, i.e. the state
            #     y_{t+K}. Feeds the sp_segment advantage estimator, never a reward, never a
            #     Q target, never readiness.
            #
            #     Lane routing: short AND full are split; AUDIT IS NOT. The audit lane is
            #     the only unbiased sample of ready-problem performance, and audit-lane
            #     qbar-rbar drift is the detector for segment-1 bootstrap gaming. Splitting
            #     audit rows would aim that detector at a lane the new gradient had itself
            #     perturbed, and it would also remove the run's untouched control.
            #
            #     gen_len > seg_k (strict): at equality segment 2 would be empty.
            #     Interior contexts are SHORTER than the `consumed` ones built below from the
            #     same row, so this adds no new ctx_limit overflow risk; an overflowed row is
            #     still returned flagged and falls back to single-segment rather than being
            #     silently scored as zero.
            if self.seg_enable and route != "audit" and gen_len_of_row(i) > self.seg_k:
                seg_bound = prefix_len + self.seg_k
                ref = self.reference_for(qid, self.trajectory_ref_proof(ei.get("sp_entry_id", "")))
                ctx = self.build_q_context_ids(qid, resp_ids_of_row(i)[:seg_bound], ref)
                rows.append({
                    "kind": "seg", "uid": uid, "qid": qid, "row": i,
                    "variant": "ref" if ref is not None else "noref",
                    "ctx_ids": ctx, "overflow": not self.fits(ctx),
                    # lane is carried so every q/seg_* metric can be reported pooled AND
                    # split short-vs-full: ~2/3 of split rows are short, where
                    # segment 2 is Q-minus-Q with no judge in the row.
                    "lane": route,
                    "seg_bound": int(seg_bound),
                })

            # --- audit-lane counterfactual: this rollout ran the FULL budget
            #     and earns a real terminal judge reward. Remember it when the short lane
            #     WOULD have cut it and consumed Q, so one member per group can be measured
            #     at that exact cut state below. Measurement only: never a reward, never a
            #     Q target, never touches PPO. ---
            if probes and route == "audit" and self.audit_cut_probes:
                cap = self.budget_cap_for("short", prefix_len)
                if cap < self.response_length - prefix_len and gen_len_of_row(i) >= cap:
                    audit_cand.setdefault(uid, []).append(i)

            # --- consumed-Q call: ready, non-audit, hit the g cap short of
            #     the global window, no extractable proof ---
            if route != "short":
                continue
            cap = self.budget_cap_for("short", prefix_len)
            if cap >= self.response_length - prefix_len:
                continue  # g cap == global cap -> final-judge route
            if gen_len_of_row(i) < cap:
                continue  # ended naturally -> final-judge route
            if _extract_proof_text(decoded_of_row(i)) is not None:
                continue  # proof extractable -> final judge
            attempt_ids = resp_ids_of_row(i)
            ref = self.reference_for(qid, self.trajectory_ref_proof(ei.get("sp_entry_id", "")))
            ctx = self.build_q_context_ids(qid, attempt_ids, ref)
            rows.append({
                "kind": "consumed", "uid": uid, "qid": qid, "row": i,
                "variant": "ref" if ref is not None else "noref",
                "ctx_ids": ctx, "overflow": not self.fits(ctx),
            })
        rows.extend(self._audit_cut_rows(step, extra, audit_cand, resp_ids_of_row))
        return rows

    def _audit_cut_rows(self, step, extra, audit_cand, resp_ids_of_row) -> list[dict]:
        """One counterfactual Q call per audit group, evaluated at the state where
        the short lane would have been cut (prefix + g tokens) — i.e. exactly the value
        that WOULD have been consumed, so the panel can compare it against the realized
        terminal reward of the SAME trajectory. The member is a pinned per-(step, uid)
        draw. Groups whose cut state already contains an extractable proof are skipped:
        those would have taken the final-judge route, not Q."""
        out = []
        for uid in sorted(audit_cand):
            cand = audit_cand[uid]
            rng = rng_for(self.rng_seed, "audit_cut", step, uid)
            i = cand[int(rng.integers(len(cand)))]
            ei = extra[i]
            prefix_len = int(ei["sp_prefix_len"])
            cut = prefix_len + self.budget_cap_for("short", prefix_len)
            cut_ids = list(resp_ids_of_row(i))[:cut]
            cut_text = self.tokenizer.decode(cut_ids, skip_special_tokens=True)
            if _extract_proof_text(cut_text) is not None:
                continue
            qid = ei.get("sp_qid")
            ref = self.reference_for(qid, self.trajectory_ref_proof(ei.get("sp_entry_id", "")))
            ctx = self.build_q_context_ids(qid, cut_ids, ref)
            out.append({
                "kind": "audit_cut", "uid": uid, "qid": qid, "row": i,
                "variant": "ref" if ref is not None else "noref",
                "ctx_ids": ctx, "overflow": not self.fits(ctx),
            })
        return out

    def record_wave_results(self, step: int, results: list[dict], append: bool = False):
        """Stash parsed wave outputs for the reward-site update. results: the wave rows
        + {'value': grid float|None, 'gen_text': str, 'fail_reason': str|None}.

        append=True (dynamic group size): a later round of the SAME step adds its rows to
        the stash instead of replacing it, so the reward site sees every round's consumed
        rows and the round-1 probes. Appending to a different step is refused."""
        if append and self._wave_result is not None:
            assert self._wave_result["step"] == step, (
                f"sp_q wave append: stash holds step {self._wave_result['step']}, got {step}")
            self._wave_result["rows"].extend(results)
            return
        self._wave_result = {"step": step, "rows": list(results)}

    def wave_result(self, step: int) -> list[dict]:
        assert self._wave_result is not None and self._wave_result["step"] == step, (
            "sp_q wave results missing for this step (wave did not run before reward?)"
        )
        return self._wave_result["rows"]

    # --------------------------------------------------------- reward-site update
    def observe_and_update(self, *, non_tensor_batch, seq_rewards, valid_flags,
                           resp_ids_of_row) -> dict:
        """Q FIFO admission + eviction, readiness errors/transitions, fallback-bank
        add-once. Runs at the REWARD site for ONE train batch; the delta record is
        BUILT here but written by append_delta(displacement) after the update, still
        before this step's checkpoint (durability ordering).

        seq_rewards[i]: the row's scalar reward (judge points/7, materialized zero, or
        consumed Q value). valid_flags[i]: whether the row is a VALID scored member
        (False = judge http/parse failure or invalid/overflowed consumed-Q -> excluded
        from targets and dropped from PPO by the caller).
        """
        extra = non_tensor_batch["extra_info"]
        uids = non_tensor_batch["uid"]
        n = len(extra)
        steps = {int(ei["sp_dataset_step"]) for ei in extra}
        assert len(steps) == 1, f"sp_q hook: mixed dataset steps: {steps}"
        step = steps.pop()
        assert step == self.next_dataset_step, (
            f"sp_q hook: batch is dataset_step {step} but cursor expects {self.next_dataset_step}"
        )

        wave = {(r["kind"], r["uid"]): r for r in self.wave_result(step)}

        # ---- group rows by uid (training prefixes only) ----
        groups: dict[str, list[int]] = {}
        for i in range(n):
            # See build_wave_rows: a replay row with a k=0 cut is an ordinary cut state
            # and MUST form a Q target; only the statement inflow lane is excluded.
            if not _is_statement_row(extra[i]):
                groups.setdefault(str(uids[i]), []).append(i)

        admitted, admitted_recs = [], []
        errors_this_step: list[float] = []
        probe_records = []
        below_min_valid = 0
        td_admitted = 0
        td_admit_capped = 0
        _td_slot_admits: dict = {}
        for uid, rows_i in sorted(groups.items()):
            i0 = rows_i[0]
            ei0 = extra[i0]
            qid = ei0["sp_qid"]
            route = ei0.get("sp_q_route", "full")
            prefix_len = int(ei0["sp_prefix_len"])
            valid_rows = [i for i in rows_i if valid_flags[i]]
            # ---- no-group TD: the BELLMAN backup. A TD row is
            # a singleton uid by construction, so the grouped lanes' min_valid=8 quality
            # floor would exclude every one of them and ready problems would exit Q
            # training entirely. Instead a valid TD singleton IS admitted: its record is
            #     (state = p_i,  z = grid(r_i))
            # with r_i the consumed Q(p_i + g continuation) -- a one-macro-step TD(0)
            # target, tagged "q_bootstrap" by the provenance logic below -- or the judge
            # score when the row finished early (a TERMINAL transition, tagged
            # "terminal_group"). The grouped floor is untouched for every other lane.
            td_row = bool(int(ei0.get("sp_q_td_row", 0)))
            if len(valid_rows) < (1 if td_row else self.min_valid):
                below_min_valid += 1
                continue
            # Volume control: each admitted record embeds its full prefix_token_ids in the
            # append-only delta log, and a ready slot yields up to 16 singletons where the
            # grouped lane yielded ONE record -- unbounded, a high-ready-fraction step
            # writes ~80MB of log. td_admit_per_slot caps admissions per (step, slot);
            # which siblings land is uuid-sort order, i.e. arbitrary-but-deterministic
            # over cuts the slot's own draw already permuted. 0 = admit all.
            if td_row and self.td_admit_per_slot > 0:
                _slot_key = (int(ei0.get("sp_dataset_step", step)),
                             int(ei0.get("sp_td_slot", -1)))
                if _td_slot_admits.get(_slot_key, 0) >= self.td_admit_per_slot:
                    td_admit_capped += 1
                    continue
                _td_slot_admits[_slot_key] = _td_slot_admits.get(_slot_key, 0) + 1
            if td_row:
                td_admitted += 1
            vals = [float(seq_rewards[i]) for i in valid_rows]
            raw_mean = sum(vals) / len(vals)
            z = round_to_grid(raw_mean)
            # provenance tag
            kinds = {extra[i].get("sp_q_route_taken", "judge") for i in valid_rows}
            if kinds == {"q_consumed"}:
                tag = "q_bootstrap"
            elif "q_consumed" in kinds:
                tag = "mixed_final_q"
            else:
                tag = "terminal_group"
            ref_proof = self.trajectory_ref_proof(ei0.get("sp_entry_id", ""))
            rec = {
                "rec_id": f"online:{step}:{uid}",
                "seq": self.q_seq_next,
                "qid": qid,
                "prefix_token_ids": [int(t) for t in resp_ids_of_row(i0)[:prefix_len]],
                "z": z,
                "raw_mean": raw_mean,
                "n_valid": len(valid_rows),
                "n_members": len(rows_i),
                "source_tag": tag,
                "step": step,
                "uid": uid,
                "ref_proof": ref_proof,
            }
            self.q_seq_next += 1
            self.fifo[rec["seq"]] = rec
            admitted.append(rec["seq"])
            admitted_recs.append(rec)

            # ---- fresh readiness point: probe error vs this group's target ----
            pr = wave.get(("probe", uid))
            if pr is not None and not pr["overflow"]:
                v = pr.get("value")
                if v is None:
                    e = 1.0  # invalid probe -> error 1.0 (the "error readiness" convention)
                else:
                    e = abs(float(v) - z)
                errors_this_step.append(e)
                # route + source_tag are PROVENANCE: a probe whose target z came
                # from consumed Q ("q_bootstrap"/"mixed_final_q") is Q measured against
                # itself and must be separable downstream from a real terminal target.
                probe_records.append({
                    "uid": uid, "qid": qid, "prediction": v, "z": z, "error": e,
                    "variant": pr["variant"], "fail_reason": pr.get("fail_reason"),
                    "route": route, "source_tag": tag,
                })

        # records admitted THIS step, for the separate-Q new-records-only draw
        self._last_admitted_recs = list(admitted_recs)

        # ---- constant-capacity FIFO eviction ----
        evicted = []
        while len(self.fifo) > self.fifo_cap:
            seq, _ = self.fifo.popitem(last=False)
            evicted.append(seq)

        # ---- pooled 5-step MAE + transitions ----
        self.mae_tail.append((step, errors_this_step))
        self.mae_tail = self.mae_tail[-self.mae_window:]
        pooled = [e for _, es in self.mae_tail for e in es]
        window_complete = len(self.mae_tail) == self.mae_window
        mae5 = (sum(pooled) / len(pooled)) if (pooled and window_complete) else None
        # global-mode routing state: the gate as of THIS step (routes the next one) and the
        # monotone latch. Updated in every mode (cheap; the metrics below report it), consulted
        # by is_ready() only when sp_q_ready_mode == "global".
        self.global_gate_open = bool(mae5 is not None and mae5 < self.ready_thresh_global)
        if self.global_gate_open and not self.global_ready_latched:
            self.global_ready_latched = True
            print(f"[sp_q] GLOBAL gate first opened at dataset_step={step} (mae5={mae5:.4f} < "
                  f"{self.ready_thresh_global}); latched"
                  + (" -> ALL replay slots route to Q from the next step on"
                     if self.ready_mode == "global" and self.ready_global_latch else ""),
                  flush=True)
        # latch nonzero-Q sightings (monotone, tracked regardless of the knob)
        for prb in probe_records:
            v = prb["prediction"]
            if v is not None and float(v) > 1e-9:
                self.q_nonzero_seen[prb["qid"]] = True
        transitions = []
        n_blocked_no_bank = 0
        if mae5 is not None and mae5 < self.ready_thresh_global:
            for prb in probe_records:
                # observability for SP_Q_READY_REQUIRE_BANK: these would have gone ready under
                # the error-only gate. If this stays 0 the flag is doing nothing.
                if (self.ready_require_bank and prb["prediction"] is not None
                        and prb["error"] < self.ready_thresh_problem
                        and not self.ready.get(prb["qid"], False)
                        and (not self.require_nonzero
                             or self.q_nonzero_seen.get(prb["qid"], False))
                        and prb["qid"] not in self.bank):
                    n_blocked_no_bank += 1
                if (prb["prediction"] is not None and prb["error"] < self.ready_thresh_problem
                        and not self.ready.get(prb["qid"], False)
                        and (not self.require_nonzero
                             or self.q_nonzero_seen.get(prb["qid"], False))
                        and (not self.ready_require_bank
                             or prb["qid"] in self.bank)):
                    self.ready[prb["qid"]] = True
                    transitions.append(prb["qid"])

        # ---- fallback bank add-once: uncovered problems with a judged-passing row ----
        bank_added = []
        pjs = non_tensor_batch.get("prover_judge_score")
        if pjs is not None:
            cand: dict[str, list[int]] = {}
            for i in range(n):
                qid = extra[i].get("sp_qid")
                if not qid or qid in self.bank:
                    continue
                try:
                    passed = int(round(float(pjs[i]))) >= 1
                except (TypeError, ValueError):
                    passed = False
                if passed and valid_flags[i]:
                    cand.setdefault(qid, []).append(i)
            for qid in sorted(cand):
                rows_i = cand[qid]
                rng = rng_for(self.rng_seed, "bank_pick", step, qid)
                i = rows_i[int(rng.integers(len(rows_i)))]
                text = self.tokenizer.decode(resp_ids_of_row(i), skip_special_tokens=True)
                proof = _extract_proof_text(text)
                if proof:
                    b = {"qid": qid, "proof": proof, "source": f"online:{step}:{uids[i]}"}
                    self.bank[qid] = b
                    bank_added.append(b)

        # ---- audit-lane calibration pairs: the Q value that WOULD have been
        #      consumed (measured at the cut state) vs the REALIZED terminal reward of the
        #      same audit rollout. This is the ground-truth check that Q consumption is not
        #      rewarding failures; it never feeds targets, readiness, or PPO. ----
        #      The measured member is found via its `sp_q_audit_cut` STAMP, not via the
        #      wave row's index: the batch is reordered (_balance_batch) between the wave
        #      and this site, so an index would pair the Q value with another rollout's
        #      reward. Stamps travel with their row; uid identifies the group.
        wave_rows = self.wave_result(step)
        wave_audit = {r["uid"]: r for r in wave_rows if r["kind"] == "audit_cut"}
        audit_pairs = []
        for i in range(n if wave_audit else 0):
            if not extra[i].get("sp_q_audit_cut"):
                continue
            r = wave_audit.get(str(uids[i]))
            if r is None or r["overflow"] or not valid_flags[i]:
                continue  # no observation / no trustworthy terminal reward to compare to
            audit_pairs.append({
                "uid": r["uid"], "qid": r["qid"],
                "q_at_cut": r.get("value"),
                "terminal_reward": float(seq_rewards[i]),
                "variant": r["variant"], "fail_reason": r.get("fail_reason"),
            })
        audit_valid = [p for p in audit_pairs if p["q_at_cut"] is not None]
        audit_mae = (sum(abs(p["q_at_cut"] - p["terminal_reward"]) for p in audit_valid)
                     / len(audit_valid)) if audit_valid else float("nan")

        # ---- pending delta: written by append_delta() after the update ----
        consumed_rows = [r for r in wave_rows if r["kind"] == "consumed"]
        self._pending_delta = {
            "dataset_step": step,
            "admitted": admitted_recs,
            "evicted_seqs": evicted,
            "bank_added": bank_added,
            "probes": probe_records,
            "audit_pairs": audit_pairs,
            "probe_overflows": sum(1 for r in wave_rows if r["kind"] == "probe" and r["overflow"]),
            "consumed_calls": len(consumed_rows),
            "consumed_invalid": sum(1 for r in consumed_rows
                                    if r["overflow"] or r.get("value") is None),
            "mae5": mae5,
            "transitions": transitions,
            "ready_total": sum(1 for v in self.ready.values() if v),
            "displacement": {},
        }

        # ---- metrics (trainer keys) ----
        routes = [extra[i].get("sp_q_route", "full") for i in range(n)
                  if not _is_statement_row(extra[i])]
        n_ready_sampled = sum(1 for r in routes if r in ("short", "audit"))
        n_audit = sum(1 for r in routes if r == "audit")
        probe_invalid = sum(1 for p in probe_records if p["prediction"] is None)
        m = {
            # Both thresholds, every step, so the dashboard plots the gate the run actually
            # used instead of a hardcoded number. Constant series, but cheap and
            # self-documenting in the metrics file.
            "q/ready_thresh_global": self.ready_thresh_global,
            "q/ready_thresh_problem": self.ready_thresh_problem,
            "q/global_gate_open": int(mae5 is not None and mae5 < self.ready_thresh_global),
            # readiness mode: 1 = every replay slot's route follows the global gate.
            # global_ready_effective is what NEXT step's route_plan will see: in problem mode
            # it is always 0 (routing is per-problem there) so the panel cannot be misread.
            "q/ready_mode_global": int(self.ready_mode == "global"),
            "q/global_ready_latched": int(self.global_ready_latched),
            "q/global_ready_effective": int(self.ready_mode == "global" and self._global_all_ready()),
            "q/readiness_mae5": mae5 if mae5 is not None else -1.0,
            "q/readiness_mae5_n": len(pooled) if window_complete else -1,
            "q/probe_mae_fresh": (sum(errors_this_step) / len(errors_this_step))
                                 if errors_this_step else float("nan"),
            "q/readiness_obs": len(probe_records),
            "q/readiness_obs_invalid": probe_invalid,
            "q/readiness_transitions": len(transitions),
            "q/ready_problems": sum(1 for v in self.ready.values() if v),
            "q/ready_blocked_no_bank": n_blocked_no_bank,
            "q/ready_require_bank": int(self.ready_require_bank),
            "q/nonzero_seen_total": sum(1 for v in self.q_nonzero_seen.values() if v),
            "q/ready_sampled_prefixes": n_ready_sampled,
            "q/ready_fraction_sampled": n_ready_sampled / max(len(routes), 1),
            "q/audit_prefixes": n_audit,
            "q/fifo_size": len(self.fifo),
            # Tier-1 references refused for not being judged-correct. THIS STEP's count (reset
            # every step), so it can legitimately rise and fall: under ungated admission with a
            # cold bank it should be high early (most attempts fail) and fall as the run starts
            # solving problems. A flat 0 means the gate is off or judge_pass is missing from the
            # entries -- i.e. failed attempts may be reaching Q as "reference correct proofs".
            # THIS STEP's rejections = the rise in the cumulative total since the previous step
            # closed. Order-independent: it counts the wave's rejections and admission's alike,
            # wherever in the step they happened.
            "q/ref_rejected_unpassed": (
                self.ref_rejected_unpassed_total - self._ref_rej_total_at_step_start
            ),
            # ...and the run-to-date total, which IS cumulative and is persisted across resumes.
            "q/ref_rejected_unpassed_total": self.ref_rejected_unpassed_total,
            "q/capacity": self.fifo_cap,
            "q/admitted": len(admitted),
            "q/evicted": len(evicted),
            "q/target_below_min_valid": below_min_valid,
            "q/td_targets_admitted": td_admitted,
            "q/td_admit_capped": td_admit_capped,
            "q/consumed_calls": len(consumed_rows),
            "q/consumed_invalid": self._pending_delta["consumed_invalid"],
            "q/bank_size": len(self.bank),
            "q/bank_added": len(bank_added),
            "q/audit_cut_calls": sum(1 for r in wave_rows if r["kind"] == "audit_cut"),
            # Segmented-Q wave COST. One extra call per split row roughly DOUBLES the wave, so this must be visible
            # from step 1. Counted here from the wave rows themselves -- q/seg_rows in the
            # estimator's panels counts rows successfully SPLIT, which excludes the
            # overflowed and unparseable ones that still cost an engine call.
            "q/seg_wave_calls": sum(1 for r in wave_rows if r["kind"] == "seg"),
            "q/seg_wave_invalid": sum(1 for r in wave_rows if r["kind"] == "seg"
                                      and (r.get("overflow") or r.get("value") is None)),
            "q/audit_cut_pairs": len(audit_pairs),
            "q/audit_cut_invalid": len(audit_pairs) - len(audit_valid),
            "q/audit_cut_mae": audit_mae,
            "q/audit_distinct_problems": len({p["qid"] for p in audit_pairs}),
        }
        tags = [r["source_tag"] for r in self.fifo.values()]
        if admitted_recs:
            adm_tags = [r["source_tag"] for r in admitted_recs]
            for t in ("terminal_group", "q_bootstrap", "mixed_final_q"):
                m[f"q/admit_frac_{t}"] = adm_tags.count(t) / len(adm_tags)
        for t in ("terminal_group", "q_bootstrap", "mixed_final_q"):
            m[f"q/fifo_frac_{t}"] = tags.count(t) / max(len(tags), 1)
        return m

    def append_delta(self, displacement: dict):
        """Write the step's delta record (built at the reward site) with the update's
        displacement norms folded in, advance the cursor, and clear per-step caches.
        Called AFTER the PPO+Q update, BEFORE _save_checkpoint."""
        delta = self._pending_delta
        assert delta is not None, "sp_q append_delta called without a pending delta"
        delta["displacement"] = displacement or {}
        dq = float(displacement.get("q/delta_q_applied", 0.0) or 0.0)
        dp = float(displacement.get("q/delta_ppo_net", 0.0) or 0.0)
        if math.isfinite(dq):
            self.cum_delta_q += dq
        if math.isfinite(dp):
            self.cum_delta_ppo_net += dp
        delta["cum_delta_q"] = self.cum_delta_q
        delta["cum_delta_ppo_net"] = self.cum_delta_ppo_net
        os.makedirs(self.delta_dir, exist_ok=True)
        with open(self._delta_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(delta, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())  # delta k durable BEFORE checkpoint k exists
        self.next_dataset_step = int(delta["dataset_step"]) + 1
        # close the per-step rejection window (see __init__)
        self._ref_rej_total_at_step_start = self.ref_rejected_unpassed_total
        self._pending_delta = None
        self._wave_result = None
        self._route_cache = None
        self._train_sample = None

    # --------------------------------------------------------- Q training sample
    def draw_train_sample(self, step: int) -> list[dict]:
        """Uniform permutation walk over B^Q with the C_Q packing check; first
        train_n fit-eligible records. Returns materialized rows
        {rec, variant, ctx_ids, target_ids, weight}.

        By default, train the WITH-REFERENCE variant only, weight 1 per record -- the
        no-reference variant is never consumed (tier-1 references resolve ~100%), so
        training it would be ~2x Q-phase cost of pure regularization, and dropping it
        makes the trained format exactly the consumed format. A record with NO
        resolvable reference is SKIPPED from training (tallied; it still probes/consumes
        via the no-reference prompt and simply stays unready until covered).
        sp_q_train_noref=1 trains both variants at weight 1/2 each as an ablation.
        """
        seqs = list(self.fifo.keys())
        rng = rng_for(self.rng_seed, "q_train_sample", step)
        perm = rng.permutation(len(seqs))
        picked, rows = [], []
        overflows = {"ref_only": 0, "noref_only": 0, "both": 0}
        skipped_no_ref = 0
        for j in perm:
            if len(picked) >= self.train_n:
                break
            rec = self.fifo[seqs[int(j)]]
            ref = self.reference_for(rec["qid"], rec.get("ref_proof"))
            prefix = [int(t) for t in rec["prefix_token_ids"]]
            z = round_to_grid(float(rec["z"]))
            if not self.train_noref:
                # ref-only training (the pinned default)
                if ref is None:
                    skipped_no_ref += 1
                    continue
                ctx_ref = self.build_q_context_ids(rec["qid"], prefix, ref)
                if self.fits(ctx_ref):
                    rows.append({"rec": rec, "variant": "ref", "ctx_ids": ctx_ref,
                                 "target_ids": self.target_ids[z], "weight": 1.0})
                    picked.append(rec["seq"])
                else:
                    overflows["ref_only"] += 1
                continue
            # ablation path (sp_q_train_noref=1): both variants at weight 1/2 each
            ctx_noref = self.build_q_context_ids(rec["qid"], prefix, None)
            fits_noref = self.fits(ctx_noref)
            if ref is not None:
                ctx_ref = self.build_q_context_ids(rec["qid"], prefix, ref)
                fits_ref = self.fits(ctx_ref)
                if fits_noref and fits_ref:
                    rows.append({"rec": rec, "variant": "noref", "ctx_ids": ctx_noref,
                                 "target_ids": self.target_ids[z], "weight": 0.5})
                    rows.append({"rec": rec, "variant": "ref", "ctx_ids": ctx_ref,
                                 "target_ids": self.target_ids[z], "weight": 0.5})
                    picked.append(rec["seq"])
                else:
                    key = ("both" if not fits_noref and not fits_ref
                           else "ref_only" if not fits_ref else "noref_only")
                    overflows[key] += 1
            else:
                if fits_noref:
                    rows.append({"rec": rec, "variant": "noref", "ctx_ids": ctx_noref,
                                 "target_ids": self.target_ids[z], "weight": 1.0})
                    picked.append(rec["seq"])
                else:
                    overflows["noref_only"] += 1
        self._train_sample = {
            "step": step, "picked_seqs": picked, "rows": rows, "overflows": overflows,
            "skipped_no_ref": skipped_no_ref,
        }
        return rows

    def draw_new_records(self, step: int) -> list[dict]:
        """Separate-Q: train ONLY on the records newly admitted to B^Q this
        step, with-reference variant, same fit/format as draw_train_sample. Sets
        self._train_sample so the driver's separate-Q phase consumes it identically."""
        recs = list(getattr(self, "_last_admitted_recs", []) or [])
        picked, rows, skipped_no_ref, overflow = [], [], 0, 0
        for rec in recs:
            ref = self.reference_for(rec["qid"], rec.get("ref_proof"))
            if ref is None:
                skipped_no_ref += 1
                continue
            prefix = [int(t) for t in rec["prefix_token_ids"]]
            z = round_to_grid(float(rec["z"]))
            ctx_ref = self.build_q_context_ids(rec["qid"], prefix, ref)
            if self.fits(ctx_ref):
                rows.append({"rec": rec, "variant": "ref", "ctx_ids": ctx_ref,
                             "target_ids": self.target_ids[z], "weight": 1.0})
                picked.append(rec["seq"])
            else:
                overflow += 1
        self._train_sample = {
            "step": step, "picked_seqs": picked, "rows": rows,
            "overflows": {"ref_only": overflow, "noref_only": 0, "both": 0},
            "skipped_no_ref": skipped_no_ref,
        }
        return rows

    def load_readiness_warmstart(self, path: str):
        """Seed the readiness gate from the SFT prequential simulation: the
        per-problem monotone `ready` flags AND the pooled `mae_tail` (so the GLOBAL gate
        initializes ready iff the SFT-sim pooled MAE5 < thresh). Same predict-then-learn
        criterion as the online gate -> authoritative, no online confirmation. Applied once
        (branch-point only); the resumed q_state.json already carries the result."""
        # RESET FIRST, before deciding whether a seed exists: a parent q_state.json written
        # by a fused (tied) Q carries that Q's ready flags, which are NOT valid for the
        # separate Q. Clearing up front guarantees that a missing/unreadable
        # warm-start can never let the fused flags leak in — the separate gate then starts
        # COLD (nothing ready) rather than inheriting an invalid table.
        self.ready = {}
        self.mae_tail = []
        if not path or not os.path.exists(path):
            print(f"[sp_q-sep] WARNING: no readiness warm-start at {path!r} — fused flags CLEARED; "
                  "gate starts COLD (0 ready). Unexpected: run_attach rebuilds+preflights the seed "
                  "every attach — check the prequential log / build step.", flush=True)
            self._readiness_warmstart = {"provenance": "cold_no_warmstart", "source": path, "n_ready": 0}
            return
        with open(path) as f:
            ws = json.load(f)
        # Threshold must match the live gate: the seed's ready/gate decisions were computed
        # against ws["thresh"]; if the run uses a different threshold the seed is inconsistent
        # with the online criterion — fail closed. Compared against the GLOBAL
        # threshold: ws["mae_tail"] feeds the pooled gate, which is the global one.
        ws_thresh = ws.get("thresh")
        if ws_thresh is not None and abs(float(ws_thresh) - self.ready_thresh_global) > 1e-9:
            raise RuntimeError(
                f"[sp_q-sep] warm-start thresh {ws_thresh} != live global readiness threshold "
                f"{self.ready_thresh_global}; rebuild the seed at the matching threshold")
        n_ready = 0
        for qid, r in (ws.get("ready") or {}).items():
            if r:
                self.ready[qid] = True
                n_ready += 1
        tail = ws.get("mae_tail")
        if tail:
            self.mae_tail = [(int(s), [float(e) for e in es]) for s, es in tail][-self.mae_window:]
        # provenance marker — persisted by save_state, so a later resume can
        # confirm the seed was applied once and never re-applies it.
        self._readiness_warmstart = {"provenance": "sft_warmstart", "source": path, "n_ready": n_ready,
                                     "source_sha256": ws.get("_source_sha256")}
        pooled = [e for _, es in self.mae_tail for e in es]
        mae5 = (sum(pooled) / len(pooled)) if pooled else None
        print(f"[sp_q-sep] readiness warm-start: {n_ready} problems ready, pooled MAE5 seed="
              f"{mae5} (n={len(pooled)}), global-gate {'OPEN' if (mae5 is not None and mae5 < self.ready_thresh_global) else 'closed'}",
              flush=True)

    def train_sample_meta(self) -> dict:
        s = self._train_sample or {"picked_seqs": [], "rows": [], "overflows": {}}
        ages = []
        for seq in s["picked_seqs"]:
            rec = self.fifo.get(seq)
            if rec is not None:
                ages.append(self.q_seq_next - seq)
        return {
            "q/q_records_trained": len(s["picked_seqs"]),
            "q/sample_overflowed": sum(s["overflows"].values()) if s["overflows"] else 0,
            "q/sample_overflow_ref_only": s["overflows"].get("ref_only", 0),
            "q/sample_overflow_noref_only": s["overflows"].get("noref_only", 0),
            "q/sample_skipped_no_reference": s.get("skipped_no_ref", 0),
            "q/sample_age_mean": (sum(ages) / len(ages)) if ages else float("nan"),
            "q/sample_age_max": max(ages) if ages else float("nan"),
        }


def _extract_proof_text(decoded_text: str):
    """Same proof-extraction convention as the reward path (<proof>...</proof> after the
    last </think>)."""
    idx = decoded_text.rfind(THINK_CLOSE)
    post = decoded_text[idx + len(THINK_CLOSE):] if idx >= 0 else decoded_text
    m = re.search(r"<proof>(.*?)</proof>", post, re.DOTALL)
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# module-level API (all no-ops unless enabled)
# ---------------------------------------------------------------------------

def install(cfg: dict, tokenizer, replay_harness, raw_prompt_of_qid) -> QHarness:
    assert enabled(), "sp_q_readiness.install called but SP_Q_ENABLE is not set"
    assert _S["harness"] is None, "sp_q_readiness.install called twice"
    _S["harness"] = QHarness(cfg, tokenizer, replay_harness, raw_prompt_of_qid)
    return _S["harness"]


def on_checkpoint_load(ckpt_dir):
    if not enabled() or not installed():
        return
    harness().finalize_from_checkpoint(ckpt_dir)


def save_state(ckpt_dir: str):
    """Persist q_state.json. Deliberately FAILS CLOSED (as does sp_replay.save_state): a
    post-branch checkpoint that lacks Q state is fatal on resume, so publishing one as
    `latest` would strand the run at the next restart. Raising here
    aborts _save_checkpoint before the tracker file is written, and auto-resume replays
    this step from the previous complete checkpoint."""
    if not enabled() or not installed():
        return
    harness().save_state(ckpt_dir)
