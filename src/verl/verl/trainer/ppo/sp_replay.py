"""Replay-prefix off-policy training -- driver-side state.

Replay buffer of judged-correct trajectories whose prefix cuts become training prompts,
following the difficulty.py pattern: a torch-free, env-gated module whose state lives
ONLY in the driver process. Everything is a no-op unless SP_REPLAY_ENABLE=1.

Mechanism per dataset step (batch n_base=256):
  * slots [0, n_orig)          -> original-statement rows, uniform WITHOUT replacement;
  * slots [n_orig, n_base)     -> replay rows: 192 DISTINCT buffer-covered problems drawn
    without replacement with weight w = 3 - 2.5*EMA(from-scratch success), then one stored
    judged-correct trajectory uniform from that problem's bucket (capacity 5), cut at a
    uniform point in [30%, 80%] of its token ids. The prefix rides in the RESPONSE with
    response_mask=0 (see prefix_agent_loop.py) so it is never trained on.
  * EMA prior = bucket occupancy k/capacity: a problem with k
    stored proofs starts at ema=k/5; materialized lazily at first touch.
  * post-reward hook (observe_and_update): EMA update from prefix-free rows only; every
    judged-passing trajectory admitted at capacity 5, replace-oldest (flat per-question policy);
    one append-only delta per step keyed by the LOCAL dataset_step.

Resume contract (never reconcile against the trainer's global counter):
  * sp_replay_state.json is saved into every checkpoint dir: the EMA table + the
    dataset-step cursor (= number of completed local batches) + the seed manifest sha.
  * On load, deltas with dataset_step >= cursor are dropped AND physically truncated from
    the on-disk log (atomic rewrite) -- without truncation a resumed run would rebuild the
    same step, append a second delta for it, and a later resume would double-apply.
  * The hook runs at the REWARD site (pre-optimizer-update): in this fork _save_checkpoint
    fires inside the step right after update_actor, so the reward-site placement is what
    guarantees "delta k is durable before checkpoint k exists".
  * Replay-enabled runs skip the dataloader state restore (like difficulty sampling); the
    dataset offsets its local step by the restored cursor instead, so batch composition is
    a pure function of (rng seed, dataset_step) across resumes.

Determinism: every draw is keyed (rng_seed, purpose, dataset_step[, slot, ...]) via a
sha256-seeded numpy Generator, so the batch stream is reproducible on resume. Requires
data.shuffle=False, SequentialSampler, dataloader_num_workers=0 (lazy driver-process row
building: batch k+1 is only built after step k's hook ran).
"""
from __future__ import annotations

import hashlib
import json
import math
import os

import numpy as np

DELTA_LOG_NAME = "replay_buffer_deltas.jsonl"
STATE_FILE_NAME = "sp_replay_state.json"
MANIFEST_NAME = "replay_buffer_manifest.json"
SHARD_GLOB = "replay_buffer/shard_*.jsonl"

_S = {
    "env_inited": False,
    "enable": False,
    "harness": None,  # ReplayHarness once install() ran (train dataset init)
}


def _env_init():
    if _S["env_inited"]:
        return
    _S["enable"] = os.environ.get("SP_REPLAY_ENABLE", "0") in ("1", "true", "True")
    _S["env_inited"] = True


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0") in ("1", "true", "True")


def _env_step_set(name: str) -> set:
    """Parse a comma-separated step list env var ("80" / "80,120") into a set of ints."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return set()
    return {int(tok) for tok in raw.split(",") if tok.strip()}


def enabled():
    _env_init()
    return _S["enable"]


def installed():
    return _S["harness"] is not None


def harness() -> "ReplayHarness":
    h = _S["harness"]
    assert h is not None, "sp_replay used before install() (train dataset never initialized?)"
    return h


def _is_statement_row(ei) -> bool:
    """True for a row from the SCRATCH INFLOW lane (sp_source_type == "statement").

    Used for the stream-split metrics (replay/pass_rate_orig, replay/*_orig), which the
    dashboard plots as "from-scratch (N stmt rows)". Those must mean the inflow LANE, so
    the panel line agrees with its own row-count label.

    NOT `sp_prefix_len == 0`. That is an exact proxy for "statement row" only while every
    replay row carries a non-empty prefix. With SP_REPLAY_CUT_GRAIN a k=0 cut is a
    legitimate draw, so a sizeable share of REPLAY rows (~27% in one measured step) have
    prefix_len == 0 and would be folded into the from-scratch stream: a stream labelled
    "32 stmt rows" would aggregate ~448 rows and report the replay pass rate instead.

    Falls back to the prefix test when sp_source_type is absent, so any dump without the
    tag keeps the prefix-based behaviour exactly.

    NOTE: statement_row_flags() below deliberately does NOT use this predicate. It feeds
    the AEC entropy control signal, where the right question is "did this row generate
    from scratch?" -- a k=0 cut has no prefix, so it is not the systematically
    lower-entropy prefix-conditioned population the controller must exclude. Different
    question, different predicate; do not unify them.
    """
    if not isinstance(ei, dict):
        return False
    st = ei.get("sp_source_type")
    if st is not None:
        # "cold_scratch" (cold bootstrap) generates from scratch too, so it belongs in
        # the from-scratch stream for these metrics even though it is PPO-trained.
        return str(st) in ("statement", "cold_scratch")
    try:
        return int(ei.get("sp_prefix_len", 0)) == 0
    except (TypeError, ValueError):
        return False


def rng_for(seed: int, *parts) -> np.random.Generator:

    """Independent generator keyed by the run rng seed plus arbitrary parts."""
    key = f"{seed}|" + "|".join(str(p) for p in parts)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest_shards(root: str, recorded: dict, discovered: list, what: str):
    """Two-way manifest check for a sharded frozen artifact.

    Forward: every DISCOVERED shard is recorded, with a matching sha256.
    Reverse: every RECORDED shard still EXISTS. The reverse direction is the one that
    catches a deleted/never-copied shard -- without it the loader happily builds a
    partial seed (fewer records, no error), which is silent science corruption rather
    than a crash.
    """
    rels = set()
    for path in discovered:
        rel = os.path.relpath(path, root)
        rels.add(rel)
        if rel not in recorded:
            raise ValueError(f"{what}: shard {rel} is not in the manifest")
        actual = sha256_file(path)
        if actual != recorded[rel]:
            raise ValueError(
                f"{what}: shard hash mismatch for {rel}: manifest {recorded[rel][:12]}... "
                f"!= actual {actual[:12]}..."
            )
    missing = sorted(set(recorded) - rels)
    if missing:
        shown = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
        raise FileNotFoundError(
            f"{what}: {len(missing)} manifest-recorded shard(s) missing under {root} "
            f"({shown}) -- refusing to load a partial artifact"
        )


def _validate_entry(e: dict, where: str) -> dict:
    for k in ("entry_id", "qid", "response_token_ids"):
        if not e.get(k):
            raise ValueError(f"replay entry missing/empty {k!r} ({where}): {str(e)[:200]}")
    if not isinstance(e["response_token_ids"], list):
        raise ValueError(f"replay entry response_token_ids not a list ({where})")
    return e


class ReplayHarness:
    """Buffer + EMA + deterministic per-step draws. Driver-process only."""

    def __init__(self, cfg: dict, num_statements: int, row_qids: list):
        # ---- config (all pinned by the runner via +data.sp_replay_*) ----
        self.seed_dir = str(cfg["sp_replay_seed_dir"])
        self.delta_dir = str(cfg["sp_replay_delta_dir"])
        self.n_base = int(cfg["train_batch_size"])
        self.n_replay = int(cfg.get("sp_replay_n", 192))
        self.n_orig = self.n_base - self.n_replay
        assert 0 < self.n_replay < self.n_base, (self.n_replay, self.n_base)
        self.capacity = int(cfg.get("sp_replay_capacity", 5))
        self.cut_low = float(cfg.get("sp_replay_cut_low", 0.30))
        self.cut_high = float(cfg.get("sp_replay_cut_high", 0.80))
        # Quantize the cut point to a multiple of this many tokens (0 = off, the
        # original continuous uniform draw). Kept OFF by default so runs that do not
        # set it are bit-for-bit unaffected.
        self.cut_grain = int(cfg.get("sp_replay_cut_grain", 0))
        if self.cut_grain < 0:
            raise ValueError("sp_replay_cut_grain must be >= 0, got %r" % self.cut_grain)
        assert 0.0 <= self.cut_low <= self.cut_high < 1.0, (self.cut_low, self.cut_high)
        self.ema_coef = float(cfg.get("sp_replay_ema_coef", 0.95))
        self.w_base = float(cfg.get("sp_replay_weight_base", 3.0))
        self.w_slope = float(cfg.get("sp_replay_weight_slope", 2.5))
        self.rng_seed = int(cfg.get("sp_replay_rng_seed", 714001))
        self.local_steps = int(cfg.get("sp_replay_local_steps", 100000))

        # ---- replay management policy (bucketing/admission/rotation).
        # Defaults are EXACTLY the original per-question behavior; every non-default combination is
        # opt-in and stamped into sp_replay_state.json (checked on resume, below). ----
        self.bucketing = str(cfg.get("sp_replay_bucketing", "per_question"))
        # Under bucketing="global" the draw is a pinned permutation with wraparound. Over WHAT
        # is the question:
        #   "entry"    (DEFAULT) -- permute ENTRIES. A problem's draw
        #              probability is therefore proportional to how many entries it currently
        #              holds, i.e. an implicit OCCUPANCY weighting with no difficulty signal
        #              in it. Under admission="ungated" occupancy tracks how recently and
        #              often a problem entered the inflow lane.
        #   "question" -- permute the DISTINCT PROBLEMS present in the buffer, then pick one of
        #              that problem's entries uniformly. Every problem in the buffer is equally
        #              likely per slot regardless of how many trajectories it stored.
        # Default is "entry" ON PURPOSE: existing checkpoints were produced with it, and
        # flipping a default would silently change a resumed run's sampling.
        self.global_sampling = str(cfg.get("sp_replay_global_sampling", "entry"))
        if self.global_sampling not in ("entry", "question"):
            raise ValueError(
                f"sp_replay_global_sampling must be 'entry' or 'question', got "
                f"{self.global_sampling!r}"
            )
        self.admission = str(cfg.get("sp_replay_admission", "judged_correct"))
        self.rotation = str(cfg.get("sp_replay_rotation", "bucket_oldest"))
        self.bound = int(cfg.get("sp_replay_bound", 0))
        self.stale_steps = int(cfg.get("sp_replay_stale_steps", 0))
        if self.bucketing not in ("per_question", "global"):
            raise ValueError(f"sp_replay_bucketing {self.bucketing!r}")
        if self.admission not in ("judged_correct", "ungated"):
            raise ValueError(f"sp_replay_admission {self.admission!r}")
        if self.rotation not in ("bucket_oldest", "global_fifo", "stale_window"):
            raise ValueError(f"sp_replay_rotation {self.rotation!r}")
        if self.bucketing == "global":
            if self.bound <= 0:
                raise ValueError("sp_replay_bucketing=global requires sp_replay_bound > 0")
            if self.rotation == "bucket_oldest":
                raise ValueError("rotation=bucket_oldest is a per_question policy")
        else:
            # The per_question branch is the frozen original code path: it always
            # admits judged-correct rows with bucket-oldest replacement. Other combos are
            # NOT implemented there — reject them loudly rather than silently running
            # different semantics.
            if self.admission != "judged_correct":
                raise ValueError(
                    "sp_replay_bucketing=per_question implements admission=judged_correct "
                    f"only (got {self.admission!r}); use bucketing=global for ungated"
                )
            if self.rotation != "bucket_oldest":
                raise ValueError(
                    "sp_replay_bucketing=per_question implements rotation=bucket_oldest "
                    f"only (got {self.rotation!r}); use bucketing=global for other rotations"
                )
        if self.rotation == "stale_window" and self.stale_steps <= 0:
            raise ValueError("rotation=stale_window requires sp_replay_stale_steps > 0")
        # ---- COLD BOOTSTRAP: tolerate a genuinely EMPTY buffer ------------------------
        # A run started from a frozen seed of judged-correct proofs can only see an empty
        # buffer through a bug, so the global draw raises on it. A from-scratch run
        # with no seed at all has an empty buffer for exactly as long as it takes the inflow
        # lane to admit its first entries (one step at ungated admission). With this on, the
        # global draw reports the shortfall instead of raising and the dataset fills those
        # slots with fresh-problem rows that ARE trained (source type "cold_scratch"), so the
        # step has a non-empty PPO batch. Default OFF: an empty buffer stays fatal everywhere
        # else, which is the behavior seeded runs rely on.
        self.cold_bootstrap = bool(int(cfg.get("sp_replay_cold_bootstrap", 0)))
        if self.cold_bootstrap and self.bucketing != "global":
            raise ValueError(
                "sp_replay_cold_bootstrap is implemented for bucketing=global only (the "
                "per_question draw already fills a coverage shortfall from statements)"
            )
        self.policy_dict = {
            "bucketing": self.bucketing,
            "admission": self.admission,
            "rotation": self.rotation,
            "capacity": self.capacity,
            "bound": self.bound,
            "stale_steps": self.stale_steps,
        }
        # Stamped ONLY when on. policy_dict is compared for exact equality against the
        # checkpoint's stamp on resume (drift guard), so unconditionally adding a key would
        # fail every resume from a checkpoint stamped without it with a spurious
        # policy-drift error.
        if self.cold_bootstrap:
            self.policy_dict["cold_bootstrap"] = True
        # Same treatment, same reason: "entry" is the default and checkpoints stamped before
        # this key existed lack it, so adding it unconditionally would fail their next resume
        # with a spurious policy-drift error. Stamped only when the non-default mode is
        # actually in use, where drift detection is wanted.
        if self.global_sampling != "entry":
            self.policy_dict["global_sampling"] = self.global_sampling
        self.buffer_seq_next = 0        # global-mode admission order (buffer_seq)
        self._reseeds: list[dict] = []  # SP_REPLAY_RESEED_STEPS stamps, carried in state
        self._global_slot_cache: tuple | None = None  # (step, [entry]*n_replay)

        # ---- problem identity join: parquet row idx <-> qid ----
        self.num_statements = int(num_statements)
        self.row_of_qid: dict[str, int] = {}
        for idx, q in enumerate(row_qids):
            if q and q not in self.row_of_qid:
                self.row_of_qid[q] = idx
        if not self.row_of_qid:
            raise ValueError("sp_replay: no parquet row produced a qid; identity join impossible")

        # ---- buffer: frozen seed (deltas applied later by on_checkpoint_load) ----
        self.entries: dict[str, list[dict]] = {}
        self.seed_manifest_sha = self._load_seed()

        # ---- EMA + resume cursor (finalized by on_checkpoint_load) ----
        self.ema: dict[str, float] = {}
        self.resume_base = 0
        self.next_dataset_step = 0
        self._finalized = False

        # ---- per-step caches (draws are pure functions of step) ----
        self._stmt_cache: tuple | None = None      # (step, [parquet_idx]*n_orig)
        self._replay_cache: tuple | None = None    # (step, [qid]*<=n_replay, fill_indices)
        self._last_weights_mean = float("nan")
        self._last_distinct_q = 0
        self._last_prefix_lens: list[int] = []
        self._last_cut_fracs: list[float] = []

    # ------------------------------------------------------------------ seed
    def _load_seed(self) -> str:
        import glob as _glob

        manifest_path = os.path.join(self.seed_dir, MANIFEST_NAME)
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"sp_replay seed manifest missing: {manifest_path}")
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        shard_paths = sorted(_glob.glob(os.path.join(self.seed_dir, SHARD_GLOB)))
        if not shard_paths:
            raise FileNotFoundError(f"no replay seed shards under {self.seed_dir}")
        verify_manifest_shards(self.seed_dir, manifest["shards"], shard_paths, "sp_replay seed")
        n = 0
        for path in shard_paths:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    e = _validate_entry(json.loads(line), where=path)
                    if e["qid"] not in self.row_of_qid:
                        raise ValueError(
                            f"seed entry {e['entry_id']} qid {e['qid'][:12]}... has no train-parquet "
                            "row (identity-join contract; rebuild the seed against this parquet)"
                        )
                    bucket = self.entries.setdefault(e["qid"], [])
                    if self.bucketing == "per_question" and len(bucket) >= self.capacity:
                        raise ValueError(
                            f"seed bucket for {e['qid'][:12]}... exceeds capacity {self.capacity}"
                        )
                    if self.bucketing == "global":
                        # honor the builder's draw-order buffer_seq; assign if absent
                        meta = e.setdefault("meta", {})
                        if "buffer_seq" not in meta:
                            meta["buffer_seq"] = self.buffer_seq_next
                        self.buffer_seq_next = max(self.buffer_seq_next, int(meta["buffer_seq"]) + 1)
                    bucket.append(e)
                    n += 1
        if self.bucketing == "global" and n > self.bound:
            raise ValueError(
                f"global seed has {n} entries > sp_replay_bound {self.bound}; "
                "rebuild the seed at the bound (no shrink schedule is implemented)"
            )
        print(
            f"[sp_replay] seed loaded: {n} entries over {len(self.entries)} problems "
            f"(capacity {self.capacity}) from {self.seed_dir}",
            flush=True,
        )
        return sha256_file(manifest_path)

    # ---------------------------------------------------------------- resume
    def _delta_path(self) -> str:
        return os.path.join(self.delta_dir, DELTA_LOG_NAME)

    def finalize_from_checkpoint(self, ckpt_dir):
        """Restore cursor + EMA from ckpt_dir (None/absent file = fresh first attach),
        then apply the delta log up to the cursor and truncate the invalidated tail."""
        assert not self._finalized, "sp_replay finalize_from_checkpoint called twice"
        state_path = os.path.join(ckpt_dir, STATE_FILE_NAME) if ckpt_dir else None
        if state_path and os.path.exists(state_path):
            with open(state_path, encoding="utf-8") as f:
                st = json.load(f)
            self._reseeds = list(st.get("reseeds", []) or [])
            self.buffer_seq_next = max(self.buffer_seq_next, int(st.get("buffer_seq_next", 0)))
            cursor = int(st["next_dataset_step"])
            reseed_steps = _env_step_set("SP_REPLAY_RESEED_STEPS")
            stamped = {int(r["at_step"]) for r in self._reseeds}
            do_reseed = cursor in reseed_steps and cursor not in stamped
            # ---- policy provenance: no silent strategy drift mid-run.
            # The UNSTAMPED reseed event is the sanctioned policy transition (the new
            # policy is stamped at the next save), so the check is skipped exactly then
            # -- never permanently via the override, which stays for manual repair. ----
            rec_pol = st.get("replay_policy")
            if (rec_pol and rec_pol != self.policy_dict and not do_reseed
                    and not _env_flag("SP_REPLAY_POLICY_OVERRIDE")):
                raise ValueError(
                    f"sp_replay policy drift: checkpoint ran {rec_pol} but this launch is "
                    f"{self.policy_dict}; set SP_REPLAY_POLICY_OVERRIDE=1 only if intentional"
                )
            if do_reseed:
                # ---- one-shot RESEED at a branch step: adopt the NEW frozen
                # seed (sha check deliberately skipped), discard the parent's EMA (prior
                # falls back to bucket occupancy), keep the cursor (replay==Q cursor
                # equality invariant). Delta-log rule: a PRE-cursor delta belongs to the
                # parent run (mixed-history misconfig -> abort); a >= cursor tail is the
                # normal crash tail of the branch step itself and the standard
                # truncation below clears it (crash-loop idempotence at the branch). ----
                dpath = self._delta_path()
                if os.path.exists(dpath):
                    with open(dpath, encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            if int(json.loads(line)["dataset_step"]) < cursor:
                                raise ValueError(
                                    f"SP_REPLAY_RESEED_STEPS includes {cursor} but the delta "
                                    f"log at {dpath} has a delta with dataset_step < {cursor} "
                                    "(parent-run deltas; a reseeded run must not inherit them)"
                                )
                self.ema = {}
                self.resume_base = cursor
                self._reseeds.append({"at_step": cursor, "new_seed_sha": self.seed_manifest_sha})
                print(
                    f"[sp_replay] RESEED fired at cursor={cursor}: adopted seed "
                    f"{self.seed_manifest_sha[:12]}..., EMA cleared; any >= {cursor} delta "
                    f"tail is truncated below "
                    f"(stamps now {sorted(int(r['at_step']) for r in self._reseeds)})",
                    flush=True,
                )
            else:
                recorded_sha = st.get("seed_manifest_sha")
                if recorded_sha and recorded_sha != self.seed_manifest_sha:
                    raise ValueError(
                        "sp_replay seed manifest drift: checkpoint recorded "
                        f"{recorded_sha[:12]}... but loaded seed is {self.seed_manifest_sha[:12]}..."
                    )
                self.ema = {str(k): float(v) for k, v in st["ema"].items()}
                self.resume_base = cursor
            print(
                f"[sp_replay] resumed cursor={self.resume_base}, {len(self.ema)} EMA entries "
                f"from {state_path}",
                flush=True,
            )
        else:
            self.resume_base = 0
            print(
                f"[sp_replay] no persisted state{' at ' + str(state_path) if state_path else ''}; "
                "fresh start (cursor=0, EMA prior = bucket k/capacity)",
                flush=True,
            )
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
                    raise ValueError(f"delta log out of order: {step} after {last_step}")
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
            os.replace(tmp, path)  # atomic; drops the invalidated tail (double-apply guard)
        print(f"[sp_replay] delta log: applied {applied}, truncated {dropped}", flush=True)

    def _apply_delta(self, delta: dict):
        replaced = set(delta.get("replaced_entry_ids", []))
        if replaced:
            for qid in list(self.entries):
                kept = [e for e in self.entries[qid] if e["entry_id"] not in replaced]
                if kept:
                    self.entries[qid] = kept
                else:
                    del self.entries[qid]
        for d in delta.get("added_entries", []):
            e = _validate_entry(d, where="delta")
            self.entries.setdefault(e["qid"], []).append(e)
            seq = e.get("meta", {}).get("buffer_seq")
            if seq is not None:
                self.buffer_seq_next = max(self.buffer_seq_next, int(seq) + 1)

    def save_state(self, ckpt_dir: str):
        path = os.path.join(ckpt_dir, STATE_FILE_NAME)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "next_dataset_step": self.next_dataset_step,
                    "ema": self.ema,
                    "seed_manifest_sha": self.seed_manifest_sha,
                    "replay_policy": self.policy_dict,
                    "reseeds": self._reseeds,
                    "buffer_seq_next": self.buffer_seq_next,
                },
                f,
            )
        os.replace(tmp, path)
        print(
            f"[sp_replay] saved state (cursor={self.next_dataset_step}, "
            f"{len(self.ema)} EMA entries) -> {path}",
            flush=True,
        )

    # ------------------------------------------------------------------- EMA
    def get_ema(self, qid: str) -> float:
        """Occupancy prior k/capacity, materialized at first touch."""
        v = self.ema.get(qid)
        if v is None:
            v = len(self.entries.get(qid, ())) / float(self.capacity)
            self.ema[qid] = v
        return v

    def observe(self, qid: str, pass_frac: float):
        cur = self.get_ema(qid)
        self.ema[qid] = self.ema_coef * cur + (1.0 - self.ema_coef) * min(max(pass_frac, 0.0), 1.0)

    def weight(self, qid: str) -> float:
        w = self.w_base - self.w_slope * self.get_ema(qid)
        if not math.isfinite(w) or w <= 0:
            raise ValueError(f"sp_replay weight {w!r} for {qid[:12]}... (ema={self.ema.get(qid)})")
        return w

    # ------------------------------------------------------------- per-step draws
    def statement_indices(self, step: int) -> list[int]:
        if self._stmt_cache is None or self._stmt_cache[0] != step:
            rng = rng_for(self.rng_seed, "orig", step)
            idx = [int(i) for i in rng.choice(self.num_statements, size=self.n_orig, replace=False)]
            self._stmt_cache = (step, idx)
        return self._stmt_cache[1]

    def _flat_entries(self) -> list[dict]:
        """Global-mode view: all live entries in buffer_seq order (deterministic)."""
        flat = [e for b in self.entries.values() for e in b]
        flat.sort(key=lambda e: int(e["meta"]["buffer_seq"]))
        return flat

    def replay_plan_for_step(self, step: int) -> tuple[list[str], list[int]]:
        """(qids in draw order, fill parquet indices for any coverage shortfall)."""
        if self.bucketing == "global":
            # Entry semantics: one pinned permutation over ENTRIES (not problems), take
            # n_replay with wraparound if the buffer is smaller. Never fills from
            # statements; duplicate qids across slots are expected and fine.
            if self._global_slot_cache is None or self._global_slot_cache[0] != step:
                flat = self._flat_entries()
                if not flat:
                    if not self.cold_bootstrap:
                        raise ValueError("sp_replay global buffer is empty")
                    # COLD BOOTSTRAP: no entries yet, so there is nothing to cut.
                    # Report a FULL shortfall -- zero replay qids plus n_replay fresh-problem
                    # fill indices, which the dataset turns into trained "cold_scratch" rows.
                    # Chosen disjointly from this step's statement lane so the batch holds
                    # n_orig + n_replay DISTINCT problems, exactly as a warm step does.
                    used = set(self.statement_indices(step))
                    candidates = [i for i in range(self.num_statements) if i not in used]
                    if len(candidates) < self.n_replay:
                        # The dataset indexes fill[rslot - len(qids)] for every replay slot,
                        # so a short fill list would IndexError mid-epoch. Unreachable with
                        # the real train parquet (thousands of problems vs 32 + 96); fail
                        # loudly rather than half-filling the batch.
                        raise ValueError(
                            f"sp_replay cold bootstrap needs {self.n_replay} fill problems but "
                            f"only {len(candidates)} are unused this step "
                            f"(num_statements={self.num_statements}, n_orig={self.n_orig})"
                        )
                    rng_f = rng_for(self.rng_seed, "cold_fill", step)
                    extra = rng_f.choice(len(candidates), size=self.n_replay, replace=False)
                    fill = [candidates[int(i)] for i in extra]
                    self._global_slot_cache = (step, [])
                    self._replay_cache = (step, [], fill)
                    self._last_weights_mean = float("nan")
                    self._last_prefix_lens, self._last_cut_fracs = [], []
                    print(f"[sp_replay] COLD BOOTSTRAP step {step}: buffer empty -> 0 replay "
                          f"rows + {len(fill)} trained cold_scratch rows", flush=True)
                    return self._replay_cache[1], self._replay_cache[2]
                if self.global_sampling == "question":
                    # Uniform over the DISTINCT problems in the buffer, not over entries, so a
                    # problem that happens to hold more trajectories is not drawn more often.
                    # Sorted for determinism (dict order is insertion order and a resume
                    # rebuilds the buffer in a different order).
                    by_q: dict[str, list] = {}
                    for e in flat:
                        by_q.setdefault(e["qid"], []).append(e)
                    # Sort each bucket by buffer_seq EXPLICITLY. _flat_entries() already sorts,
                    # so this is currently redundant -- but it makes the determinism local
                    # instead of inherited from a distant method: without it, changing
                    # _flat_entries' ordering would silently change which trajectory every
                    # slot draws, with nothing failing.
                    for _b in by_q.values():
                        _b.sort(key=lambda e: int(e["meta"]["buffer_seq"]))
                    qs = sorted(by_q)
                    rng = rng_for(self.rng_seed, "global_question_perm", step)
                    perm = rng.permutation(len(qs))
                    picks = []
                    for i in range(self.n_replay):
                        # Pinned permutation with wraparound: the first pass covers every
                        # problem once before any repeats, exactly as the entry version does.
                        q = qs[int(perm[i % len(qs)])]
                        bucket = by_q[q]
                        # Which of that problem's trajectories: uniform, keyed per (step,slot)
                        # so it is reproducible on resume.
                        r2 = rng_for(self.rng_seed, "global_question_entry", step, i)
                        picks.append(bucket[int(r2.integers(len(bucket)))])
                    self._last_distinct_q = len(qs)
                else:
                    rng = rng_for(self.rng_seed, "global_entry_perm", step)
                    perm = rng.permutation(len(flat))
                    picks = [flat[int(perm[i % len(flat)])] for i in range(self.n_replay)]
                    self._last_distinct_q = len({e["qid"] for e in flat})
                self._global_slot_cache = (step, picks)
                self._replay_cache = (step, [e["qid"] for e in picks], [])
                self._last_weights_mean = float("nan")
                self._last_prefix_lens, self._last_cut_fracs = [], []
            return self._replay_cache[1], self._replay_cache[2]
        if self._replay_cache is None or self._replay_cache[0] != step:
            eligible = sorted(q for q, v in self.entries.items() if v)
            if not eligible:
                raise ValueError("sp_replay buffer has no covered problems")
            ws = np.asarray([self.weight(q) for q in eligible], dtype=np.float64)
            probs = ws / ws.sum()
            k = min(self.n_replay, len(eligible))
            rng = rng_for(self.rng_seed, "weighted_replay_problems", step)
            chosen = rng.choice(len(eligible), size=k, replace=False, p=probs)
            qids = [eligible[int(i)] for i in chosen]
            fill: list[int] = []
            if k < self.n_replay:  # coverage shortfall -> extra uniform statement rows
                used = set(self.statement_indices(step))
                candidates = [i for i in range(self.num_statements) if i not in used]
                rng_f = rng_for(self.rng_seed, "orig_fill", step)
                extra = rng_f.choice(len(candidates), size=self.n_replay - k, replace=False)
                fill = [candidates[int(i)] for i in extra]
            self._last_weights_mean = float(ws.mean())
            self._replay_cache = (step, qids, fill)
            self._last_prefix_lens, self._last_cut_fracs = [], []
        return self._replay_cache[1], self._replay_cache[2]

    def entry_for_slot(self, qid: str, step: int, rslot: int) -> dict:
        if self.bucketing == "global":
            assert self._global_slot_cache and self._global_slot_cache[0] == step, (
                "entry_for_slot before replay_plan_for_step (global mode)"
            )
            entry = self._global_slot_cache[1][rslot]
            assert entry["qid"] == qid, (entry["qid"], qid, rslot)
            return entry
        bucket = self.entries.get(qid)
        if not bucket:
            raise KeyError(f"sp_replay: problem {qid[:12]}... has an empty bucket (step {step})")
        rng = rng_for(self.rng_seed, "replay_entry", step, rslot)
        return bucket[int(rng.integers(len(bucket)))]

    def prefix_cut(self, entry: dict, step: int, rslot: int) -> tuple[list[int], float]:
        ids = entry["response_token_ids"]
        t = len(ids)
        lo = int(math.floor(self.cut_low * t))
        hi = max(lo, int(math.floor(self.cut_high * t)))
        rng = rng_for(self.rng_seed, "cut", step, rslot, entry["entry_id"])
        if self.cut_grain > 0:
            # Quantized cut: the prefix ends on a multiple of cut_grain tokens, so the
            # model always resumes from a round offset into the parent trajectory.
            # k ranges over the multiples lying inside [lo, hi]. cut_low=0 makes k=0 a
            # legal draw -- that is the from-scratch row (empty prefix, the model gets
            # only the problem and generates everything), which observe_and_update
            # correctly folds into the from-scratch EMA. Any cut_low large enough to
            # push lo past 0 excludes it instead. Exactly one rng.integers() call on
            # either branch, so the generator stream stays structurally identical.
            k_lo = -(-lo // self.cut_grain)   # ceil(lo / grain)
            k_hi = hi // self.cut_grain       # floor(hi / grain)
            if k_hi < k_lo:
                # No multiple of cut_grain lies inside [lo, hi] -- only possible when
                # cut_low > 0 and the trajectory is too short for even one full grain
                # under cut_high. Grain is the primary constraint here (an off-grid
                # prefix would silently violate the whole point of the knob), so fall
                # back to the largest multiple that still respects cut_high, i.e.
                # k_hi * grain, which is 0 when hi < grain. That relaxes cut_low
                # rather than the grid.
                a = max(0, k_hi) * self.cut_grain
            else:
                a = int(rng.integers(k_lo, k_hi + 1)) * self.cut_grain
        else:
            a = int(rng.integers(lo, hi + 1))
        a = min(a, t - 1)  # always leave >=1 token to generate toward
        self._last_prefix_lens.append(a)
        self._last_cut_fracs.append(a / t if t else 0.0)
        return [int(x) for x in ids[:a]], (a / t if t else 0.0)

    def prefix_cuts_td(self, entry: dict, step: int, rslot: int, n: int,
                       grain: int | None = None) -> list[int]:
        """n prefix cut LENGTHS for one no-group-TD slot (sp_q_td_enable).

        Same [cut_low, cut_high] window as prefix_cut, but the slot's budget of n
        single-rollout siblings is spread over the cut grid: a pinned permutation covers
        every grid point once before any repeats (the global replay draw's own wraparound
        idiom), so a trajectory with >= n grid points yields n DISTINCT prefixes and a
        shorter one cycles through all it has. `grain` defaults to the replay cut_grain;
        the TD arm passes a finer one (sp_q_td_cut_grain) because at grain 10k a 30k
        trajectory has only ~3 grid points and the whole point of this mode is prefix
        diversity -- and unlike the grouped lanes, no sibling shares a prefix here, so the
        coarse grain buys nothing.

        Keyed on its own rng purpose ("td_cut"): the existing "cut" stream is not
        consumed for a TD slot, and no other stream's keying changes -- with the mode off
        this method is never called and existing runs replay byte-identically.
        """
        ids = entry["response_token_ids"]
        t = len(ids)
        lo = int(math.floor(self.cut_low * t))
        hi = max(lo, int(math.floor(self.cut_high * t)))
        g = self.cut_grain if grain is None else int(grain)
        if g < 0:
            raise ValueError("prefix_cuts_td grain must be >= 0, got %r" % g)
        if g > 0:
            k_lo = -(-lo // g)   # ceil(lo / grain)
            k_hi = hi // g       # floor(hi / grain)
            if k_hi < k_lo:
                # No multiple of grain inside [lo, hi] (short trajectory under cut_low>0):
                # same fallback as prefix_cut -- the largest multiple respecting cut_high.
                grid = [max(0, k_hi) * g]
            else:
                grid = [k * g for k in range(k_lo, k_hi + 1)]
        else:
            grid = list(range(lo, hi + 1))
        rng = rng_for(self.rng_seed, "td_cut", step, rslot, entry["entry_id"])
        perm = rng.permutation(len(grid))
        cuts = [min(int(grid[int(perm[j % len(grid)])]), t - 1) for j in range(int(n))]
        for a in cuts:
            self._last_prefix_lens.append(a)
            self._last_cut_fracs.append(a / t if t else 0.0)
        return cuts

    # ------------------------------------------------------ post-reward hook
    def observe_and_update(self, non_tensor_batch, seq_rewards, get_row_token_ids) -> dict:
        """EMA observe + capacity-K admission + delta append, for ONE train batch.

        seq_rewards: per-row scalar rewards (degenerate-group metric only).
        get_row_token_ids: row index -> full valid response token ids (incl. prefix),
        called only for rows actually admitted (bounded by capacity * problems).
        """
        extra = non_tensor_batch["extra_info"]
        uids = non_tensor_batch["uid"]
        pjs = non_tensor_batch.get("prover_judge_score")
        if pjs is None:
            raise ValueError("sp_replay hook: batch has no prover_judge_score column")
        n = len(extra)

        steps = {int(ei["sp_dataset_step"]) for ei in extra}
        assert len(steps) == 1, f"sp_replay hook: mixed dataset steps in one batch: {steps}"
        step = steps.pop()
        assert step == self.next_dataset_step, (
            f"sp_replay hook: batch is dataset_step {step} but cursor expects "
            f"{self.next_dataset_step} (dataloader ordering broken?)"
        )

        def _pass(i):
            try:
                return 1 if int(round(float(pjs[i]))) >= 1 else 0
            except (TypeError, ValueError):
                return 0

        passes = [_pass(i) for i in range(n)]

        # ---- EMA: from-scratch rows only (prefix-free; incl. undercoverage fill rows) ----
        per_q: dict[str, list[int]] = {}
        for i, ei in enumerate(extra):
            if int(ei["sp_prefix_len"]) == 0:
                per_q.setdefault(ei["sp_qid"], []).append(passes[i])
        for q, ps in per_q.items():
            self.observe(q, sum(ps) / len(ps))

        # ---- admissions ----
        if self.bucketing == "global":
            added_entries, replaced_ids, n_from_prefix = self._admit_global(
                extra, passes, pjs, uids, step, get_row_token_ids
            )
            return self._finish_step(
                step, extra, uids, passes, seq_rewards, non_tensor_batch,
                added_entries, replaced_ids, n_from_prefix,
            )

        # per_question (default): judged-passing rows, capacity-K replace-oldest per problem
        # (the flat per-question policy).
        cand: dict[str, list[int]] = {}
        for i, ei in enumerate(extra):
            if passes[i]:
                cand.setdefault(ei["sp_qid"], []).append(i)
        added_entries, replaced_ids = [], []
        n_from_prefix = 0
        for qid in sorted(cand):
            rows = cand[qid]
            if len(rows) > self.capacity:
                rng = rng_for(self.rng_seed, "update", step, qid)
                pick = rng.choice(len(rows), size=self.capacity, replace=False)
                rows = [rows[int(i)] for i in sorted(pick)]
            bucket = self.entries.setdefault(qid, [])
            for i in rows:
                ids = [int(x) for x in get_row_token_ids(i)]
                if not ids:
                    continue  # judged-pass on an empty response would be a judge bug; skip
                entry = {
                    "entry_id": f"online:{step}:{uids[i]}:{i}",
                    "qid": qid,
                    "response_token_ids": ids,
                    "meta": {
                        "dataset_step": step,
                        "prefix_len": int(extra[i]["sp_prefix_len"]),
                        "source_entry_id": extra[i].get("sp_entry_id", ""),
                    },
                }
                if int(extra[i]["sp_prefix_len"]) > 0:
                    n_from_prefix += 1
                if len(bucket) < self.capacity:
                    bucket.append(entry)
                else:
                    oldest = min(
                        range(len(bucket)),
                        key=lambda j: (
                            bucket[j].get("meta", {}).get("dataset_step", -1),
                            bucket[j]["entry_id"],
                        ),
                    )
                    replaced_ids.append(bucket[oldest]["entry_id"])
                    bucket[oldest : oldest + 1] = []
                    bucket.append(entry)
                added_entries.append(entry)
        return self._finish_step(
            step, extra, uids, passes, seq_rewards, non_tensor_batch,
            added_entries, replaced_ids, n_from_prefix,
        )

    def _admit_global(self, extra, passes, pjs, uids, step, get_row_token_ids):
        """Global-buffer admission: candidates are the SCRATCH stream only
        (sp_source_type=="statement" -- the buffer inflow lane; empty-prefix REPLAY rows
        are trained rows, not inflow), gated per the admission axis; eviction per the
        rotation axis over the global buffer_seq order."""
        added_entries, replaced_ids = [], []
        for i, ei in enumerate(extra):
            if ei.get("sp_source_type") != "statement":
                continue
            if self.admission == "judged_correct" and not passes[i]:
                continue
            ids = [int(x) for x in get_row_token_ids(i)]
            if not ids:
                continue  # empty response is never a useful trajectory
            try:
                raw_score = float(pjs[i])
            except (TypeError, ValueError):
                raw_score = float("nan")
            entry = {
                "entry_id": f"online:{step}:{uids[i]}:{i}",
                "qid": ei["sp_qid"],
                "response_token_ids": ids,
                "meta": {
                    "dataset_step": step,
                    "prefix_len": 0,
                    "source_entry_id": "",
                    "buffer_seq": self.buffer_seq_next,
                    "judge_pass": int(passes[i]),
                    "judge_score": raw_score,
                },
            }
            self.buffer_seq_next += 1
            self.entries.setdefault(ei["sp_qid"], []).append(entry)
            added_entries.append(entry)

        # ---- rotation ----
        if self.rotation == "stale_window":
            floor_step = step - self.stale_steps
            for e in self._flat_entries():
                if int(e["meta"].get("dataset_step", -1)) < floor_step:
                    replaced_ids.append(e["entry_id"])
                    self._remove_entry(e)
        total = sum(len(b) for b in self.entries.values())
        if self.bound > 0:
            flat = self._flat_entries()
            while total > self.bound:
                oldest = flat.pop(0)  # smallest buffer_seq = global FIFO head
                replaced_ids.append(oldest["entry_id"])
                self._remove_entry(oldest)
                total -= 1
        return added_entries, replaced_ids, 0

    def _remove_entry(self, entry: dict):
        qid = entry["qid"]
        bucket = self.entries.get(qid, [])
        kept = [e for e in bucket if e["entry_id"] != entry["entry_id"]]
        if kept:
            self.entries[qid] = kept
        else:
            self.entries.pop(qid, None)

    def _finish_step(self, step, extra, uids, passes, seq_rewards, non_tensor_batch,
                     added_entries, replaced_ids, n_from_prefix) -> dict:
        os.makedirs(self.delta_dir, exist_ok=True)
        with open(self._delta_path(), "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "dataset_step": step,
                        "added_entries": added_entries,
                        "replaced_entry_ids": replaced_ids,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        self.next_dataset_step = step + 1

        # ---- metrics ----
        def _group_stats(rows_sel):
            groups: dict[str, list[float]] = {}
            for i in rows_sel:
                groups.setdefault(str(uids[i]), []).append(float(seq_rewards[i]))
            if not groups:
                return float("nan"), float("nan")
            degen = sum(1 for v in groups.values() if max(v) - min(v) < 1e-9) / len(groups)
            pr = sum(passes[i] for i in rows_sel) / len(rows_sel)
            return degen, pr

        # Split on the inflow LANE, not on prefix length -- see _is_statement_row. A
        # replay row that drew a k=0 cut belongs to the replay stream even though it has
        # no prefix, otherwise the "from-scratch (N stmt rows)" panel plots N rows in its
        # label and ~14x that many in its line.
        orig_rows = [i for i, ei in enumerate(extra) if _is_statement_row(ei)]
        repl_rows = [i for i, ei in enumerate(extra) if not _is_statement_row(ei)]
        degen_o, pass_o = _group_stats(orig_rows)
        degen_r, pass_r = _group_stats(repl_rows)

        # Stream-split correctness in the SAME units the main dashboard's panel-1 uses
        # (rubric grade mean = points/7; best@n = any pass in the GRPO group; strict-7
        # rate), so the panel can show the from-scratch stream only and stay
        # comparable across runs.
        pts = non_tensor_batch.get("rubric_points")

        def _stream_correctness(rows_sel):
            if not rows_sel:
                return float("nan"), float("nan"), float("nan")
            groups: dict[str, list[int]] = {}
            for i in rows_sel:
                groups.setdefault(str(uids[i]), []).append(i)
            best = sum(1 for g in groups.values() if any(passes[i] for i in g)) / len(groups)
            grade_mean = strict7 = float("nan")
            if pts is not None:
                vals = []
                for i in rows_sel:
                    try:
                        vals.append(float(pts[i]))
                    except (TypeError, ValueError):
                        pass
                if vals:
                    grade_mean = sum(vals) / len(vals) / 7.0
                    strict7 = sum(1 for v in vals if v >= 6.999) / len(vals)
            return best, grade_mean, strict7

        best_o, grade_o, s7_o = _stream_correctness(orig_rows)
        best_r, grade_r, s7_r = _stream_correctness(repl_rows)
        covered = [q for q, v in self.entries.items() if v]
        m = {
            "replay/coverage": len(covered),
            "replay/buffer_size": sum(len(v) for v in self.entries.values()),
            "replay/admitted": len(added_entries),
            "replay/admitted_from_prefix": n_from_prefix,
            "replay/replaced": len(replaced_ids),
            "replay/ema_mean": (
                sum(self.get_ema(q) for q in covered) / len(covered) if covered else float("nan")
            ),
            "replay/mean_weight": self._last_weights_mean,
            # Distinct problems available in the buffer at draw time, and how many times the
            # average problem had to be reused to fill n_replay slots. Under
            # global_sampling="question" a repeat factor > 1 means some problems appear more
            # than once per step (with different prefixes) -- visible rather than silent.
            "replay/distinct_questions": self._last_distinct_q,
            "replay/question_repeat_factor": (
                self.n_replay / self._last_distinct_q if self._last_distinct_q else float("nan")
            ),
            "replay/pass_rate_orig": pass_o,
            "replay/pass_rate_replay": pass_r,
            "replay/degenerate_group_frac_orig": degen_o,
            "replay/degenerate_group_frac_replay": degen_r,
            "replay/best_at_n_orig": best_o,
            "replay/best_at_n_replay": best_r,
            "replay/rubric_grade_mean_orig": grade_o,
            "replay/rubric_grade_mean_replay": grade_r,
            "replay/strict7_frac_orig": s7_o,
            "replay/strict7_frac_replay": s7_r,
            "replay/dataset_step": step,
        }
        if self._last_prefix_lens:
            m["replay/prefix_len_mean"] = sum(self._last_prefix_lens) / len(self._last_prefix_lens)
            m["replay/cut_fraction_mean"] = sum(self._last_cut_fracs) / len(self._last_cut_fracs)
        return m


# ---------------------------------------------------------------------------
# module-level API used by the dataset and ray_trainer (all no-ops unless enabled)
# ---------------------------------------------------------------------------

def install(cfg: dict, num_statements: int, row_qids: list) -> ReplayHarness:
    """Called once from the train dataset __init__ (driver process)."""
    assert enabled(), "sp_replay.install called but SP_REPLAY_ENABLE is not set"
    assert _S["harness"] is None, "sp_replay.install called twice"
    _S["harness"] = ReplayHarness(cfg, num_statements, row_qids)
    return _S["harness"]


def on_checkpoint_load(ckpt_dir):
    """ray_trainer._load_checkpoint hook (both the resume and the from-scratch path)."""
    if not enabled() or not installed():
        return
    harness().finalize_from_checkpoint(ckpt_dir)


def save_state(ckpt_dir: str):
    """ray_trainer._save_checkpoint hook.

    Fail-closed: a persistence failure MUST abort the checkpoint (it raises,
    before ray_trainer writes latest_checkpointed_iteration.txt) rather than being swallowed.
    A swallowed failure would publish `latest` with a stale replay cursor; on resume the Q /
    replay cursor-consistency check rejects the mismatch, leaving the run unrecoverable. A
    crash here is instead recoverable — resume falls back to the previous good checkpoint."""
    if not enabled() or not installed():
        return
    harness().save_state(ckpt_dir)


def statement_row_flags(non_tensor_batch):
    """Per-row True = from-scratch (prefix-free) row; None when the batch does not carry
    the replay schema (e.g. validation). Used by the AEC entropy split."""
    if not enabled() or not installed():
        return None
    extra = non_tensor_batch.get("extra_info")
    if extra is None:
        return None
    # DELIBERATELY the prefix test, not _is_statement_row: this feeds the AEC entropy
    # control signal, which must exclude PREFIX-CONDITIONED rows (systematically lower
    # entropy). A replay row with a k=0 cut generates from scratch and belongs here.
    flags = []
    for ei in extra:
        if not isinstance(ei, dict) or "sp_prefix_len" not in ei:
            return None
        flags.append(int(ei["sp_prefix_len"]) == 0)
    return flags


def observe_and_update(non_tensor_batch, seq_rewards, get_row_token_ids):
    if not enabled() or not installed():
        return {}
    return harness().observe_and_update(non_tensor_batch, seq_rewards, get_row_token_ids)
