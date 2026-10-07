"""SPQReadinessDataset: the AC2 training dataset (loaded by file path via ``data.custom_cls``).

A training step is a deterministic (step, slot) stream of n_base = n_orig + n_replay rows:

  * statement rows (slot < n_orig): fresh problems. Under SP_SCRATCH_INFLOW_ONLY=1 they are
    rolled out once (``sp_rollout_n`` = 1) to refill the replay buffer and are dropped before
    the loss;
  * replay rows: a prefix cut from a stored trajectory (sp_replay) with a per-row critic route
    in {"full", "short", "audit"} from the readiness table and the pinned per-step audit draw
    (sp_q_readiness). "short" rows carry the action-chunk cap ``sp_q_max_new_tokens`` =
    min(b, C - a_i), consumed by sp_prefix_agent (b = SP_Q_BUDGET_G, C = SP_MAX_RESPONSE_LEN,
    a_i = prefix length); every other row carries 0 (no cap);
  * fill rows: when the buffer cannot cover every replay slot, the shortfall is filled with
    fresh problems. With an empty buffer at a cold start they are tagged ``cold_scratch`` and
    trained at the full group size instead of being dropped as inflow.

The critic state (critic buffer, reference bank, readiness table) is installed right after
sp_replay.install, so it lives in the same driver process as the replay state. Validation
delegates to the plain RLHFDataset. Requires SP_REPLAY_ENABLE=1 and SP_Q_ENABLE=1 (asserted).
"""
from __future__ import annotations

import os

from verl.trainer.ppo import difficulty as _difficulty
from verl.trainer.ppo import sp_q_readiness, sp_replay
from verl.utils.dataset.rl_dataset import RLHFDataset


def _normalize_files(files) -> list[str]:
    if files is None:
        return []
    if isinstance(files, str):
        return [files]
    return [str(f) for f in files]


class SPQReadinessDataset(RLHFDataset):
    def __init__(self, data_files, tokenizer, config, processor=None, max_samples: int = -1):
        super().__init__(
            data_files=data_files,
            tokenizer=tokenizer,
            config=config,
            processor=processor,
            max_samples=max_samples,
        )
        train_files = _normalize_files(config.get("train_files"))
        self.sp_train_mode = _normalize_files(data_files) == train_files and bool(train_files)
        print(
            f"[sp_q] SPQReadinessDataset mode: "
            f"{'TRAIN (step,slot)+Q routing' if self.sp_train_mode else 'VAL (plain RLHFDataset delegation)'} "
            f"for data_files={_normalize_files(data_files)}",
            flush=True,
        )
        if not self.sp_train_mode:
            return

        assert sp_replay.enabled(), (
            "SPQReadinessDataset requires SP_REPLAY_ENABLE=1 (the Q mechanism rides on the "
            "replay dataset's (step,slot) stream)."
        )
        assert sp_q_readiness.enabled(), (
            "SPQReadinessDataset is the train dataset but SP_Q_ENABLE is not set — the Q "
            "trainer hooks would silently no-op; refuse to run half-wired."
        )
        assert not _difficulty.enabled(), (
            "SP_DIFF_SAMPLING is on: incompatible with the deterministic (step,slot) dataset."
        )
        assert not config.get("shuffle", True), "replay dataset requires data.shuffle=False"
        assert int(config.get("dataloader_num_workers", 8)) == 0, (
            "replay dataset requires data.dataloader_num_workers=0 (lazy driver-process rows)"
        )

        n_rows = super().__len__()
        row_qids = [
            _difficulty.qid_from_messages(self.dataframe[i][self.prompt_key]) for i in range(n_rows)
        ]
        self.harness = sp_replay.install(config, num_statements=n_rows, row_qids=row_qids)
        self._row_qids = row_qids

        def _raw_prompt_of_qid(qid: str):
            idx = self.harness.row_of_qid.get(qid)
            assert idx is not None, f"sp_q: qid {qid[:12]}... has no train-parquet row"
            return self.dataframe[idx][self.prompt_key]

        self.q_harness = sp_q_readiness.install(
            config, tokenizer=tokenizer, replay_harness=self.harness,
            raw_prompt_of_qid=_raw_prompt_of_qid,
        )
        print(
            f"[sp_q] train mode: {n_rows} statements, n_base={self.harness.n_base} "
            f"(orig {self.harness.n_orig} + replay {self.harness.n_replay}), "
            f"g={self.q_harness.budget_g}, ready_thresh={self.q_harness.ready_thresh}, "
            f"fifo_cap={self.q_harness.fifo_cap}, train_n={self.q_harness.train_n}, "
            f"ctx_limit={self.q_harness.ctx_limit}, q_rng_seed={self.q_harness.rng_seed}",
            flush=True,
        )

    # -- length / rows -------------------------------------------------------

    def __len__(self):
        if not getattr(self, "sp_train_mode", False):
            return super().__len__()
        return self.harness.local_steps * self.harness.n_base

    def __getitem__(self, item):
        if not getattr(self, "sp_train_mode", False):
            return super().__getitem__(item)

        h = self.harness
        assert h._finalized, (
            "sp_replay row build before on_checkpoint_load finalized the cursor — "
            "_load_checkpoint must run before the first dataloader iteration"
        )
        local, slot = divmod(int(item), h.n_base)
        step = h.resume_base + local

        if slot < h.n_orig:
            parquet_idx = h.statement_indices(step)[slot]
            return self._finalize_row(
                super().__getitem__(parquet_idx),
                step=step,
                qid=self._row_qids[parquet_idx],
                source_type="statement",
                prefix_ids=[],
                cut_fraction=0.0,
                entry_id="",
                fill=False,
                q_route="full",
                q_cap=0,
            )

        rslot = slot - h.n_orig
        qids, fill_indices = h.replay_plan_for_step(step)
        if rslot >= len(qids):  # coverage shortfall -> uniform statement fill row
            parquet_idx = fill_indices[rslot - len(qids)]
            # COLD BOOTSTRAP: an EMPTY buffer (no seed, before the first admission)
            # yields zero replay qids, so every replay slot becomes a fill row. Those
            # rows must be TRAINED -- tagged "statement" they would be inflow-only,
            # dropped before the loss, and the step would have nothing left to train
            # on. "cold_scratch" = a fresh problem generated at the full rollout count
            # and trained like a replay row, but with no prefix and no parent trajectory.
            _cold = bool(getattr(h, "cold_bootstrap", False)) and not qids
            return self._finalize_row(
                super().__getitem__(parquet_idx),
                step=step,
                qid=self._row_qids[parquet_idx],
                source_type="cold_scratch" if _cold else "statement",
                prefix_ids=[],
                cut_fraction=0.0,
                entry_id="",
                fill=True,
                q_route="full",
                q_cap=0,
            )

        qid = qids[rslot]
        entry = h.entry_for_slot(qid, step, rslot)
        prefix_ids, cut_fraction = h.prefix_cut(entry, step, rslot)
        parquet_idx = h.row_of_qid[qid]
        # Critic routing: readiness + pinned audit draw, per replay slot.
        route = self.q_harness.route_plan_for_step(step)[rslot]
        q_cap = self.q_harness.budget_cap_for(route, len(prefix_ids)) if route == "short" else 0
        return self._finalize_row(
            super().__getitem__(parquet_idx),
            step=step,
            qid=qid,
            source_type="replay",
            prefix_ids=prefix_ids,
            cut_fraction=cut_fraction,
            entry_id=entry["entry_id"],
            fill=False,
            q_route=route,
            q_cap=q_cap,
        )

    _INFLOW_ONLY = os.environ.get("SP_SCRATCH_INFLOW_ONLY", "0") in ("1", "true", "True")

    @staticmethod
    def _finalize_row(row: dict, *, step, qid, source_type, prefix_ids, cut_fraction,
                      entry_id, fill, q_route, q_cap) -> dict:
        row["sp_prefix_token_ids"] = [int(t) for t in prefix_ids]
        row["sp_q_max_new_tokens"] = int(q_cap)   # uniform schema; 0 = no cap
        row["agent_name"] = "sp_prefix_agent"
        if SPQReadinessDataset._INFLOW_ONLY:
            # Statement rows are inflow-only -> 1 rollout; 0 = config default n.
            # Stamped BOTH top-level and in extra_info: the top-level key has been
            # observed to vanish from the live dataloader batch (cause unresolved)
            # while extra_info survives to the trainer, so extra_info is authoritative.
            row["sp_rollout_n"] = 1 if source_type == "statement" else 0
        ei = dict(row.get("extra_info") or {})
        ei.update(
            {
                "sp_dataset_step": int(step),
                "sp_qid": qid,
                "sp_source_type": source_type,
                "sp_prefix_len": len(prefix_ids),
                "sp_cut_fraction": float(cut_fraction),
                "sp_entry_id": entry_id,
                "sp_fill": bool(fill),
                "sp_q_route": q_route,
            }
        )
        if SPQReadinessDataset._INFLOW_ONLY:
            ei["sp_rollout_n"] = 1 if source_type == "statement" else 0
        row["extra_info"] = ei
        return row
