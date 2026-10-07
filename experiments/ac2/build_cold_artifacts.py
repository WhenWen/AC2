"""build_cold_artifacts.py -- write the three EMPTY artifacts a from-scratch run starts from: the
replay-buffer seed, the critic-buffer (Q FIFO) seed and the reference bank.

The loaders (`sp_replay._load_seed`, `sp_q_readiness._load_q_seed` / `_load_ref_bank`) require
a manifest and at least one shard file even when there are no records (an empty shard is legal,
zero shards is not). This script writes exactly that, with the same manifest/shard/sha layout as
a non-empty seed, so the sha pins, `verify_manifest_shards` and the state-file drift checks
behave identically.

How the run behaves while they are empty:
  * replay buffer: step 1 has nothing to cut; with `sp_replay_cold_bootstrap=1` its replay
    slots are filled with fresh `cold_scratch` problems (full group size, trained), and the
    inflow lane fills the buffer from step 1's own trajectories;
  * critic buffer: records are admitted only from replay rows, so the first ones land at step
    2; until then the critic phase is skipped (`q/q_phase_skipped=1`) and the LR ladder treats
    the skip as a no-op (streak reset, no reduction);
  * reference bank: filled add-once from the run's own judge-passing proofs; with
    SP_Q_TRAIN_NOREF=1 a record without a reference trains the no-reference prompt instead of
    being skipped. Readiness stays false for at least the first 5 steps (the MAE window).

Run once before the first launch:

  python experiments/ac2/build_cold_artifacts.py \
      --out experiments/ac2

Idempotent (no RNG and no input data, so the manifest shas do not move and a resume cannot trip
a drift check); refuses to overwrite a non-empty seed unless --force is given.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "ac2",
    ),
)
from q_build_common import sha256_file, write_shards  # noqa: E402


def _existing_record_count(out: str, subdir: str) -> int:
    n = 0
    for path in glob.glob(os.path.join(out, subdir, "shard_*.jsonl")):
        with open(path, encoding="utf-8") as f:
            n += sum(1 for line in f if line.strip())
    return n


def _clear_shards(base: str, subdir: str) -> int:
    """Delete every existing shard_*.jsonl before writing.

    `write_shards` writes shard_0000..shard_N for the records it is
    given and NEVER deletes anything, so rebuilding a previously multi-shard artifact as an
    empty one would leave shard_0001+ in place: their records would still be loaded (the
    loaders glob the directory), and they are not in the new manifest, so
    verify_manifest_shards fails -- or worse, an older loader picks them up. Clearing first
    makes the rebuild actually a rebuild."""
    removed = 0
    for path in sorted(glob.glob(os.path.join(base, subdir, "shard_*.jsonl"))):
        os.remove(path)
        removed += 1
    return removed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True,
                    help="experiment dir (holds replay_seed_cold/, q_seed/, reference_bank/)")
    ap.add_argument("--replay-seed-name", default="replay_seed_cold",
                    help="subdir for the empty replay seed; must match SP_REPLAY_SEED_DIR")
    ap.add_argument("--fifo-cap", type=int, default=1920,
                    help="recorded in the manifest for provenance only; the FIFO capacity the "
                         "run enforces is SP_Q_FIFO_CAP (20-step horizon x 96 groups = 1920)")
    ap.add_argument("--min-valid", type=int, default=8)
    ap.add_argument("--force", action="store_true",
                    help="overwrite even if an existing seed/bank has records")
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    replay_out = os.path.join(out, args.replay_seed_name)

    for base, subdir in ((out, "q_seed"), (out, "reference_bank"),
                         (replay_out, "replay_buffer")):
        have = _existing_record_count(base, subdir)
        if have and not args.force:
            raise SystemExit(
                f"refusing to overwrite {base}/{subdir}: it already holds {have} records "
                "(pass --force only if you really mean to discard them)"
            )

    # Clear any pre-existing shards first (see _clear_shards): write_shards only overwrites
    # shard_0000, so a stale shard_0001+ would survive a "rebuild" and be loaded anyway.
    for _base, _sub in ((replay_out, "replay_buffer"), (out, "q_seed"), (out, "reference_bank")):
        _rm = _clear_shards(_base, _sub)
        if _rm:
            print(f"[cold] cleared {_rm} pre-existing shard(s) from {_base}/{_sub}", flush=True)

    # ---- empty REPLAY buffer seed --------------------------------------------------------
    # Same layout sp_replay._load_seed expects: replay_buffer_manifest.json + at least one
    # replay_buffer/shard_*.jsonl. Zero entries means the run has no trajectory to cut until
    # its own inflow lane admits one, which is what sp_replay_cold_bootstrap=1 permits.
    os.makedirs(replay_out, exist_ok=True)
    rshards = write_shards([], replay_out, "replay_buffer")
    rmanifest = {
        "builder": "build_cold_artifacts.py",
        "cold_start": True,
        "reason": "truly empty buffer: the run bootstraps its own trajectories from step 1",
        "entries": 0,
        "problems": 0,
        "capacity": None,
        "shards": rshards,
    }
    rpath = os.path.join(replay_out, "replay_buffer_manifest.json")
    with open(rpath, "w", encoding="utf-8") as f:
        json.dump(rmanifest, f, indent=2)
    print(f"[cold] empty replay seed -> {rpath}\n  sha256 = {sha256_file(rpath)}", flush=True)

    # ---- empty Q FIFO seed ----
    shards = write_shards([], out, "q_seed")
    manifest = {
        "builder": "build_cold_artifacts.py",
        "cold_start": True,
        "reason": "from-scratch base Qwen3-4B-Thinking: no parent dumps to seed from",
        "cap": args.fifo_cap,
        "min_valid": args.min_valid,
        "total_qualifying": 0,
        "seeded": 0,
        "seeded_seq_range": None,
        "seeded_steps_range": None,
        "skip_reasons": {},
        "ref_resolved": 0,
        "z_histogram": {},
        "prefix_len_histogram_2k": {},
        "shards": shards,
    }
    mpath = os.path.join(out, "q_seed_manifest.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[cold] empty q seed -> {mpath}\n  sha256 = {sha256_file(mpath)}", flush=True)

    # ---- empty reference bank ----
    bshards = write_shards([], out, "reference_bank")
    bmanifest = {
        "builder": "build_cold_artifacts.py",
        "cold_start": True,
        "reason": "from-scratch: the bank fills add-once online from this run's own proofs",
        "covered": 0,
        "skipped_no_proof": 0,
        "shards": bshards,
    }
    bpath = os.path.join(out, "reference_bank_manifest.json")
    with open(bpath, "w", encoding="utf-8") as f:
        json.dump(bmanifest, f, indent=2)
    print(f"[cold] empty reference bank -> {bpath}\n  sha256 = {sha256_file(bpath)}", flush=True)
    print("[cold] done. The Q phase will be SKIPPED until the first online admission "
          "(~step 1-2) and readiness stays false for >= 5 steps by construction.", flush=True)


if __name__ == "__main__":
    main()
