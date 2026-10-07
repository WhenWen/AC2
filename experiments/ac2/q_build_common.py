"""Shared helpers for the critic-seed / reference-bank builders.

Reconstructs the replay-buffer state at any dataset step from its seed shards + append-only
delta log (the same replay the trainer does on resume), and provides proof extraction, grid
rounding, sha256 and shard-writing utilities (build_cold_artifacts.py uses `write_shards` and
`sha256_file`). Offline only.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re

GRID = [round(i / 10.0, 1) for i in range(11)]

_PROOF_RE = re.compile(r"<proof>(.*?)</proof>", re.DOTALL)
THINK_CLOSE = "</think>"


def round_to_grid(x: float) -> float:
    return min(GRID, key=lambda g: (abs(g - float(x)), g))


def extract_proof_text(decoded_text: str):
    idx = decoded_text.rfind(THINK_CLOSE)
    post = decoded_text[idx + len(THINK_CLOSE):] if idx >= 0 else decoded_text
    m = _PROOF_RE.search(post)
    return m.group(1).strip() if m else None


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def qid_from_text(text: str) -> str:
    """MUST match verl.trainer.ppo.difficulty.qid_from_text (marker-slice sha1)."""
    text = text or ""
    a = text.find("mathematical problem:")
    b = text.find("Solve the problem")
    core = text[a:b] if (a != -1 and b != -1 and b > a) else text
    return hashlib.sha1(core.encode("utf-8", "ignore")).hexdigest()


class BufferReplayer:
    """Replay-buffer state as a function of dataset_step: seed + deltas < step."""

    def __init__(self, seed_dir: str, delta_log_path: str):
        self.entries: dict[str, list[dict]] = {}
        n = 0
        for path in sorted(glob.glob(os.path.join(seed_dir, "replay_buffer/shard_*.jsonl"))):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    e = json.loads(line)
                    self.entries.setdefault(e["qid"], []).append(e)
                    n += 1
        assert n > 0, f"no replay seed entries under {seed_dir}"
        self.deltas: list[dict] = []
        skipped_tail = 0
        if os.path.exists(delta_log_path):
            with open(delta_log_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self.deltas.append(json.loads(line))
                    except json.JSONDecodeError:
                        skipped_tail += 1  # torn final line from a live-run copy
        assert skipped_tail <= 1, f"{skipped_tail} unparseable delta lines (expected <=1 torn tail)"
        self._applied_upto = 0  # deltas[0:_applied_upto] applied
        print(f"[q_build] replay seed {n} entries, {len(self.deltas)} deltas "
              f"(+{skipped_tail} torn tail skipped)", flush=True)

    def state_at(self, dataset_step: int) -> dict[str, list[dict]]:
        """Advance (monotone) to seed + all deltas with delta.dataset_step < dataset_step."""
        while self._applied_upto < len(self.deltas):
            d = self.deltas[self._applied_upto]
            if int(d["dataset_step"]) >= dataset_step:
                break
            replaced = set(d.get("replaced_entry_ids", []))
            if replaced:
                for qid in list(self.entries):
                    kept = [e for e in self.entries[qid] if e["entry_id"] not in replaced]
                    if kept:
                        self.entries[qid] = kept
                    else:
                        del self.entries[qid]
            for e in d.get("added_entries", []):
                self.entries.setdefault(e["qid"], []).append(e)
            self._applied_upto += 1
        return self.entries


def iter_dump_rows(path: str):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_shards(records: list[dict], out_dir: str, subdir: str, shard_size: int = 256) -> dict:
    os.makedirs(os.path.join(out_dir, subdir), exist_ok=True)
    shards = {}
    for si in range(0, max((len(records) + shard_size - 1) // shard_size, 1)):
        chunk = records[si * shard_size:(si + 1) * shard_size]
        rel = f"{subdir}/shard_{si:04d}.jsonl"
        path = os.path.join(out_dir, rel)
        with open(path, "w", encoding="utf-8") as f:
            for r in chunk:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        shards[rel] = sha256_file(path)
    return shards
