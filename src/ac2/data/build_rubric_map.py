"""Build the rubric map for ``qednano_rubric_judge``: {sha1(normalized problem): rubric}.

Two sources, matching the two splits the reward grades:
  * TRAIN — ``lm-provers/FineProofs-RL`` ``rubrics`` column (QED-Nano's per-problem grading
    schemes, arXiv:2604.04898 §Setup). Our train parquet is built from this exact dataset
    (``prepare_fineproofs``), so keys join on raw ``problem`` text.
  * VAL — IMO-ProofBench ``Grading guidelines`` (+ reference ``Solution``, appended, since the
    strict grader prompt has no separate reference slot) from google-deepmind/superhuman
    ``proofbench_v2.csv``. Our val parquet problems come from ``lm-provers/IMOProofBench``;
    both are keyed after whitespace-collapse+lowercase normalization so formatting drift
    between the two republications doesn't break the join.

Usage (needs internet; run on the cluster login node inside the project venv):
  python -m ac2.data.build_rubric_map --out ~/data/fineproofs/rubric_map.json
"""

import argparse
import csv
import hashlib
import io
import json
import re
import urllib.request
from pathlib import Path

PROOFBENCH_CSV_URL = (
    "https://raw.githubusercontent.com/google-deepmind/superhuman/main/imobench/proofbench_v2.csv"
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text or "").lower()


def _key(text: str) -> str:
    return hashlib.sha1(_norm(text).encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="~/data/fineproofs/rubric_map.json")
    ap.add_argument("--val-out", default="~/data/fineproofs/val_map.json",
                    help="ProofAutoGrader val map: {key: {solution, guidelines}} (separate slots, "
                         "for the reward's VAL branch)")
    ap.add_argument("--train-source", default="lm-provers/FineProofs-RL")
    ap.add_argument("--val-source", default="lm-provers/IMOProofBench")
    args = ap.parse_args()

    import datasets  # heavy import kept local

    rubric_map: dict[str, str] = {}

    # --- TRAIN: FineProofs-RL (problem, rubrics) ---
    train = datasets.load_dataset(args.train_source, split="train")
    n_empty = 0
    for row in train:
        rub = (row.get("rubrics") or "").strip()
        if not rub:
            n_empty += 1
            continue
        rubric_map[_key(row["problem"])] = rub
    print(f"[rubric-map] train: {len(train)} rows -> {len(rubric_map)} rubrics ({n_empty} empty)")

    # --- VAL: IMO-ProofBench guidelines, keyed by the lm-provers/IMOProofBench problem text ---
    csv_text = urllib.request.urlopen(PROOFBENCH_CSV_URL, timeout=120).read().decode()
    bench = {  # normalized deepmind problem -> (guidelines, solution)
        _norm(r["Problem"]): (r["Grading guidelines"], r["Solution"])
        for r in csv.DictReader(io.StringIO(csv_text))
    }
    val = datasets.load_dataset(args.val_source, split="train")
    val_map: dict[str, dict] = {}
    n_val = n_miss = 0
    for row in val:
        prob = row["problem"]
        npb = _norm(prob)
        hit = bench.get(npb)
        if hit is None:  # containment fallback for wrapper-text drift between republications
            hit = next((v for k, v in bench.items() if k and (k in npb or npb in k)), None)
        if hit is None:
            n_miss += 1
            continue
        guidelines, solution = hit
        # legacy combined entry (kept so a rubric-branch lookup on a val problem still works)
        rubric_map[_key(prob)] = (
            f"{guidelines.strip()}\n\n(Reference solution, as an anchor for sufficiency, "
            f"not exclusivity:)\n{solution.strip()}"
        )
        # structured entry for the ProofAutoGrader val branch (separate prompt slots)
        val_map[_key(prob)] = {"solution": solution.strip(), "guidelines": guidelines.strip()}
        n_val += 1
    print(f"[rubric-map] val: {len(val)} rows -> {n_val} matched, {n_miss} unmatched")

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(rubric_map, f)
    print(f"[rubric-map] wrote {len(rubric_map)} entries -> {out}")
    vout = Path(args.val_out).expanduser()
    vout.parent.mkdir(parents=True, exist_ok=True)
    with open(vout, "w") as f:
        json.dump(val_map, f)
    print(f"[rubric-map] wrote {len(val_map)} val entries -> {vout}")


if __name__ == "__main__":
    main()
