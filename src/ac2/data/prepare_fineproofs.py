"""Prepare the FineProofs prover train + IMOProofBench val parquet files (verl RL schema).

  Train: lm-provers/FineProofs-RL   (column: problem)
  Val:   lm-provers/IMOProofBench   (column: problem)

The data is **prover-only**: each source theorem becomes ONE prover row whose prompt
asks for a ``<proof>...</proof>`` block; ``ac2.rewards.prover_judge`` then LLM-judges that
proof (binary 1/0). Both splits are prover-only, so verl's ``val-core/<ds>/acc/mean@N`` headline
is a clean binary direct-proof signal.

``extra_info.theorem`` carries the raw problem text — the reward reads it as the judged problem,
and ``ac2.viz`` uses it to label proof cards.

Usage:
  python -m ac2.data.prepare_fineproofs                 # both, to ~/data/fineproofs
  python -m ac2.data.prepare_fineproofs --out-dir /scr/data/fineproofs --split train
"""

import argparse
import json
from pathlib import Path

import datasets

TRAIN_SOURCE = "lm-provers/FineProofs-RL"
VAL_SOURCE = "lm-provers/IMOProofBench"
TRAIN_DATA_SOURCE = "fineproofs-rl"   # honest label; the custom reward ignores it for routing
VAL_DATA_SOURCE = "imoproofbench"

_PROVER_TEMPLATE_PATH = Path(__file__).parent / "templates" / "prover_prompt.txt"


def _load_template() -> str:
    tmpl = _PROVER_TEMPLATE_PATH.read_text()
    if "{problem}" not in tmpl:
        raise ValueError(f"prover prompt template {_PROVER_TEMPLATE_PATH} is missing the {{problem}} slot.")
    return tmpl


def to_verl_record(problem: str, idx: int, split: str, data_source: str, template: str) -> dict:
    """Map a source theorem to a verl RL parquet row (prover mode)."""
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": template.format(problem=problem)}],
        "ability": "Proof",
        # No rule-based ground truth: correctness is decided by the LLM proof judge.
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {
            "split": split,
            "index": idx,
            "question": problem,
            "theorem": problem,   # read by the reward (judged problem) and the viz (card label)
            "mode": "prover",
        },
    }


def _build(hf_source: str, split: str, data_source: str, template: str) -> datasets.Dataset:
    ds = datasets.load_dataset(hf_source, split="train")
    ds = ds.map(
        lambda ex, idx: to_verl_record(ex["problem"], idx, split, data_source, template),
        with_indices=True,
        remove_columns=ds.column_names,
    )
    return ds


def build_train(out_dir: Path, template: str) -> Path:
    ds = _build(TRAIN_SOURCE, "train", TRAIN_DATA_SOURCE, template)
    out = out_dir / "train.parquet"
    ds.to_parquet(str(out))
    print(f"[train] {len(ds)} prover rows from {TRAIN_SOURCE} -> {out}")
    return out


def build_val(out_dir: Path, template: str) -> Path:
    ds = _build(VAL_SOURCE, "test", VAL_DATA_SOURCE, template)
    out = out_dir / "test.parquet"
    ds.to_parquet(str(out))
    print(f"[val] {len(ds)} prover rows from {VAL_SOURCE} -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="~/data/fineproofs", help="dir for train.parquet / test.parquet")
    ap.add_argument("--split", choices=["train", "val", "both"], default="both")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    template = _load_template()

    provenance = {"train_source": TRAIN_SOURCE, "val_source": VAL_SOURCE, "mode": "prover",
                  "prover_template_path": str(_PROVER_TEMPLATE_PATH)}
    if args.split in ("train", "both"):
        provenance["train_path"] = str(build_train(out_dir, template))
    if args.split in ("val", "both"):
        provenance["val_path"] = str(build_val(out_dir, template))
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
