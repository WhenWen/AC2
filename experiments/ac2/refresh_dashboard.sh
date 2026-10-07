#!/usr/bin/env bash
# refresh_dashboard.sh — render the ac2 dashboard PDF (run on the login node).
#
# The run starts at step 1 from the base model, so there is no parent history to merge and no
# branch divider to draw: a staging dir with manifest/run_data symlinks, an incremental parse
# into one canonical fig_data, render, and copy the PDF to the experiment root.
set -euo pipefail
# Site configuration (paths, accounts) comes from the repository's .env; see .env.example.
_AC2_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.env"
[ -f "$_AC2_ENV" ] && { set -a; . "$_AC2_ENV"; set +a; }
REPO="${AC2_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
EXP="$REPO/experiments/ac2"
RUN_ID=ac2
S="$EXP/.dash_$RUN_ID"
A="$S/analysis"
CANON="$A/${RUN_ID}_fig_data.json"

mkdir -p "$A"
[ -e "$S/manifest" ] || ln -s "$EXP/manifest" "$S/manifest"
[ -e "$S/run_data" ] || ln -s "$EXP/run_data" "$S/run_data"

cd "$REPO" && source .venv/bin/activate

# The canonical's train watermark: parse only from there on, so per-refresh cost stays
# O(1 step) instead of re-reading every multi-hundred-MB rollout dump.
W=0
if [ -f "$CANON" ]; then
  W=$(python3 - "$CANON" <<'PY'
import json, sys
b = json.load(open(sys.argv[1]))
ps = b.get("per_step", {})
print(max((int(s) for s, v in zip(ps.get("steps", []), ps.get("v7__train__prover__rows_generated", [])) if v), default=0))
PY
)
fi

if [ ! -f "$CANON" ]; then
  echo "[dash] full build (no canonical yet)"
  python -m ac2.viz.parse_fig_data --run-dir "$S" --run-id "${RUN_ID}_live" \
      --out "$A/_live_fig.json" 2>&1 | tail -1
  python -m ac2.viz.merge_fig_data --parse-output "$A/_live_fig.json" \
      --canonical "$CANON" 2>&1 | tail -1
else
  echo "[dash] incremental parse from train watermark $W"
  python -m ac2.viz.parse_fig_data --run-dir "$S" --run-id "${RUN_ID}_live" \
      --start-step "$W" --out "$A/_live_fig.json" 2>&1 | tail -1
  python -m ac2.viz.merge_fig_data --parse-output "$A/_live_fig.json" \
      --canonical "$CANON" 2>&1 | tail -1
fi

python -m ac2.viz.render_dashboard "$CANON" "$A" \
    --experiment-folder "$RUN_ID" 2>&1 | tail -1
cp "$A/${RUN_ID}_dashboard.pdf" "$EXP/${RUN_ID}_dashboard.pdf"
CKPT_NOW=$(cat "$EXP/run_data/checkpoints/latest_checkpointed_iteration.txt")
echo "DASH_OK thru ckpt $CKPT_NOW $(date -Is)"
