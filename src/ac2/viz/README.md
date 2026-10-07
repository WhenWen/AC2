# Training dashboards (`ac2.viz`)

Tools that turn a training run's outputs into a multi-page PDF dashboard: validation scores,
training reward and response length, judge health, critic and readiness metrics, optimizer
statistics, and throughput. Every experiment folder's `refresh_dashboard.sh` uses this pipeline,
and the `*_dashboard.pdf` files in the experiment folders were rendered with it.

## Pipeline

The parser and the merge step use only the standard library, so they can run on a cluster login
node next to the raw rollouts. The renderer needs matplotlib and numpy.

```bash
# 1. Parse <exp>/manifest and <exp>/run_data into a compact fig_data JSON.
python -m ac2.viz.parse_fig_data --run-dir experiments/<exp> --run-id <exp> \
    --out /tmp/<exp>_parse.json            # add --start-step N for an incremental parse

# 2. Merge the parse output into the run's single canonical fig_data file.
#    The merge checks that the overlapping step agrees and fails without writing on conflict.
python -m ac2.viz.merge_fig_data --parse-output /tmp/<exp>_parse.json \
    --canonical experiments/<exp>/analysis/<exp>_fig_data.json

# 3. Render the dashboard PDF into the analysis directory.
python -m ac2.viz.render_dashboard experiments/<exp>/analysis/<exp>_fig_data.json \
    experiments/<exp>/analysis --experiment-folder <exp>
```

For a run that branches from a parent checkpoint, `merge_fig_data --seed <parent fig_data>`
starts the child's canonical file from the parent's history.

## Files

| File | Role |
|---|---|
| `parse_fig_data.py` | Parser. Reads `manifest/config.yaml`, `run_data/metrics.jsonl`, `run_data/rollouts/<step>.jsonl` and `run_data/val_rollouts/<step>.jsonl`, and writes a `fig_data` JSON with per-step series. |
| `merge_fig_data.py` | Appends a parse output to the canonical `fig_data` file after checking the overlapping step. |
| `render_dashboard.py` | Renderer entry point. Writes `<exp>_dashboard.pdf`. |
| `single_run_dashboard.py` | Main dashboard page. |
| `diagnostics_dashboard.py` | Diagnostics page. |
| `dashboard_common.py` | Shared plotting helpers (palette, series accessors, header). |
| `hintgap_dashboard.py` | Optional extra page (`render_dashboard --hintgap-sidecar`); not used by the experiments in this repository. |
| `test_*.py`, `tests/` | Unit tests for the parser, merge and dashboard helpers. |
