# Exporting per-run metrics for the paper

`export_paper_metrics.py` (wrapper: `export_paper_metrics.sh`) reads finished training runs and
writes the per-step quantities behind the paper's Decoding-FLOPs accounting (App. B)
and the logged validation metrics. It runs on CPU, uses only the Python standard library, and
only reads the run directories. The output contains numerical summaries, source paths and
checksums. It contains no prompts, completions, sample IDs, checkpoints or credentials.

## Usage

```bash
bash scripts/export_paper_metrics.sh                       # all default runs, read from ./experiments
bash scripts/export_paper_metrics.sh --run ac2
bash scripts/export_paper_metrics.sh --run NAME=/absolute/path/to/experiment/dir
bash scripts/export_paper_metrics.sh --metrics-only        # logged metrics only, no trajectory scans
```

| Option | Meaning |
|---|---|
| `--repo-root PATH` | Repository whose `experiments/` holds the runs (default: this checkout). |
| `--run NAME` or `--run NAME=DIR` | Select runs; repeatable. `DIR` overrides the run directory. |
| `--output DIR` | New output directory (default: a timestamped directory under `/tmp`). |
| `--train-only` | Skip validation-rollout scans; logged validation metrics are still exported. |
| `--metrics-only` | Skip all trajectory scans; export `metrics.jsonl` values and dashboard means only. |

The script prints the path of a ZIP holding one JSON per run plus `manifest.json`. A run whose
directory or steps are incomplete is marked `partial` or `error`; missing values are never
estimated.

Default run and the minimum training-step coverage it must reach:

| Run | Paper name | Minimum steps |
|---|---|---:|
| `ac2` | AC2 | 202 |

## Exported quantities

For each training step, the exporter records the request count, invalid and missing counts,
generated- and prefix-token sums and sums of squares, decode-forward sums and sums of squares,
the prefix × forward cross-product, and the exact growing-context attention sum. For one request,

```text
p = prompt_length + sp_prefix_len        # original prompt plus replayed prefix
g = response_length - sp_prefix_len      # newly generated tokens
m = max(g - 1, 0)                        # incremental decoding forwards (the first token comes from prefill)
attention_context_sum = m*p + m*(m+1)/2
```

The decoding cost of Eq. (6) is then `A * sum(m) + B * sum(attention_context_sum)`.
Validation, critic and judge calls, prefill and training updates are not counted.

Logged training metrics (`response_length/mean`, `prompt_length/mean`, replay-prefix statistics,
and others) are kept in `reported_metrics.per_step`, with per-metric coverage in
`reported_metrics.training_metric_steps`. Missing values are filled from the run's dashboard
cache (`*_fig_data.json`) when one exists. Values from the original logs take precedence, and any
conflict is reported. Validation metrics keep their logged names: `acc/mean@16` is the mean score
and `acc/best@16/mean` the best-of-16 score.

## Tests

```bash
python3 -m unittest discover -s tests/local -p test_export_paper_metrics.py -v
```
