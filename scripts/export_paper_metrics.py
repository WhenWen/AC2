#!/usr/bin/env python3
"""Read-only, standard-library export of length moments and logged paper metrics."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile
import zipfile

# Runs exported by default, with the minimum training-step coverage each must reach.
DEFAULT_RUNS = {
    'ac2': 202,
}
STEP_FILE = re.compile(r'(?:step[_-]?)?(\d+)\.jsonl(?:\.gz)?$')
TRAIN_DIRS = ['run_data/rollouts', 'run_data/rollouts/train', 'rollouts/train']
VAL_DIRS = ['run_data/val_rollouts', 'run_data/rollouts/val', 'rollouts/val']
METRIC_FILES = ['run_data/metrics.jsonl', 'summaries/metrics.jsonl', 'metrics.jsonl']
MOMENTS = ['prompt_tokens_sum', 'replay_prefix_tokens_sum', 'prefix_tokens_sum',
           'prefix_tokens_squared_sum', 'generated_tokens_sum',
           'generated_tokens_squared_sum', 'decode_forwards_sum',
           'decode_forwards_squared_sum', 'decode_prefix_product_sum',
           'decode_context_sum', 'response_tokens_sum']
DEFINITION = {
    'scope': 'policy rollout decoding; validation is exported separately and never added',
    'prefix_p': 'prompt_length + sp_prefix_len',
    'generated_g': 'response_length - sp_prefix_len (stored response includes replay)',
    'decode_m': 'max(g - 1, 0); first sampled token belongs to prefill',
    'context_sum_per_request': 'm*p + m*(m+1)//2',
    'flops_from_moments': 'A * decode_forwards_sum + B * decode_context_sum',
    'architecture_note': 'A and B are model-specific; no model size or judge architecture is assumed here',
    'excluded': ['validation from training cost', 'prefill', 'Q and judge model calls',
                 'training forward/backward', 'failed requests absent from the stored logs'],
}


def integer(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    if not math.isfinite(v) or v < 0 or int(v) != v:
        return None
    return int(v)


def stream(path):
    return gzip.open(path, 'rb') if path.name.endswith('.gz') else path.open('rb')


def scan_rollout(path, step):
    """Export only numerical summaries; never prompt/completion text or IDs."""
    before = path.stat()
    out = dict(step=step, rows=0, valid_length_rows=0, missing_length_rows=0,
               malformed_rows=0, zero_generated_rows=0, replay_field_absent_rows=0,
               **{k: 0 for k in MOMENTS})
    errors = Counter()
    digest = hashlib.sha256()
    with stream(path) as handle:
        for line in handle:
            digest.update(line)
            if not line.strip():
                continue
            out['rows'] += 1
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                out['malformed_rows'] += 1
                continue
            if not isinstance(row, dict):
                out['malformed_rows'] += 1
                continue
            prompt = integer(row.get('prompt_length'))
            response = integer(row.get('response_length'))
            if response is None:
                response = integer(row.get('response_length_tokens'))
            # Older pure-GRPO dumps omit this field; report how often the default applies.
            if 'sp_prefix_len' not in row:
                out['replay_field_absent_rows'] += 1
            replay = integer(row.get('sp_prefix_len', 0))
            if prompt is None or response is None or replay is None or response < replay:
                out['missing_length_rows'] += 1
                errors['invalid_or_missing_length'] += 1
                continue
            if 'step' in row and integer(row['step']) != step:
                out['missing_length_rows'] += 1
                errors['row_step_disagrees_with_filename'] += 1
                continue
            p, g = prompt + replay, response - replay
            m = max(g - 1, 0)
            out['valid_length_rows'] += 1
            out['zero_generated_rows'] += int(g == 0)
            values = [prompt, replay, p, p*p, g, g*g, m, m*m, m*p,
                      m*p + m*(m+1)//2, response]
            for key, value in zip(MOMENTS, values):
                out[key] += value
    after = path.stat()
    out['source_bytes'] = before.st_size
    out['source_mtime_ns'] = before.st_mtime_ns
    out['uncompressed_sha256'] = digest.hexdigest()
    out['file_changed_during_read'] = (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
    out['complete'] = (out['rows'] > 0 and out['rows'] == out['valid_length_rows']
                       and not out['file_changed_during_read'])
    out['errors'] = dict(errors)
    n = out['valid_length_rows']
    out['response_length_mean_valid_rows'] = out['response_tokens_sum']/n if n else None
    return out


def discover(run_dir, directories):
    result = {}
    for directory in directories:
        for path in sorted((run_dir/directory).glob('*.jsonl*')):
            match = STEP_FILE.fullmatch(path.name)
            if not match:
                continue
            step = int(match[1])
            if step in result and path.resolve() != result[step].resolve():
                raise ValueError(f'Ambiguous step {step}: {result[step]} and {path}; select a single source layout')
            result[step] = path
    return result


def flatten(obj, prefix=''):
    for key, value in obj.items():
        key = (prefix + '__' + key if prefix else key).replace('/', '__')
        if isinstance(value, dict):
            yield from flatten(value, key)
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            yield key, value


# Preserve logged populations/names; never substitute validation for training.
TRAIN_METRIC_PREFIXES = (
    'response_length__', 'response_length_non_aborted__', 'prompt_length__',
    'v7__train__prover__resp_len_', 'v7__train__prover__prefix_tokens_',
)
TRAIN_METRIC_KEYS = {
    'response__aborted_ratio', 'perf__total_num_tokens',
    'v7__train__prover__generated_tokens_sum', 'v7__train__prover__rows_generated',
}


def wanted_metric(key):
    return (key.startswith(('val-', 'val_', 'validation__', 'v7__val__')) or
            key.startswith(TRAIN_METRIC_PREFIXES) or key in TRAIN_METRIC_KEYS)


def metric_caches(run_dir):
    patterns = [directory + suffix for directory in ['', 'analysis/', '.dash*/analysis/']
                for suffix in ['*fig_data.json', '*canonical.json']]
    # Parent/recent caches can describe another run or a truncated window.
    return sorted({p for pattern in patterns for p in run_dir.glob(pattern)
                   if p.is_file() and 'recent' not in p.name and not p.name.startswith('parent_')})


def read_metric_cache(path):
    data = json.loads(path.read_text())['per_step']
    rows = {}
    for i, value in enumerate(data.get('steps', [])):
        step = integer(value)
        if step is None:
            continue
        metrics = {}
        for key, values in data.items():
            if isinstance(values, list) and i < len(values):
                metrics.update({k: v for k, v in flatten({key: values[i]}) if wanted_metric(k)})
        if metrics:
            rows.setdefault(step, {}).update(metrics)
    return rows


def reported_metrics(run_dir):
    """Retain numerical metrics, supplementing raw logs with a full dashboard cache."""
    paths = [run_dir/p for p in METRIC_FILES if (run_dir/p).is_file()]
    if len(paths) > 1:
        raise ValueError('Multiple metrics.jsonl sources; select one run directory: ' + str(paths))
    rows, issues, sources, provenance = {}, [], [], {}
    if paths:
        path = paths[0]
        source = str(path.relative_to(run_dir))
        sources.append(source)
        with stream(path) as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    step = integer(record.get('step'))
                    values = record.get('data', record)
                    if step is None or not isinstance(values, dict):
                        raise ValueError('Missing step/data')
                    flat = {k: v for k, v in flatten(values) if wanted_metric(k)}
                    target = rows.setdefault(step, {})
                    if any(k in target and target[k] != v for k, v in flat.items()):
                        issues.append(f'Conflicting logged values at step {step}; last record retained')
                    target.update(flat)
                    provenance.setdefault(step, {}).update({k: source for k in flat})
                except (ValueError, AttributeError, UnicodeDecodeError):
                    issues.append(f'Malformed metric record at line {line_no}')
    caches = []
    for path in metric_caches(run_dir):
        try:
            cached = read_metric_cache(path)
            caches.append((path, cached))
        except (OSError, ValueError, KeyError, AttributeError, TypeError) as exc:
            issues.append(f'Unreadable metric cache {path.relative_to(run_dir)}: {type(exc).__name__}')
    if caches:
        # A freshly rebuilt last-20-step cache must not displace a full canonical.
        path, cached = max(caches, key=lambda item: (
            sum(any(k.startswith(TRAIN_METRIC_PREFIXES) or k in TRAIN_METRIC_KEYS
                    for k in metrics) for step, metrics in item[1].items() if step > 0),
            len(item[1]), sum(len(v) for v in item[1].values()),
            item[0].stat().st_mtime_ns, str(item[0])))
        source = str(path.relative_to(run_dir))
        added = conflicts = 0
        for step, metrics in cached.items():
            target = rows.setdefault(step, {})
            for key, value in metrics.items():
                if key in target:
                    conflicts += target[key] != value
                else:
                    target[key] = value
                    provenance.setdefault(step, {})[key] = source
                    added += 1
        if added:
            sources.append(source)
            issues.append(f'Dashboard cache supplied {added} missing step/metric values: {source}')
        if conflicts:
            issues.append(f'{conflicts} cache/log conflicts; original metrics.jsonl values retained')
    training_steps = {}
    for step, metrics in sorted(rows.items()):
        for key in metrics:
            if step > 0 and not key.startswith(('val-', 'val_', 'validation__', 'v7__val__')):
                training_steps.setdefault(key, []).append(step)
    return {'sources': sources, 'issues': issues, 'training_metric_steps': training_steps,
            'per_step': [{'step': step, 'metrics': rows[step],
                          'metric_sources': provenance.get(step, {})} for step in sorted(rows)]}


def choose_run(root, name, override=None):
    if override:
        return Path(override).expanduser().resolve()
    return root/'experiments'/name


def export_run(root, name, override=None, train_only=False, metrics_only=False):
    run_dir = choose_run(root, name, override)
    report = dict(schema='q_reasoning_paper_metrics_v1', run_id=name,
                  source_run_dir=str(run_dir), definition=DEFINITION, issues=[])
    if not run_dir.is_dir():
        report['issues'].append('Run directory not found')
    report['reported_metrics'] = reported_metrics(run_dir)
    for split, directories in [('train', TRAIN_DIRS), ('validation', VAL_DIRS)]:
        files = discover(run_dir, directories) if not metrics_only and (split == 'train' or not train_only) else {}
        rows = []
        for i, (step, path) in enumerate(sorted(files.items()), 1):
            print(f'[{name}] {split} step {step} ({i}/{len(files)})', flush=True)
            row = scan_rollout(path, step)
            if split == 'train' and name in DEFAULT_RUNS and row['replay_field_absent_rows']:
                row['complete'] = False
                row['errors']['replay_length_absent_in_replay_experiment'] = row['replay_field_absent_rows']
            row['source'] = str(path.relative_to(run_dir))
            rows.append(row)
        report[split] = {'per_step': rows, 'scan_skipped': metrics_only or (split == 'validation' and train_only)}
    train = report['train']['per_step']
    expected = max([DEFAULT_RUNS.get(name, 0)] + [r['step'] for r in train] +
                   [r['step'] for r in report['reported_metrics']['per_step']])
    complete = {r['step'] for r in train if r['complete']}
    missing = [s for s in range(1, expected+1) if s not in complete]
    report['train'].update(required_through_step=expected, missing_or_incomplete_steps=missing,
                           complete=bool(train) and not missing)
    reported = {r['step']: r['metrics'] for r in report['reported_metrics']['per_step']}
    mismatches = []
    for row in train:
        token_sum = reported.get(row['step'], {}).get('v7__train__prover__generated_tokens_sum')
        if row['complete'] and token_sum is not None and token_sum != row['generated_tokens_sum']:
            mismatches.append(row['step'])
    report['train']['token_sum_mismatch_steps'] = mismatches
    if mismatches:
        report['train']['complete'] = False
        report['issues'].append('Rollout token totals disagree with logged prover totals; verify lineage/mode scope')
    needed = {
        'pass_at_1': 'val-core__imoproofbench__acc__mean@16',
        'best_at_16': 'val-core__imoproofbench__acc__best@16__mean',
        'response_length': 'val-aux__imoproofbench__response_length_tokens__mean@16',
    }
    coverage = {label: sorted(s for s, metrics in reported.items() if key in metrics)
                for label, key in needed.items()}
    raw_validation = {r['step'] for r in report['validation']['per_step'] if r['complete']}
    eval_steps = sorted(set(coverage['pass_at_1']) | set(coverage['best_at_16']))
    report['validation']['reported_metric_steps'] = coverage
    report['validation']['missing_response_length_steps'] = [
        s for s in eval_steps if s not in coverage['response_length'] and s not in raw_validation]
    val_ready = bool(coverage['pass_at_1'] and coverage['best_at_16']) and not report['validation']['missing_response_length_steps']
    report['status'] = 'complete' if report['train']['complete'] and val_ready else 'partial'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, help='New output directory; default is a timestamped directory under /tmp')
    parser.add_argument('--run', action='append', help='Run name, or name=/absolute/experiment/directory; repeat to select runs')
    parser.add_argument('--train-only', action='store_true', help='Skip validation rollout scans (logged validation metrics still exported)')
    parser.add_argument('--metrics-only', action='store_true', help='Export logged metrics and dashboard means without scanning rollout trajectories')
    args = parser.parse_args()
    root = args.repo_root.expanduser().resolve()
    out = args.output.expanduser().resolve() if args.output else Path(tempfile.mkdtemp(prefix='q-reasoning-paper-metrics-'))
    if args.output:
        out.mkdir(parents=True, exist_ok=False)
    specs = args.run or list(DEFAULT_RUNS)
    manifest = {'schema': 'q_reasoning_paper_metrics_bundle_v1', 'generated_at_utc': datetime.now(timezone.utc).isoformat(),
                'definition': DEFINITION, 'runs': []}
    try:
        manifest['exporter_git_commit'] = subprocess.check_output(['git', '-C', str(Path(__file__).resolve().parents[1]), 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        manifest['exporter_git_commit'] = None
    names = set()
    for spec in specs:
        name, sep, override = spec.partition('=')
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or name in names:
            parser.error('Run names must be unique, simple directory names')
        names.add(name)
        try:
            result = export_run(root, name, override if sep else None, args.train_only, args.metrics_only)
        except (OSError, ValueError, KeyError) as e:
            result = {'schema': 'q_reasoning_paper_metrics_v1', 'run_id': name, 'status': 'error', 'issues': [str(e)]}
        filename = name+'.json'
        (out/filename).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        manifest['runs'].append({'run_id': name, 'file': filename, 'status': result['status'], 'issues': result.get('issues', [])})
        print(f'[{name}] {result["status"]}', flush=True)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    archive = out.with_name(out.name+'.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.glob('*.json')):
            z.write(path, path.name)
    counts = Counter(r['status'] for r in manifest['runs'])
    print('Status:', dict(counts), flush=True)
    print('UPLOAD_THIS_ZIP='+str(archive), flush=True)
    print('Partial/error runs are recorded in manifest.json; no missing lengths were estimated.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
