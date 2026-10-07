"""No cluster, model, torch or network dependencies."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

SCRIPT = Path(__file__).resolve().parents[2]/'scripts/export_paper_metrics.py'
spec = importlib.util.spec_from_file_location('exporter', SCRIPT)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write_rows(self, relative, rows):
        path = self.root/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return path

    def test_exact_attention_replay_zero_and_prefill(self):
        data = [dict(prompt_length=7, response_length=9, sp_prefix_len=5),
                dict(prompt_length=2, response_length=1),
                dict(prompt_length=2, response_length=0)]
        path = self.write_rows('1.jsonl', data)
        row = exporter.scan_rollout(path, 1)
        self.assertTrue(row['complete'])
        # First request: p=12, g=4, m=3; attended lengths are 13,14,15.
        self.assertEqual(row['decode_context_sum'], sum([13, 14, 15]))
        self.assertEqual(row['decode_forwards_sum'], 3)
        self.assertEqual(row['generated_tokens_sum'], 5)
        self.assertEqual(row['zero_generated_rows'], 1)
        self.assertEqual(row['decode_prefix_product_sum'], 36)

    def test_invalid_lengths_never_become_complete(self):
        path = self.write_rows('2.jsonl', [dict(prompt_length=3, response_length=2, sp_prefix_len=5),
                                           dict(prompt_length=True, response_length=4),
                                           dict(prompt_length=3, response_length=5, step=9),
                                           dict(prompt_length=3, response_length=4, sp_prefix_len=None)])
        with path.open('a') as f:
            f.write('{broken\n')
        row = exporter.scan_rollout(path, 2)
        self.assertFalse(row['complete'])
        self.assertEqual(row['missing_length_rows'], 4)
        self.assertEqual(row['malformed_rows'], 1)
        self.assertEqual(row['decode_context_sum'], 0)

    def test_bundle_gap_validation_separation_and_numeric_only(self):
        run = 'fixture'
        train = dict(prompt_length=2, response_length=4, input='PRIVATE_PROMPT_SENTINEL', output='PRIVATE_COMPLETION_SENTINEL')
        self.write_rows('experiments/fixture/run_data/rollouts/1.jsonl', [train])
        self.write_rows('experiments/fixture/run_data/rollouts/3.jsonl', [train])
        self.write_rows('experiments/fixture/run_data/val_rollouts/0.jsonl', [dict(prompt_length=2, response_length=500)])
        self.write_rows('experiments/fixture/run_data/metrics.jsonl', [dict(step=0, data={
            'val-core/imoproofbench/acc/mean@16': .25,
            'val-core/imoproofbench/acc/best@16/mean': .75})])
        before = {p: p.read_bytes() for p in (self.root/'experiments').rglob('*.jsonl')}
        out = self.root/'export'
        result = subprocess.run([sys.executable, str(SCRIPT), '--repo-root', str(self.root), '--output', str(out), '--run', run, '--run', 'missing'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        d = json.loads((out/'fixture.json').read_text())
        self.assertEqual(d['train']['missing_or_incomplete_steps'], [2])
        self.assertEqual(d['train']['per_step'][0]['generated_tokens_sum'], 4)
        self.assertEqual(d['validation']['per_step'][0]['generated_tokens_sum'], 500)
        self.assertEqual(d['validation']['reported_metric_steps']['pass_at_1'], [0])
        self.assertEqual(d['validation']['missing_response_length_steps'], [])
        self.assertEqual(json.loads((out/'missing.json').read_text())['status'], 'partial')
        with zipfile.ZipFile(out.with_suffix('.zip')) as z:
            self.assertEqual(set(z.namelist()), {'fixture.json', 'missing.json', 'manifest.json'})
            for filename in z.namelist():
                self.assertNotIn(b'PRIVATE_', z.read(filename))
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_run_dir_and_duplicate_layout(self):
        self.write_rows('experiments/fixture_run/run_data/rollouts/1.jsonl', [{}])
        chosen = exporter.choose_run(self.root, 'fixture_run')
        self.assertEqual(chosen.name, 'fixture_run')
        self.write_rows('experiments/fixture_run/run_data/rollouts/train/1.jsonl', [{}])
        with self.assertRaisesRegex(ValueError, 'Ambiguous step'):
            exporter.discover(chosen, exporter.TRAIN_DIRS)

    def test_token_total_mismatch_flags_lineage(self):
        self.write_rows('experiments/fixture/run_data/rollouts/1.jsonl', [dict(prompt_length=1, response_length=4)])
        self.write_rows('experiments/fixture/run_data/metrics.jsonl', [dict(step=1, data={'v7/train/prover/generated_tokens_sum': 99})])
        d = exporter.export_run(self.root, 'fixture', train_only=True)
        self.assertEqual(d['train']['token_sum_mismatch_steps'], [1])
        self.assertFalse(d['train']['complete'])

    def test_training_means_survive_export_without_rollouts(self):
        self.write_rows('experiments/fixture/run_data/metrics.jsonl', [dict(step=1, data={
            'response_length/mean': 12345.5, 'prompt_length/mean': 180.25,
            'response/aborted_ratio': 0, 'response_length_non_aborted/mean': 12345.5,
            'v7/train/prover/resp_len_mean': 12345.5,
            'v7/train/prover/rows_generated': 4096,
            'v7/train/prover/resp_len_replay_n': 0,
            'v7/train/prover/prefix_tokens_replay_mean': 0,
            'v7/train/prover/generated_tokens_sum': 50567168,
            'val-aux/imoproofbench/response_length_tokens/mean@16': 999,
            'sample_id': 123456789, 'input': 'PRIVATE_PROMPT_SENTINEL'})])
        report = exporter.export_run(self.root, 'fixture', metrics_only=True)
        logged = report['reported_metrics']['per_step'][0]['metrics']
        self.assertEqual(logged['response_length__mean'], 12345.5)
        self.assertEqual(logged['prompt_length__mean'], 180.25)
        self.assertEqual(logged['v7__train__prover__rows_generated'], 4096)
        self.assertEqual(logged['val-aux__imoproofbench__response_length_tokens__mean@16'], 999)
        self.assertNotIn('sample_id', logged)
        self.assertNotIn('PRIVATE_', json.dumps(report))
        self.assertEqual(report['reported_metrics']['training_metric_steps']['response_length__mean'], [1])
        self.assertNotIn('val-aux__imoproofbench__response_length_tokens__mean@16', report['reported_metrics']['training_metric_steps'])
        self.assertFalse(report['train']['complete'])  # Means are not exact moments.
        self.assertTrue(report['train']['scan_skipped'])

    def test_canonical_supplements_existing_logs_without_overriding(self):
        run = self.root/'experiments/fixture'
        self.write_rows('experiments/fixture/run_data/metrics.jsonl', [dict(step=2, data={
            'response_length/mean': 20, 'val-core/imoproofbench/acc/mean@16': .3})])
        cache = run/'.dash/analysis/fixture_canonical.json'
        cache.parent.mkdir(parents=True)
        cache.write_text(json.dumps({'per_step': {'steps': [1, 2, 3],
            'response_length__mean': [10, 999, None], 'prompt_length__mean': [2, 3, True],
            'v7__train__prover__rows_generated': [4096, 4096, float('nan')],
            'input': ['PRIVATE_CACHE', 'PRIVATE_CACHE', 'PRIVATE_CACHE']}}))
        # Newer partial render cache should not erase the full history.
        (cache.parent/'fixture_fig_data.json').write_text(json.dumps({'per_step': {
            'steps': [2], 'response_length__mean': [999]}}))
        report = exporter.reported_metrics(run)
        rows = {r['step']: r for r in report['per_step']}
        self.assertEqual(rows[1]['metrics']['response_length__mean'], 10)
        self.assertEqual(rows[2]['metrics']['response_length__mean'], 20)
        self.assertEqual(rows[2]['metrics']['prompt_length__mean'], 3)
        self.assertEqual(rows[2]['metric_sources']['response_length__mean'], 'run_data/metrics.jsonl')
        self.assertEqual(rows[1]['metric_sources']['response_length__mean'], '.dash/analysis/fixture_canonical.json')
        self.assertNotIn(3, rows)
        self.assertTrue(any('conflicts' in s for s in report['issues']))
        self.assertNotIn('PRIVATE_', json.dumps(report, allow_nan=False))

    def test_cache_only_run(self):
        cache = self.root/'experiments/fixture_run/.dash/analysis/run_canonical.json'
        cache.parent.mkdir(parents=True)
        cache.write_text(json.dumps({'per_step': {'steps': [1], 'response_length__mean': [123]}}))
        chosen = exporter.choose_run(self.root, 'fixture_run')
        self.assertEqual(chosen.name, 'fixture_run')
        report = exporter.reported_metrics(chosen)
        self.assertEqual(report['per_step'][0]['metrics']['response_length__mean'], 123)
        self.assertEqual(report['sources'], ['.dash/analysis/run_canonical.json'])

    def test_metrics_only_cli_skips_rollouts_and_bundles_means(self):
        self.write_rows('experiments/fixture/run_data/rollouts/1.jsonl', [dict(prompt_length=2, response_length=5)])
        self.write_rows('experiments/fixture/run_data/metrics.jsonl', [dict(step=1, data={
            'response_length/mean': 5, 'prompt_length/mean': 2})])
        out = self.root/'means'
        result = subprocess.run([sys.executable, str(SCRIPT), '--repo-root', str(self.root),
            '--output', str(out), '--run', 'fixture', '--metrics-only'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        with zipfile.ZipFile(out.with_suffix('.zip')) as z:
            report = json.loads(z.read('fixture.json'))
        self.assertEqual(report['train']['per_step'], [])
        self.assertEqual(report['validation']['per_step'], [])
        self.assertTrue(report['train']['scan_skipped'])
        self.assertEqual(report['reported_metrics']['per_step'][0]['metrics']['response_length__mean'], 5)


if __name__ == '__main__':
    unittest.main()
