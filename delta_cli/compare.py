"""Paired comparison of two local evaluation reports."""
import json
import math
import random
import re
import statistics
from pathlib import Path

import typer

from . import local

NOTE = 'Advisory local comparison. Official validation and private test scores are authoritative.'
SUMMARY = ('source_version_id', 'artifact', 'generation', 'validation_score')


def load(path):
    report = json.loads(Path(path).read_text())
    if not isinstance(report, dict):
        raise ValueError('Report must be a JSON object.')
    if 'public_validation' in report:
        return load_public(report)
    rows = local.generations_path(path)
    if not rows.exists():
        raise ValueError(f'Generations file not found: {rows}')
    with rows.open(encoding='utf-8') as f:
        means = local.example_means(json.loads(l) for l in f if l.strip())
    if not means:
        raise ValueError('Generations file has no rows.')
    return report, means


def load_public(report):
    """Validate the frozen public contract before pairing its stable item IDs."""
    from .public_eval.suite import score_public

    public = report['public_validation']
    if not isinstance(public, dict):
        raise ValueError('public_validation must be an object.')
    content_hash = public.get('content_hash')
    if not isinstance(content_hash, str) or not re.fullmatch(r'[0-9a-f]{64}', content_hash):
        raise ValueError('Public report must identify its checkpoint with a content hash.')
    # This also rejects reference fixtures, wrong suites, missing/duplicate IDs,
    # incomplete benchmark coverage and non-finite/out-of-range scores.
    score, _ = score_public(public, content_hash)
    normalized = {'report_format': 'public', 'protocol': public['protocol'],
                  'suite_sha256': public['suite_sha256'], 'content_hash': content_hash,
                  'validation_score': score}
    means = {name: {'examples': {row['id']: row['score'] for row in rows},
                    'truncation_rate': None, 'mean_tokens': None}
             for name, rows in public['groups'].items()}
    return normalized, means


def side(path, report):
    return {'path': str(path), **{k: report.get(k) for k in SUMMARY},
            **{k: report[k] for k in ('protocol', 'suite_sha256', 'content_hash') if k in report}}


def sign_test(wins, losses):
    n = wins + losses
    if not n:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def interval(values):
    values = sorted(values)
    return [values[int(.025 * len(values))], values[min(len(values) - 1, int(.975 * len(values)))]]


def compare_reports(baseline, candidate, resamples=10000, seed=0):
    (base, base_b), (cand, cand_b) = load(baseline), load(candidate)
    public = base.get('report_format') == 'public'
    if public != (cand.get('report_format') == 'public'):
        raise ValueError('Cannot compare public-evaluator and legacy local reports.')
    if base.get('config_id') != cand.get('config_id') or base.get('validation_sha256') != cand.get('validation_sha256'):
        raise ValueError('Reports use a different evaluation configuration.')
    if base.get('model_id') != cand.get('model_id'):
        raise ValueError('Reports belong to a different model_id.')
    if base_b.keys() != cand_b.keys() or any(base_b[b]['examples'].keys() != cand_b[b]['examples'].keys() for b in base_b):
        raise ValueError('Benchmark or example sets differ; rerun both with the same --limit and --benchmark.')
    names = sorted(base_b)
    diffs = {b: [(cand_b[b]['examples'][k] - base_b[b]['examples'][k]) * 100 for k in sorted(base_b[b]['examples'])] for b in names}
    rng = random.Random(seed)
    boot = {b: [] for b in names}
    overall = []
    for _ in range(resamples):
        samples = [rng.choices(diffs[b], k=len(diffs[b])) for b in names]
        means = [math.fsum(sample) / len(sample) for sample in samples]
        for b, mean in zip(names, means):
            boot[b].append(mean)
        overall.append(statistics.mean(means))
    benchmarks = {}
    for b in names:
        wins, losses = sum(d > 0 for d in diffs[b]), sum(d < 0 for d in diffs[b])
        n = len(diffs[b])
        benchmarks[b] = {
            'n': n, 'baseline_mean': statistics.mean(base_b[b]['examples'].values()) * 100,
            'candidate_mean': statistics.mean(cand_b[b]['examples'].values()) * 100,
            'diff': statistics.mean(diffs[b]), 'ci95': interval(boot[b]),
            'wins': wins, 'losses': losses, 'ties': n - wins - losses, 'sign_test_p': sign_test(wins, losses),
            'baseline_truncation_rate': base_b[b]['truncation_rate'], 'candidate_truncation_rate': cand_b[b]['truncation_rate'],
            'baseline_mean_tokens': base_b[b]['mean_tokens'], 'candidate_mean_tokens': cand_b[b]['mean_tokens'],
            'run_worst_case_se': 50 / math.sqrt(n)}
    ci = interval(overall)
    pooled = [d for b in names for d in diffs[b]]
    wins, losses = sum(d > 0 for d in pooled), sum(d < 0 for d in pooled)
    sign_p = sign_test(wins, losses)
    return {'baseline': side(baseline, base), 'candidate': side(candidate, cand),
            'model_id': base.get('model_id'), 'config_id': base.get('config_id'),
            'validation_sha256': base.get('validation_sha256'),
            **({'protocol': base['protocol'], 'suite_sha256': base['suite_sha256']} if public else {}),
            'generation_differs': None if public else base.get('generation') != cand.get('generation'),
            'benchmarks': benchmarks,
            'overall': {'baseline_mean': statistics.mean(v['baseline_mean'] for v in benchmarks.values()),
                        'candidate_mean': statistics.mean(v['candidate_mean'] for v in benchmarks.values()),
                        'diff': statistics.mean(v['diff'] for v in benchmarks.values()), 'ci95': ci,
                        'run_worst_case_se': 100 * math.sqrt(sum(.25 / v['n'] for v in benchmarks.values())) / len(names),
                        'wins': wins, 'losses': losses, 'ties': len(pooled) - wins - losses, 'sign_test_p': sign_p,
                        'separable': (ci[0] > 0 or ci[1] < 0) and sign_p < .05},
            'resamples': resamples, 'seed': seed,
            'note': NOTE + (' Public reports identify the declared checkpoint and pinned suite, but do not '
                            'attest endpoint weights or matching serving runtime. Token/truncation metrics '
                            'are unavailable. Intervals resample paired items within each benchmark and '
                            'do not measure run-to-run generation variability.' if public else '')}


def render(result):
    lines = []
    for name, v in result['benchmarks'].items():
        truncation = ('trunc n/a' if v['baseline_truncation_rate'] is None or v['candidate_truncation_rate'] is None
                      else f"trunc {v['baseline_truncation_rate']:.2f}->{v['candidate_truncation_rate']:.2f}")
        lines.append(f"{name} n={v['n']} {v['baseline_mean']:.1f} {v['candidate_mean']:.1f} {v['diff']:+.1f} "
                     f"[{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}] p={v['sign_test_p']:.3f} "
                     f"{truncation}")
    o = result['overall']
    lines.append(f"overall {o['baseline_mean']:.1f} {o['candidate_mean']:.1f} {o['diff']:+.1f} [{o['ci95'][0]:+.1f}, {o['ci95'][1]:+.1f}]")
    se = o['run_worst_case_se']
    spread = (f"Worst-case standard error of one run's mean: {se:.2f} points; "
              f"two runs separate at about 2.8\u00d7 that ({1.96 * math.sqrt(2) * se:.2f} points).")
    lines.append(f"Candidate is separable from baseline (95% CI excludes 0; sign test p={o['sign_test_p']:.3f}). {spread}"
                 if o['separable'] else f'Not separable at this sample size. {spread}')
    if result['generation_differs']:
        lines.append('Generation settings differ between runs; the comparison includes decoding effects.')
    if result['generation_differs'] is None:
        lines.append(result['note'])
    return '\n'.join(lines)


def compare(baseline: Path, candidate: Path, resamples: int = typer.Option(10000, min=1), seed: int = 0,
            output: Path | None = None, json_output: bool = typer.Option(False, '--json')):
    """Compare legacy local reports or public-evaluator result.json files with paired statistics."""
    try:
        output = output and output.resolve()
        if output and output.exists():
            raise ValueError('Output already exists. Choose a new --output path to preserve previous results.')
        result = compare_reports(baseline, candidate, resamples, seed)
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open('x') as stream:
                json.dump(result, stream, indent=2)
        typer.echo(json.dumps(result, indent=2) if json_output else render(result))
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        typer.echo(f'Compare failed: {exc}', err=True)
        raise typer.Exit(1)
