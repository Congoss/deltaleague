"""Paired comparison of two local evaluation reports."""
import json
import math
import random
import statistics
from pathlib import Path

import typer

NOTE = 'Advisory local comparison. Official validation and private test scores are authoritative.'
SUMMARY = ('path', 'source_version_id', 'artifact', 'generation', 'validation_score')


def load(path):
    path = Path(path)
    report = json.loads(path.read_text())
    candidates = [Path(report['generations_file'])] if report.get('generations_file') else []
    candidates.append(path.with_suffix('.generations.jsonl'))
    rows = next((p for p in candidates if p.exists()), None)
    if rows is None:
        raise ValueError(f'Generations file not found: {candidates[-1]}')
    scores, truncated, tokens = {}, {}, {}
    for line in rows.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            name = row['benchmark']
            scores.setdefault(name, {}).setdefault(row['example_id'], []).append(row['score'])
            truncated.setdefault(name, []).append(bool(row['truncated']))
            tokens.setdefault(name, []).append(row['tokens'])
    examples = {name: {k: statistics.mean(v) for k, v in items.items()} for name, items in scores.items()}
    return report, examples, truncated, tokens


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
    (base, base_scores, base_trunc, base_tokens), (cand, cand_scores, cand_trunc, cand_tokens) = load(baseline), load(candidate)
    if base.get('config_id') != cand.get('config_id') or base.get('validation_sha256') != cand.get('validation_sha256'):
        raise ValueError('Reports use a different evaluation configuration.')
    if base.get('model_id') != cand.get('model_id'):
        raise ValueError('Reports belong to a different model_id.')
    if base_scores.keys() != cand_scores.keys() or any(base_scores[b].keys() != cand_scores[b].keys() for b in base_scores):
        raise ValueError('Benchmark or example sets differ; rerun both with the same --limit and --benchmark.')
    names = sorted(base_scores)
    diffs = {b: [(cand_scores[b][k] - base_scores[b][k]) * 100 for k in sorted(base_scores[b])] for b in names}
    rng = random.Random(seed)
    boot = {b: [] for b in names}
    overall = []
    for _ in range(resamples):
        means = [statistics.mean(rng.choices(diffs[b], k=len(diffs[b]))) for b in names]
        for b, mean in zip(names, means):
            boot[b].append(mean)
        overall.append(statistics.mean(means))
    benchmarks = {}
    for b in names:
        wins, losses = sum(d > 0 for d in diffs[b]), sum(d < 0 for d in diffs[b])
        n = len(diffs[b])
        benchmarks[b] = {
            'n': n, 'baseline_mean': statistics.mean(base_scores[b].values()) * 100,
            'candidate_mean': statistics.mean(cand_scores[b].values()) * 100,
            'diff': statistics.mean(diffs[b]), 'ci95': interval(boot[b]),
            'wins': wins, 'losses': losses, 'ties': n - wins - losses, 'sign_test_p': sign_test(wins, losses),
            'baseline_truncation_rate': statistics.mean(base_trunc[b]), 'candidate_truncation_rate': statistics.mean(cand_trunc[b]),
            'baseline_mean_tokens': statistics.mean(base_tokens[b]), 'candidate_mean_tokens': statistics.mean(cand_tokens[b]),
            'worst_case_se': 50 / math.sqrt(n)}
    ci = interval(overall)
    return {'baseline': {'path': str(baseline), **{k: base.get(k) for k in SUMMARY[1:]}},
            'candidate': {'path': str(candidate), **{k: cand.get(k) for k in SUMMARY[1:]}},
            'model_id': base.get('model_id'), 'config_id': base.get('config_id'),
            'validation_sha256': base.get('validation_sha256'),
            'generation_differs': base.get('generation') != cand.get('generation'),
            'benchmarks': benchmarks,
            'overall': {'baseline_mean': statistics.mean(v['baseline_mean'] for v in benchmarks.values()),
                        'candidate_mean': statistics.mean(v['candidate_mean'] for v in benchmarks.values()),
                        'diff': statistics.mean(v['diff'] for v in benchmarks.values()), 'ci95': ci,
                        'worst_case_se': 100 * math.sqrt(sum(.25 / v['n'] for v in benchmarks.values())) / len(names),
                        'separable': ci[0] > 0 or ci[1] < 0},
            'resamples': resamples, 'seed': seed, 'note': NOTE}


def render(result):
    lines = []
    for name, v in result['benchmarks'].items():
        lines.append(f"{name} n={v['n']} {v['baseline_mean']:.1f} {v['candidate_mean']:.1f} {v['diff']:+.1f} "
                     f"[{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}] p={v['sign_test_p']:.3f} "
                     f"trunc {v['baseline_truncation_rate']:.2f}->{v['candidate_truncation_rate']:.2f}")
    o = result['overall']
    lines.append(f"overall {o['baseline_mean']:.1f} {o['candidate_mean']:.1f} {o['diff']:+.1f} [{o['ci95'][0]:+.1f}, {o['ci95'][1]:+.1f}]")
    lines.append('Candidate is separable from baseline (95% CI excludes 0).' if o['separable'] else
                 f"Not separable at this sample size: worst-case standard error {o['worst_case_se']:.2f} points overall.")
    if result['generation_differs']:
        lines.append('Generation settings differ between runs; the comparison includes decoding effects.')
    return '\n'.join(lines)


def compare(baseline: Path, candidate: Path, resamples: int = typer.Option(10000, min=1), seed: int = 0,
            output: Path | None = None, json_output: bool = typer.Option(False, '--json')):
    """Compare two local evaluation reports with paired statistics; advisory, not a leaderboard score."""
    try:
        if output and output.exists():
            raise ValueError('Output already exists. Choose a new --output path to preserve previous results.')
        result = compare_reports(baseline, candidate, resamples, seed)
        if output:
            with output.open('x') as stream:
                json.dump(result, stream, indent=2)
        typer.echo(json.dumps(result, indent=2) if json_output else render(result))
    except (OSError, ValueError, KeyError) as exc:
        typer.echo(f'Compare failed: {exc}', err=True)
        raise typer.Exit(1)
