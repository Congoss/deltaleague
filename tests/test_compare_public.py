import json

import pytest
from typer.testing import CliRunner

from delta_cli import compare, main
from delta_cli.public_eval.suite import PROTOCOL, expected_items, manifest


def write_public(root, name, *, content='a', gain=None, reverse=False):
    suite = manifest()
    groups = {name: [{'id': item, 'score': int(name == gain)} for item in sorted(ids)]
              for name, ids in expected_items(suite).items()}
    if reverse:
        groups = {name: list(reversed(rows)) for name, rows in reversed(list(groups.items()))}
    path = root / name
    path.mkdir()
    result = path / 'result.json'
    result.write_text(json.dumps({'public_validation': {
        'protocol': PROTOCOL, 'suite_sha256': suite['sha256'], 'content_hash': content * 64,
        'reference_check': False, 'groups': groups}, 'validation_score': -123}))
    return result


def test_public_pairing_weighting_and_checkpoint_identity(tmp_path):
    groups = expected_items(manifest())
    smallest = min(groups, key=lambda name: len(groups[name]))
    base = write_public(tmp_path, 'base')
    cand = write_public(tmp_path, 'candidate', content='b', gain=smallest, reverse=True)
    result = compare.compare_reports(base, cand, resamples=100, seed=7)
    assert sum(v['n'] for v in result['benchmarks'].values()) == 1110
    assert len(result['benchmarks']) == 18
    assert result['overall']['diff'] == pytest.approx(100 / 18)
    assert result['overall']['diff'] != pytest.approx(100 * len(groups[smallest]) / 1110)
    assert result['overall']['ci95'] == pytest.approx([100 / 18, 100 / 18])
    assert result['overall']['wins'] == len(groups[smallest])
    assert result['baseline']['validation_score'] == 0  # Ignore stale summary scores.
    assert result['candidate']['validation_score'] == pytest.approx(100 / 18)
    assert result['baseline']['content_hash'] == 'a' * 64
    assert result['candidate']['content_hash'] == 'b' * 64
    assert result['suite_sha256'] == manifest()['sha256']
    assert result['generation_differs'] is None
    assert all(v['baseline_mean_tokens'] is None for v in result['benchmarks'].values())
    assert 'trunc n/a' in compare.render(result)
    assert 'matching serving runtime' in compare.render(result)
    same = write_public(tmp_path, 'same', gain=smallest)
    identical = compare.compare_reports(cand, same, resamples=50)
    assert identical['overall']['ci95'] == [0, 0]
    assert identical['overall']['ties'] == 1110


@pytest.mark.parametrize('problem', [
    'protocol', 'suite', 'reference', 'hash', 'missing_group', 'missing_item',
    'duplicate', 'unknown_item', 'nan', 'infinity', 'boolean', 'range', 'shape', 'huge_integer',
])
def test_invalid_public_contract_fails_before_comparison(tmp_path, problem):
    base = write_public(tmp_path, 'base')
    cand = write_public(tmp_path, 'candidate')
    report = json.loads(cand.read_text())
    public = report['public_validation']
    group = next(iter(public['groups']))
    rows = public['groups'][group]
    if problem == 'protocol': public['protocol'] = 'other'
    elif problem == 'suite': public['suite_sha256'] = 'b' * 64
    elif problem == 'reference': public['reference_check'] = True
    elif problem == 'hash': public['content_hash'] = 'not-a-hash'
    elif problem == 'missing_group': del public['groups'][group]
    elif problem == 'missing_item': rows.pop()
    elif problem == 'duplicate': rows[-1] = rows[0]
    elif problem == 'unknown_item': rows[0]['id'] = 'unknown'
    elif problem == 'shape': report['public_validation'] = []
    elif problem == 'huge_integer': rows[0]['score'] = 10 ** 1000
    else: rows[0]['score'] = {'nan': float('nan'), 'infinity': float('inf'), 'boolean': True, 'range': 1.1}[problem]
    cand.write_text(json.dumps(report))
    output = tmp_path / 'comparison.json'
    result = CliRunner().invoke(main.app, ['compare', str(base), str(cand), '--output', str(output)])
    assert result.exit_code == 1
    assert 'Compare failed' in result.output
    assert not output.exists()


def test_public_cli_output_and_legacy_mixing(tmp_path):
    base = write_public(tmp_path, 'base')
    cand = write_public(tmp_path, 'candidate', content='b')
    runner = CliRunner()
    output = tmp_path / 'comparison.json'
    args = ['compare', str(base), str(cand), '--resamples', '20', '--json', '--output', str(output)]
    result = runner.invoke(main.app, args)
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == json.loads(output.read_text())
    assert runner.invoke(main.app, args).exit_code == 1
    text = runner.invoke(main.app, ['compare', str(base), str(cand), '--resamples', '20'])
    assert text.exit_code == 0 and 'trunc n/a' in text.output
    legacy = tmp_path / 'legacy.json'
    legacy.write_text('{}')
    legacy.with_suffix('.generations.jsonl').write_text(json.dumps({
        'benchmark': 'test', 'example_index': 0, 'score': 0, 'truncated': False, 'tokens': 1}))
    for a, b in [(base, legacy), (legacy, base)]:
        result = runner.invoke(main.app, ['compare', str(a), str(b)])
        assert result.exit_code == 1 and 'Cannot compare public-evaluator and legacy' in result.output
