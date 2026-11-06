import copy
import hashlib
import json
from pathlib import Path
import sys
import tomllib

import pytest
from typer.testing import CliRunner

from delta_cli.main import app
from delta_cli.public_eval import coding, runner, text, tools
from delta_cli.public_eval.suite import PROTOCOL, expected_items, manifest, score_public

CONTENT = 'a' * 64


def public_result():
    return {'protocol': PROTOCOL, 'suite_sha256': manifest()['sha256'], 'content_hash': CONTENT,
            'reference_check': False, 'groups': {name: [{'id': i, 'score': 1.0} for i in sorted(ids)]
                                               for name, ids in expected_items(manifest()).items()}}


def test_aggregation_matches_production_contract():
    result = public_result()
    for row in result['groups']['humaneval']:
        row['score'] = 0
    total, groups = score_public(result, CONTENT)
    assert len(groups) == 18
    assert total == pytest.approx(100 * 17 / 18)
    for change in ('missing', 'duplicate', 'nan', 'reference', 'hash', 'suite', 'protocol', 'boolean'):
        bad = copy.deepcopy(result)
        rows = bad['groups']['humaneval']
        if change == 'missing':
            rows.pop()
        if change == 'duplicate':
            rows[0] = rows[1]
        if change == 'nan':
            rows[0]['score'] = float('nan')
        if change == 'boolean':
            rows[0]['score'] = True
        if change == 'reference':
            bad['reference_check'] = True
        if change == 'hash':
            bad['content_hash'] = 'b' * 64
        if change == 'suite':
            bad['suite_sha256'] = 'b' * 64
        if change == 'protocol':
            bad['protocol'] = 'unknown'
        with pytest.raises(ValueError):
            score_public(bad, CONTENT)


@pytest.fixture
def adapters(monkeypatch):
    monkeypatch.setattr(runner, 'check_versions', lambda: None)
    monkeypatch.setattr(runner, 'version', lambda name: manifest()['versions'][name])
    calls = []

    def text_run(suite, group, endpoint, model, tokenizer, output, **kwargs):
        calls.append(group)
        metric, selected_filter = runner.PRIMARY[group]
        samples = {name: [{'doc_id': i, 'filter': selected_filter, metric: float(i % 2),
                          'doc': 'omitted', 'arguments': 'omitted'} for i in spec['indices']]
                   for name, spec in suite['text'][group]['selection'].items()}
        return {'samples': samples}

    def code_run(suite, group, endpoint, model, image, output, **kwargs):
        calls.append(group)
        assert kwargs['sandbox_backend'] == 'wasi'
        return {'samples': [{'id': row['task_id'], 'passed': True} for row in suite['code'][group]['docs']]}

    def tools_run(suite, endpoint, model, tokenizer, output, **kwargs):
        calls.append('bfcl')
        directory = output.with_suffix('')
        directory.mkdir()
        for category, spec in suite['bfcl'].items():
            entries = [{'correct_count': spec['count'] - 1}, {'id': spec['ids'][0]}]
            (directory / f'BFCL_v4_{category}_score.json').write_text('\n'.join(map(json.dumps, entries)))
            (directory / f'{category}.responses.json').write_text('[]')
        return {'scores': {}}

    monkeypatch.setattr(text, 'run', text_run)
    monkeypatch.setattr(coding, 'run', code_run)
    monkeypatch.setattr(tools, 'run', tools_run)
    return calls


def evaluate(output, **kwargs):
    return runner.evaluate_public(endpoint='http://localhost:8000/v1', model='candidate', tokenizer='model',
                                  content_hash=CONTENT, output=output, **kwargs)


def test_compare_consumes_completed_public_evaluator_reports(tmp_path, adapters):
    baseline = evaluate(tmp_path / 'baseline')
    evaluate(tmp_path / 'candidate')
    result = CliRunner().invoke(app, ['compare', str(tmp_path / 'baseline/result.json'),
        str(tmp_path / 'candidate/result.json'), '--resamples', '20', '--json'])
    assert result.exit_code == 0, result.output
    comparison = json.loads(result.stdout)
    assert comparison['overall']['baseline_mean'] == baseline['validation_score']
    assert comparison['overall']['ci95'] == [0, 0]
    assert comparison['overall']['ties'] == 1110


def test_complete_report_preserves_production_metric_selection(tmp_path, adapters):
    progress = []
    result = evaluate(tmp_path / 'result', progress=lambda done, total: progress.append((done, total)))
    suite = manifest()
    assert adapters == [*suite['text'], *suite['code'], 'bfcl']
    expected = public_result()
    for name, spec in suite['text'].items():
        expected['groups'][name] = [{'id': f'{task}:{i}', 'score': float(i % 2)}
                                   for task, selection in spec['selection'].items() for i in selection['indices']]
    failed = {spec['ids'][0] for spec in suite['bfcl'].values()}
    for name in ('bfcl_single_turn', 'bfcl_multi_turn'):
        for row in expected['groups'][name]:
            row['score'] = float(row['id'] not in failed)
    assert result['validation_score'] == score_public(expected, CONTENT)[0]
    assert result['validation_benchmarks'] == score_public(expected, CONTENT)[1]
    assert progress[-1] == (1110, 1110)
    state = json.loads((tmp_path / 'result/run.json').read_text())
    assert state['status'] == 'complete' and state['local_only']
    assert 'generations' not in result
    assert json.loads((tmp_path / 'result/result.json').read_text()) == result
    assert all('doc' not in r and 'arguments' not in r
               for rows in result['public_evidence']['gsm8k'].values() for r in rows)
    with pytest.raises(FileExistsError):
        evaluate(tmp_path / 'result')


@pytest.mark.parametrize('failure', ['infrastructure', 'incomplete', 'interrupted'])
def test_failures_never_publish_complete_results(tmp_path, adapters, monkeypatch, failure):
    def fail(*args, **kwargs):
        if failure == 'incomplete':
            return {'samples': []}
        if failure == 'interrupted':
            raise KeyboardInterrupt()
        raise RuntimeError('https://user:secret@endpoint.invalid')
    monkeypatch.setattr(coding, 'run', fail)
    with pytest.raises((RuntimeError, ValueError, KeyboardInterrupt)):
        evaluate(tmp_path / 'result')
    state = (tmp_path / 'result/run.json').read_text()
    assert json.loads(state)['status'] == 'failed'
    assert 'secret' not in state
    assert not (tmp_path / 'result/result.json').exists()


def test_manifest_identity_rejected_before_inference(tmp_path, monkeypatch):
    suite = manifest()
    suite['seed'] += 1
    from delta_cli.public_eval.common import digest
    suite['sha256'] = digest({k: v for k, v in suite.items() if k != 'sha256'})
    changed = tmp_path / 'validation.json'
    changed.write_text(json.dumps(suite))
    monkeypatch.setattr(runner, 'check_versions', lambda: pytest.fail('Must reject before loading dependencies'))
    with pytest.raises(ValueError, match='Unsupported public suite'):
        evaluate(tmp_path / 'result', manifest_path=changed)
    assert not (tmp_path / 'result').exists()


def test_config_is_available_without_optional_benchmark_libraries(monkeypatch):
    monkeypatch.setitem(sys.modules, 'lm_eval', None)
    monkeypatch.setitem(sys.modules, 'bfcl_eval', None)
    result = CliRunner().invoke(app, ['public-eval', 'config'])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert sum(data['benchmark_items'].values()) == 1110
    assert data['protocol'] == PROTOCOL
    assert data['runtime']['coding']['sandbox'] == 'cpython-3.12.10-wasi-wasmtime-50.0.1'


def workspace(root):
    (root / 'model').mkdir()
    (root / 'delta-workspace.json').write_text(json.dumps({'config_id': 'public', 'model_id': 'tiny',
                                                         'parent_version_id': 'parent'}))
    (root / 'model/delta-model.json').write_text(json.dumps({'id': 'parent', 'model_id': 'tiny'}))
    raw = json.dumps(manifest()).encode()
    (root / 'validation.json').write_bytes(raw)
    (root / 'evaluation.json').write_text(json.dumps({'id': 'public', 'model_id': 'tiny',
        'protocol_version': PROTOCOL, 'validation_sha256': hashlib.sha256(raw).hexdigest()}))


def test_evaluate_routes_current_downloaded_manifest(tmp_path, adapters):
    workspace(tmp_path)
    result = CliRunner().invoke(app, ['evaluate', '--workspace', str(tmp_path),
        '--endpoint', 'http://localhost:8000/v1', '--content-hash', CONTENT, '--output', str(tmp_path / 'results')])
    assert result.exit_code == 0, result.output
    assert (tmp_path / 'results/result.json').exists()


def test_evaluate_explains_current_protocol_without_inventory_error(tmp_path):
    workspace(tmp_path)
    result = CliRunner().invoke(app, ['evaluate', '--workspace', str(tmp_path)])
    assert result.exit_code == 1
    assert 'delta public-eval run' in result.output
    assert 'inventory mismatch' not in result.output


def test_runtime_dependency_pins_stay_in_sync():
    root = Path(__file__).parents[1]
    project = tomllib.loads((root / 'pyproject.toml').read_text())
    extra = set(project['project']['optional-dependencies']['public-eval'])
    requirements = {line for line in (root / 'delta_cli/public_eval/requirements.txt').read_text().splitlines()
                    if line and not line.startswith('#')}
    assert requirements | {'torch==2.10.0'} == extra


@pytest.mark.parametrize('extra', [['--limit', '1'], ['--thinking'], ['--benchmark', 'gsm8k']])
def test_pinned_evaluation_rejects_protocol_overrides(tmp_path, monkeypatch, extra):
    workspace(tmp_path)
    monkeypatch.setattr(runner, 'evaluate_public', lambda **kwargs: pytest.fail('Must reject before inference'))
    result = CliRunner().invoke(app, ['evaluate', '--workspace', str(tmp_path),
        '--endpoint', 'http://localhost:8000/v1', '--content-hash', CONTENT, *extra])
    assert result.exit_code == 1
    assert 'complete frozen suite' in result.output


def test_downloaded_manifest_hash_checked_before_dispatch(tmp_path, monkeypatch):
    workspace(tmp_path)
    (tmp_path / 'validation.json').write_text('{}')
    monkeypatch.setattr(runner, 'evaluate_public', lambda **kwargs: pytest.fail('Must reject before inference'))
    result = CliRunner().invoke(app, ['evaluate', '--workspace', str(tmp_path),
        '--endpoint', 'http://localhost:8000/v1', '--content-hash', CONTENT])
    assert result.exit_code == 1
    assert 'Public validation data changed' in result.output
