import json

import pytest
import torch
from safetensors.torch import save_file
from typer.testing import CliRunner

from delta_cli import compare, local, main
from delta_cli.artifacts import file_hash

GENERATION = {'max_tokens': 20, 'temperature': .6}


def write_run(root, name, scores, generation=None, config_id='tiny-v1', model_id='tiny'):
    """scores: {benchmark: [[sample scores] per example]}"""
    rows = []
    for bench, examples in scores.items():
        for index, samples in enumerate(examples):
            for sample, score in enumerate(samples):
                rows.append({'benchmark': bench, 'example_index': index, 'sample': sample, 'seed': 1000 + sample,
                             'example_id': f'{bench}-{index}', 'score': score, 'text': '', 'tokens': 5,
                             'truncated': False, 'generation_seconds': 1.0})
    generations = root / f'{name}.generations.jsonl'
    generations.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    report = root / f'{name}.json'
    report.write_text(json.dumps({'model_id': model_id, 'config_id': config_id, 'validation_sha256': 'a' * 64,
                                  'generation': generation or GENERATION, 'generations_file': str(generations),
                                  'source_version_id': 'v', 'artifact': {}, 'validation_score': 0}))
    return report


def flat(n, value, benchmarks=('a',)):
    return {b: [[value]] * n for b in benchmarks}


def test_identical_runs_are_not_separable(tmp_path):
    scores = {'a': [[1], [0], [1], [0]]}
    result = compare.compare_reports(write_run(tmp_path, 'x', scores), write_run(tmp_path, 'y', scores))
    assert result['overall']['diff'] == 0
    assert result['overall']['ci95'] == [0, 0]
    assert result['benchmarks']['a']['sign_test_p'] == 1.0
    assert not result['overall']['separable']


def test_samples_average_before_pairing(tmp_path):
    result = compare.compare_reports(write_run(tmp_path, 'x', {'a': [[1, 0]]}), write_run(tmp_path, 'y', {'a': [[1, 1]]}))
    assert result['benchmarks']['a']['diff'] == pytest.approx(50)
    assert result['benchmarks']['a']['n'] == 1


def test_consistent_gain_is_separable_and_seeded(tmp_path):
    base = {b: [[0]] * 40 for b in 'abc'}
    cand = {b: [[1]] * 30 + [[0]] * 10 for b in 'abc'}
    one = compare.compare_reports(write_run(tmp_path, 'x', base), write_run(tmp_path, 'y', cand), seed=3)
    two = compare.compare_reports(write_run(tmp_path, 'x', base), write_run(tmp_path, 'y', cand), seed=3)
    assert one['overall']['ci95'][0] > 0 and one['overall']['separable']
    assert one['benchmarks']['a']['wins'] == 30 and one['benchmarks']['a']['ties'] == 10
    assert one['overall']['ci95'] == two['overall']['ci95']


def test_mismatched_inputs_are_refused(tmp_path):
    base = write_run(tmp_path, 'x', {'a': [[1], [0]]})
    with pytest.raises(ValueError, match='example'):
        compare.compare_reports(base, write_run(tmp_path, 'y', {'a': [[1], [0], [1]]}))
    with pytest.raises(ValueError, match='configuration'):
        compare.compare_reports(base, write_run(tmp_path, 'z', {'a': [[1], [0]]}, config_id='other'))


def test_cli_json_and_output_never_overwrites(tmp_path):
    base = write_run(tmp_path, 'x', flat(5, 0))
    cand = write_run(tmp_path, 'y', flat(5, 1))
    runner = CliRunner()
    result = runner.invoke(main.app, ['compare', str(base), str(cand), '--resamples', '200', '--json'])
    assert result.exit_code == 0
    assert json.loads(result.stdout)['overall']['diff'] == 100
    out = tmp_path / 'compare.json'
    args = ['compare', str(base), str(cand), '--resamples', '200', '--output', str(out)]
    assert runner.invoke(main.app, args).exit_code == 0
    assert 'Candidate is separable' in runner.invoke(main.app, ['compare', str(base), str(cand)]).stdout
    again = runner.invoke(main.app, args)
    assert again.exit_code == 1
    assert 'already exists' in again.output


def test_worst_case_standard_errors(tmp_path):
    result = compare.compare_reports(write_run(tmp_path, 'x', flat(10, 0, 'abc')), write_run(tmp_path, 'y', flat(10, 1, 'abc')), resamples=50)
    assert result['benchmarks']['a']['worst_case_se'] == pytest.approx(15.81, abs=.005)
    assert result['overall']['worst_case_se'] == pytest.approx(9.13, abs=.005)


def test_generation_differs_flag(tmp_path):
    base = write_run(tmp_path, 'x', flat(3, 1))
    same = write_run(tmp_path, 'y', flat(3, 1))
    other = write_run(tmp_path, 'z', flat(3, 1), generation={**GENERATION, 'max_tokens': 64})
    assert not compare.compare_reports(base, same, resamples=10)['generation_differs']
    assert compare.compare_reports(base, other, resamples=10)['generation_differs']


def test_round_trip_from_evaluate_reports(tmp_path, monkeypatch):
    model = tmp_path / 'model'
    model.mkdir()
    save_file({'layer.weight': torch.tensor([1., 2.])}, str(model / 'model.safetensors'))
    (model / 'config.json').write_text('{}')
    metadata = {'id': '00000000-0000-0000-0000-000000000002', 'model_id': 'tiny', 'content_hash': 'a' * 64, 'files': [
        {'name': p.name, 'size_bytes': p.stat().st_size, 'sha256': file_hash(p)} for p in model.iterdir()]}
    (model / 'delta-model.json').write_text(json.dumps(metadata))
    data = {'a': [{'id': 'a1', 'prompt': 'A', 'answer': 'yes', 'samples': 2}, {'id': 'a2', 'prompt': 'B', 'answer': 'yes'}]}
    (tmp_path / 'validation.json').write_text(json.dumps(data))
    config = {'id': 'tiny-v1', 'model_id': 'tiny', 'generation': GENERATION, 'validation_names': ['a'],
              'validation_sha256': file_hash(tmp_path / 'validation.json')}
    (tmp_path / 'evaluation.json').write_text(json.dumps(config))
    (tmp_path / 'delta-workspace.json').write_text(json.dumps({'model_id': 'tiny', 'config_id': 'tiny-v1', 'parent_version_id': metadata['id']}))
    answers = {}
    class FakeRunner:
        details = {'device': 'cpu'}
        def __init__(self, *args):
            pass
        def generate(self, prompt, generation, seed):
            return {'text': answers['text'], 'tokens': 3, 'generation_seconds': .1, 'truncated': answers['text'] == 'no'}
    monkeypatch.setattr(local, 'Runner', FakeRunner)
    runner = CliRunner()
    for name, text in (('base', 'no'), ('edit', 'yes')):
        answers['text'] = text
        result = runner.invoke(main.app, ['evaluate', '--workspace', str(tmp_path), '--output', str(tmp_path / f'{name}.json')])
        assert result.exit_code == 0, result.output
    result = runner.invoke(main.app, ['compare', str(tmp_path / 'base.json'), str(tmp_path / 'edit.json'), '--resamples', '200', '--json'])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report['benchmarks']['a']['n'] == 2 and report['benchmarks']['a']['diff'] == 100
    assert report['benchmarks']['a']['baseline_truncation_rate'] == 1 and report['benchmarks']['a']['candidate_truncation_rate'] == 0
    assert report['baseline']['source_version_id'] == metadata['id'] and not report['generation_differs']
