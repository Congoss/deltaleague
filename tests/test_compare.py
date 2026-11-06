import json
import math
import time

import pytest
from typer.testing import CliRunner

from delta_cli import compare, main

GENERATION = {'max_tokens': 20, 'temperature': .6}


def write_run(root, name, scores, generation=None, config_id='tiny-v1', model_id='tiny', tokens=5):
    """scores: {benchmark: [[sample scores] per example]}"""
    rows = []
    for bench, examples in scores.items():
        for index, samples in enumerate(examples):
            for sample, score in enumerate(samples):
                rows.append({'benchmark': bench, 'example_index': index, 'sample': sample, 'seed': 1000 + sample,
                             'example_id': f'{bench}-{index}', 'score': score, 'text': '', 'tokens': tokens,
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


def test_sign_test_matches_the_binomial_tail_and_stays_linear():
    def slow(wins, losses):
        n = wins + losses
        if not n:
            return 1.0
        tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1))
        return min(1.0, 2 * tail / 2 ** n)

    for wins, losses in [(0, 0), (0, 5), (3, 3), (1, 20), (20, 1), (8, 7)]:
        assert compare.sign_test(wins, losses) == slow(wins, losses)
    started = time.perf_counter()
    assert compare.sign_test(5000, 5000) == 1.0
    assert time.perf_counter() - started < 1


def test_control_characters_in_a_benchmark_name_are_rejected(tmp_path):
    report = write_run(tmp_path, 'x', {'a\nsep': [[1]]})
    with pytest.raises(ValueError, match='control characters'):
        compare.load(report)


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


def test_consistent_regression_is_separable(tmp_path):
    base = {b: [[1]] * 40 for b in 'abc'}
    cand = {b: [[0]] * 30 + [[1]] * 10 for b in 'abc'}
    result = compare.compare_reports(write_run(tmp_path, 'x', base), write_run(tmp_path, 'y', cand), seed=3)
    o = result['overall']
    assert o['diff'] < 0 and o['ci95'][1] < 0
    assert o['losses'] == 90 and o['wins'] == 0 and o['separable']
    assert 'Candidate is separable' in compare.render(result)


def test_render_not_separable_and_generation_warning(tmp_path):
    base = write_run(tmp_path, 'x', flat(4, 1))
    cand = write_run(tmp_path, 'y', flat(4, 1), generation={**GENERATION, 'max_tokens': 64})
    text = compare.render(compare.compare_reports(base, cand, resamples=50))
    assert 'Not separable at this sample size' in text
    assert 'Generation settings differ' in text


def test_different_benchmark_names_are_refused(tmp_path):
    with pytest.raises(ValueError, match='example'):
        compare.compare_reports(write_run(tmp_path, 'x', {'a': [[1]]}), write_run(tmp_path, 'y', {'b': [[1]]}))


def test_malformed_inputs_fail_cleanly(tmp_path):
    base = write_run(tmp_path, 'x', flat(3, 1))
    runner = CliRunner()
    bad = tmp_path / 'bad.json'
    bad.write_text('{not json')
    result = runner.invoke(main.app, ['compare', str(base), str(bad)])
    assert result.exit_code == 1 and 'Compare failed' in result.output
    cand = write_run(tmp_path, 'z', flat(3, 1))
    (tmp_path / 'z.generations.jsonl').write_text(json.dumps({'benchmark': 'a', 'example_index': 0, 'sample': 0}) + '\n')
    result = runner.invoke(main.app, ['compare', str(base), str(cand)])
    assert result.exit_code == 1 and 'Compare failed' in result.output


def test_collapsed_interval_without_sign_evidence_is_not_separable(tmp_path):
    one = compare.compare_reports(write_run(tmp_path, 'x', {'a': [[0]]}), write_run(tmp_path, 'y', {'a': [[1]]}), resamples=50)
    assert one['overall']['ci95'] == [100, 100]
    assert one['overall']['sign_test_p'] == 1.0 and not one['overall']['separable']
    two = compare.compare_reports(write_run(tmp_path, 'p', {'a': [[0], [0]]}), write_run(tmp_path, 'q', {'a': [[1], [1]]}), resamples=50)
    assert two['overall']['sign_test_p'] == 0.5 and not two['overall']['separable']
    assert two['overall']['wins'] == 2 and two['overall']['losses'] == 0 and two['overall']['ties'] == 0


def test_duplicate_example_ids_stay_separate_examples(tmp_path):
    base = write_run(tmp_path, 'x', {'a': [[0], [1]]})
    rows = [json.loads(line) for line in (tmp_path / 'x.generations.jsonl').read_text().splitlines()]
    (tmp_path / 'x.generations.jsonl').write_text(''.join(json.dumps({**r, 'example_id': 'dup'}) + '\n' for r in rows))
    assert compare.load(base)[1]['a']['examples'] == {0: 0, 1: 1}
    result = compare.compare_reports(base, write_run(tmp_path, 'y', {'a': [[1], [1]]}), resamples=10)
    assert result['benchmarks']['a']['n'] == 2


def test_mismatched_inputs_are_refused(tmp_path):
    base = write_run(tmp_path, 'x', {'a': [[1], [0]]})
    with pytest.raises(ValueError, match='example'):
        compare.compare_reports(base, write_run(tmp_path, 'y', {'a': [[1], [0], [1]]}))
    with pytest.raises(ValueError, match='configuration'):
        compare.compare_reports(base, write_run(tmp_path, 'z', {'a': [[1], [0]]}, config_id='other'))


def test_cli_json_and_output_never_overwrites(tmp_path):
    base = write_run(tmp_path, 'x', flat(6, 0))
    cand = write_run(tmp_path, 'y', flat(6, 1))
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
    assert result['benchmarks']['a']['run_worst_case_se'] == pytest.approx(15.81, abs=.005)
    assert result['overall']['run_worst_case_se'] == pytest.approx(9.13, abs=.005)
    tied = compare.compare_reports(write_run(tmp_path, 'p', flat(10, 1, 'abc')), write_run(tmp_path, 'q', flat(10, 1, 'abc')), resamples=50)
    text = compare.render(tied)
    assert "Worst-case standard error of one run's mean: 9.13 points" in text
    assert 'independence-based unpaired 95% band: 25.30 points' in text
    assert 'not a threshold for this paired comparison' in text


def test_generation_differs_flag(tmp_path):
    base = write_run(tmp_path, 'x', flat(3, 1))
    same = write_run(tmp_path, 'y', flat(3, 1))
    other = write_run(tmp_path, 'z', flat(3, 1), generation={**GENERATION, 'max_tokens': 64})
    assert not compare.compare_reports(base, same, resamples=10)['generation_differs']
    assert compare.compare_reports(base, other, resamples=10)['generation_differs']


def test_sign_test_exact():
    assert compare.sign_test(9, 1) == pytest.approx(22 / 1024)
    assert compare.sign_test(5, 5) == 1.0
    assert compare.sign_test(0, 0) == 1.0
    assert compare.sign_test(1, 9) == compare.sign_test(9, 1)


def test_interval_percentiles():
    assert compare.interval(list(range(1000))) == [25, 975]
    assert compare.interval([7]) == [7, 7]


def test_overall_is_mean_of_benchmark_means(tmp_path):
    base = write_run(tmp_path, 'x', {'a': [[0]], 'b': [[0]] * 9})
    cand = write_run(tmp_path, 'y', {'a': [[1]], 'b': [[0]] * 9})
    overall = compare.compare_reports(base, cand, resamples=50)['overall']
    assert overall['diff'] == pytest.approx(50)
    assert overall['baseline_mean'] == pytest.approx(0)
    assert overall['candidate_mean'] == pytest.approx(50)


def test_opposite_pooled_signs_do_not_certify_macro_gain(tmp_path):
    base = {'small_a': [[0]] * 10, 'small_b': [[0]] * 10, 'large': [[1]] * 100}
    cand = {'small_a': [[1]] * 10, 'small_b': [[1]] * 10, 'large': [[0]] * 100}
    result = compare.compare_reports(write_run(tmp_path, 'x', base), write_run(tmp_path, 'y', cand), resamples=50)
    overall = result['overall']
    assert overall['diff'] > 0 and overall['ci95'][0] > 0
    assert overall['sign_test_p'] < .05 and overall['losses'] > overall['wins']
    assert not overall['pooled_sign_agrees'] and not overall['separable']


@pytest.mark.parametrize('resamples', [0, -1, True, 1.5])
def test_invalid_resample_count_fails_before_reading_reports(resamples):
    with pytest.raises(ValueError, match='positive integer'):
        compare.compare_reports('unused', 'unused', resamples=resamples)


def test_mismatched_model_id_and_validation_hash_are_refused(tmp_path):
    base = write_run(tmp_path, 'x', flat(2, 1))
    with pytest.raises(ValueError, match='model_id'):
        compare.compare_reports(base, write_run(tmp_path, 'y', flat(2, 1), model_id='other'))
    other = write_run(tmp_path, 'z', flat(2, 1))
    report = json.loads(other.read_text())
    other.write_text(json.dumps({**report, 'validation_sha256': 'b' * 64}))
    with pytest.raises(ValueError, match='configuration'):
        compare.compare_reports(base, other)


def test_generations_file_missing_and_cli_errors(tmp_path):
    base = write_run(tmp_path, 'x', flat(3, 1))
    cand = write_run(tmp_path, 'y', flat(3, 0))
    expected = compare.compare_reports(base, cand, resamples=50)
    generations = tmp_path / 'y.generations.jsonl'
    lines = generations.read_text().splitlines(keepends=True)
    generations.write_text(lines[0] + '\n' + ''.join(lines[1:]))
    assert compare.compare_reports(base, cand, resamples=50) == expected
    generations.unlink()
    with pytest.raises(ValueError, match='Generations file not found'):
        compare.compare_reports(base, cand)
    result = CliRunner().invoke(main.app, ['compare', str(base), str(tmp_path / 'missing.json')])
    assert result.exit_code == 1
    assert 'Compare failed' in result.output


def test_malformed_inputs_fail_with_a_sentence(tmp_path):
    base = write_run(tmp_path, 'x', flat(3, 1))
    cand = write_run(tmp_path, 'y', flat(3, 1))
    runner = CliRunner()
    cand.write_text('[]')
    result = runner.invoke(main.app, ['compare', str(base), str(cand)])
    assert result.exit_code == 1 and 'Report must be' in result.output
    cand = write_run(tmp_path, 'z', flat(3, 1))
    (tmp_path / 'z.generations.jsonl').write_text('')
    result = runner.invoke(main.app, ['compare', str(base), str(cand)])
    assert result.exit_code == 1 and 'no rows' in result.output


def test_mean_tokens_reported(tmp_path):
    result = compare.compare_reports(write_run(tmp_path, 'x', flat(3, 1), tokens=5), write_run(tmp_path, 'y', flat(3, 1), tokens=9), resamples=10)
    assert result['benchmarks']['a']['baseline_mean_tokens'] == 5
    assert result['benchmarks']['a']['candidate_mean_tokens'] == 9


@pytest.fixture
def validation_data():
    return {'a': [{'id': 'a1', 'prompt': 'A', 'answer': 'yes', 'samples': 2}, {'id': 'a2', 'prompt': 'B', 'answer': 'yes'}]}


def test_round_trip_from_evaluate_reports(workspace, fake_runner):
    root, metadata, _ = workspace
    runner = CliRunner()
    for name, text in (('base', 'no'), ('edit', 'yes')):
        fake_runner(lambda generation: text)
        result = runner.invoke(main.app, ['evaluate', '--workspace', str(root), '--output', str(root / f'{name}.json')])
        assert result.exit_code == 0, result.output
    result = runner.invoke(main.app, ['compare', str(root / 'base.json'), str(root / 'edit.json'), '--resamples', '200', '--json'])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report['benchmarks']['a']['n'] == 2 and report['benchmarks']['a']['diff'] == 100
    assert report['benchmarks']['a']['baseline_truncation_rate'] == 1 and report['benchmarks']['a']['candidate_truncation_rate'] == 0
    assert report['baseline']['source_version_id'] == metadata['id'] and not report['generation_differs']
