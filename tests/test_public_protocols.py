import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from delta_cli.public_eval.common import allocate, digest, read_manifest, select
from delta_cli.public_eval.coding import extract_code, program
from delta_cli.public_eval.tools import safe_simulator_call

MANIFEST = Path(__file__).parents[1] / 'delta_cli/public_eval/data/qwen-small-public-v1.json'


def test_frozen_suite_covers_every_subject():
    suite = read_manifest(MANIFEST)
    assert len(suite['text']) == 14
    assert len(suite['text']['mmlu']['selection']) == 57
    assert len(suite['text']['bbh']['selection']) == 27
    assert sum(g['count'] for g in suite['text'].values()) == 942
    assert sum(g['count'] for g in suite['code'].values()) == 64
    assert sum(g['count'] for g in suite['bfcl'].values()) == 104
    for group in suite['text'].values():
        assert sum(len(x['indices']) for x in group['selection'].values()) == group['count']
        for task, selected in group['selection'].items():
            assert len(selected['indices']) == len(set(selected['indices'])) == len(selected['doc_sha256'])
    assert suite['scope'] == 'public-validation-only'
    assert 'test' not in suite


def test_sampling_is_deterministic_stratified_and_not_first_n():
    rows = [{'id': i} for i in range(100)]
    ids = select(rows, 10, 'test', excluded={2, 3, 4})
    assert ids == select(rows, 10, 'test', excluded={2, 3, 4})
    assert not set(ids) & {2, 3, 4}
    reversed_rows = rows[::-1]
    assert {digest(rows[i]) for i in ids} == {digest(reversed_rows[i]) for i in select(
        reversed_rows, 10, 'test', excluded={95, 96, 97})}
    assert allocate(5, ['b', 'a']) == {'a': 3, 'b': 2}
    with pytest.raises(ValueError):
        allocate(1, ['a', 'b'])


def test_manifest_tampering_rejected(tmp_path):
    import json
    data = read_manifest(MANIFEST)
    data['seed'] += 1
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='integrity'):
        read_manifest(path)


def test_coding_uses_original_tests_and_complete_function():
    code = extract_code('<think>noise</think>\n```python\ndef add(a,b): return a+b\n```')
    assert code == 'def add(a,b): return a+b'
    source = program('humaneval', {'test': 'def check(f): assert f(1,2)==3', 'entry_point': 'add'}, code)
    assert source.endswith('check(add)\n')
    ast.parse(source)
    assert 'assert x == 4' in program('mbpp', {'test_list': ['assert x == 4']}, 'x=4')


def test_tool_calls_are_literal_public_simulator_methods_only():
    class Simulator:
        def add(self, x, y):
            return x + y
    Simulator.__module__ = 'bfcl_eval.eval_checker.multi_turn_eval.func_source_code.math_api'
    namespace = {'sim': Simulator()}
    assert safe_simulator_call('sim.add(x=2,y=3)', namespace) == 5
    for expression in ['__import__("os").system("id")', 'sim.__class__()',
                       'sim.add(x=__import__("os"),y=1)', 'sim.add(**{})',
                       'sim.add.__call__(1,2)', '(lambda: 1)()', 'open("/etc/passwd")']:
        with pytest.raises((ValueError, SyntaxError)):
            safe_simulator_call(expression, namespace)
    with pytest.raises(ValueError, match='Unknown simulator'):
        safe_simulator_call('sim.add(1,2)', {'sim': SimpleNamespace(add=lambda x, y: x+y)})


def test_coding_inference_concurrency_does_not_expand_sandbox_limit(tmp_path, monkeypatch):
    import threading
    import time
    from types import SimpleNamespace
    from delta_cli.public_eval import coding
    barrier = threading.Barrier(16)
    lock = threading.Lock()
    active = peak = 0

    def response(*args, **kwargs):
        barrier.wait(timeout=5)  # Generation must be able to exceed four active requests.
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
            'choices': [{'message': {'content': 'pass'}, 'finish_reason': 'stop'}]})

    def sandbox(*args, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(.01)
        with lock:
            active -= 1
        return {'passed': True}

    monkeypatch.setattr(coding.httpx, 'post', response)
    monkeypatch.setattr(coding, 'sandbox', sandbox)
    docs = [{'task_id': str(i), 'text': 'Example', 'test_list': ['assert True']} for i in range(16)]
    result = coding.run({'sha256': 'fixture', 'seed': 1, 'code': {'mbpp': {'docs': docs}}},
        'mbpp', 'http://test', 'fixture', 'fixture', tmp_path / 'result.json', concurrency=128)
    assert result['count'] == 16
    assert result['score'] == 1
    assert 1 <= peak <= 4
