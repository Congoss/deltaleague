"""Dependency-free protocol contract shared by controller and GPU suite adapter."""
import hashlib
import json
import math
from pathlib import Path
import statistics

PROTOCOL = 'delta-public-v1'
SUITE_ID = 'qwen-small-public-v1'
PRIMARY = {
    'gsm8k': ('exact_match', 'flexible-extract'),
    'math500': ('math_verify', 'none'),
    'arc_easy': ('acc_norm', 'none'), 'arc_challenge': ('acc_norm', 'none'),
    'hellaswag': ('acc_norm', 'none'), 'piqa': ('acc_norm', 'none'),
    'winogrande': ('acc', 'none'), 'boolq': ('acc', 'none'),
    'openbookqa': ('acc_norm', 'none'), 'commonsense_qa': ('acc', 'none'),
    'mmlu': ('acc', 'none'), 'mmlu_pro': ('exact_match', 'custom-extract'),
    'bbh': ('exact_match', 'get-answer'), 'ifeval': ('prompt_level_strict_acc', 'none'),
}


def manifest():
    path = Path(__file__).parent / 'data' / (SUITE_ID + '.json')
    value = json.loads(path.read_text())
    raw = json.dumps({k: v for k, v in value.items() if k != 'sha256'}, sort_keys=True,
                     ensure_ascii=False, separators=(',', ':')).encode()
    if hashlib.sha256(raw).hexdigest() != value['sha256']:
        raise ValueError('public_suite_integrity')
    return value


def expected_items(suite):
    groups = {}
    for name, spec in suite['text'].items():
        groups[name] = {f'{task}:{i}' for task, s in spec['selection'].items() for i in s['indices']}
    for name, spec in suite['code'].items():
        groups[name] = {str(d['task_id']) for d in spec['docs']}
    for name in ['bfcl_single_turn', 'bfcl_multi_turn']:
        groups[name] = {i for cat, spec in suite['bfcl'].items()
                       if cat.startswith('multi_turn_') == (name == 'bfcl_multi_turn') for i in spec['ids']}
    return groups


def score_public(result, content_hash):
    suite = manifest()
    if (result.get('suite_sha256') != suite['sha256'] or result.get('protocol') != PROTOCOL
            or result.get('content_hash') != content_hash or result.get('reference_check') is not False):
        raise ValueError('public_result_identity')
    expected = expected_items(suite)
    groups = result.get('groups', {})
    if set(groups) != set(expected):
        raise ValueError('public_result_groups')
    components = {}
    for name, ids in expected.items():
        rows = groups[name]
        if not isinstance(rows, list) or len(rows) != len(ids) or {r.get('id') for r in rows} != ids:
            raise ValueError('public_result_coverage')
        for r in rows:
            value = r.get('score')
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('public_result_score')
        components[name] = statistics.mean(r['score'] for r in rows) * 100
    return statistics.mean(components.values()), components
