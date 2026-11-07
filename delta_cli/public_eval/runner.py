"""Public-only evaluator API shared by the CLI and platform workers."""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import time

from .common import check_versions, read_manifest
from .suite import PROTOCOL, PRIMARY, expected_items, manifest, score_public


def load_suite(path=None):
    """Accept the pinned public manifest, including copies downloaded by competition init."""
    bundled = manifest()
    suite = read_manifest(Path(path)) if path is not None else bundled
    if (suite.get('id') != bundled['id'] or suite.get('sha256') != bundled['sha256']
            or suite.get('scope') != 'public-validation-only'):
        raise ValueError('Unsupported public suite. Install the evaluator version matching this configuration.')
    return suite


def configuration():
    root = Path(__file__).parent
    suite = manifest()
    runtime = json.loads((root / 'data/runtime.json').read_text())
    sources = sorted(root.glob('*.py')) + [root / 'requirements.txt', root / 'data/runtime.json']
    return {'protocol': PROTOCOL, 'suite_id': suite['id'], 'suite_sha256': suite['sha256'],
            'seed': suite['seed'], 'benchmark_items': {k: len(v) for k, v in expected_items(suite).items()},
            'primary_metrics': PRIMARY, 'runtime': runtime,
            'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}


def evaluate_public(*, endpoint, model, tokenizer, content_hash, output, manifest_path=None,
                    concurrency=32, progress=None, generation=None, config_id=None):
    """Return public_validation/public_evidence; never accepts platform/private job payloads.

    The content hash identifies the checkpoint the caller serves. It is recorded,
    not remotely attested. Platform callers must retain their own trusted lifecycle.
    """
    if not re.fullmatch(r'[0-9a-f]{64}', content_hash):
        raise ValueError('content_hash must identify the served checkpoint (64 lowercase hex characters)')
    if type(concurrency) is not int or not 1 <= concurrency <= 256:
        raise ValueError('concurrency must be between 1 and 256')
    suite = load_suite(manifest_path)
    check_versions()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    reports = output / 'reports'
    reports.mkdir()
    state = {**configuration(), 'local_only': True, 'model': model, 'content_hash': content_hash,
             'tokenizer': str(tokenizer), 'concurrency': concurrency, 'generation': generation or {}, 'config_id': config_id,
             'dependencies': {name: version(name) for name in suite['versions']},
             'status': 'running', 'completed': 0, 'total': sum(map(len, expected_items(suite).values()))}
    state_path = output / 'run.json'
    groups, evidence = {}, {}
    started = time.monotonic()

    def record():
        state['completed'] = sum(len(rows) for rows in groups.values())
        state_path.write_text(json.dumps(state, indent=2))
        if progress:
            progress(state['completed'], state['total'])

    try:
        record()
        for group, spec in suite['text'].items():
            from .text import run as text_run
            report = text_run(suite, group, endpoint, model, str(tokenizer), reports / (group + '.json'),
                              concurrency=concurrency, **({'generation':generation} if generation else {}))
            metric, selected_filter = PRIMARY[group]
            groups[group] = [
                {'id': f"{task}:{sample['doc_id']}", 'score': float(sample[metric])}
                for task in spec['selection'] for sample in report['samples'][task]
                if sample['filter'] == selected_filter]
            evidence[group] = {task: [{k: v for k, v in row.items() if k not in ('doc', 'arguments')}
                                      for row in samples] for task, samples in report['samples'].items()}
            record()
        for group in suite['code']:
            from .coding import run as code_run
            report = code_run(suite, group, endpoint, model, '', reports / (group + '.json'),
                              sandbox_backend='wasi', concurrency=concurrency, **({'generation':generation} if generation else {}))
            groups[group] = [{'id': str(r['id']), 'score': float(r['passed'])} for r in report['samples']]
            evidence[group] = report
            record()
        from .tools import run as tools_run
        report = tools_run(suite, endpoint, model, str(tokenizer), reports / 'bfcl.json', concurrency=concurrency, **({'generation':generation} if generation else {}))
        groups['bfcl_single_turn'], groups['bfcl_multi_turn'] = [], []
        evidence['bfcl'] = report
        for category, spec in suite['bfcl'].items():
            paths = list((reports / 'bfcl').rglob(f'BFCL_v4_{category}_score.json'))
            if len(paths) != 1:
                raise ValueError('Missing BFCL score evidence')
            entries = [json.loads(line) for line in paths[0].read_text().splitlines() if line.strip()]
            failed = {entry['id'] for entry in entries[1:]}
            if not failed <= set(spec['ids']) or entries[0]['correct_count'] != spec['count'] - len(failed):
                raise ValueError('BFCL evidence coverage mismatch')
            name = 'bfcl_multi_turn' if category.startswith('multi_turn_') else 'bfcl_single_turn'
            groups[name].extend({'id': i, 'score': float(i not in failed)} for i in spec['ids'])
            evidence['bfcl'][category] = json.loads((reports / 'bfcl' / f'{category}.responses.json').read_text())
        public = {'protocol': PROTOCOL, 'suite_sha256': suite['sha256'], 'content_hash': content_hash,
                  'reference_check': False, 'groups': groups}
        score, components = score_public(public, content_hash)
        record()
        result = {'public_validation': public, 'public_evidence': evidence,
                  'validation_score': score, 'validation_benchmarks': components,
                  'local_only': True, 'generation': generation or {}, 'config_id': config_id, 'duration_seconds': time.monotonic() - started}
        (output / 'result.json').write_text(json.dumps(result, indent=2))
        state['status'] = 'complete'
        return result
    except BaseException as exc:
        state['status'] = 'failed'
        # Exception text may contain endpoint credentials; preserve only the type.
        state['error_type'] = type(exc).__name__
        raise
    finally:
        state_path.write_text(json.dumps(state, indent=2))
