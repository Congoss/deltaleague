"""HumanEval/MBPP pass@1 against original tests, executed in disposable containers."""
from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore
import json
import re
import subprocess
import tempfile
import uuid
from pathlib import Path
import httpx


def extract_code(response):
    final = response.split('</think>')[-1].strip()
    blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)```', final, re.DOTALL)
    return blocks[-1].strip() if blocks else final


def program(benchmark, doc, code):
    if benchmark == 'humaneval':
        return code + '\n\n' + doc['test'] + '\ncheck(' + doc['entry_point'] + ')\n'
    if benchmark == 'mbpp':
        return doc.get('test_setup_code', '') + '\n' + code + '\n' + '\n'.join(doc['test_list']) + '\n'
    raise ValueError('Unknown coding benchmark')


def sandbox(source, image, *, timeout=12):
    if '@sha256:' not in image:
        raise ValueError('Sandbox image must be digest pinned')
    name = 'delta-validation-' + uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix='delta-code-') as td:
        path = Path(td) / 'candidate.py'
        # Successful interpreter exit alone is insufficient (e.g. sys.exit(0)).
        nonce = uuid.uuid4().hex
        path.write_text(source + f'\nprint({nonce!r}, flush=True)\n')
        Path(td).chmod(0o755)
        path.chmod(0o444)
        args = ['docker', 'run', '--rm', '--name', name, '--network', 'none', '--read-only',
                '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--pids-limit', '32',
                '--memory', '256m', '--memory-swap', '256m', '--cpus', '1', '--user', '65534:65534',
                '--ulimit', 'cpu=5:5', '--ulimit', 'fsize=1048576:1048576',
                '--tmpfs', '/tmp:rw,noexec,nosuid,size=16m',
                '--mount', f'type=bind,src={path},dst=/candidate.py,readonly',
                image, 'python', '-I', '-B', '/candidate.py']
        try:
            # Bound output on disk; generated programs cannot fill controller memory.
            with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
                process = subprocess.Popen(['prlimit', '--fsize=2097152:2097152', '--', *args], stdout=stdout, stderr=stderr)
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    subprocess.run(['docker', 'kill', name], capture_output=True, timeout=10)
                    process.kill()
                    process.wait()
                    return {'passed': False, 'reason': 'timeout'}
                stdout.seek(0)
                data = stdout.read(2 * 1024**2)
                if process.returncode in (125, 126, 127):
                    stderr.seek(0)
                    raise RuntimeError('Sandbox infrastructure failure: ' + stderr.read(2048).decode(errors='replace'))
                return {'passed': process.returncode == 0 and nonce.encode() in data.splitlines(),
                        'reason': 'passed' if process.returncode == 0 and nonce.encode() in data.splitlines()
                                  else 'test_failure'}
        finally:
            subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=10)


def prompt(benchmark, doc):
    if benchmark == 'humaneval':
        return 'Complete this Python function. Return the complete function in a Python code block.\n```python\n' + doc['prompt'] + '\n```'
    return ('Write Python code solving this task. Return code in a Python code block.\n' + doc['text']
            + '\nThe code must pass:\n' + '\n'.join(doc['test_list']))


def run(manifest, benchmark, endpoint, model, image, output, *, reference_check=False, sandbox_backend="docker", concurrency=32, generation=None):
    from .generation import sampling, final_answer
    generation = generation or {}
    docs = manifest['code'][benchmark]['docs']
    if sandbox_backend == 'wasi':
        from .wasi import sandbox as wasi_sandbox, SANDBOX_ID
        if not wasi_sandbox('assert 1 + 1 == 2')['passed']:
            raise RuntimeError('WASI sandbox preflight failed')
        evaluate = lambda source: wasi_sandbox(source)
        image = SANDBOX_ID
    elif sandbox_backend == 'docker':
        evaluate = lambda source: sandbox(source, image)
    else:
        raise ValueError('Unknown sandbox backend')

    sandbox_slots = BoundedSemaphore(4)
    def one(doc):
        if reference_check:
            code = doc['prompt'] + doc['canonical_solution'] if benchmark == 'humaneval' else doc['code']
            usage, finish = {}, 'reference'
        else:
            response = httpx.post(endpoint.rstrip('/') + '/chat/completions', json={
                'model': model, 'messages': [{'role': 'user', 'content': prompt(benchmark, doc)}],
                'temperature': 0, **sampling(generation), 'max_tokens': generation.get('max_tokens', 4096), 'seed': manifest['seed'],
                'chat_template_kwargs': {'enable_thinking': generation.get('enable_thinking', False)}}, timeout=600)
            response.raise_for_status()
            data = response.json()
            if generation.get('enable_thinking'):
                message = data['choices'][0]['message']
                message['content'] = final_answer(message.get('content') or '')
            code = extract_code(data['choices'][0]['message']['content'])
            usage, finish = data.get('usage', {}), data['choices'][0]['finish_reason']
        with sandbox_slots:
            result = evaluate(program(benchmark, doc, code))
        return {'id': doc['task_id'], 'code': code, 'usage': usage, 'finish_reason': finish, **result}

    with ThreadPoolExecutor(max_workers=4 if reference_check else concurrency) as pool:
        results = list(pool.map(one, docs))
    report = {'suite_sha256': manifest['sha256'], 'benchmark': benchmark, 'model': model,
              'sandbox_image': image, 'reference_check': reference_check,
              'metric': 'pass@1', 'score': sum(r['passed'] for r in results) / len(results),
              'count': len(results), 'samples': results}
    output.write_text(json.dumps(report, indent=2))
    return report
