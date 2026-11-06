"""Opt-in adapter integration tests; use the isolated benchmark environment."""
import copy
import json
import os
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytestmark = pytest.mark.skipif(os.getenv('DELTA_TEST_PUBLIC_NATIVE') != '1',
                               reason='Requires isolated standard benchmark dependencies and cached sources')


def test_native_text_api_roundtrip(tmp_path):
    from transformers import AutoTokenizer
    from delta_cli.public_eval.common import read_manifest
    from delta_cli.public_eval.text import run
    root = Path(__file__).parents[1]
    tokenizer_dir = Path(os.getenv('DELTA_TEST_TOKENIZER', str(root / 'var/qwen-base')))
    if not tokenizer_dir.exists():
        pytest.skip('Requires local Qwen tokenizer fixture')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append(body)
            prompt = body['prompt']
            prompts = prompt if isinstance(prompt, list) and isinstance(prompt[0], (str, list)) else [prompt]
            choices = []
            for i, prompt in enumerate(prompts):
                choice = {'index': i, 'text': 'The answer is 42.\n#### 42', 'finish_reason': 'stop'}
                if body.get('echo'):
                    tokens = tokenizer.encode(prompt, add_special_tokens=False) if isinstance(prompt, str) else prompt
                    choice['logprobs'] = {'tokens': ['x'] * (len(tokens) + 1),
                        'token_logprobs': [-.5] * (len(tokens) + 1),
                        'top_logprobs': [{'x': -.5}] * (len(tokens) + 1)}
                choices.append(choice)
            raw = json.dumps({'id': 'mock', 'object': 'text_completion', 'created': 0,
                'model': 'connector-fixture', 'choices': choices,
                'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        manifest = read_manifest(root / 'delta_cli/public_eval/data/qwen-small-public-v1.json')
        fixture = copy.deepcopy(manifest)
        fixture['sha256'] = 'connector-fixture-not-a-model-result'
        for group in ['arc_challenge', 'gsm8k']:
            for spec in fixture['text'][group]['selection'].values():
                spec['indices'], spec['doc_sha256'] = spec['indices'][:1], spec['doc_sha256'][:1]
            result = run(fixture, group, f'http://127.0.0.1:{server.server_port}/v1',
                         'connector-fixture', str(tokenizer_dir), tmp_path / (group + '.json'))
            assert len({r['doc_id'] for r in result['samples'][fixture['text'][group]['task']]}) == 1
        assert any(r.get('echo') and r.get('logprobs') == 1 for r in requests)
        assert any(not r.get('echo') and '<|im_start|>' in str(r['prompt']) for r in requests)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_native_bfcl_api_roundtrip_with_no_tool_calls(tmp_path, monkeypatch):
    import httpx
    from openai import OpenAI
    from delta_cli.public_eval import tools
    from delta_cli.public_eval.common import read_manifest
    root = Path(__file__).parents[1]
    tokenizer_dir = Path(os.getenv('DELTA_TEST_TOKENIZER', str(root / 'var/qwen-base')))
    if not tokenizer_dir.exists():
        pytest.skip('Requires local Qwen tokenizer fixture')
    original = tools.handler_for
    requests = []

    def handle(request):
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(200, json={'id': 'mock', 'object': 'text_completion', 'created': 0,
            'model': 'connector-fixture', 'choices': [{'index': 0, 'text': 'I cannot do that.', 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}})

    def handler_for(*args, **kwargs):
        handler, alias = original(*args, **kwargs)
        handler.client.close()
        handler.client = OpenAI(base_url='http://fixture.invalid/v1', api_key='fixture',
            http_client=httpx.Client(transport=httpx.MockTransport(handle)))
        return handler, alias

    monkeypatch.setattr(tools, 'handler_for', handler_for)
    fixture = copy.deepcopy(read_manifest(root / 'delta_cli/public_eval/data/qwen-small-public-v1.json'))
    fixture['sha256'] = 'connector-fixture-not-a-model-result'
    fixture['bfcl'] = {k: v for k, v in fixture['bfcl'].items() if k in ('simple_python', 'multi_turn_base')}
    for spec in fixture['bfcl'].values():
        spec['ids'], spec['doc_sha256'], spec['count'] = spec['ids'][:1], spec['doc_sha256'][:1], 1
    result = tools.run(fixture, 'http://fixture.invalid/v1', 'connector-fixture', str(tokenizer_dir), tmp_path / 'bfcl.json')
    assert result['scores'] == {'multi_turn_base': {'accuracy': 0, 'count': 1},
                               'simple_python': {'accuracy': 0, 'count': 1}}
    assert requests and all('<|im_start|>' in r['prompt'] for r in requests)
