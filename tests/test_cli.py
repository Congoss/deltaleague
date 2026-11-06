from contextlib import nullcontext
import hashlib
import json

import httpx
import pytest
from typer.testing import CliRunner

from delta_cli import main

runner = CliRunner()


def artifact(data=b'weights'):
    return {"name": "model.safetensors", "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(), "url": "https://blob.invalid/?secret=hidden"}


def test_download_integrity_preserves_existing_file(tmp_path, monkeypatch):
    path = tmp_path / 'model.safetensors'
    path.write_bytes(b'original')
    monkeypatch.setattr(main.httpx, 'stream', lambda *a, **k: nullcontext(httpx.Response(200, content=b'bad', request=httpx.Request('GET', a[1]))))
    with pytest.raises(main.typer.Exit):
        main.download_artifact(artifact(), tmp_path, force=True)
    assert path.read_bytes() == b'original'
    assert not list(tmp_path.glob('*.part'))


def test_download_reuses_only_verified_files(tmp_path, monkeypatch):
    path = tmp_path / 'model.safetensors'
    path.write_bytes(b'weights')
    monkeypatch.setattr(main.httpx, 'stream', lambda *a, **k: pytest.fail('Unnecessary network request'))
    main.download_artifact(artifact(), tmp_path)
    path.write_bytes(b'changed')
    with pytest.raises(main.typer.BadParameter, match='different contents'):
        main.download_artifact(artifact(), tmp_path)


def test_init_records_reproducible_workspace_without_signed_urls(tmp_path, monkeypatch):
    validation = b'{"examples": []}'
    config = {"id": "config-v1", "validation_names": ["math"],
              "validation_download": "https://blob.invalid/validation?secret=hidden",
              "validation_sha256": hashlib.sha256(validation).hexdigest(), "validation_size_bytes": len(validation)}
    def call(method, path, **kwargs):
        if path == 'leaderboard':
            return {"config_id": "config-v1", "rows": [{"version_id": "parent", "parent_id": None}]}
        if path.startswith('evaluation-configs/'):
            return config
        return {"id": "parent", "model_id": "model", "content_hash": "hash", "files": [artifact()]}
    monkeypatch.setattr(main, 'call', call)
    monkeypatch.setattr(main.httpx, 'stream', lambda *a, **k: nullcontext(httpx.Response(200, content=validation if '/validation' in a[1] else b'weights', request=httpx.Request('GET', a[1]))))
    result = runner.invoke(main.app, ['competition', 'init', 'model', '--output', str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert json.loads((tmp_path / 'delta-workspace.json').read_text())['parent_version_id'] == 'parent'
    assert (tmp_path / 'validation.json').read_bytes() == validation
    for path in tmp_path.rglob('*.json'):
        assert 'secret=hidden' not in path.read_text()
    # A different model cannot silently replace the workspace's parent.
    result = runner.invoke(main.app, ['competition', 'init', 'other', '--parent', 'parent', '--output', str(tmp_path)])
    assert result.exit_code != 0
    assert 'different model' in result.output


def device_flow():
    return {'verification_url': 'https://deltaleague.ai/device', 'user_code': 'ABCD-1234', 'interval': 0, 'id': 'dev', 'secret': 'hidden'}


def test_login_polls_for_token_and_writes_0600_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv('DELTA_CONFIG', str(tmp_path / 'auth.json'))
    monkeypatch.setattr(main.time, 'sleep', lambda *args: None)
    opened = []
    monkeypatch.setattr(main.webbrowser, 'open', lambda url: opened.append(url))
    polls = {'n': 0}

    def call(method, path, **kwargs):
        if path == 'cli-auth':
            return device_flow()
        polls['n'] += 1
        return {'token': 'token-value'} if polls['n'] > 1 else {}

    monkeypatch.setattr(main, 'call', call)
    result = runner.invoke(main.app, ['login', '--no-browser'])
    assert result.exit_code == 0, result.output
    assert 'ABCD-1234' in result.output and opened == []
    auth = tmp_path / 'auth.json'
    assert json.loads(auth.read_text()) == {'api': 'https://deltaleague.ai', 'token': 'token-value'}
    assert auth.stat().st_mode & 0o777 == 0o600
    monkeypatch.setattr(main, 'call', lambda *args, **kwargs: device_flow() if args[1] == 'cli-auth' else {})
    timed_out = runner.invoke(main.app, ['login', '--no-browser'])
    assert timed_out.exit_code == 1
    assert 'timed out' in timed_out.output and 'Traceback' not in timed_out.output


def test_login_retries_a_transient_poll_error(tmp_path, monkeypatch):
    monkeypatch.setenv('DELTA_CONFIG', str(tmp_path / 'auth.json'))
    monkeypatch.setattr(main.time, 'sleep', lambda *args: None)
    monkeypatch.setattr(main.webbrowser, 'open', lambda url: None)
    attempts = {'n': 0}

    def handler(request):
        if request.url.path.endswith('/cli-auth'):
            return httpx.Response(200, json=device_flow())
        attempts['n'] += 1
        if attempts['n'] == 1:
            return httpx.Response(503, json={'error': 'busy'})
        return httpx.Response(200, json={'token': 'tok'})

    monkeypatch.setattr(main, 'client', lambda: httpx.Client(transport=httpx.MockTransport(handler), base_url='https://deltaleague.ai'))
    result = runner.invoke(main.app, ['login', '--no-browser'])
    assert result.exit_code == 0, result.output
    assert attempts['n'] == 2
    assert json.loads((tmp_path / 'auth.json').read_text())['token'] == 'tok'


def test_request_retries_429_and_stops_on_4xx_or_exhausted_5xx(monkeypatch):
    monkeypatch.setattr(main.time, 'sleep', lambda *args: None)
    calls = {'n': 0}

    def transport(status):
        def handler(request):
            calls['n'] += 1
            if status == 429 and calls['n'] > 1:
                return httpx.Response(200, json={'ok': True})
            return httpx.Response(status, json={'ok': True} if status == 200 else {'no': True})
        return lambda: httpx.Client(transport=httpx.MockTransport(handler), base_url='https://deltaleague.ai')

    monkeypatch.setattr(main, 'client', transport(429))
    assert main.call('GET', 'me', retries=2) == {'ok': True}
    assert calls['n'] == 2
    calls['n'] = 0
    monkeypatch.setattr(main, 'client', transport(400))
    with pytest.raises(main.typer.Exit):
        main.call('GET', 'me', retries=5)
    assert calls['n'] == 1
    calls['n'] = 0

    def down(request):
        calls['n'] += 1
        return httpx.Response(503, text='down')

    monkeypatch.setattr(main, 'client', lambda: httpx.Client(transport=httpx.MockTransport(down), base_url='https://deltaleague.ai'))
    with pytest.raises(main.typer.Exit):
        main.call('GET', 'me', retries=2)
    assert calls['n'] == 3


def test_call_reports_authentication_and_generic_http_errors(monkeypatch):
    def client_for(status, body=None, raw=None):
        def handler(request):
            if raw is not None:
                return httpx.Response(status, content=raw)
            return httpx.Response(status, json=body or {})
        return lambda: httpx.Client(transport=httpx.MockTransport(handler), base_url='https://deltaleague.ai')

    monkeypatch.setattr(main, 'client', client_for(401, {'detail': 'nope'}))
    denied = runner.invoke(main.app, ['whoami'])
    assert denied.exit_code == 1 and 'delta login' in denied.output
    monkeypatch.setattr(main, 'client', client_for(500, {'detail': 'boom'}))
    failed = runner.invoke(main.app, ['whoami'])
    assert failed.exit_code == 1 and 'HTTP 500' in failed.output and 'boom' in failed.output
    monkeypatch.setattr(main, 'client', client_for(500, raw=b'<html>'))
    plain = runner.invoke(main.app, ['whoami'])
    assert plain.exit_code == 1 and 'HTTP 500' in plain.output and 'Traceback' not in plain.output


def test_status_watch_stops_on_terminal_state(monkeypatch):
    monkeypatch.setattr(main.time, 'sleep', lambda *args: None)
    states = iter([{'state': 'pending', 'id': 's'}, {'state': 'verified', 'id': 's'}])
    seen = []

    def call(method, path, **kwargs):
        seen.append(kwargs.get('retries', 0))
        return next(states)

    monkeypatch.setattr(main, 'call', call)
    watched = runner.invoke(main.app, ['submission', 'status', 's', '--watch'])
    assert watched.exit_code == 0, watched.output
    assert seen == [5, 5] and 'verified' in watched.output
    seen.clear()
    monkeypatch.setattr(main, 'call', lambda *args, **kwargs: seen.append(kwargs.get('retries', 0)) or {'state': 'pending'})
    once = runner.invoke(main.app, ['submission', 'status', 's'])
    assert once.exit_code == 0, once.output
    assert seen == [0]


def test_client_scopes_saved_token_to_api_and_prefers_delta_token(tmp_path, monkeypatch):
    auth = tmp_path / 'auth.json'
    auth.write_text(json.dumps({'api': 'https://deltaleague.ai', 'token': 'saved'}))
    monkeypatch.setenv('DELTA_CONFIG', str(auth))
    monkeypatch.delenv('DELTA_TOKEN', raising=False)
    with main.client() as http:
        assert http.headers['Authorization'] == 'Bearer saved'
    monkeypatch.setattr(main, 'API', 'https://other.example')
    with main.client() as http:
        assert 'Authorization' not in http.headers
    monkeypatch.setenv('DELTA_TOKEN', 'env-token')
    with main.client() as http:
        assert http.headers['Authorization'] == 'Bearer env-token'


def test_api_connection_failure_does_not_print_credentials(monkeypatch):
    def fail():
        raise httpx.ConnectError('https://user:secret@server.invalid')
    monkeypatch.setattr(main, 'client', fail)
    result = runner.invoke(main.app, ['competition', 'list'])
    assert result.exit_code == 1
    assert 'secret' not in result.output
    assert 'Could not reach' in result.output
