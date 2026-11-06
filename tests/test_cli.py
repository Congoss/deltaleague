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


def submission_dir(tmp_path, payload=b'weights', model='tiny'):
    root = tmp_path / 'submission'
    root.mkdir()
    (root / 'weights.safetensors').write_bytes(payload)
    manifest = {
        'schema_version': 'delta.submission.v1', 'format': 'sparse-diff', 'operation': 'replace',
        'reconstruction_version': 'delta-reconstruct-v1',
        'parent_version_id': '00000000-0000-0000-0000-000000000002',
        'parent_content_hash': 'a' * 64, 'model_config_id': model,
        'tensors': {'layer.weight': {'indices': 't0.indices', 'values': 't0.values'}},
        'artifacts': [{'name': 'weights.safetensors', 'sha256': hashlib.sha256(payload).hexdigest(), 'size_bytes': len(payload)}],
    }
    (root / 'manifest.json').write_text(json.dumps(manifest))
    return root


def upload_session(url='https://blob.invalid/upload?sig=secret'):
    return {'id': 'upload-one', 'files': [{'name': 'weights.safetensors', 'url': url}]}


def test_submit_uploads_once_and_reuses_idempotency_state(tmp_path, monkeypatch):
    root = submission_dir(tmp_path)
    calls = []
    puts = []

    def call(method, path, **kwargs):
        calls.append((path, kwargs))
        if path == 'upload-sessions':
            return upload_session()
        return {'id': 'submission-one', 'title': kwargs['json']['title']}

    def put(url, **kwargs):
        puts.append(kwargs['content'].read())
        return httpx.Response(200, request=httpx.Request('PUT', url))

    monkeypatch.setattr(main, 'call', call)
    monkeypatch.setattr(main.httpx, 'put', put)
    first = runner.invoke(main.app, ['submit', str(root), '--title', 'First'])
    assert first.exit_code == 0, first.output
    assert 'secret' not in first.output
    state = json.loads((root / '.delta-upload.json').read_text())
    assert (root / '.delta-upload.json').stat().st_mode & 0o777 == 0o600
    second = runner.invoke(main.app, ['submit', str(root), '--title', 'First'])
    assert second.exit_code == 0, second.output
    assert [path for path, _ in calls].count('upload-sessions') == 1
    assert puts == [b'weights']
    posted = [kwargs for path, kwargs in calls if path == 'submissions']
    assert posted[0]['headers']['Idempotency-Key'] == posted[1]['headers']['Idempotency-Key'] == state['idempotency_key']
    retitled = runner.invoke(main.app, ['submit', str(root), '--title', 'Second'])
    assert retitled.exit_code == 1 and 'original title' in retitled.output
    assert [path for path, _ in calls].count('upload-sessions') == 1
    posted = [kwargs for path, kwargs in calls if path == 'submissions']
    assert len(posted) == 2
    (tmp_path / 'other').mkdir()
    changed = submission_dir(tmp_path / 'other', model='tiny-2')
    (changed / '.delta-upload.json').write_text((root / '.delta-upload.json').read_text())
    other = runner.invoke(main.app, ['--api', 'http://127.0.0.1:9', 'submit', str(changed), '--title', 'First'])
    assert other.exit_code == 0, other.output
    assert [path for path, _ in calls].count('upload-sessions') == 2


def test_submit_upload_errors_do_not_report_success(tmp_path, monkeypatch):
    root = submission_dir(tmp_path)
    calls = []

    def call(method, path, **kwargs):
        calls.append(path)
        if path == 'upload-sessions':
            return upload_session()
        raise AssertionError('submission posted after a failed upload')

    monkeypatch.setattr(main, 'call', call)

    def refuse(status):
        def put(url, **kwargs):
            kwargs['content'].read()
            return httpx.Response(status, headers={'Location': url}, request=httpx.Request('PUT', url))
        return put

    monkeypatch.setattr(main.httpx, 'put', refuse(302))
    redirected = runner.invoke(main.app, ['submit', str(root)])
    assert redirected.exit_code == 1
    assert 'secret' not in redirected.output and 'Traceback' not in redirected.output
    assert 'HTTP 302' in redirected.output
    assert json.loads((root / '.delta-upload.json').read_text())['pending_files']
    monkeypatch.setattr(main.httpx, 'put', lambda url, **kwargs: (_ for _ in ()).throw(httpx.ConnectError(url)))
    failed = runner.invoke(main.app, ['submit', str(root)])
    assert failed.exit_code == 1
    assert 'secret' not in failed.output and 'Traceback' not in failed.output
    assert calls == ['upload-sessions']


def test_submit_ignores_server_filenames_and_rejects_bad_urls(tmp_path, monkeypatch):
    root = submission_dir(tmp_path)
    puts = []
    calls = []

    def call(method, path, **kwargs):
        calls.append(path)
        if path == 'upload-sessions':
            return {'id': 'upload-one', 'files': [
                {'name': '../escape.bin', 'url': 'https://blob.invalid/escape?sig=secret'},
                {'name': 'other.bin', 'url': 'http://evil.example/object'},
                {'name': 'weights.safetensors', 'url': 'https://blob.invalid/real'},
            ]}
        return {'id': 'submission-one'}

    monkeypatch.setattr(main, 'call', call)
    monkeypatch.setattr(main.httpx, 'put', lambda url, **kwargs: puts.append(url) or httpx.Response(200, request=httpx.Request('PUT', url)))
    result = runner.invoke(main.app, ['submit', str(root)])
    assert result.exit_code == 0, result.output
    assert puts == ['https://blob.invalid/real']
    assert 'secret' not in result.output and 'escape.bin' not in result.output
    (tmp_path / 'only').mkdir()
    hostile = submission_dir(tmp_path / 'only')

    def only_hostile(method, path, **kwargs):
        if path == 'upload-sessions':
            return {'id': 'upload-two', 'files': [{'name': '../escape.bin', 'url': 'https://blob.invalid/escape?sig=secret'}]}
        raise AssertionError(path)

    monkeypatch.setattr(main, 'call', only_hostile)
    puts.clear()
    refused = runner.invoke(main.app, ['submit', str(hostile)])
    assert refused.exit_code == 1 and puts == []
    assert 'not allowed' in refused.output and 'secret' not in refused.output and 'Traceback' not in refused.output
    for url in ('http://blob.invalid/x', 'https://user:pw@blob.invalid/x', 'http://127.0.0.1/x', 'http://[bad'):
        (root / '.delta-upload.json').unlink(missing_ok=True)
        monkeypatch.setattr(main, 'call', lambda method, path, url=url, **kwargs: {'id': 'u', 'files': [{'name': 'weights.safetensors', 'url': url}]} if path == 'upload-sessions' else {'id': 's'})
        bad = runner.invoke(main.app, ['submit', str(root)])
        assert bad.exit_code == 1 and 'not allowed' in bad.output and 'Traceback' not in bad.output, url


def test_submit_keeps_the_idempotency_key_when_old_state_has_no_title(tmp_path, monkeypatch):
    root = submission_dir(tmp_path)
    manifest = json.loads((root / 'manifest.json').read_text())
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    (root / '.delta-upload.json').write_text(json.dumps({
        'upload_id': 'kept', 'idempotency_key': 'key-1', 'fingerprint': fingerprint, 'api': main.API}))
    calls = []

    def call(method, path, **kwargs):
        calls.append((path, kwargs))
        return {'id': 'submission-one'}

    monkeypatch.setattr(main, 'call', call)
    result = runner.invoke(main.app, ['submit', str(root), '--title', 'Weight modification'])
    assert result.exit_code == 0, result.output
    assert [path for path, _ in calls] == ['submissions']
    assert calls[0][1]['headers']['Idempotency-Key'] == 'key-1'


def test_api_connection_failure_does_not_print_credentials(monkeypatch):
    def fail():
        raise httpx.ConnectError('https://user:secret@server.invalid')
    monkeypatch.setattr(main, 'client', fail)
    result = runner.invoke(main.app, ['competition', 'list'])
    assert result.exit_code == 1
    assert 'secret' not in result.output
    assert 'Could not reach' in result.output


@pytest.mark.parametrize('failure', ['transfer', 'submission_response'])
def test_submit_retries_saved_session(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(main, 'validate', lambda path: None)
    (tmp_path / 'manifest.json').write_text(json.dumps({'artifacts': [artifact()]}))
    (tmp_path / 'model.safetensors').write_bytes(b'weights')
    state_path = tmp_path / '.delta-upload.json'
    calls = []
    puts = []

    def call(method, path, **kwargs):
        calls.append((path, kwargs))
        if path == 'upload-sessions':
            return {'id': 'upload-one', 'files': [artifact()]}
        if failure == 'submission_response' and sum(p == 'submissions' for p, _ in calls) == 1:
            raise main.typer.Exit(1)
        return {'id': 'submission-one'}

    def put(url, **kwargs):
        state = json.loads(state_path.read_text())
        assert state['upload_id'] == 'upload-one'
        assert state['pending_files'][0]['url'] == url
        assert state_path.stat().st_mode & 0o777 == 0o600
        puts.append(kwargs['content'].read())
        if failure == 'transfer' and len(puts) == 1:
            raise httpx.ReadError('secret=hidden')
        return httpx.Response(200, request=httpx.Request('PUT', url))

    monkeypatch.setattr(main, 'call', call)
    monkeypatch.setattr(main.httpx, 'put', put)
    first = runner.invoke(main.app, ['submit', str(tmp_path)])
    assert first.exit_code == 1
    assert 'secret=hidden' not in first.output
    original = json.loads(state_path.read_text())
    second = runner.invoke(main.app, ['submit', str(tmp_path)])
    assert second.exit_code == 0, second.output
    assert sum(p == 'upload-sessions' for p, _ in calls) == 1
    assert puts == [b'weights'] * (2 if failure == 'transfer' else 1)
    assert json.loads(state_path.read_text())['pending_files'] == []
    for path, kwargs in calls:
        if path == 'submissions':
            assert kwargs['headers']['Idempotency-Key'] == original['idempotency_key']


def test_submit_replaces_expired_incomplete_session(tmp_path, monkeypatch):
    monkeypatch.setattr(main, 'validate', lambda path: None)
    (tmp_path / 'manifest.json').write_text(json.dumps({'artifacts': [artifact()]}))
    (tmp_path / 'model.safetensors').write_bytes(b'weights')
    state = {'upload_id': 'expired', 'api': main.API,
             'fingerprint': hashlib.sha256(json.dumps({'artifacts': [artifact()]}, sort_keys=True).encode()).hexdigest(),
             'idempotency_key': 'expired-key',
             'pending_files': [artifact()], 'expires_at': 1}
    (tmp_path / '.delta-upload.json').write_text(json.dumps(state))
    calls = []

    def call(method, path, **kwargs):
        calls.append(path)
        if path == 'upload-sessions':
            return {'id': 'new', 'files': [artifact()]}
        assert kwargs['json']['upload_id'] == 'new'
        return {'id': 'submission'}

    monkeypatch.setattr(main, 'call', call)
    monkeypatch.setattr(main.httpx, 'put', lambda url, **kwargs: httpx.Response(200, request=httpx.Request('PUT', url)))
    result = runner.invoke(main.app, ['submit', str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert calls == ['upload-sessions', 'submissions']


def test_pending_upload_can_be_retitled_but_lost_submission_response_cannot(tmp_path, monkeypatch):
    root = submission_dir(tmp_path)
    calls, puts = [], []

    def call(method, path, **kwargs):
        calls.append((path, kwargs))
        if path == 'upload-sessions':
            return upload_session()
        if sum(p == 'submissions' for p, _ in calls) == 1:
            raise main.typer.Exit(1)  # Server may already have finalized the upload.
        return {'id': 'same-submission'}

    def put(url, **kwargs):
        puts.append(url)
        if len(puts) == 1:
            raise httpx.ConnectError('secret')
        return httpx.Response(200, request=httpx.Request('PUT', url))

    monkeypatch.setattr(main, 'call', call)
    monkeypatch.setattr(main.httpx, 'put', put)
    assert runner.invoke(main.app, ['submit', str(root), '--title', 'First']).exit_code == 1
    assert runner.invoke(main.app, ['submit', str(root), '--title', 'Second']).exit_code == 1
    state = json.loads((root / '.delta-upload.json').read_text())
    assert state['title'] == 'Second' and state['submission_started']
    refused = runner.invoke(main.app, ['submit', str(root), '--title', 'Third'])
    assert refused.exit_code == 1 and 'original title' in refused.output
    assert runner.invoke(main.app, ['submit', str(root), '--title', 'Second']).exit_code == 0
    posted = [kw for p, kw in calls if p == 'submissions']
    assert len(posted) == 2
    assert all(kw['headers']['Idempotency-Key'] == state['idempotency_key'] for kw in posted)
    assert sum(p == 'upload-sessions' for p, _ in calls) == 1
    assert len(puts) == 2


def test_saved_session_cannot_select_an_unrelated_local_file(tmp_path, monkeypatch):
    root = submission_dir(tmp_path)
    manifest = json.loads((root / 'manifest.json').read_text())
    state = {'upload_id': 'u', 'idempotency_key': 'k', 'api': main.API,
             'fingerprint': hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
             'pending_files': [{'name': '../private.txt', 'url': 'https://blob.invalid/secret'}],
             'expires_at': main.time.time() + 3600}
    (root / '.delta-upload.json').write_text(json.dumps(state))
    monkeypatch.setattr(main, 'call', lambda *a, **k: pytest.fail('Unexpected API request'))
    monkeypatch.setattr(main.httpx, 'put', lambda *a, **k: pytest.fail('Unexpected upload'))
    result = runner.invoke(main.app, ['submit', str(root)])
    assert result.exit_code == 1 and 'unexpected artifact' in result.output
    assert 'secret' not in result.output


def test_atomic_upload_state_failure_preserves_previous_intent(tmp_path, monkeypatch):
    state = tmp_path / '.delta-upload.json'
    state.write_text('{"idempotency_key":"old"}')
    original = state.read_bytes()
    monkeypatch.setattr(main.os, 'fsync', lambda fd: (_ for _ in ()).throw(OSError('disk failed')))
    with pytest.raises(OSError):
        main.write_upload_state(state, {'idempotency_key': 'new'})
    assert state.read_bytes() == original
    assert not list(tmp_path.glob('.delta-upload-*.tmp'))


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


def test_logout_removes_credentials_when_the_api_call_fails(tmp_path, monkeypatch):
    auth = tmp_path / 'auth.json'
    auth.write_text(json.dumps({'api': 'https://deltaleague.ai', 'token': 'saved'}))
    monkeypatch.setenv('DELTA_CONFIG', str(auth))

    def client():
        def handler(request):
            return httpx.Response(401, json={'detail': 'expired'})
        return httpx.Client(transport=httpx.MockTransport(handler), base_url='https://deltaleague.ai')

    monkeypatch.setattr(main, 'client', client)
    denied = runner.invoke(main.app, ['logout'])
    assert denied.exit_code == 0, denied.output
    assert 'Signed out' in denied.output and not auth.exists()
    auth.write_text(json.dumps({'api': 'https://deltaleague.ai', 'token': 'saved'}))

    def down():
        raise httpx.ConnectError('https://user:secret@server.invalid')

    monkeypatch.setattr(main, 'client', down)
    offline = runner.invoke(main.app, ['logout'])
    assert offline.exit_code == 0, offline.output
    assert 'could not be revoked' in offline.output and 'Signed out' not in offline.output
    assert 'secret' not in offline.output and not auth.exists()


def test_validate_reports_malformed_submissions_without_a_traceback(tmp_path):
    missing = runner.invoke(main.app, ['validate', str(tmp_path)])
    assert missing.exit_code != 0 and 'Traceback' not in missing.output and 'manifest.json' in ''.join(missing.output.replace('│', '').split())
    submitted = runner.invoke(main.app, ['submit', str(tmp_path)])
    assert submitted.exit_code != 0 and 'Traceback' not in submitted.output
    (tmp_path / 'manifest.json').write_text('{')
    broken = runner.invoke(main.app, ['validate', str(tmp_path)])
    assert broken.exit_code != 0 and 'Traceback' not in broken.output
    (tmp_path / 'manifest.json').write_text(json.dumps({'schema_version': 'nope'}))
    schema = runner.invoke(main.app, ['validate', str(tmp_path)])
    assert schema.exit_code != 0 and 'unsupported_schema' in schema.output and 'Traceback' not in schema.output
    payload = b'weights'
    manifest = {
        'schema_version': 'delta.submission.v1', 'format': 'sparse-diff', 'operation': 'replace',
        'reconstruction_version': 'delta-reconstruct-v1',
        'parent_version_id': '00000000-0000-0000-0000-000000000002',
        'parent_content_hash': 'a' * 64, 'model_config_id': 'tiny',
        'tensors': {'layer.weight': {'indices': 't0.indices', 'values': 't0.values'}},
        'artifacts': [{'name': 'weights.safetensors', 'sha256': hashlib.sha256(payload).hexdigest(), 'size_bytes': len(payload)}],
    }
    listed = dict(manifest, format=['sparse-diff'])
    (tmp_path / 'manifest.json').write_text(json.dumps(listed))
    typed = runner.invoke(main.app, ['validate', str(tmp_path)])
    assert typed.exit_code != 0 and 'Traceback' not in typed.output
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    absent = runner.invoke(main.app, ['validate', str(tmp_path)])
    assert absent.exit_code != 0 and 'Traceback' not in absent.output
    (tmp_path / 'weights.safetensors').write_bytes(b'other')
    mismatch = runner.invoke(main.app, ['validate', str(tmp_path)])
    assert mismatch.exit_code != 0 and 'integrity' in mismatch.output and 'Traceback' not in mismatch.output
    (tmp_path / 'weights.safetensors').write_bytes(payload)
    valid = runner.invoke(main.app, ['validate', str(tmp_path)])
    assert valid.exit_code == 0, valid.output
    assert 'valid' in valid.output


def test_download_rejects_a_traversing_filename(tmp_path):
    bad = dict(artifact(), name='../auth.json')
    with pytest.raises(main.typer.BadParameter, match='filename'):
        main.download_artifact(bad, tmp_path)
    assert not (tmp_path / 'auth.json').exists()


def test_results_reads_the_json_flag(monkeypatch):
    monkeypatch.setattr(main, 'call', lambda *args, **kwargs: {'result': {'validation_score': 1, 'rows': [1]}})
    text = runner.invoke(main.app, ['submission', 'results', 'sub'])
    assert text.exit_code == 0, text.output
    assert 'validation_score: 1' in text.output and 'rows' not in text.output
    raw = runner.invoke(main.app, ['submission', 'results', 'sub', '--json'])
    assert '"rows"' in raw.output


def test_api_option_requires_https_outside_localhost(monkeypatch):
    seen = []

    def client():
        seen.append(main.API)
        raise httpx.ConnectError('stop')

    monkeypatch.setattr(main, 'client', client)
    for origin in ('http://evil.example', 'http://127.0.0.1:80@evil.example', 'http://localhost', 'http://127.0.0.1',
                   'https://user:secret@example.com', 'http://localhost:abc', 'https://example.com/extra'):
        result = runner.invoke(main.app, ['--api', origin, 'whoami'])
        assert result.exit_code != 0 and 'HTTPS' in result.output, origin
    assert seen == []
    local = runner.invoke(main.app, ['--api', 'http://127.0.0.1:9', 'whoami'])
    named = runner.invoke(main.app, ['--api', 'http://localhost:9/', 'whoami'])
    remote = runner.invoke(main.app, ['--api', 'https://example.com/', 'whoami'])
    assert 'HTTPS' not in local.output and 'HTTPS' not in named.output and 'HTTPS' not in remote.output
    assert seen == ['http://127.0.0.1:9', 'http://localhost:9', 'https://example.com']


def test_competition_init_rejects_missing_config_parent_and_validation(tmp_path, monkeypatch):
    downloads = []
    monkeypatch.setattr(main, 'pull_model', lambda *args, **kwargs: downloads.append(args))

    def invoke(board, config=None, version=None):
        def call(method, path, **kwargs):
            if path == 'leaderboard':
                return board
            if path.startswith('evaluation-configs/'):
                return config or {}
            if path.startswith('model-versions/'):
                return version or {'id': 'parent', 'model_id': 'model', 'files': []}
            raise AssertionError(path)
        monkeypatch.setattr(main, 'call', call)
        return runner.invoke(main.app, ['competition', 'init', 'model', '--output', str(tmp_path / 'ws')])

    missing_config = invoke({'config_id': None, 'rows': []})
    assert missing_config.exit_code != 0 and 'no active competition' in missing_config.output
    missing_parent = invoke({'config_id': 'config', 'rows': [{'version_id': 'child', 'parent_id': 'parent'}]})
    assert missing_parent.exit_code != 0 and 'No original checkpoint' in missing_parent.output
    ready = {'config_id': 'config', 'rows': [{'version_id': 'parent', 'parent_id': None}]}
    incomplete = [
        {'id': 'config', 'validation_download': 'https://blob.invalid/v'},
        {'id': 'config', 'validation_download': 'https://blob.invalid/v', 'validation_sha256': 'ab', 'validation_size_bytes': 2},
        {'id': 'config', 'validation_download': 'https://blob.invalid/v', 'validation_sha256': 'ab', 'validation_size_bytes': 0, 'validation_names': ['math']},
        {'id': 'config', 'validation_download': 'https://blob.invalid/v', 'validation_sha256': 'ab', 'validation_size_bytes': 2, 'validation_names': []},
        {'id': 'config', 'validation_sha256': 'ab', 'validation_size_bytes': 2, 'validation_names': ['math']},
    ]
    for config in incomplete:
        result = invoke(ready, config)
        assert result.exit_code != 0 and 'verifiable' in result.output, config
    assert downloads == []
    workspace = tmp_path / 'ws'
    workspace.mkdir()
    (workspace / 'delta-workspace.json').write_text(json.dumps({
        'api': 'https://deltaleague.ai', 'model_id': 'model', 'parent_version_id': 'parent', 'config_id': 'old'}))
    occupied = invoke(ready, {'id': 'config', 'validation_download': 'https://blob.invalid/v', 'validation_sha256': 'a' * 64,
                              'validation_size_bytes': 2, 'validation_names': ['math']})
    assert occupied.exit_code != 0 and 'another' in occupied.output and downloads == []


def test_pull_model_refuses_a_different_parent(tmp_path, monkeypatch):
    output = tmp_path / 'model'
    output.mkdir()
    (output / 'delta-model.json').write_text(json.dumps({'id': 'other'}))
    monkeypatch.setattr(main, 'call', lambda *args, **kwargs: {'id': 'parent', 'model_id': 'model', 'files': []})
    with pytest.raises(main.typer.BadParameter, match='different parent'):
        main.pull_model('parent', output)


@pytest.mark.parametrize('contents', ['[]', 'null', '42', '"token"', '{'])
def test_logout_clears_malformed_credentials(tmp_path, monkeypatch, contents):
    config = tmp_path / 'auth.json'
    config.write_text(contents)
    monkeypatch.setenv('DELTA_CONFIG', str(config))
    result = runner.invoke(main.app, ['logout'])
    assert result.exit_code == 0, result.output
    assert not config.exists()
    assert 'could not be revoked' in result.output


@pytest.mark.parametrize('url,digest', [('ftp://example.org/a', 'a' * 64), ('https://example.org/a', 'invalid'), ('https://[invalid/a', 'a' * 64)])
def test_public_validation_rejects_invalid_download_metadata(url, digest):
    with pytest.raises(main.typer.BadParameter):
        main.public_validation({'validation_download': url, 'validation_sha256': digest,
                                'validation_size_bytes': 1, 'validation_names': ['math']})


def test_login_lost_token_response_explains_restart(monkeypatch, tmp_path):
    polls = []
    def handle(request):
        if request.url.path == '/api/v1/cli-auth':
            return httpx.Response(200, json={'id': 'device', 'secret': 'secret', 'verification_url': 'https://example.org', 'interval': 0})
        polls.append(request)
        if len(polls) == 1:
            raise httpx.ReadError('lost token response', request=request)
        return httpx.Response(400, json={'detail': 'Authorization expired or consumed'})
    monkeypatch.setattr(main, 'client', lambda: httpx.Client(base_url='https://example.org', transport=httpx.MockTransport(handle)))
    monkeypatch.setattr(main.time, 'sleep', lambda _: None)
    monkeypatch.setenv('DELTA_CONFIG', str(tmp_path / 'auth.json'))
    result = runner.invoke(main.app, ['login', '--no-browser'])
    assert result.exit_code == 1
    assert len(polls) == 2
    assert 'Run `delta login` again' in result.output
    assert not (tmp_path / 'auth.json').exists()
