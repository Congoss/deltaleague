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
    (tmp_path / 'manifest.json').write_text('{}')
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
    (tmp_path / 'manifest.json').write_text('{}')
    (tmp_path / 'model.safetensors').write_bytes(b'weights')
    state = {'upload_id': 'expired', 'api': main.API,
             'fingerprint': hashlib.sha256(b'{}').hexdigest(),
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
