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
    assert retitled.exit_code == 0, retitled.output
    assert [path for path, _ in calls].count('upload-sessions') == 1
    posted = [kwargs for path, kwargs in calls if path == 'submissions']
    assert posted[2]['json']['title'] == 'Second'
    assert posted[2]['headers']['Idempotency-Key'] != state['idempotency_key']
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
    assert not (root / '.delta-upload.json').exists()
    monkeypatch.setattr(main.httpx, 'put', lambda url, **kwargs: (_ for _ in ()).throw(httpx.ConnectError(url)))
    failed = runner.invoke(main.app, ['submit', str(root)])
    assert failed.exit_code == 1
    assert 'secret' not in failed.output and 'Traceback' not in failed.output
    assert calls == ['upload-sessions', 'upload-sessions']


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
