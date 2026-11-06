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
    occupied = invoke(ready, {'id': 'config', 'validation_download': 'https://blob.invalid/v', 'validation_sha256': 'ab',
                              'validation_size_bytes': 2, 'validation_names': ['math']})
    assert occupied.exit_code != 0 and 'another' in occupied.output and downloads == []


def test_pull_model_refuses_a_different_parent(tmp_path, monkeypatch):
    output = tmp_path / 'model'
    output.mkdir()
    (output / 'delta-model.json').write_text(json.dumps({'id': 'other'}))
    monkeypatch.setattr(main, 'call', lambda *args, **kwargs: {'id': 'parent', 'model_id': 'model', 'files': []})
    with pytest.raises(main.typer.BadParameter, match='different parent'):
        main.pull_model('parent', output)


def test_api_connection_failure_does_not_print_credentials(monkeypatch):
    def fail():
        raise httpx.ConnectError('https://user:secret@server.invalid')
    monkeypatch.setattr(main, 'client', fail)
    result = runner.invoke(main.app, ['competition', 'list'])
    assert result.exit_code == 1
    assert 'secret' not in result.output
    assert 'Could not reach' in result.output
