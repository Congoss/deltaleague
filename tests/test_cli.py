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


def test_validate_reports_malformed_submissions_without_a_traceback(tmp_path):
    missing = runner.invoke(main.app, ['validate', str(tmp_path)])
    assert missing.exit_code != 0 and 'Traceback' not in missing.output and 'manifest.json' in missing.output
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


def test_api_connection_failure_does_not_print_credentials(monkeypatch):
    def fail():
        raise httpx.ConnectError('https://user:secret@server.invalid')
    monkeypatch.setattr(main, 'client', fail)
    result = runner.invoke(main.app, ['competition', 'list'])
    assert result.exit_code == 1
    assert 'secret' not in result.output
    assert 'Could not reach' in result.output
