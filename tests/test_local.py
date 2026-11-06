import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from typer.testing import CliRunner

from delta_cli import local, main
from delta_cli.artifacts import build_sparse, file_hash


def test_patch_build_rejects_a_modified_parent_and_an_existing_output(workspace):
    root, _, _ = workspace
    parent = root / 'model'
    modified = root / 'edited.safetensors'
    save_file({'layer.weight': torch.tensor([1., 9.])}, str(modified))
    config = parent / 'config.json'
    original = config.read_text()
    config.write_text('changed')
    output = root / 'built'
    args = ['patch', 'build', '--parent', str(parent), '--modified', str(modified), '--output', str(output)]
    result = CliRunner().invoke(main.app, args)
    assert result.exit_code != 0
    assert 'changed' in result.output and 'Traceback' not in result.output
    assert not output.exists()
    config.write_text(original)
    result = CliRunner().invoke(main.app, args)
    assert result.exit_code == 0, result.output
    assert (output / 'manifest.json').exists()
    again = CliRunner().invoke(main.app, args)
    assert again.exit_code != 0 and 'already exists' in again.output
    previous = (output / 'manifest.json').read_bytes()
    (output / 'stale.txt').write_text('keep')
    save_file({'layer.weight': torch.tensor([1., 2.])}, str(modified))
    failed = CliRunner().invoke(main.app, args + ['--force'])
    assert failed.exit_code != 0 and 'Traceback' not in failed.output
    assert (output / 'manifest.json').read_bytes() == previous
    assert (output / 'stale.txt').read_text() == 'keep'
    save_file({'layer.weight': torch.tensor([1., 9.])}, str(modified))
    forced = CliRunner().invoke(main.app, args + ['--force'])
    assert forced.exit_code == 0, forced.output
    assert not (output / 'stale.txt').exists()


def test_parent_checkpoint_is_hashed_once(workspace, monkeypatch):
    root, metadata, _ = workspace
    calls = []
    real = file_hash

    def wrapped(path):
        calls.append(Path(path).name)
        return real(path)

    monkeypatch.setattr(local, 'file_hash', wrapped)
    monkeypatch.setattr('delta_cli.artifacts.file_hash', wrapped)
    with local.model_directory(root, metadata):
        pass
    assert calls.count('model.safetensors') == 1


def test_patch_overlay_reconstructs_without_mutating_parent(workspace):
    root, metadata, _ = workspace
    modified = root / 'edited.safetensors'
    save_file({'layer.weight': torch.tensor([1., 9.])}, str(modified))
    build_sparse(root / 'model/model.safetensors', modified, root / 'patch', '00000000-0000-0000-0000-000000000002', 'a' * 64, 'tiny')
    original = file_hash(root / 'model/model.safetensors')
    with local.model_directory(root, metadata, patch=root / 'patch') as (directory, identity):
        assert torch.equal(load_file(str(directory / 'model.safetensors'))['layer.weight'], torch.tensor([1., 9.]))
        assert identity['kind'] == 'patch'
        assert (directory / 'config.json').read_text() == '{}'
    assert not directory.exists()
    assert file_hash(root / 'model/model.safetensors') == original
    manifest = root / 'patch/manifest.json'
    data = json.loads(manifest.read_text())
    data['parent_version_id'] = '00000000-0000-0000-0000-000000000003'
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='parent'):
        with local.model_directory(root, metadata, patch=root / 'patch'):
            pass


def test_public_scoring_averages_samples_then_problems_then_benchmarks(workspace):
    root, _, config = workspace
    tasks, names, full = local.public_examples(root, config)
    assert [t['seed'] for t in tasks] == [1000, 1001, 1000, 1000]
    answers = iter(['yes', 'no', 'yes', 'yes'])
    runner = SimpleNamespace(generate=lambda *a: {'text': next(answers), 'tokens': 3, 'generation_seconds': 1, 'truncated': False})
    stream = io.StringIO()
    report = local.evaluate(runner, tasks, config['generation'], stream, lambda *args: None)
    assert full and names == ['a', 'b']
    assert report['validation_benchmarks'] == {'a': 50, 'b': 100}
    assert report['validation_score'] == 75
    assert len(stream.getvalue().splitlines()) == 4
    assert local.public_examples(root, config, limit=1)[2] is False
    (root / 'validation.json').write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        local.public_examples(root, config)


def test_thinking_and_choice_scoring():
    assert local.score_answer({'answer': 'yes'}, 'wrong reasoning</think> yes') == 1
    assert local.score_answer({'answer': 'B', 'scorer': 'choice'}, r'Reasoning. \boxed{B}') == 1
    assert local.score_answer({'answer': '4', 'scorer': 'math'}, r'\boxed{4}') == 1


def test_device_selection_and_unavailable_device():
    fake = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False), backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)))
    assert local.select_device(fake, 'auto') == 'mps'
    assert local.select_device(fake, 'cpu') == 'cpu'
    with pytest.raises(ValueError, match='CUDA'):
        local.select_device(fake, 'cuda')


def test_cli_evaluation_report_and_existing_output_protection(workspace, monkeypatch):
    root, _, _ = workspace
    class FakeRunner:
        details = {'device': 'cpu'}
        def __init__(self, *args):
            pass
        def generate(self, *args):
            return {'text': 'yes', 'tokens': 2, 'generation_seconds': .1, 'truncated': False}
    monkeypatch.setattr(local, 'Runner', FakeRunner)
    runner = CliRunner()
    output = root / 'result.json'
    args = ['evaluate', '--workspace', str(root), '--limit', '1', '--max-new-tokens', '4', '--output', str(output)]
    result = runner.invoke(main.app, args)
    assert result.exit_code == 0, result.output
    report = json.loads(output.read_text())
    assert report['local_only'] and report['generation_overridden']
    assert report['full_validation_set'] is False
    assert report['validation_score'] == 100
    assert report['source_version_id'] == '00000000-0000-0000-0000-000000000002'
    original = output.read_bytes()
    result = runner.invoke(main.app, args)
    assert result.exit_code == 1 and 'already exists' in result.output
    assert output.read_bytes() == original


def test_leaderboard_version_model_mismatch_is_rejected(workspace, monkeypatch):
    root, _, _ = workspace
    monkeypatch.setattr(main, 'call', lambda *a, **kw: {'model_id': 'other'})
    monkeypatch.setattr(main, 'pull_model', lambda *a: pytest.fail('Should not download another model'))
    result = CliRunner().invoke(main.app, ['run', 'Hi', '--workspace', str(root), '--version', '00000000-0000-0000-0000-000000000001'])
    assert result.exit_code == 1 and 'another model' in result.output


def test_leaderboard_version_download_is_used_for_inference(workspace, monkeypatch):
    root, metadata, _ = workspace
    selected = '00000000-0000-0000-0000-000000000009'
    monkeypatch.setattr(main, 'call', lambda *a, **kw: {'model_id': 'tiny'})
    def pull(version_id, directory):
        assert version_id == selected
        directory.mkdir(parents=True)
        for artifact in metadata['files']:
            (directory / artifact['name']).write_bytes((root / 'model' / artifact['name']).read_bytes())
        return {**metadata, 'id': selected}
    monkeypatch.setattr(main, 'pull_model', pull)
    class FakeRunner:
        details = {'device': 'cpu'}
        def __init__(self, path, device):
            assert path == root / '.delta/models' / selected
        def generate(self, *args):
            return {'text': 'yes', 'tokens': 1, 'truncated': False}
    monkeypatch.setattr(local, 'Runner', FakeRunner)
    result = CliRunner().invoke(main.app, ['run', 'Hi', '--workspace', str(root), '--version', selected, '--json'])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)['source_version_id'] == selected


def test_patch_build_requires_checkpoint_in_inventory(workspace, monkeypatch):
    root, _, _ = workspace
    parent = root / 'model'
    path = parent / 'delta-model.json'
    metadata = json.loads(path.read_text())
    metadata['files'] = []
    path.write_text(json.dumps(metadata))
    monkeypatch.setattr(main, 'build_sparse', lambda *args: pytest.fail('Unverified checkpoint used'))
    result = CliRunner().invoke(main.app, ['patch', 'build', '--parent', str(parent), '--modified', str(root / 'edited.safetensors'), '--output', str(root / 'built')])
    assert result.exit_code != 0
    assert 'inventory' in result.output
    assert not (root / 'built').exists()


@pytest.mark.parametrize('fail_publish', [False, True])
def test_publish_preserves_unrelated_backup_and_rolls_back(tmp_path, monkeypatch, fail_publish):
    output, staging, unrelated = (tmp_path / name for name in ['submission', 'staging', '.submission.replaced'])
    for directory in (output, staging, unrelated):
        directory.mkdir()
    (output / 'old').write_text('old')
    (staging / 'new').write_text('new')
    (unrelated / 'keep').write_text('keep')
    rename = Path.rename
    def failing_rename(path, target):
        if path == staging:
            raise OSError('cannot publish')
        return rename(path, target)
    if fail_publish:
        monkeypatch.setattr(Path, 'rename', failing_rename)
        with pytest.raises(OSError, match='cannot publish'):
            main.publish_directory(staging, output)
        assert (output / 'old').read_text() == 'old'
    else:
        main.publish_directory(staging, output)
        assert (output / 'new').read_text() == 'new'
    assert (unrelated / 'keep').read_text() == 'keep'
    assert not list(tmp_path.glob('.submission.backup-*'))
