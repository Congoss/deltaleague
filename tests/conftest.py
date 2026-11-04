import json

import pytest
import torch
from safetensors.torch import save_file

from delta_cli import local
from delta_cli.artifacts import file_hash


@pytest.fixture
def validation_data():
    return {'a': [{'id': 'a1', 'prompt': 'A', 'answer': 'yes', 'samples': 2}],
            'b': [{'id': 'b1', 'prompt': 'B', 'answer': 'yes'}, {'id': 'b2', 'prompt': 'C', 'answer': 'yes'}]}


@pytest.fixture
def workspace(tmp_path, validation_data):
    model = tmp_path / 'model'
    model.mkdir()
    save_file({'layer.weight': torch.tensor([1., 2.])}, str(model / 'model.safetensors'))
    (model / 'config.json').write_text('{}')
    metadata = {'id': '00000000-0000-0000-0000-000000000002', 'model_id': 'tiny', 'content_hash': 'a' * 64, 'files': [
        {'name': p.name, 'size_bytes': p.stat().st_size, 'sha256': file_hash(p)} for p in model.iterdir()]}
    (model / 'delta-model.json').write_text(json.dumps(metadata))
    data = validation_data
    (tmp_path / 'validation.json').write_text(json.dumps(data))
    config = {'id': 'tiny-v1', 'model_id': 'tiny', 'generation': {'max_tokens': 20, 'temperature': .6},
              'validation_names': list(data), 'validation_sha256': file_hash(tmp_path / 'validation.json')}
    (tmp_path / 'evaluation.json').write_text(json.dumps(config))
    (tmp_path / 'delta-workspace.json').write_text(json.dumps({'model_id': 'tiny', 'config_id': 'tiny-v1', 'parent_version_id': '00000000-0000-0000-0000-000000000002'}))
    return tmp_path, metadata, config


@pytest.fixture
def fake_runner(monkeypatch):
    def install(answer, seen=None):
        class FakeRunner:
            details = {'device': 'cpu'}

            def __init__(self, *args):
                pass

            def generate(self, prompt, generation, seed):
                if seen is not None:
                    seen.append(generation)
                text = answer(generation)
                return {'text': text, 'tokens': 2, 'generation_seconds': .1, 'truncated': text == 'no'}
        monkeypatch.setattr(local, 'Runner', FakeRunner)
    return install
