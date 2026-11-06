from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from delta_cli.artifacts import build_sparse, reconstruct, validate_manifest

PARENT = '00000000-0000-0000-0000-000000000002'


def manifest(**overrides):
    value = {
        'schema_version': 'delta.submission.v1', 'format': 'sparse-diff', 'operation': 'replace',
        'reconstruction_version': 'delta-reconstruct-v1', 'parent_version_id': PARENT,
        'parent_content_hash': 'a' * 64, 'model_config_id': 'tiny',
        'tensors': {'edit': {'indices': 'i', 'values': 'v'}},
        'artifacts': [{'name': 'weights.safetensors', 'sha256': 'b' * 64, 'size_bytes': 1}],
    }
    value.update(overrides)
    return value


def test_validate_manifest_rejects_malformed_manifests():
    validate_manifest(manifest())
    cases = [
        (dict(manifest(), schema_version='nope'), 'unsupported_schema'),
        (dict(manifest(), format='qlora'), 'unsupported_format'),
        (dict(manifest(), format=['sparse-diff']), 'unsupported_format'),
        (dict(manifest(), reconstruction_version='v0'), 'unsupported_reconstruction'),
        (dict(manifest(), parent_version_id='parent'), 'invalid_parent_version'),
        (dict(manifest(), parent_content_hash='abc'), 'invalid_parent_hash'),
        (dict(manifest(), model_config_id=''), 'invalid_model_config'),
        (dict(manifest(), artifacts=[{'name': 'w.bin', 'sha256': 'b' * 64, 'size_bytes': 1}]), 'expected_weights_safetensors'),
        (dict(manifest(), artifacts=[{'name': 'weights.safetensors', 'sha256': 'zz', 'size_bytes': 1}]), 'invalid_hash'),
        (dict(manifest(), artifacts=[{'name': 'weights.safetensors', 'sha256': 'b' * 64, 'size_bytes': 0}]), 'invalid_size'),
        (dict(manifest(), artifacts=[{'name': 'weights.safetensors', 'sha256': 'b' * 64, 'size_bytes': True}]), 'invalid_size'),
        (dict(manifest(), operation='scale'), 'unsupported_operation'),
        (dict(manifest(), operation=['replace']), 'unsupported_operation'),
        (dict(manifest(), tensors=[]), 'invalid_tensor_mapping'),
        (dict(manifest(), tensors={'edit': {'indices': 1, 'values': 'v'}}), 'invalid_tensor_mapping'),
        (dict(manifest(), tensors={'x' * 1025: {'indices': 'i', 'values': 'v'}}), 'invalid_tensor_name'),
        (dict(manifest(), tensors={f't{i}': {'indices': 'i', 'values': 'v'} for i in range(10001)}), 'too_many_tensors'),
        ([], 'invalid_manifest'),
    ]
    for value, message in cases:
        with pytest.raises(ValueError, match=message):
            validate_manifest(value)


def test_validate_manifest_requires_a_finite_lora_alpha():
    spec = {'a': 'A', 'b': 'B', 'alpha': 1.5}
    validate_manifest(manifest(format='lora', tensors={'edit': spec}))
    for alpha in (None, [], float('nan'), float('inf'), True, 10 ** 400):
        with pytest.raises(ValueError, match='invalid_lora_alpha'):
            validate_manifest(manifest(format='lora', tensors={'edit': {**spec, 'alpha': alpha}}))


def store(tmp_path, name, tensors):
    path = tmp_path / name
    save_file(tensors, str(path))
    return path


def test_reconstruct_lora_full_and_add_operations(tmp_path, monkeypatch):
    parent = store(tmp_path, 'parent.safetensors', {
        'keep': torch.tensor([4.]), 'edit': torch.tensor([1., 2.]),
        'lora': torch.tensor([[1., 0.], [0., 1.]]),
    })
    original = parent.read_bytes()
    sparse = store(tmp_path, 'sparse.safetensors', {'i': torch.tensor([1], dtype=torch.int64), 'v': torch.tensor([9.])})
    pointers = {}
    import safetensors.torch as st
    real = st.load_file

    def wrapped(path, *args, **kwargs):
        tensors = real(path, *args, **kwargs)
        if Path(path).name == 'parent.safetensors':
            pointers.update({name: tensor.data_ptr() for name, tensor in tensors.items()})
        return tensors

    monkeypatch.setattr(st, 'load_file', wrapped)
    replaced = reconstruct(parent, sparse, manifest())
    assert replaced['edit'].tolist() == [1., 9.]
    assert replaced['keep'].data_ptr() == pointers['keep']
    assert replaced['edit'].data_ptr() != pointers['edit']
    assert parent.read_bytes() == original

    added = reconstruct(parent, sparse, manifest(operation='add'))
    assert added['edit'].tolist() == [1., 11.]

    dense = store(tmp_path, 'dense.safetensors', {'v': torch.tensor([3., 4.])})
    dense_manifest = manifest(format='dense-diff', tensors={'edit': {'values': 'v'}})
    assert reconstruct(parent, dense, dense_manifest)['edit'].tolist() == [3., 4.]
    assert reconstruct(parent, dense, manifest(format='dense-diff', operation='add', tensors={'edit': {'values': 'v'}}))['edit'].tolist() == [4., 6.]

    full_patch = store(tmp_path, 'full.safetensors', {
        'keep': torch.tensor([4.]), 'edit': torch.tensor([8., 9.]), 'lora': torch.tensor([[1., 0.], [0., 1.]]),
    })
    full = reconstruct(parent, full_patch, manifest(format='full', tensors={}))
    assert full['edit'].tolist() == [8., 9.]
    assert full['keep'].tolist() == [4.]

    low = torch.tensor([[1., 0.]])
    high = torch.tensor([[0.], [2.]])
    lora = store(tmp_path, 'lora.safetensors', {'A': low, 'B': high})
    lora_manifest = manifest(format='lora', tensors={'lora': {'a': 'A', 'b': 'B', 'alpha': 1}})
    assert torch.equal(reconstruct(parent, lora, lora_manifest)['lora'], torch.tensor([[1., 0.], [2., 1.]]))

    bf = store(tmp_path, 'bf-parent.safetensors', {'edit': torch.tensor([1., 2.], dtype=torch.bfloat16)})
    bf_patch = store(tmp_path, 'bf-patch.safetensors', {'i': torch.tensor([0], dtype=torch.int64), 'v': torch.tensor([3.5])})
    bf_result = reconstruct(bf, bf_patch, manifest())
    expected = torch.tensor([1., 2.], dtype=torch.bfloat16).clone()
    expected.view(-1)[0] = torch.tensor(3.5, dtype=torch.bfloat16)
    assert torch.equal(bf_result['edit'], expected)

    with pytest.raises(ValueError, match='frozen_tensor'):
        reconstruct(parent, sparse, manifest(), allowed_prefix='keep.')
    with pytest.raises(ValueError, match='unknown_tensor'):
        reconstruct(parent, sparse, manifest(tensors={'missing': {'indices': 'i', 'values': 'v'}}))
    extra = store(tmp_path, 'extra.safetensors', {'i': torch.tensor([0], dtype=torch.int64), 'v': torch.tensor([1.]), 'extra': torch.tensor([1.])})
    with pytest.raises(ValueError, match='unlisted_tensors'):
        reconstruct(parent, extra, manifest())
    bad_rank = store(tmp_path, 'rank.safetensors', {'A': torch.tensor([1., 0.]), 'B': high})
    with pytest.raises(ValueError, match='invalid_lora_rank'):
        reconstruct(parent, bad_rank, lora_manifest)
    poisoned = store(tmp_path, 'nan.safetensors', {'i': torch.tensor([0], dtype=torch.int64), 'v': torch.tensor([float('nan')])})
    with pytest.raises(ValueError, match='nonfinite_patch'):
        reconstruct(parent, poisoned, manifest())
    with pytest.raises(ValueError, match='empty_patch'):
        reconstruct(parent, sparse, manifest(tensors={}))
    with pytest.raises(ValueError, match='invalid_sparse_indices'):
        reconstruct(parent, store(tmp_path, 'oob.safetensors', {'i': torch.tensor([5], dtype=torch.int64), 'v': torch.tensor([1.])}), manifest())


def test_sparse_replace_keeps_precision_the_parent_dtype_can_hold(tmp_path):
    precise = 1.0 + 2 ** -40
    parent = store(tmp_path, 'wide-parent.safetensors', {'edit': torch.zeros(2, dtype=torch.float64)})
    patch = store(tmp_path, 'wide-patch.safetensors', {'i': torch.tensor([0], dtype=torch.int64), 'v': torch.tensor([precise], dtype=torch.float64)})
    result = reconstruct(parent, patch, manifest())
    assert result['edit'].dtype == torch.float64
    assert result['edit'][0].item() == precise


def test_lora_scales_by_alpha_over_rank(tmp_path):
    parent = store(tmp_path, 'lora-parent.safetensors', {'lora': torch.eye(2)})
    patch = store(tmp_path, 'lora-scale.safetensors', {
        'A': torch.tensor([[1., 0.], [0., 1.]]),
        'B': torch.tensor([[2., 0.], [0., 0.]]),
    })
    spec = manifest(format='lora', tensors={'lora': {'a': 'A', 'b': 'B', 'alpha': 4}})
    assert torch.equal(reconstruct(parent, patch, spec)['lora'], torch.tensor([[5., 0.], [0., 1.]]))


def test_build_sparse_rejects_inventory_shape_and_empty_patch(tmp_path):
    parent = store(tmp_path, 'parent.safetensors', {'w': torch.tensor([1., 2.])})

    def build(modified, folder):
        build_sparse(parent, modified, tmp_path / folder, PARENT, 'a' * 64, 'tiny')

    with pytest.raises(ValueError, match='empty_patch'):
        build(store(tmp_path, 'same.safetensors', {'w': torch.tensor([1., 2.])}), 'empty')
    with pytest.raises(ValueError, match='tensor_inventory'):
        build(store(tmp_path, 'other.safetensors', {'v': torch.tensor([1., 2.])}), 'inventory')
    with pytest.raises(ValueError, match='tensor_shape_or_dtype'):
        build(store(tmp_path, 'shaped.safetensors', {'w': torch.tensor([[1., 2.]])}), 'shape')
    with pytest.raises(ValueError, match='tensor_shape_or_dtype'):
        build(store(tmp_path, 'typed.safetensors', {'w': torch.tensor([1., 2.], dtype=torch.float64)}), 'dtype')
    built = tmp_path / 'built'
    manifest_out = build_sparse(parent, store(tmp_path, 'changed.safetensors', {'w': torch.tensor([1., 9.])}), built, PARENT, 'a' * 64, 'tiny')
    assert set(manifest_out['tensors']) == {'w'}
    assert (built / 'weights.safetensors').exists()
