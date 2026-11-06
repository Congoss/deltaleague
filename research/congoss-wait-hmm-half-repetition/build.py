"""Combine the verified Wait/Hmm artifact with two published half-dose tensors."""
import hashlib
import json
from pathlib import Path
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

parent, control, donor, output = map(Path, sys.argv[1:])
manifest = json.loads((control / "manifest.json").read_text())
identity = json.loads((parent / "delta-model.json").read_text())
assert manifest["parent_version_id"] == identity["id"]
assert manifest["parent_content_hash"] == identity["content_hash"]
assert json.loads((donor / "delta-model.json").read_text())["id"] == "344c8376-66a1-413b-abca-da2b5b986044"
artifact = control / "weights.safetensors"
assert hashlib.sha256(artifact.read_bytes()).hexdigest() == manifest["artifacts"][0]["sha256"]
payload = load_file(artifact)
with (
    safe_open(parent / "model.safetensors", framework="pt") as base,
    safe_open(donor / "model.safetensors", framework="pt") as edited,
):
    for layer in (15, 19):
        key = f"model.language_model.layers.{layer}.mlp.down_proj.weight"
        original = base.get_tensor(key)
        replacement = edited.get_tensor(key)
        assert original.shape == replacement.shape and original.dtype == replacement.dtype
        indices = torch.where((original != replacement).flatten())[0]
        index_key, value_key = f"l{layer}.indices", f"l{layer}.values"
        payload[index_key] = indices
        payload[value_key] = replacement.flatten()[indices]
        manifest["tensors"][key] = {"indices": index_key, "values": value_key}
        print(layer, indices.numel())

output.mkdir()
artifact = output / "weights.safetensors"
save_file(payload, artifact)
manifest["artifacts"] = [{"name": artifact.name, "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                          "size_bytes": artifact.stat().st_size}]
(output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(manifest["artifacts"][0]["sha256"])
