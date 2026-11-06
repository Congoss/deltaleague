"""Build the validated sparse final RMSNorm edit from a registered parent."""
import hashlib
import json
from pathlib import Path
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

model, output = map(Path, sys.argv[1:])
identity = json.loads((model / "delta-model.json").read_text())
assert identity["id"] == "344c8376-66a1-413b-abca-da2b5b986044"
key = "model.language_model.norm.weight"
with safe_open(model / "model.safetensors", framework="pt") as weights:
    original = weights.get_tensor(key)
    candidate = ((original.float() + 1) * 1.05 - 1).to(original.dtype)
    indices = torch.where((original != candidate).flatten())[0]

output.mkdir()
artifact = output / "weights.safetensors"
save_file({"indices": indices, "values": candidate.flatten()[indices]}, artifact)
manifest = {
    "schema_version": "delta.submission.v1",
    "reconstruction_version": "delta-reconstruct-v1",
    "model_config_id": identity["model_id"],
    "parent_version_id": identity["id"],
    "parent_content_hash": identity["content_hash"],
    "format": "sparse-diff",
    "operation": "replace",
    "artifacts": [{"name": artifact.name, "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                   "size_bytes": artifact.stat().st_size}],
    "tensors": {key: {"indices": "indices", "values": "values"}},
}
(output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
