"""Build a sparse patch using seven half-scale embeddings from the original model."""
import hashlib
import json
from pathlib import Path
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

original, parent, output = map(Path, sys.argv[1:])
key = "model.language_model.embed_tokens.weight"
token_ids = sorted([13784, 13428, 77264, 85152, 11158, 3655, 84485])
with (
    safe_open(original / "model.safetensors", framework="pt") as base,
    safe_open(parent / "model.safetensors", framework="pt") as leader,
):
    source = base.get_slice(key)
    width = source.get_shape()[1]
    old = torch.cat([leader.get_slice(key)[i:i+1] for i in token_ids])
    rows = torch.cat([source[i:i+1] for i in token_ids])
    replacement = (rows.float() * 0.5).to(old.dtype)
    assert (replacement.float().norm(dim=1) < old.float().norm(dim=1)).all()
    changed = (old != replacement).flatten()
    offsets = torch.tensor(token_ids).unsqueeze(1) * width + torch.arange(width)
    indices = offsets.flatten()[changed]
    values = replacement.flatten()[changed]

identity = json.loads((parent / "delta-model.json").read_text(encoding="utf-8"))
output.mkdir(exist_ok=False)
weights = output / "weights.safetensors"
save_file({"indices": indices, "values": values}, weights)
manifest = {
    "schema_version": "delta.submission.v1",
    "reconstruction_version": "delta-reconstruct-v1",
    "model_config_id": "qwen35-08b-v1",
    "parent_version_id": identity["id"],
    "parent_content_hash": identity["content_hash"],
    "format": "sparse-diff",
    "operation": "replace",
    "artifacts": [{"name": weights.name, "sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
                   "size_bytes": weights.stat().st_size}],
    "tensors": {key: {"indices": "indices", "values": "values"}},
}
(output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(f"{len(indices)} changed entries across {len(token_ids)} rows")
