"""Roll back ten percent of the late-block GRPO update."""
import hashlib
import json
from pathlib import Path
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

leader, pre_grpo, output = map(Path, sys.argv[1:])
identity = json.loads((leader / "delta-model.json").read_text())
assert identity["id"] == "79be46a9-cba5-4535-9076-1d97ba749b29"
assert json.loads((pre_grpo / "delta-model.json").read_text())["id"] == "344c8376-66a1-413b-abca-da2b5b986044"
payload = {}
mapping = {}
with (
    safe_open(leader / "model.safetensors", framework="pt") as trained,
    safe_open(pre_grpo / "model.safetensors", framework="pt") as baseline,
):
    for key in trained.keys():
        if not key.startswith(("model.language_model.layers.22.", "model.language_model.layers.23.")):
            continue
        current, prior = trained.get_tensor(key), baseline.get_tensor(key)
        assert current.shape == prior.shape and current.dtype == prior.dtype
        edited = torch.lerp(current.float(), prior.float(), 0.10).to(current.dtype)
        if torch.equal(current, edited):
            continue
        name = f"t{len(mapping)}.values"
        payload[name] = edited
        mapping[key] = {"values": name}
        print(key, torch.count_nonzero(current != edited).item(), flush=True)

assert mapping, "BF16 rounding removed the entire rollback"
output.mkdir(exist_ok=True)
artifact = output / "weights.safetensors"
save_file(payload, artifact)
with artifact.open("rb") as file:
    digest = hashlib.file_digest(file, "sha256").hexdigest()
manifest = {
    "schema_version": "delta.submission.v1",
    "reconstruction_version": "delta-reconstruct-v1",
    "model_config_id": identity["model_id"],
    "parent_version_id": identity["id"],
    "parent_content_hash": identity["content_hash"],
    "format": "dense-diff",
    "operation": "replace",
    "artifacts": [{"name": artifact.name, "sha256": digest, "size_bytes": artifact.stat().st_size}],
    "tensors": mapping,
}
(output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(len(mapping), digest)
