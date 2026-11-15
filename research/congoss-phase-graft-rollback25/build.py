"""Build a partial rollback of the two phase-gated neurons on the current leader."""
import hashlib
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

MODELS, ROOT = map(Path, sys.argv[1:])
EXPECTED = {
    "leader": "fb984a19-6262-48db-bd14-a5d68e246ea6",
    "grafted": "fdad0374-29f3-49c6-911d-d0ce9531f7a6",
    "before": "ac147fa6-6755-4f1b-a6b2-697aef3552db",
}
identities = {}
for name, expected in EXPECTED.items():
    identity = json.loads((MODELS / name / "model/delta-model.json").read_text())
    assert identity["id"] == expected
    identities[name] = identity

payload = {}
mapping = {}
counts = {}
neurons = None
with (
    safe_open(MODELS / "leader/model/model.safetensors", framework="pt") as leader,
    safe_open(MODELS / "grafted/model/model.safetensors", framework="pt") as grafted,
    safe_open(MODELS / "before/model/model.safetensors", framework="pt") as before,
):
    for part in ("gate", "up", "down"):
        key = f"model.language_model.layers.23.mlp.{part}_proj.weight"
        current = leader.get_tensor(key)
        reference = before.get_tensor(key)
        donor = grafted.get_tensor(key)
        axis = 0 if part == "down" else 1
        affected = torch.where((reference != donor).any(dim=axis))[0]
        assert affected.numel() == 2, (part, affected.tolist())
        assert neurons is None or torch.equal(neurons, affected)
        neurons = affected
        candidate = current.clone()
        if part == "down":
            assert torch.equal(current[:, neurons], donor[:, neurons]), "Leader changed graft columns"
            candidate[:, neurons] = torch.lerp(
                current[:, neurons].float(), reference[:, neurons].float(), 0.25
            ).to(current.dtype)
        else:
            assert torch.equal(current[neurons], donor[neurons]), "Leader changed graft rows"
            candidate[neurons] = torch.lerp(
                current[neurons].float(), reference[neurons].float(), 0.25
            ).to(current.dtype)
        indices = torch.where((candidate != current).flatten())[0]
        assert indices.numel() > 0
        payload[f"{part}.indices"] = indices
        payload[f"{part}.values"] = candidate.flatten()[indices]
        mapping[key] = {"indices": f"{part}.indices", "values": f"{part}.values"}
        counts[key] = indices.numel()
        print(part, "neurons", neurons.tolist(), "changed", indices.numel(), flush=True)

output = ROOT / "submission"
output.mkdir(parents=True, exist_ok=True)
artifact = output / "weights.safetensors"
save_file(payload, artifact)
digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
identity = identities["leader"]
manifest = {
    "schema_version": "delta.submission.v1",
    "reconstruction_version": "delta-reconstruct-v1",
    "model_config_id": identity["model_id"],
    "parent_version_id": identity["id"],
    "parent_content_hash": identity["content_hash"],
    "format": "sparse-diff",
    "operation": "replace",
    "artifacts": [{"name": artifact.name, "sha256": digest, "size_bytes": artifact.stat().st_size}],
    "tensors": mapping,
}
(output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
(ROOT / "build-report.json").write_text(
    json.dumps({"neurons": neurons.tolist(), "rollback": 0.25, "changed": counts,
                "sources": identities, "artifact_sha256": digest}, indent=2), encoding="utf-8"
)
print("Artifact", digest, flush=True)
