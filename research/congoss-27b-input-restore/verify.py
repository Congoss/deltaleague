"""Independently read and verify the small safetensors patch."""
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

root = Path(__file__).resolve().parent
report = json.loads((root / "build-report.json").read_text())
manifest = json.loads((root / "submission/manifest.json").read_text())
path = root / "submission/weights.safetensors"
assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["artifacts"][0]["sha256"]
assert set(manifest["tensors"]) == {"model.language_model.embed_tokens.weight"}
with safe_open(path, framework="pt") as payload:
    assert set(payload.keys()) == {"indices", "values"}
    indices = payload.get_tensor("indices")
    values = payload.get_tensor("values")
    assert indices.dtype == torch.int64 and values.dtype == torch.bfloat16
    assert torch.isfinite(values).all()
    assert indices.shape == values.shape == (40960,)
    assert torch.all(indices[1:] > indices[:-1])
    raw = bytes(values.view(torch.uint8).tolist())
    for offset, row in enumerate(report["rows"]):
        begin, end = offset * 5120 * 2, (offset + 1) * 5120 * 2
        assert hashlib.sha256(raw[begin:end]).hexdigest() == row["original_sha256"]
        assert torch.equal(indices[offset*5120:(offset+1)*5120],
                           torch.arange(row["token_id"]*5120, (row["token_id"]+1)*5120, dtype=torch.int64))
print("All 8 restored original rows verified with an independent safetensors reader")
