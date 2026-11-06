"""Portable weight artifact contract. This module never imports the platform."""
import hashlib
import json
import math
import re
import uuid
from pathlib import Path


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def verify_artifact_bytes(directory, files, skip_names=()):
    """Check recorded sizes and SHA-256 digests. Return the digests that were computed."""
    directory = Path(directory)
    if not isinstance(files, list):
        raise ValueError("invalid_artifact_inventory")
    digests = {}
    for artifact in files:
        if not isinstance(artifact, dict):
            raise ValueError("invalid_artifact_inventory")
        name = artifact.get("name")
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("Invalid model artifact filename.")
        if name in skip_names:
            continue
        path = directory / name
        size = path.stat().st_size
        digest = file_hash(path)
        if size != artifact.get("size_bytes") or digest != artifact.get("sha256"):
            raise ValueError(f"Parent artifact changed: {name}. Keep the parent untouched and use --weights or --patch.")
        digests[name] = digest
    return digests


def validate_manifest(manifest):
    if not isinstance(manifest, dict):
        raise ValueError("invalid_manifest")
    if manifest.get("schema_version") != "delta.submission.v1":
        raise ValueError("unsupported_schema")
    if manifest.get("format") not in {"sparse-diff", "dense-diff", "lora", "full"}:
        raise ValueError("unsupported_format")
    if manifest.get("reconstruction_version") != "delta-reconstruct-v1":
        raise ValueError("unsupported_reconstruction")
    try:
        uuid.UUID(str(manifest["parent_version_id"]))
    except (ValueError, KeyError):
        raise ValueError("invalid_parent_version")
    if not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("parent_content_hash", ""))):
        raise ValueError("invalid_parent_hash")
    if not re.fullmatch(r"[-a-zA-Z0-9_]{1,100}", str(manifest.get("model_config_id", ""))):
        raise ValueError("invalid_model_config")
    files = manifest.get("artifacts", [])
    if not isinstance(files, list) or not all(isinstance(f, dict) for f in files):
        raise ValueError("invalid_artifact_inventory")
    if len(files) != 1 or files[0].get("name") != "weights.safetensors":
        raise ValueError("expected_weights_safetensors")
    for f in files:
        if not re.fullmatch(r"[0-9a-f]{64}", f.get("sha256", "")):
            raise ValueError("invalid_hash")
        if type(f.get("size_bytes")) is not int or f["size_bytes"] <= 0:
            raise ValueError("invalid_size")
    if manifest.get("operation", "replace") not in {"replace", "add"}:
        raise ValueError("unsupported_operation")
    mapping = manifest.get("tensors", {})
    if not isinstance(mapping, dict) or not all(isinstance(k, str) and isinstance(v, dict) for k, v in mapping.items()):
        raise ValueError("invalid_tensor_mapping")
    for target, spec in mapping.items():
        if len(target) > 1024:
            raise ValueError("invalid_tensor_name")
        required = {"a", "b", "alpha"} if manifest["format"] == "lora" else ({"indices", "values"} if manifest["format"] == "sparse-diff" else {"values"})
        if set(spec) != required or any(not isinstance(v, str) for k, v in spec.items() if k != "alpha"):
            raise ValueError("invalid_tensor_mapping")
    if len(mapping) > 10000:
        raise ValueError("too_many_tensors")


def content_hash(tensors, identity_hash):
    import torch
    h = hashlib.sha256(canonical_json({"schema": "delta-content-v1", "identity": identity_hash}))
    for name, tensor in sorted(tensors.items()):
        t = tensor.detach().cpu().contiguous()
        header = canonical_json([name, list(t.shape), str(t.dtype)])
        h.update(len(header).to_bytes(8, "little"))
        h.update(header)
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def reconstruct(parent_path, artifact_path, manifest, allowed_prefix=""):
    import torch
    from safetensors.torch import load_file
    validate_manifest(manifest)
    base = load_file(str(parent_path))
    patch = load_file(str(artifact_path))
    if any(not torch.isfinite(t).all() for t in patch.values()):
        raise ValueError("nonfinite_patch")
    result = {k: v.clone() for k, v in base.items()}
    used = set()
    if manifest["format"] == "full":
        if set(patch) != set(base):
            raise ValueError("tensor_inventory")
        mapping = {k: {"values": k} for k in patch}
    else:
        mapping = manifest.get("tensors", {})
        if not mapping:
            raise ValueError("empty_patch")
    for name, spec in mapping.items():
        if name not in base:
            raise ValueError("unknown_tensor")
        original = base[name]
        if not name.startswith(allowed_prefix):
            # Full checkpoints may carry unchanged frozen tensors.
            if manifest["format"] == "full" and torch.equal(patch[name], original):
                used.add(name)
                continue
            raise ValueError("frozen_tensor")
        kind = manifest["format"]
        if kind == "lora":
            a, b = patch[spec["a"]], patch[spec["b"]]
            used.update([spec["a"], spec["b"]])
            if a.ndim != 2 or b.ndim != 2 or original.ndim != 2 or a.shape[0] != b.shape[1] or not 1 <= a.shape[0] <= 256:
                raise ValueError("invalid_lora_rank")
            alpha = float(spec["alpha"])
            if not math.isfinite(alpha) or (b.shape[0], a.shape[1]) != tuple(original.shape):
                raise ValueError("invalid_lora_shape")
            updated = original.float() + (b.float() @ a.float()) * (alpha / a.shape[0])
        else:
            values = patch[spec["values"]]
            used.add(spec["values"])
            if kind == "sparse-diff":
                indices = patch[spec["indices"]]
                used.add(spec["indices"])
                if indices.dtype != torch.int64 or indices.ndim != 1 or values.ndim != 1 or indices.numel() != values.numel():
                    raise ValueError("invalid_sparse_shape")
                if indices.numel() and (indices[0] < 0 or indices[-1] >= original.numel() or not (indices[1:] > indices[:-1]).all()):
                    raise ValueError("invalid_sparse_indices")
                updated = original.float().clone().reshape(-1)
                updated[indices] = values.float() if manifest.get("operation", "replace") == "replace" else updated[indices] + values.float()
                updated = updated.reshape(original.shape)
            else:
                if values.shape != original.shape:
                    raise ValueError("tensor_shape")
                updated = values if kind == "full" or manifest.get("operation", "replace") == "replace" else original.float() + values.float()
        result[name] = updated.to(original.dtype).contiguous()
        if not torch.isfinite(result[name]).all():
            raise ValueError("nonfinite_result")
    if used != set(patch):
        raise ValueError("unlisted_tensors")
    return result


def build_sparse(parent, modified, output, parent_id, parent_hash, model_id):
    import torch
    from safetensors.torch import load_file, save_file
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    before, after = load_file(str(parent)), load_file(str(modified))
    if set(before) != set(after):
        raise ValueError("tensor_inventory")
    tensors, mapping = {}, {}
    for number, name in enumerate(sorted(before)):
        a, b = before[name], after[name]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError("tensor_shape_or_dtype")
        idx = torch.nonzero(a.reshape(-1) != b.reshape(-1)).flatten()
        if idx.numel():
            ik, vk = f"t{number}.indices", f"t{number}.values"
            tensors[ik], tensors[vk] = idx, b.reshape(-1)[idx]
            mapping[name] = {"indices": ik, "values": vk}
    if not tensors:
        raise ValueError("empty_patch")
    path = output / "weights.safetensors"
    save_file(tensors, str(path))
    manifest = {"schema_version": "delta.submission.v1", "format": "sparse-diff", "operation": "replace",
                "reconstruction_version": "delta-reconstruct-v1", "parent_version_id": str(parent_id),
                "parent_content_hash": parent_hash, "model_config_id": model_id, "tensors": mapping,
                "artifacts": [{"name": path.name, "sha256": file_hash(path), "size_bytes": path.stat().st_size}]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
