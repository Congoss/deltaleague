# Delta submission format version 1

A directory contains `manifest.json` and `weights.safetensors`. The immutable parent version and its canonical content hash are mandatory. All tensor payloads use safetensors; no pickle or participant code is loaded.

```json
{
  "schema_version": "delta.submission.v1",
  "reconstruction_version": "delta-reconstruct-v1",
  "model_config_id": "qwen35-08b-v1",
  "parent_version_id": "UUID",
  "parent_content_hash": "64 lowercase hex characters",
  "format": "sparse-diff",
  "operation": "replace",
  "artifacts": [{"name": "weights.safetensors", "sha256": "64 lowercase hex characters", "size_bytes": 1024}],
  "tensors": {"model.language_model.layers.0.mlp.down_proj.weight": {"indices": "t0.indices", "values": "t0.values"}}
}
```

Sparse indices are int64, flat, strictly increasing and in bounds. Values are one-dimensional and match index count. Replacement sets those elements; addition accumulates parent and delta in float32 before casting to the registered parent dtype. Untouched tensors inherit the parent. `dense-diff` uses a `values` tensor matching the full target shape. `lora` maps each target to `a`, `b`, and finite `alpha`; apply `W + (alpha/r) * (B @ A)` in float32, then cast. A is `[r, in]`, B is `[out, r]`, rank is 1–256. `full` uses exact parent tensor names and shapes; changed frozen tensors are rejected.

Canonical identity is SHA-256 over canonical JSON identity metadata and sorted named tensors, each with a length-prefixed JSON header (name, shape, dtype) followed by contiguous raw CPU bytes. Model/config/tokenizer identity participates in the hash. Server reconstruction is authoritative. Different submissions may resolve to one model content hash without sharing provenance.

Client requests use `Authorization: Bearer <delta-token>`. Request `/api/v1/upload-sessions` with the manifest, PUT bytes to returned URLs, then POST `/api/v1/submissions` with `upload_id`, `title`, and an `Idempotency-Key` header. Retrying a submission uses the same key and upload. Signed URLs are credentials; do not publish them. Single-object uploads support the first model; multipart/resumable uploads remain a launch-completion task.

All metrics are JSON. Validation is the equally weighted mean of configured benchmark scores on a 0–100 scale. Private test is a single overall score. A missing score is null, never zero. Private example text and generations are not returned. Public validation can be downloaded through `/api/v1/evaluation-configs/<id>`.
