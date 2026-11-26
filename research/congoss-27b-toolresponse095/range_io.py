"""Restore original input embedding rows on the 27B leader using bounded range reads."""
import hashlib
import json
import math
from pathlib import Path
import re
import struct
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
API = "https://deltaleague.ai/api/v1/"
MODEL = "qwen38-27b-nvfp4-ref"
ORIGINAL = "47f586c3-5e3f-42ac-9991-e3db49e09103"
PARENT = "a610fbdb-eb7e-42bc-ba43-19f7922df81f"
EMBED = "model.language_model.embed_tokens.weight"
TOKENS = {
    3655: ("Ġwait", 0.5), 11158: ("wait", 0.5), 13428: ("ĠWait", 0.5),
    13784: ("Wait", 0.5), 77264: ("Hmm", 0.5), 84485: ("Ġhmm", 0.5),
    85152: ("ĠHmm", 0.5), 248069: ("</think>", 1.3),
}
network_bytes = 0

def fetch(url, maximum, start=None, end=None, total=None):
    global network_bytes
    if not url.startswith("https://"):
        raise ValueError("Only HTTPS asset URLs are accepted")
    headers = {"Accept-Encoding": "identity", "User-Agent": "DeltaLeague-research-input-restore"}
    if start is not None:
        headers["Range"] = f"bytes={start}-{end}"
    try:
        with urlopen(Request(url, headers=headers), timeout=60) as response:
            if start is not None:
                expected = f"bytes {start}-{end}/{total}"
                if response.status != 206 or response.headers.get("Content-Range") != expected:
                    raise ValueError("Asset server did not honor the exact byte range")
            if response.headers.get("Content-Encoding", "identity") != "identity":
                raise ValueError("Encoded asset response is incompatible with byte ranges")
            data = response.read(maximum + 1)
    except HTTPError as exc:
        raise RuntimeError(f"HTTP request failed with status {exc.code}") from None
    except URLError:
        raise RuntimeError("HTTPS request failed") from None
    if len(data) > maximum or (start is not None and len(data) != end - start + 1):
        raise ValueError("Unexpected response length")
    network_bytes += len(data)
    return data

def metadata(version):
    value = json.loads(fetch(API + "model-versions/" + version, 2 * 1024**2))
    assert value["id"] == version and value["model_id"] == MODEL
    assert re.fullmatch(r"[a-f0-9]{64}", value["content_hash"])
    return value

def file_info(meta, name):
    matches = [item for item in meta["files"] if item["name"] == name]
    assert len(matches) == 1
    return matches[0]

def header(info):
    first = fetch(info["url"], 8, 0, 7, info["size_bytes"])
    size, = struct.unpack("<Q", first)
    assert 0 < size <= 16 * 1024**2 and 8 + size < info["size_bytes"]
    raw = fetch(info["url"], size, 8, 7 + size, info["size_bytes"])
    value = json.loads(raw)
    spec = value[EMBED]
    assert spec["dtype"] == "BF16"
    assert spec["shape"][1] == 5120 and max(TOKENS) < spec["shape"][0]
    begin, finish = spec["data_offsets"]
    assert finish - begin == math.prod(spec["shape"]) * 2
    assert finish + 8 + size <= info["size_bytes"]
    return value, 8 + size, hashlib.sha256(raw).hexdigest()

def embedding_row(info, specs, offset, token):
    spec = specs[EMBED]
    width = spec["shape"][1]
    begin = offset + spec["data_offsets"][0] + token * width * 2
    return fetch(info["url"], width * 2, begin, begin + width * 2 - 1, info["size_bytes"])

def scale_bf16(bits, factor):
    value, = struct.unpack("<f", struct.pack("<I", bits << 16))
    assert math.isfinite(value)
    factor32, = struct.unpack("<f", struct.pack("<f", factor))
    multiplied = struct.pack("<f", value * factor32)
    full, = struct.unpack("<I", multiplied)
    assert full & 0x7F800000 != 0x7F800000
    return ((full + 0x7FFF + ((full >> 16) & 1)) >> 16) & 0xFFFF

def main():
    original, parent = metadata(ORIGINAL), metadata(PARENT)
    assert parent["parent_id"] == ORIGINAL
    original_info, parent_info = file_info(original, "model.safetensors"), file_info(parent, "model.safetensors")
    original_specs, original_offset, original_header_hash = header(original_info)
    parent_specs, parent_offset, parent_header_hash = header(parent_info)
    assert original_specs[EMBED]["shape"] == parent_specs[EMBED]["shape"]
    width = original_specs[EMBED]["shape"][1]
    tokenizer_info = file_info(original, "tokenizer.json")
    assert tokenizer_info["sha256"] == file_info(parent, "tokenizer.json")["sha256"]
    tokenizer_raw = fetch(tokenizer_info["url"], 32 * 1024**2)
    assert len(tokenizer_raw) == tokenizer_info["size_bytes"]
    assert hashlib.sha256(tokenizer_raw).hexdigest() == tokenizer_info["sha256"]
    tokenizer = json.loads(tokenizer_raw)
    vocab = tokenizer["model"]["vocab"]
    lookup = dict(vocab)
    lookup.update({item["content"]: item["id"] for item in tokenizer["added_tokens"]})
    for token, (text, _) in TOKENS.items():
        assert lookup[text] == token

    indices, restored_values, rows = [], bytearray(), []
    for token, (text, factor) in sorted(TOKENS.items()):
        source = embedding_row(original_info, original_specs, original_offset, token)
        current = embedding_row(parent_info, parent_specs, parent_offset, token)
        original_bits = struct.unpack(f"<{width}H", source)
        parent_bits = struct.unpack(f"<{width}H", current)
        expected = tuple(scale_bf16(bits, factor) for bits in original_bits)
        assert parent_bits == expected, f"Unexpected parent scaling at token {token}"
        reconstructed = bytearray(current)
        changed = 0
        for column, (before, after) in enumerate(zip(original_bits, parent_bits, strict=True)):
            if before != after:
                indices.append(token * width + column)
                restored_values.extend(struct.pack("<H", before))
                struct.pack_into("<H", reconstructed, column * 2, before)
                changed += 1
        assert bytes(reconstructed) == source
        rows.append({"token_id": token, "token": text, "parent_scale": factor, "changed": changed,
                     "original_sha256": hashlib.sha256(source).hexdigest(),
                     "parent_sha256": hashlib.sha256(current).hexdigest()})
        print("Verified original scaling and restored row", token, "entries", changed, flush=True)

    assert indices and all(a < b for a, b in zip(indices, indices[1:]))
    assert len(restored_values) == len(indices) * 2
    index_bytes = struct.pack(f"<{len(indices)}q", *indices)
    specs = {
        "indices": {"dtype": "I64", "shape": [len(indices)], "data_offsets": [0, len(index_bytes)]},
        "values": {"dtype": "BF16", "shape": [len(indices)],
                   "data_offsets": [len(index_bytes), len(index_bytes) + len(restored_values)]},
    }
    raw_header = json.dumps(specs, separators=(",", ":")).encode()
    raw_header += b" " * (-len(raw_header) % 8)
    payload = struct.pack("<Q", len(raw_header)) + raw_header + index_bytes + bytes(restored_values)
    out = ROOT / "submission"
    out.mkdir(exist_ok=True)
    weights = out / "weights.safetensors"
    weights.write_bytes(payload)
    manifest = {
        "schema_version": "delta.submission.v1", "reconstruction_version": "delta-reconstruct-v1",
        "model_config_id": MODEL, "parent_version_id": PARENT,
        "parent_content_hash": parent["content_hash"], "format": "sparse-diff", "operation": "replace",
        "artifacts": [{"name": weights.name, "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}],
        "tensors": {EMBED: {"indices": "indices", "values": "values"}},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = {
        "model": MODEL, "parent": PARENT, "parent_content_hash": parent["content_hash"],
        "original": ORIGINAL, "rows": rows, "embedding_shape": original_specs[EMBED]["shape"],
        "parent_file_sha256_from_api": parent_info["sha256"],
        "original_file_sha256_from_api": original_info["sha256"],
        "parent_header_sha256": parent_header_hash, "original_header_sha256": original_header_hash,
        "tokenizer_sha256_verified": tokenizer_info["sha256"], "network_bytes_read": network_bytes,
        "artifact_sha256": manifest["artifacts"][0]["sha256"],
        "artifact_size_bytes": len(payload), "changed_entries": len(indices),
        "checks": {"parent_scaling_matches": True, "target_rows_reconstruct_exactly": True,
                   "indices_sorted_unique": True, "full_parent_file_hash_computed": False,
                   "inference_run": False},
    }
    (ROOT / "build-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Built", len(indices), "entries; artifact bytes", len(payload), "network bytes", network_bytes, flush=True)
    print("SHA256", report["artifact_sha256"], flush=True)

if __name__ == "__main__":
    main()
