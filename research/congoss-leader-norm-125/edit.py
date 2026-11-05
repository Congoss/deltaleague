"""Retune final RMSNorm gain on the Delta League leader."""
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

directory = Path(__file__).resolve().parent
with safe_open(directory / "model/model.safetensors", framework="pt") as weights:
    key, = (name for name in weights.keys() if name.endswith("language_model.norm.weight"))
    tensors = {name: weights.get_tensor(name) for name in weights.keys()}
    original = tensors[key]
    tensors[key] = ((original.float() + 1) * (1.25 / 1.4) - 1).to(original.dtype)
    save_file(tensors, directory / "edited.safetensors", metadata={"format": "pt"})
