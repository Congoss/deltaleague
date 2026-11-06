"""Apply independently specified doubt-projection and norm gains."""
import json
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent

def prepare(model):
    parameters = dict(model.named_parameters())
    direction = torch.tensor(json.loads((ROOT / "doubt_direction_L13.json").read_text())["values"])
    direction /= direction.norm()
    originals = {"model.norm.weight": parameters["model.norm.weight"].detach().clone()}
    replacements = {}
    with safe_open(ROOT.parent / "delta-experiment/model/model.safetensors", framework="pt") as base:
        for layer in range(14):
            for part in ("self_attn.o_proj", "linear_attn.out_proj", "mlp.down_proj"):
                key = f"model.layers.{layer}.{part}.weight"
                if key not in parameters:
                    continue
                original = base.get_tensor(key.replace("model.", "model.language_model.", 1))
                weights = original.float()
                projection = torch.outer(direction, direction @ weights)
                expected = (weights - projection).to(original.dtype).float()
                error = (parameters[key].detach() - expected).norm() / expected.norm()
                assert error < 0.001, (key, error.item())
                print(key, "relative reconstruction error", error.item(), flush=True)
                replacements[key] = (weights - 0.85 * projection).to(original.dtype).float()
                originals[key] = parameters[key].detach().clone()
    return parameters, originals, replacements

def apply(arm, state):
    parameters, originals, replacements = state
    with torch.no_grad():
        for key, value in originals.items():
            parameters[key].copy_(value)
        if arm in ("soft_doubt", "soft_doubt_norm105"):
            for key, value in replacements.items():
                parameters[key].copy_(value)
        if arm in ("norm105", "soft_doubt_norm105"):
            norm = originals["model.norm.weight"]
            parameters["model.norm.weight"].copy_(((norm + 1) * 1.05 - 1).to(torch.bfloat16).float())
