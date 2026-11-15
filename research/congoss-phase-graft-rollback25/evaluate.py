"""Paired native-format code/IFEval diagnostic of the two-neuron rollback."""
import collections
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
# Set DELTA_WASI_ROOT to the pinned WASI runtime before running.
sys.path[:0] = [
    os.environ["DELTA_IFEVAL_PATH"],
    os.environ["DELTA_REPO"],
]
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, StoppingCriteria
from delta_cli.public_eval.coding import extract_code, program, prompt as code_prompt
from delta_cli.public_eval.wasi import sandbox
from delta_cli.scoring import score_answer

root = Path(__file__).resolve().parent
checkpoint = Path(os.environ["DELTA_NEURON_MODELS"]) / "leader"
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 20261115
manifest = json.loads((root / "submission/manifest.json").read_text(encoding="utf-8"))
identity = json.loads((checkpoint / "model/delta-model.json").read_text())
assert manifest["parent_version_id"] == identity["id"]
assert manifest["parent_content_hash"] == identity["content_hash"]
artifact = root / "submission/weights.safetensors"
assert hashlib.sha256(artifact.read_bytes()).hexdigest() == manifest["artifacts"][0]["sha256"]
suite = json.loads((checkpoint / "validation.json").read_text())
cases = []
for doc in suite["code"]["humaneval"]["docs"][:2]:
    reference = doc["prompt"] + doc["canonical_solution"]
    assert sandbox(program("humaneval", doc, reference))["passed"], doc["task_id"]
    cases.append({"id": doc["task_id"], "kind": "humaneval",
                  "prompt": code_prompt("humaneval", doc), "doc": doc})
for item in json.loads((root / "neuron-ifeval.json").read_text(encoding="utf-8-sig")):
    doc = item["row"]
    assert item["row_idx"] in suite["text"]["ifeval"]["selection"]["ifeval"]["indices"]
    cases.append({"id": f"IFEval/{item['row_idx']}", "kind": "ifeval",
                  "prompt": doc["prompt"], "doc": doc})
assert len(cases) == 4
(root / "native-cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
print("Reference-code preflight passed. Cases:", [case["id"] for case in cases], flush=True)

torch.set_num_threads(4)
tokenizer = AutoTokenizer.from_pretrained(checkpoint / "model", local_files_only=True, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(checkpoint / "model", dtype=torch.float32, local_files_only=True).eval()
parameters = dict(model.named_parameters())
originals = {}
for key in manifest["tensors"]:
    local_key = key.replace("model.language_model.", "model.", 1)
    originals[local_key] = parameters[local_key].detach().clone()
config = json.loads((checkpoint / "evaluation.json").read_text())["generation"]
prompts = [tokenizer.apply_chat_template(
    [{"role": "user", "content": case["prompt"]}], tokenize=False,
    add_generation_prompt=True, enable_thinking=config["enable_thinking"]) for case in cases]
inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
start_tokens = inputs.input_ids.shape[1]
eos = model.generation_config.eos_token_id
eos_ids = {eos} if isinstance(eos, int) else set(eos or [])

class PresencePenalty(LogitsProcessor):
    def __call__(self, input_ids, scores):
        for row in range(input_ids.shape[0]):
            seen = input_ids[row, start_tokens:].unique()
            scores[row, seen] -= config["presence_penalty"]
        return scores

class Deadline(StoppingCriteria):
    def __init__(self):
        self.end = time.monotonic() + 600

    def __call__(self, input_ids, scores, **kwargs):
        return time.monotonic() >= self.end

for arm in ("parent", "rollback25"):
    output = root / f"native-{arm}-{seed}.json"
    if output.exists():
        print("Existing completed report:", output.name, flush=True)
        continue
    with torch.no_grad():
        for key, tensor in originals.items():
            parameters[key].copy_(tensor)
        if arm == "rollback25":
            with safe_open(artifact, framework="pt") as tensors:
                for key, mapping in manifest["tensors"].items():
                    local_key = key.replace("model.language_model.", "model.", 1)
                    parameters[local_key].view(-1)[tensors.get_tensor(mapping["indices"])] = tensors.get_tensor(mapping["values"]).float()
    torch.manual_seed(seed)
    started = time.monotonic()
    with torch.inference_mode():
        generated = model.generate(
            **inputs, do_sample=True, temperature=config["temperature"],
            top_k=config["top_k"], top_p=config["top_p"],
            max_new_tokens=8192, logits_processor=[PresencePenalty()],
            stopping_criteria=[Deadline()], pad_token_id=tokenizer.pad_token_id,
        )[:, start_tokens:].tolist()
    elapsed = time.monotonic() - started
    (root / f"native-{arm}-{seed}-tokens.json").write_text(json.dumps(generated), encoding="utf-8")
    results = []
    for case, tokens in zip(cases, generated, strict=True):
        end = next((index + 1 for index, token in enumerate(tokens) if token in eos_ids), None)
        tokens = tokens[:end]
        raw = tokenizer.decode(tokens)
        final_ids = [token for token in tokens if token not in eos_ids and token != tokenizer.pad_token_id]
        final = tokenizer.decode(final_ids).split("</think>")[-1].strip()
        row = {"id": case["id"], "tokens": len(tokens), "closed": "</think>" in raw,
               "generation": raw, "status": "cap" if len(tokens) >= 8192 else "time_limit"}
        if end is not None and row["closed"]:
            if case["kind"] == "humaneval":
                result = sandbox(program("humaneval", case["doc"], extract_code(final)))
                row["status"] = "correct" if result["passed"] else result["reason"]
            else:
                item = {**case["doc"], "scorer": "ifeval"}
                passed = score_answer(item, final)
                row["status"] = "correct" if passed else "wrong"
        elif end is not None:
            row["status"] = "missing_final"
        results.append(row)
        print(arm, seed, case["id"], row["tokens"], row["status"], flush=True)
    report = {"parent": identity["id"], "arm": arm, "seed": seed, "elapsed_seconds": elapsed,
              "sampling": config, "wall_time_limit_seconds": 600, "dtype": "float32",
              "scorer_source": "local Delta League public_eval/scoring",
              "results": results}
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(arm, dict(collections.Counter(row["status"] for row in results)), flush=True)
