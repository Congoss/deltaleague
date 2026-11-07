"""Paired thinking-mode pilot with matched 768-token prefixes and 8192-token limits."""
import json
import os
from pathlib import Path
import re
import sys
import time

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, StoppingCriteria

from delta_cli.public_eval.coding import extract_code, program, prompt as code_prompt
from delta_cli.public_eval.wasi import sandbox
from delta_cli.scoring import score_answer

root = Path(__file__).resolve().parent
checkpoint = Path(os.environ["DELTA_LEADER_WORKSPACE"])
arm = sys.argv[1]
assert arm in ("leader", "rollback10")
assert sandbox("assert 1 + 1 == 2")["passed"]

def read_row(path, index):
    data = json.loads(path.read_text(encoding="utf-8"))
    return next(item["row"] for item in data["rows"] if item["row_idx"] == index)

math = read_row(root / "math.json", 29)
code = read_row(root / "source-code.json", 0)
ifeval = read_row(root / "ifeval.json", 5)
tasks = [
    ("GSM8K:29", math["question"] + "\nGive the final number in a box.", math["answer"].split("####")[-1].strip()),
    ("HumanEval:0", code_prompt("humaneval", code), code),
    ("IFEval:5", ifeval["prompt"], ifeval),
]

torch.set_num_threads(4)
tokenizer = AutoTokenizer.from_pretrained(checkpoint / "model", local_files_only=True, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(checkpoint / "model", dtype=torch.float32, local_files_only=True).eval()
if arm == "rollback10":
    patch = root / "submission"
    manifest = json.loads((patch / "manifest.json").read_text())
    identity = json.loads((checkpoint / "model/delta-model.json").read_text())
    assert identity["content_hash"] == manifest["parent_content_hash"]
    parameters = dict(model.named_parameters())
    with torch.no_grad(), safe_open(patch / "weights.safetensors", framework="pt") as tensors:
        for key, spec in manifest["tensors"].items():
            parameters[key.replace("model.language_model.", "model.", 1)].copy_(tensors.get_tensor(spec["values"]))

config = json.loads((checkpoint / "evaluation.json").read_text())["generation"]
assert config["max_tokens"] == 8192
chats = [tokenizer.apply_chat_template([{"role": "user", "content": question}],
         tokenize=False, add_generation_prompt=True, enable_thinking=config["enable_thinking"])
         for _, question, _ in tasks]
inputs = tokenizer(chats, return_tensors="pt", padding=True, add_special_tokens=False)
prompt_length = inputs.input_ids.shape[1]
eos = model.generation_config.eos_token_id
eos_ids = {eos} if isinstance(eos, int) else set(eos or ())

class PresencePenalty(LogitsProcessor):
    def __call__(self, input_ids, scores):
        seen = input_ids[:, prompt_length:]
        scores.scatter_(1, seen, scores.gather(1, seen) - config["presence_penalty"])
        return scores

class Deadline(StoppingCriteria):
    def __init__(self, seconds):
        self.end = time.monotonic() + seconds

    def __call__(self, input_ids, scores, **kwargs):
        return time.monotonic() >= self.end

def score(task, reference, tokens, cap):
    prefix = tokens[:cap]
    end = next((i + 1 for i, token in enumerate(prefix) if token in eos_ids), None)
    raw = tokenizer.decode(prefix[:end])
    final = re.sub(r"<\|[^|]+\|>", "", raw).split("</think>")[-1].strip()
    closed = "</think>" in raw
    result = {"tokens": len(prefix[:end]), "closed": closed, "raw": raw,
              "status": "cap" if len(prefix) == cap else "time_limit"}
    if end is None:
        return result
    if not closed or not final:
        result["status"] = "missing_final"
    elif task.startswith("GSM8K"):
        matches = re.findall(r"\\boxed\{([^{}]+)\}", final)
        answer = matches[-1].strip() if matches else final
        result["status"] = "correct" if answer.replace(",", "") == reference.replace(",", "") else "wrong"
        result["final"] = answer
    elif task.startswith("HumanEval"):
        test = sandbox(program("humaneval", reference, extract_code(final)))
        result["status"] = "correct" if test["passed"] else test["reason"]
    else:
        passed = score_answer({**reference, "scorer": "ifeval"}, final)
        result["status"] = "correct" if passed else "wrong"
    return result

seeds = tuple(map(int, sys.argv[2:])) or (20261104, 20261105)
for seed in seeds:
    path = root / f"paired-8k-{arm}-{seed}.json"
    if sys.argv[2:]:
        path = path.with_name(f"{path.stem}-retry.json")
    if path.exists():
        continue
    torch.manual_seed(seed)
    with torch.inference_mode():
        continuations = model.generate(
            **inputs, do_sample=True, temperature=config["temperature"], top_k=config["top_k"],
            top_p=config["top_p"], max_new_tokens=config["max_tokens"],
            logits_processor=[PresencePenalty()], stopping_criteria=[Deadline(900 if sys.argv[2:] else 720)],
            pad_token_id=tokenizer.pad_token_id,
        )[:, prompt_length:].tolist()
    path.with_name(f"{path.stem}-tokens.json").write_text(json.dumps(continuations), encoding="utf-8")
    results = []
    for (task, _, reference), tokens in zip(tasks, continuations, strict=True):
        row = {"arm": arm, "seed": seed, "task": task,
               **{str(cap): score(task, reference, tokens, cap) for cap in (768, 8192)}}
        results.append(row)
        print(task, arm, seed, [(row[str(cap)]["status"], row[str(cap)]["tokens"]) for cap in (768, 8192)], flush=True)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
