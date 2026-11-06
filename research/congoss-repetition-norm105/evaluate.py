"""Compare answer-choice probabilities on 24 public tasks, without thinking."""
import json
import math
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

root = Path(__file__).resolve().parent
workspace = root.parent / "delta-dose/experiment"
tasks = []
for dataset, config, split, kind in (
    ("allenai/ai2_arc", "ARC-Challenge", "test", "arc"),
    ("aps/super_glue", "boolq", "validation", "boolq"),
):
    query = urlencode({"dataset": dataset, "config": config, "split": split, "offset": 0, "length": 12})
    rows = json.loads((root / f"{kind}.json").read_text())["rows"]
    for item in rows:
        row = item["row"]
        if kind == "arc":
            choices = row["choices"]
            labels = [chr(65 + i) for i in range(len(choices["text"]))]
            prompt = row["question"] + "\n" + "\n".join(
                label + ". " + answer for label, answer in zip(labels, choices["text"])
            )
            correct = labels[choices["label"].index(row["answerKey"])]
        else:
            labels = ["A", "B"]
            prompt = row["passage"] + "\n" + row["question"] + "\nA. Yes\nB. No"
            correct = "A" if row["label"] == 1 else "B"
        tasks.append({"kind": kind, "index": item["row_idx"], "prompt": prompt,
                      "labels": labels, "correct": correct})
(root / "public-tasks.json").write_text(json.dumps(tasks, indent=2))

torch.set_num_threads(4)
tokenizer = AutoTokenizer.from_pretrained(workspace / "model", local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(workspace / "model", dtype=torch.float32, local_files_only=True)
model.eval()
from variants import prepare, apply
state = prepare(model)
results = []
started = time.monotonic()
for arm in ("parent", "soft_doubt", "norm105", "soft_doubt_norm105"):
    apply(arm, state)
    for task in tasks:
        if time.monotonic() - started > 1100:
            raise RuntimeError("Local compute limit reached; partial results saved")
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": task["prompt"] + "\nAnswer with the letter only."}],
            add_generation_prompt=True, enable_thinking=False, tokenize=False,
        )
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in task["labels"]]
        assert all(len(tokens) == 1 for tokens in ids)
        with torch.inference_mode():
            logits = model(**inputs, logits_to_keep=1).logits[0, -1].float()
        scores = logits[[tokens[0] for tokens in ids]]
        probabilities = scores.softmax(dim=0)
        predicted = task["labels"][scores.argmax().item()]
        expected = task["labels"].index(task["correct"])
        row = {"arm": arm, "kind": task["kind"], "index": task["index"],
               "prediction": predicted, "correct": predicted == task["correct"],
               "loss": -math.log(max(probabilities[expected].item(), 1e-30))}
        results.append(row)
        (root / "results.json").write_text(json.dumps(results, indent=2))
    arm_results = [row for row in results if row["arm"] == arm]
    print(json.dumps({"arm": arm, "correct": sum(row["correct"] for row in arm_results),
                      "total": len(arm_results), "loss": sum(row["loss"] for row in arm_results) / len(arm_results)}), flush=True)
