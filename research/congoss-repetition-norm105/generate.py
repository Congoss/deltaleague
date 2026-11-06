"""Bounded paired public smoke test of neighboring repetition doses."""
import json
import re
import time
from pathlib import Path
from urllib.request import urlopen

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, StoppingCriteria

root = Path(__file__).resolve().parent
workspace = root.parent / "delta-dose/experiment"
selection = json.loads((workspace / "validation.json").read_text())["text"]["gsm8k"]["selection"]["gsm8k"]
revision = json.loads((workspace / "validation.json").read_text())["dataset_revisions"]["openai/gsm8k"]
indices = selection["indices"][2:3]
docs = []
for index in indices:
    url = (f"https://datasets-server.huggingface.co/rows?dataset=openai/gsm8k"
           f"&config=main&split=test&offset={index}&length=1&revision={revision}")
    with urlopen(url, timeout=30) as response:
        docs.append(json.load(response)["rows"][0]["row"])

torch.set_num_threads(4)
tokenizer = AutoTokenizer.from_pretrained(workspace / "model", local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(workspace / "model", dtype=torch.float32, local_files_only=True)
model.eval()
deadline = time.monotonic() + 1100

class Deadline(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs):
        return time.monotonic() >= deadline

class Presence(LogitsProcessor):
    def __init__(self, prompt_length):
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores):
        seen = input_ids[0, self.prompt_length:].unique()
        scores[:, seen] -= 1.5
        return scores

from variants import prepare, apply
state = prepare(model)
results = []
for dose in ("parent", "soft_doubt", "norm105", "soft_doubt_norm105"):
    apply(dose, state)
    for index, doc in zip(indices, docs):
        if time.monotonic() >= deadline:
            break
        torch.manual_seed(1000 + index)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": doc["question"] + " Give the final number in a box."}],
            add_generation_prompt=True, enable_thinking=True, tokenize=False,
        )
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        with torch.inference_mode():
            output = model.generate(
                **inputs, max_new_tokens=1024, do_sample=True, temperature=1.0, top_k=20, top_p=0.95,
                logits_processor=[Presence(inputs.input_ids.shape[1])], stopping_criteria=[Deadline()],
            )
        generated = output[0, inputs.input_ids.shape[1]:]
        answer = tokenizer.decode(generated)
        closed = "</think>" in answer
        matches = re.findall(re.escape(chr(92) + "boxed{") + "([^{}]+)" + re.escape("}"), answer.split("</think>")[-1]) if closed else []
        expected = doc["answer"].split("####")[-1].strip().replace(",", "")
        row = {"dose": dose, "index": index, "tokens": len(generated), "closed": closed,
               "correct": bool(matches and matches[-1].replace(",", "").strip() == expected),
               "deadline_hit": time.monotonic() >= deadline, "generation": answer, "expected": expected}
        results.append(row)
        (root / "smoke-results.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(row), flush=True)
