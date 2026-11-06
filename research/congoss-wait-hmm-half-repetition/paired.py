import json
from pathlib import Path
from transformers import LogitsProcessor

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

root = Path(__file__).resolve().parent
model_dir = root.parent / "delta-leader/model"
torch.set_num_threads(4)
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32, local_files_only=True).eval()
with safe_open(root.parent / "delta-leader/submission-wait-hmm/weights.safetensors", framework="pt") as patch:
    with torch.no_grad():
        model.get_input_embeddings().weight.view(-1)[patch.get_tensor("indices")] = patch.get_tensor("values").float()

sources = [
    ("math", "openai/gsm8k", "main", "test", 25, "question", "answer"),
    ("logic", "allenai/ai2_arc", "ARC-Challenge", "test", 12, "question", "answerKey"),
    ("code", "openai/openai_humaneval", "openai_humaneval", "test", 0, "prompt", "test"),
]
class Presence(LogitsProcessor):
    def __init__(self, start):
        self.start = start

    def __call__(self, input_ids, scores):
        scores[:, input_ids[0, self.start:].unique()] -= 1.5
        return scores

keys = [f"model.layers.{layer}.mlp.down_proj.weight" for layer in (15, 19)]
params = dict(model.named_parameters())
originals = {key: params[key].detach().clone() for key in keys}
leader_path = root.parent / "delta-dose/experiment/model/model.safetensors"
with safe_open(leader_path, framework="pt") as leader:
    edits = {key: leader.get_tensor(key.replace("model.", "model.language_model.", 1)).float()
             for key in keys}
for arm in ("control", "half_rep"):
    with torch.no_grad():
        for key in keys:
            params[key].copy_(edits[key] if arm == "half_rep" else originals[key])
    for kind, dataset, config, split, index, field, reference in sources:
        row = next(item["row"] for item in json.loads((root / f"source-{kind}.json").read_text(encoding="utf-8"))["rows"] if item["row_idx"] == index)
        prompt = row[field]
        if kind == "logic":
            prompt += chr(10) + chr(10).join(f"{label}. {value}" for label, value in zip(row["choices"]["label"], row["choices"]["text"]))
        chat = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
        inputs = tokenizer(chat, return_tensors="pt", add_special_tokens=False)
        torch.manual_seed(1000 + index)
        with torch.inference_mode():
            tokens = model.generate(**inputs, max_new_tokens=1024, logits_processor=[Presence(inputs.input_ids.shape[1])], do_sample=True, temperature=1.0, top_k=20, top_p=0.95)[0, inputs.input_ids.shape[1]:]
        answer = tokenizer.decode(tokens)
        result = {"kind": kind, "index": index, "tokens": tokens.tolist(), "answer": answer, "reference": row[reference]}
        (root / f"paired-{arm}-{kind}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(arm, kind, len(tokens), "</think>" in answer, flush=True)
