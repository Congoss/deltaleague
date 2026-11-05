"""Paired CPU smoke test on public GSM8K validation selections."""
from pathlib import Path
import json
import re
from urllib.request import urlopen

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

root = Path(__file__).resolve().parent
manifest = json.loads((root / "validation.json").read_text(encoding="utf-8"))
indices = manifest["text"]["gsm8k"]["selection"]["gsm8k"]["indices"][:2]
docs = []
for index in indices:
    url = f"https://datasets-server.huggingface.co/rows?dataset=openai/gsm8k&config=main&split=test&offset={index}&length=1"
    with urlopen(url, timeout=30) as response:
        docs.append(json.load(response)["rows"][0]["row"])

torch.set_num_threads(4)
tokenizer = AutoTokenizer.from_pretrained(root / "model", local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(root / "model", dtype=torch.float32, local_files_only=True)
model.eval()
for name in ("leader", "candidate"):
    if name == "candidate":
        with safe_open(root / "edited.safetensors", framework="pt") as weights:
            key, = (key for key in weights.keys() if key.endswith("language_model.norm.weight"))
            with torch.no_grad():
                model.model.norm.weight.copy_(weights.get_tensor(key))
    for index, doc in zip(indices, docs):
        torch.manual_seed(1000 + index)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": doc["question"] + " Give the final number in a box."}],
            add_generation_prompt=True, enable_thinking=True, tokenize=False,
        )
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=768, do_sample=True,
                                    temperature=1.0, top_k=20, top_p=0.95, repetition_penalty=1.0)
        generated = output[0, inputs.input_ids.shape[1]:]
        answer = tokenizer.decode(generated)
        matches = re.findall(r"\\boxed\{([^{}]+)\}", answer.split("</think>")[-1])
        expected = doc["answer"].split("####")[-1].strip().replace(",", "")
        print(json.dumps({"model": name, "index": index, "tokens": len(generated),
                          "closed": "</think>" in answer,
                          "correct": bool(matches and matches[-1].replace(",", "").strip() == expected)}),
              flush=True)
