"""Rebuild the authors' published sparse edit; no fitting or private evaluation access."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file,save_file
root=Path.cwd()
manifest=json.loads((root/'upstream/manifest.json').read_text())
weights=load_file(str(root/'model/model.safetensors'))
with np.load(root/'upstream/edit.npz',allow_pickle=False) as edits:
 for i,entry in enumerate(manifest['tensors']):
  original=weights[entry['name']]
  assert original.dtype==torch.bfloat16 and list(original.shape)==entry['shape']
  bits=original.view(torch.uint16).numpy().copy()
  assert hashlib.sha256(bits.tobytes()).hexdigest()==entry['base_sha256']
  assert entry['form']=='sparse'
  bits.reshape(-1)[edits[f't{i}_idx']]=edits[f't{i}_val']
  assert hashlib.sha256(bits.tobytes()).hexdigest()==entry['edited_sha256']
  changed=int(np.count_nonzero(bits!=original.view(torch.uint16).numpy()))
  assert changed==entry['changed_entries']
  weights[entry['name']]=torch.from_numpy(bits).view(torch.bfloat16)
  print(entry['name'],changed,'changed entries; upstream hash verified',flush=True)
save_file(weights,str(root/'edited.safetensors'))
print('Saved exact published edit',flush=True)
