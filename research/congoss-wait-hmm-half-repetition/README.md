# Combine selective Wait/Hmm damping with a half-strength repetition edit

The Wait/Hmm checkpoint can reach a plausible answer and keep checking it, or begin rewriting a function before finishing. This experiment tests whether the published repetition edit helps this checkpoint finish correctly. It transfers only the two edited MLP down projections from PR #24, at the published 0.5 dose.

- Registered parent: ab592312-ec77-4f98-b17f-6ea17723b74e (Qwen3.5 0.8B).
- Local control: ffd90ff5-3ef8-46ce-8275-8a041a8fe684 (Congoss, PR #21, seven Wait/Hmm embeddings at 0.5 of original).
- Donor: 344c8376-66a1-413b-abca-da2b5b986044 (SpacexCeo, PR #24). Only layers 15 and 19 are transferred. The published repetition edit originates with Rikiya Takahashi and collaborators.
- Evaluation: public-ladder500-8k-128-v1.

## Paired public CPU diagnostic

Both arms used float32 Transformers inference, thinking enabled, temperature 1.0, top-k 20, top-p 0.95, a generated-token presence penalty of 1.5, and a 1024-token cap. The seeds were 1025 for GSM8K row 25, 1012 for ARC-Challenge row 12 and 1000 for HumanEval row 0.

| Task | Wait/Hmm control | With repetition edit |
| --- | --- | --- |
| GSM8K | Hit 1024-token cap | Hit 1024-token cap |
| ARC-Challenge | Hit 1024-token cap | Hit 1024-token cap |
| HumanEval | Hit 1024-token cap | Closed reasoning and produced code at 990 tokens |

The code was inspected but did not undergo execution-based scoring in this diagnostic. The small sample and shorter local token budget support only a completion observation. The official evaluation will measure accuracy. Raw generations remain local. Only public checkpoints and public examples were used.

## Reproduction

Download the registered parent and donor with delta competition init qwen35-08b-v1 --parent VERSION_ID. Rebuild the Wait/Hmm control artifact using the instructions in PR #21. Run python build.py PARENT/model CONTROL_SUBMISSION DONOR/model submission, then delta validate submission. The deterministic sparse replacement changes seven embedding rows and 64,904 and 64,448 entries in the layer 15 and 19 tensors, respectively.

Artifact SHA-256: 8df345cb9f64055828377cc7b4036469c37622ac0ce221016d087aa7f2ef7eff.

For paired.py, put the parent and control artifact in sibling delta-leader/model and delta-leader/submission-wait-hmm directories, and the donor in sibling delta-dose/experiment/model. Save the relevant Hugging Face dataset viewer responses in source-math.json, source-logic.json and source-code.json beside paired.py. Use Python 3.13, Torch 2.15, Transformers 5.22.1 and safetensors 0.8.
