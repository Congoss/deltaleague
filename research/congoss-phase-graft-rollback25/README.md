# Test a 25% rollback of the phase-gated layer-23 neurons on fb984a19

## Hypothesis and scope

The current leader inherited a phase-gated ready-to-answer circuit before its newer distillation and GRPO updates. This experiment tests whether partially restoring its two repurposed neurons improves final-answer quality on the newer backbone.

- **Model** — qwen35-08b-v1, Qwen3.5 0.8B.
- **Parent** — fb984a19-6262-48db-bd14-a5d68e246ea6.
- **Graft donor** — fdad0374-29f3-49c6-911d-d0ce9531f7a6, PR #66.
- **Pre-graft reference** — ac147fa6-6755-4f1b-a6b2-697aef3552db.
- **Change** — interpolate the gate/up rows and down columns of neurons 1998 and 3529 by 25% toward the pre-graft reference in FP32, then cast to the parent dtype. The combined functional effect is nonlinear.
- **Evaluation** — private1000-v6-thinking8k-20261111.
- **Attribution** — Congoss selected the experiment, Codex implemented and measured it, and the existing backbone/graft came from the public entries above. The graft research is pinned at 17fcb2bb6e0ca18266a5dc717a3ef082d85b343f. Only public examples and checkpoints were available.

## Reproduction

Download the three checkpoints into sibling directories named leader, grafted and before.

~~~sh
delta competition init qwen35-08b-v1 --parent fb984a19-6262-48db-bd14-a5d68e246ea6 --output MODELS/leader
delta competition init qwen35-08b-v1 --parent fdad0374-29f3-49c6-911d-d0ce9531f7a6 --output MODELS/grafted
delta competition init qwen35-08b-v1 --parent ac147fa6-6755-4f1b-a6b2-697aef3552db --output MODELS/before
python build.py MODELS .
delta validate submission
~~~

The builder verifies that exactly two neurons distinguish donor and reference in each affected projection, and that their grafted weights were preserved exactly in the current leader. The deterministic sparse replacement changes 6,133 entries in three tensors. A separate rebuild matched the original artifact bit-for-bit.

Artifact SHA-256 — 8023ab11177327c7e740b7273b0af3d7f7863876c65d0412a3ce6874a8e340fe.

Local versions were Python 3.13, PyTorch 2.15, Transformers 5.22.1 and safetensors 0.8. The scorer and code-prompt helpers came from the Delta League checkout at 525d3d979b4b73813c9f49c27e11d2d1f04aa114. Code executed in the pinned CPython 3.12.10 WASI runtime with Wasmtime 50.0.1. IFEval used lm-eval 0.4.13 rule checkers.

To run evaluate.py, set DELTA_NEURON_MODELS to MODELS, DELTA_REPO to that Delta League checkout, DELTA_IFEVAL_PATH to the environment directory containing the IFEval rule-checker package, and DELTA_WASI_ROOT to the pinned runtime. Run python evaluate.py 20261115. The script checks both canonical code solutions in the sandbox before evaluating generated code.

## Evidence from local public tasks

This paired pilot used four preselected public examples from the suite inventory. HumanEval used the competition code-prompt helper and execution tests. IFEval used the original instruction prompts without additional suffixes. Both arms used the same batch order and seed 20261115.

Sampling matched the downloaded configuration with thinking enabled, temperature 1, top-k 20, top-p 0.95, presence penalty 1.5 and an 8,192-token ceiling. CPU float32 inference had a separate 600-second deadline per arm.

| Public task | Parent | 25% rollback |
| --- | --- | --- |
| HumanEval/4 | Tests passed, 682 tokens | Tests passed, 732 tokens |
| HumanEval/11 | Tests passed, 642 tokens | Tests passed, 1,412 tokens |
| IFEval/7 | Time limit, 1,541 tokens | Time limit, 1,512 tokens |
| IFEval/14 | Time limit, 1,541 tokens | Time limit, 1,512 tokens |

**No local improvement was demonstrated.** Both code completions passed in both arms, and the rollback produced longer responses. The two IFEval cases were interrupted during reasoning, so their final-answer accuracy remains unresolved. A timeout is recorded separately from an incorrect completed answer.

The pilot covers four examples and one seed. CPU float32 Transformers differs from the official GPU engine. These observations cannot establish a leaderboard improvement. The official submission is an exploratory test of the hypothesis beyond this pilot.

## Submission

Official submission pending.

## Boundaries

- [x] Changes are contained in one research directory.
- [x] Model artifacts, credentials and raw generations are excluded.
- [x] Inputs are public, with no private test material.
- [x] Local measurements and pending official results are distinguished.
