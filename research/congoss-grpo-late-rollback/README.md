## Hypothesis and scope

This experiment tests whether a small rollback in the last two transformer blocks improves answer quality on the GRPO checkpoint from PR #55. In each tensor of blocks 22 and 23, the builder computes 90% of the leader weight plus 10% of the pre-GRPO reference in FP32, then casts back to the parent dtype. Eighteen tensors differ after rounding. The final RMSNorm comes from the registered parent.

- **Model:** Qwen3.5 0.8B, registered as qwen35-08b-v1
- **Parent:** 79be46a9-cba5-4535-9076-1d97ba749b29, GRPO with the final norm step from PR #55
- **Pre-GRPO reference:** 344c8376-66a1-413b-abca-da2b5b986044, PR #24
- **Evaluation:** public-thinking8k-ladder500-v1
- **Attribution:** Congoss selected the experiment. Codex built and evaluated the patch using public checkpoints and public tasks.

## Reproduction

Download the two checkpoints with delta competition init qwen35-08b-v1 --parent VERSION_ID --output DIRECTORY. Run the following from this research directory with Python 3.13, PyTorch 2.15 and safetensors 0.8.

    python build.py PARENT/model PRE_GRPO/model submission
    delta validate submission

The builder emits a deterministic dense replacement. The independently rebuilt artifact matched SHA-256 a138e79e6913f481047a87ba6e77c1be1b19f0b892616f18c7434f9f75f177b1.

For the local diagnostics, set DELTA_LEADER_WORKSPACE to PARENT. Use Transformers 5.22.1. Save Hugging Face dataset-viewer rows as arc.json and boolq.json for ARC-Challenge test and BoolQ validation, both at offset 36 and length 24. For generation, save GSM8K main test row 29 as math.json, HumanEval test row 0 as source-code.json, and google/IFEval default train rows 0–7 as ifeval.json. Set PYTHONPATH to the Delta League repository plus the lm-eval 0.4.13 IFEval scorer and its dependencies. Set DELTA_WASI_ROOT to the pinned CPython WASI runtime used by Delta League with Wasmtime 50.0.1.

    python evaluate.py
    python paired-8k.py leader
    python paired-8k.py rollback10

The retry command is python paired-8k.py leader 20261105. It extends the CPU time budget from 12 to 15 minutes for that seed and writes a separate result file.

## Evidence from local public tasks

The answer-choice diagnostic used 24 ARC-Challenge and 24 BoolQ questions with thinking disabled. The generation pilot used one public prompt each from GSM8K, HumanEval and IFEval with seeds 20261104 and 20261105. Both arms used the same prompts and sampling parameters in CPU float32 Transformers. The 768-token condition scores the exact prefix of the longer generation. HumanEval ran its tests in WASI, IFEval used the competition scorer, and GSM8K used exact numeric matching.

| Check | Parent | 10% rollback |
| --- | ---: | ---: |
| Answer choice, thinking disabled | 37/48 | 37/48 |
| Thinking, 768-token prefix | 2/6 | 2/6 |
| Thinking, up to 8192 tokens | 5/6 | 5/6 |

The longer budget recovered two IFEval answers and one GSM8K answer in both versions. One HumanEval completion failed the tests. After retrying a CPU timeout, all six generated token sequences matched across the two versions. The pilot showed no local gain from rollback and covers only three generation prompts. The official evaluation tests whether the edit generalizes beyond this small sample.

## Submission

[Delta League submission](https://deltaleague.ai/submissions/10427ba2-5f4a-49fc-b9ae-e9a9499c8d71/). Official evaluation pending.

## Boundaries

This PR changes only the research record and reproduction scripts. The experiment used public data. Weight artifacts and raw generations are kept outside the PR.
