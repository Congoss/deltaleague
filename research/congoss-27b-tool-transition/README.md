# Norm-preserving post-thinking tool-transition hypothesis

Parent: a610fbdb-eb7e-42bc-ba43-19f7922df81f (PR #91).
Evaluation: qwen38-27b-nvfp4-agentic300-8k-v1.

## Hypothesis
A small directional change to the input representation of </think> might improve the subsequent transition into a tool call. This is speculative embedding geometry, not a trained steering vector. Similarity of token embeddings does not establish a causal action direction. The edit only acts after the closing token is consumed; it cannot directly resolve thinking that never closes. It may impair ordinary answers or tool-call syntax and may cause unnecessary calls.

## Fixed experiment
Change only input row 248069, using input row 248058 (<tool_call>) as a guide. Normalize both rows, project the guide orthogonal to the target, add 0.03 times the normalized projection, then restore the original target norm. Round to BF16. Parent output head, all other input rows, and transformer weights are inherited. Alpha was chosen beforehand as an exploratory small perturbation, not selected on evaluation scores. No seeds or training, no parameter search.

## Reproduction
Run build.py with Python 3.13, torch 2.15.0, safetensors 0.8.0 and the sibling range_io.py. Then run delta validate submission. range_io.py is copied from the preceding input-restore experiment; its original main function is not invoked by this builder.

The builder performs bounded checked HTTP range reads of the public parent, verifies the full tokenizer hash and token IDs, and stores hashes of downloaded rows. Full parent file identity is API-provided, not locally verified. No full checkpoint, signed URLs or credentials are recorded. The artifact is a sparse replacement, not a complete edited checkpoint.

## Checks and findings
See build-report.json for measured BF16 perturbation, norm error, hash and sparse reconstruction checks. These are structural checks only. No model inference or public task evaluation has run. No private-test items were accessed. No score improvement is claimed. No evaluation quota consumed by this build.

Before any submission, publish a research PR and obtain user approval for the quota-consuming evaluation. A meaningful behavioral test must compare the parent and candidate on identical public tool-use examples plus no-tool controls, scoring valid calls, correct arguments, task success and unwanted calls separately. Hardware for feasible native 27B inference is unavailable locally; do not substitute embedding similarity for task accuracy.

## Reproduction commands and measured checks

    python -m pip install torch==2.15.0 safetensors==0.8.0 numpy==2.5.5
    python build.py
    delta validate submission

Python 3.13; CLI 0.4.0. CPU-only bounded construction, two embedding rows and tokenizer/header downloads (13,119,478 bytes); no training or generation. Seed: none. Official generation budget: 8192 tokens, thinking enabled, inherited evaluation configuration. No local generation settings or accuracy estimates.

Artifact SHA256: 8e77b4490b52646d909108313407a9ff96d4cd223b640fb24ef58b223a90d6ea.
Sparse replacements: 4,801; artifact: 48,162 bytes. Relative row L2 change: 0.0299890; relative norm error after BF16: 0.00003523. CLI integrity and sparse row reconstruction passed.

Parent verified scores: validation 74.2557589963%, private 29% on the pinned configuration. Candidate scores: unmeasured. The preceding eight-row restoration (PR #96) scored 73.9689308147% validation and 27.6666666667% private; this candidate starts from PR #91, not that restoration.

## Attribution and access

Congoss requested an independent small experiment. Codex proposed and implemented this untrained directional edit. Public tokenizer, checkpoint rows, aggregate scores and model template were available; no private examples or private generations were accessed. Parent recipe credited to PR #91; range-reading helper reused from Congoss PR #96.

## Submission

User approved submission. Submission ID will be recorded after upload.

## Boundaries

- [x] Only research/congoss-27b-tool-transition/ is changed.
- [x] No weights, credentials, private test material, signed URLs or raw generation logs.
- [x] Structural checks and verified parent scores are distinguished from unmeasured candidate quality.
