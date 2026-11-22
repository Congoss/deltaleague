# Final RMSNorm gain 1.3 on the 30% Wait/Hmm leader

## Hypothesis and scope

Experiment directory: research/congoss-27b-norm13/.
Congoss requested continued bounded weight experiments without rented compute. Codex selected this single-parameter ablation. Available information was public leaderboard aggregates, immutable checkpoint metadata, selected tensor bytes, and prior research. No hidden items or private generations were accessed.

Model: qwen38-27b-nvfp4-ref.
Parent: 5065b8a8-e87d-4f06-b52b-183fcb3bab8c (30% private score; Wait/Hmm output rows x0.25 according to its leaderboard title).
Original reference: 47f586c3-5e3f-42ac-9991-e3db49e09103.
Evaluation: qwen38-27b-nvfp4-agentic300-8k-v1.

Hypothesis: the parent's stronger selective output suppression may interact with its inherited final normalization gain. Reducing the original-relative final RMSNorm gain from 1.4 to 1.3 may improve calibration. This does not establish that sharper logits caused the parent's errors. It can also weaken correct predictions. The quantized output path means the change need not act as an exact temperature adjustment.

Only model.language_model.norm.weight is changed. For the zero-centered norm, the target is BF16((1 + original_weight) * 1.3 - 1). The builder first verifies that the parent equals BF16((1 + original_weight) * 1.4 - 1) exactly. Every other tensor is inherited, including Wait/Hmm changes and the thinking token. The leader's source-free output-row edit is inherited, not independently reimplemented or inferred from its title.

Gain 1.3 was chosen in advance, with no search or local accuracy selection. This is a calibration ablation, not a claimed novel training method. Attribution: parent by mihaikoodo, earlier normalization recipe PR #91 by ilblackdragon; range reader adapted from Congoss PR #96.

## Reproduction

Python 3.13, torch 2.15.0, safetensors 0.8.0, numpy 2.5.5; Delta CLI 0.4.0.

    python -m pip install torch==2.15.0 safetensors==0.8.0 numpy==2.5.5 deltaleague==0.4.0
    python build.py
    delta validate submission

Keep build.py and range_io.py together. The helper's original standalone experiment is not invoked. CPU-only build reads two small tensors and safetensors headers through checked HTTP ranges. No full checkpoint download or model inference. Seed: none; deterministic calculation. No sampling settings are used locally; official evaluation uses its pinned thinking-enabled 8192-token configuration.

See build-report.json for artifact SHA256, size, tensor source hashes, parent content hash and verification findings. The patch is a sparse replacement; no full edited.safetensors checkpoint is generated. Full parent identity is API-reported rather than locally rehashed. Official reconstruction remains authoritative.

## Evidence

Parent verified: 30% private (90/300), 74.71210820260528% validation. Candidate: unmeasured. The earlier tool-direction experiment PR #99 scored 26.333333333333332% private versus its 29% parent; it is not inherited here.

Structural verification checks the known parent norm, finite values, unique sparse indices and exact target reconstruction. None establishes a quality gain. No local public generations were run because the installed runtime/hardware does not support a feasible exact 27B execution. No claim is made that this recipe is globally unique; no matching entry was visible among the seven leaderboard rows inspected.

## Submission

Submitted with user approval on 2026-11-22. [Official submission](https://deltaleague.ai/submissions/aeecf33a-828c-4a4b-abf2-e4a2510209a5/). Status at update: verifying. Validation and private-test scores pending.

## Boundaries

- [x] Only this experiment directory is changed.
- [x] No weights, credentials, temporary signed URLs, private items or raw generations are committed.
- [x] Structural checks, verified parent scores and unmeasured candidate quality are distinguished.
