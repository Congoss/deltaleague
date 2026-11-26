# Tool-response closing input embedding x0.95 on the 27B leader

## Hypothesis and scope

Experiment directory: research/congoss-27b-toolresponse095/.
Congoss requested a new bounded experiment and explicitly authorized submission. Codex selected a one-row amplitude ablation. Available evidence: public aggregate leaderboard scores, tokenizer/template, parent tensor bytes and earlier research. No private examples or generations were accessed.

Model: qwen38-27b-nvfp4-ref.
Parent: 5065b8a8-e87d-4f06-b52b-183fcb3bab8c, the 30% Wait/Hmm leader by mihaikoodo, derived from ilblackdragon's PR #91.
Evaluation: qwen38-27b-nvfp4-agentic300-8k-v1.

Hypothesis: mildly attenuating the closing tool-response marker's input representation might change post-tool continuation beneficially. This is not evidence that the marker causes excessive reasoning. Scaling may be partly canceled by normalization, may have no benefit, or may impair boundary recognition and correct use of tool results. No causal link between this marker and truncation has been measured.

Only row 248067 (</tool_response>) of model.language_model.embed_tokens.weight is replaced with BF16(float32(parent_row) * 0.95). Token identity is verified against the hashed parent tokenizer and the literal marker is checked in the hashed parent chat template. This does not prove the private evaluator uses that template verbatim. All output rows, final normalization and transformer weights are inherited unchanged. No guarantee is made about where the token appears in actual prompts.

Scale 0.95 was chosen beforehand for one small amplitude ablation. No search, training or cross-token direction mixing. The earlier </think>-toward-tool-call patch PR #99 failed to improve and is not inherited. No matching title appeared in the inspected leaderboard, which does not establish global uniqueness or exclude an unpublished duplicate.

## Reproduction

Python 3.13; torch 2.15.0, safetensors 0.8.0, numpy 2.5.5, CLI 0.4.0.

    python -m pip install torch==2.15.0 safetensors==0.8.0 numpy==2.5.5 deltaleague==0.4.0
    python build.py
    delta validate submission

Keep range_io.py alongside build.py. It is the checked HTTP range reader from Congoss PR #96; its older standalone main is not used. The build runs on CPU with bounded header/one-row reads plus tokenizer and template downloads. No full checkpoint is downloaded. No seed is needed. No local generation is run; the official evaluation has thinking enabled and an 8192-token budget.

Artifact hash, exact size, changed-entry count, source row hash, metadata parent hash and structural checks are recorded in build-report.json. The output is a sparse replacement patch, not a complete edited checkpoint. Full parent hash is obtained from public API metadata, not locally recomputed. Server reconstruction remains authoritative.

## Evidence

Parent verified score at preparation: 30% private (90/300), validation 74.71210820260528%. Candidate quality is unmeasured. Local checks cover finite BF16 values, unique in-range sparse indices, exact target reconstruction and manifest integrity. No local public task accuracy is available because this PC has no feasible configured runtime for the exact 27B model. Structural validity is not evidence of improvement. Private truncations do not prove looping.

## Submission

User approved one submission on 2026-11-26. URL and state will be recorded after upload. No result claimed yet.

## Boundaries

- [x] Only this experiment directory is changed.
- [x] No weights, credentials, signed URLs, private items or raw generations in the PR.
- [x] Unmeasured candidate quality is separated from verified parent scores and local structural checks.
