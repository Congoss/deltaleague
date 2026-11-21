# Input-embedding ablation on Qwen 3.8 27B NVFP4

## Hypothesis and scope

PR #91 scales both input embeddings and output rows for seven Wait/Hmm tokens and the closing thinking token, together with a final RMSNorm adjustment. Because the 27B model has untied input and output embeddings, their contributions can be separated.

This experiment restores only the eight input embedding rows to the original checkpoint. It tests whether retaining the parent's output-policy calibration while restoring the original input representations improves multi-turn tool-use performance.

- Experiment directory — research/congoss-27b-input-restore/
- Model — qwen38-27b-nvfp4-ref
- Parent — a610fbdb-eb7e-42bc-ba43-19f7922df81f, the verified PR #91 entry
- Original reference — 47f586c3-5e3f-42ac-9991-e3db49e09103
- Evaluation — qwen38-27b-nvfp4-agentic300-8k-v1
- Attribution — Congoss selected this ablation, Codex built it, and the parent recipe is credited to PR #91 and the 0.8B research it cites.

The public leaderboard at preparation time showed 29.0% private for the parent and 22.0% for the original. These are platform results for those checkpoints, not candidate estimates.

## Reproduction

Run the builder with Python 3.13. It uses only the standard library.

~~~sh
python build.py
delta validate submission
python verify.py
~~~

verify.py additionally uses torch 2.15.0 and safetensors 0.8.0. Local CLI version was deltaleague 0.4.0.

The builder obtains public immutable model-version metadata and reads only the safetensors headers and selected rows through checked HTTP Range requests. It downloads and verifies the common tokenizer by its complete SHA-256 and asserts all eight token IDs. It checks that the parent rows equal the original BF16 rows scaled by the published factors, using float32 multiplication and round-to-nearest-even BF16 conversion.

Rows restored to their exact original values are 3655, 11158, 13428, 13784, 77264, 84485, 85152 and 248069. The deterministic sparse replacement changes 40,960 entries in one tensor.

- Tensor — model.language_model.embed_tokens.weight
- Artifact size — 409,752 bytes
- Artifact SHA-256 — 5669cd1ae3aaba053482e3fac7b38de48e252d9e5c0e1ea72642163bd0ac154f
- Edit seeds — none, the build is deterministic

Temporary signed asset URLs are used in memory only. The report stores public model identities and hashes without those URLs.

## Evidence and limitations

| Structural check | Result |
| --- | --- |
| Tokenizer identity and selected token IDs | Verified |
| Published parent scaling on all eight input rows | Exact match |
| Applying replacements to the fetched parent rows | Bitwise match to the original rows |
| Independent safetensors reader | Correct I64 indices, BF16 values, shapes and finite values |
| Sorted unique sparse indices | Verified |
| CLI manifest and artifact integrity | Passed |

The initial build read 13,552,439 bytes over the network. No complete 21.9 GB parent file was downloaded or locally hashed. The full-file hashes in build-report.json come from the public API, while fetched header and row hashes were computed locally. Server-side parent reconstruction remains authoritative.

No local model inference or public accuracy evaluation was run. The local machine has a Radeon RX 470, while the parent's documented native NVFP4 inference path requires a Blackwell GPU. Structural validity does not establish a score improvement.

The ablation may lose benefits of the parent's input-row scaling. Restoring the closing token's input embedding also changes the state transition after that token, so the hypothesis cannot be established from weight inspection alone.

## Submission

Official submission awaits user approval.

## Boundaries

- [x] One research directory only.
- [x] Public metadata and checkpoint bytes only.
- [x] No weight artifacts, credentials, temporary URLs, private test data or raw generations in the PR.
- [x] Structural checks and unmeasured behaviour are distinguished.
