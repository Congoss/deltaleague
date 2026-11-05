# Selectively damp seven Wait/Hmm tokens on the leader

Congoss authorized an experiment on public checkpoints. Codex compared the original and leader weights and used the seven public token IDs from PR #18. The hypothesis is that half-scale filler embeddings reduce interruptions while preserving the leader's other token edits and final RMSNorm gain.

Model qwen35-08b-v1. Original 0ea62f40-7cd8-4d98-9ebb-484192fa05eb. Parent ab592312-ec77-4f98-b17f-6ea17723b74e. Evaluation public-ladder500-8k-v2.

Initialize both checkpoints with delta competition init, passing the parent ID with --parent for the leader. With Python 3.13, torch 2.15, safetensors 0.8 and NumPy, run python build.py ORIGINAL/model LEADER/model submission and delta validate submission. The edit is deterministic and has no seed. Each selected row is replaced with 0.5 times the original row, cast to the parent dtype. The builder emits a sparse replacement directly to avoid a full checkpoint copy.

Local structural checks verified that each selected row's norm decreased relative to the leader and that the CLI accepted the manifest and hashes. The patch changes 7,153 entries in seven rows of one tensor. Artifact SHA-256: 0c79de379bc5d067deff3933a3c60985b6c0931a4e238e4ea8911371991b2789.

No candidate generation benchmark was run. The public precedent from PR #18 used 0.4 on the original and scored 24.6 private. The registered leader scored 28.2 private and 41.02 public. These scores do not establish a candidate improvement.

## Official results

Verified submission https://deltaleague.ai/submissions/424b062f-6a44-4401-8274-c1a67f2a5dfb/. Version ffd90ff5-3ef8-46ce-8275-8a041a8fe684. Rank 2. Private test 27.8 vs 18.6 original and 28.2 parent. Public validation 41.71 vs 41.83 original and 41.02 parent. Truncation 53% vs 52.6% parent.
