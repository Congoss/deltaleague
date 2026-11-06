# Four-way checkpoint composition screen

Parent 344c8376-66a1-413b-abca-da2b5b986044 on public-ladder500-8k-128-v1.
Compared parent, a 0.85 doubt projection with the existing half-strength repetition edit, a final RMSNorm gain multiplier 1.05 on the parent, and both changes.
Projection source PR #22 revision 0f93e2f340ff74f1f409f25aae78cd6cd2389da4. The 28 reconstructed fully projected tensors differed from the downloaded parent at relative Frobenius errors roughly 2e-6 to 2e-5, within the recorded 0.001 guard. The 0.85 arm is a local reconstruction, not a claim of bit-exact identity with leaderboard entry #2.

CPU float32 inference, four threads. Deterministic public no-thinking answer-choice diagnostic used ARC-Challenge test rows 12-35 and BoolQ validation rows 12-35, 48 total, preserving the scoring method of the earlier screen. The files and individual diagnostic scores are local.

| Arm | Correct out of 48 | Mean option NLL |
| --- | ---: | ---: |
| Parent | 31 | 0.803645 |
| Soft doubt | 30 | 0.771804 |
| Norm x1.05 | 31 | 0.817921 |
| Both | 30 | 0.785271 |

Generation used public GSM8K index 28, seed 1028, thinking enabled, temperature 1, top-k 20, top-p 0.95, generated-token presence penalty 1.5, cap 1024 versus the official 8192. Expected answer 25.
Parent reached 1024 tokens without closing. Soft doubt finished correctly at 1009 tokens. Norm x1.05 finished correctly at 906 tokens. Both finished at 685 tokens but gave an incorrect answer.

The running generation script initially mis-scored boxed answers because of an incorrectly escaped regex. Raw public generations were retained locally for this experiment, the extractor was corrected, and the table above uses rescored answers. Earlier delta-dose and delta-asymmetric generation correctness flags are invalid and their records now disclose this. Their token counts and answer-choice scores are unaffected.

A single mathematical example and 48 no-thinking diagnostic questions cannot establish improved official accuracy. Norm x1.05 is the leading candidate from this limited screen because it retained parent answer-choice accuracy and produced the correct final answer on the generation example. A sparse patch was built and CLI integrity validation passed. The official submission was rejected with duplicate_content. Delta League already has a checkpoint with these weights. The server returned no score for this submission.

variants.py defines the two component edits. evaluate.py and generate.py reproduce inference using the cached parent and original paths. submission-norm105/manifest.json identifies the exact parent and artifact hash.

## Reproduction

Initialize the parent with delta competition init qwen35-08b-v1 --parent 344c8376-66a1-413b-abca-da2b5b986044 --output delta-dose/experiment and the original with delta competition init qwen35-08b-v1 --output delta-experiment. Run python build.py delta-dose/experiment/model submission from the directory holding those workspaces, with the path to this build.py. Use Python 3.13, PyTorch 2.15, safetensors 0.8 and NumPy. Expected artifact SHA-256 a185d3d4be5e508cd994ee83edee601364f2b6930d949c69e73f13a7c9af7453. The edit is deterministic.

For the diagnostics, copy evaluate.py, generate.py and variants.py into a sibling delta-combinations folder. Use Transformers 5.22.1. Place doubt_direction_L13.json from PR #22 at the pinned revision above in that folder. Fetch dataset viewer rows with offset 12 and length 24 from allenai/ai2_arc config ARC-Challenge split test into arc.json and aps/super_glue config boolq split validation into boolq.json. Run both diagnostic scripts. Keep generation output local.

