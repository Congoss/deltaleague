# Retune the leader's final RMSNorm gain

Congoss selected a single experiment after reviewing the public leaderboard and prior research PRs. Codex implemented the edit and ran the public CPU comparison. The available evidence was the public leaderboard, public validation selections, and downloadable checkpoints.

## Hypothesis

Preserving the leader's token edits while moving the final RMSNorm gain from 1.4 to 1.25 may improve private-test accuracy. Earlier final-norm experiments on the original model motivated this choice. Prior reductions in truncation alone did not reliably improve rank.

- Model: qwen35-08b-v1
- Parent: ab592312-ec77-4f98-b17f-6ea17723b74e
- Original: 0ea62f40-7cd8-4d98-9ebb-484192fa05eb
- Evaluation: public-ladder500-8k-v2
- Deterministic edit: for stored zero-centered gain w, set w to (1 + w) * (1.25 / 1.4) - 1 and cast back to the original dtype.
- Measured gain vector norms relative to the original: 1.4001 for the parent and 1.2501 for the candidate.
- Sparse patch SHA-256: db81bd983b321b1c7f7f8b1ce646e9a65a08a76e802dd76d501ae5098235df67

## Reproduction

Use Python 3.13, deltaleague 0.1.1, torch 2.15.0, transformers 5.22.1 and NumPy. Initialize an experiment with the registered parent and copy edit.py and smoke.py into its root.

```sh
delta competition init qwen35-08b-v1 --parent ab592312-ec77-4f98-b17f-6ea17723b74e --output experiment
cd experiment
python edit.py
delta patch build --parent model --modified edited.safetensors --output submission
delta validate submission
python smoke.py
```

Local validation uses GSM8K indices 3 and 25 from the public manifest, with seeds 1003 and 1025 for both versions. CPU float32 generation uses top-k 20, top-p 0.95, temperature 1.0, thinking enabled and a 768-token cap. It uses the Transformers sampler without the official presence penalty and a shorter cap than the official 8192. It scores a boxed final answer by exact numeric string match after removing commas.

## Local results

| Public index | Leader tokens | Candidate tokens | Leader closed | Candidate closed | Correct (leader / candidate) |
| --- | ---: | ---: | --- | --- | --- |
| 3 | 748 | 768 | Yes | No | No / No |
| 25 | 768 | 768 | No | No | No / No |

The candidate hit the cap on both prompts. The parent hit it on one. Neither scored a correct boxed answer. This tiny CPU check gives no evidence of an improvement and does not estimate leaderboard accuracy. An earlier 256-token pilot only finished its parent arm and is excluded from the paired comparison. The original checkpoint was used for gain verification. Its official private score is 18.6 versus the parent's 28.2 on this evaluation. The candidate has no official score.

Local delta validate passed manifest and artifact hash verification.

## Official submission

[Verified submission](https://deltaleague.ai/submissions/97a0aec5-1a1d-4b53-9ff9-4042295d9d31/), version 86968401-a83c-43fc-808c-7e701d14a8c4. Rank at verification: 3. Private test: 26.0 vs 18.6 original and 28.2 parent. Public validation: 42.17 vs 41.83 original and 41.02 parent. Truncation: 54.8% vs 52.6% parent. All official scores use public-ladder500-8k-v2.

