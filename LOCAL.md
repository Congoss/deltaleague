# Run and evaluate locally

Local inference uses Transformers and PyTorch. Apple Silicon Macs select MPS, Linux machines with a compatible PyTorch GPU build select CUDA, and other machines use CPU. `--device cpu`, `--device mps`, or `--device cuda` overrides selection. There is no rented compute, server evaluation quota, or automatic submission involved.

## Install

From the CLI repository, in a Python 3.12+ virtual environment:

```sh
pip install ".[local]"
delta competition init qwen35-08b-v1 --output experiment
cd experiment
```

The extra installs the tested Transformers version and math scorer. For Linux CUDA acceleration, install a compatible CUDA build of PyTorch using [the official installer](https://pytorch.org/get-started/locally/). If your existing PyTorch build is CPU-only, installing the CLI extra will not replace it with a CUDA build. For IFEval configurations, additionally install the CLI's `[ifeval]` extra.

The portable implementation loads full weights, without automatic quantization. Allow several GB of free RAM for Qwen3.5-0.8B and more for long contexts. CPU/MPS use float32; CUDA uses BF16 where supported. On Macs, unsupported MPS operations can fall back to CPU. If your model encounters an unsupported operator or memory error, try `--device cpu` and lower the output budget. macOS acceleration has not been verified on physical Apple hardware in this release; Linux CPU has been exercised with the actual Qwen checkpoint. CUDA selection is implemented but has not been hardware-tested for this CLI release.

## Ask the model

```sh
delta run "What is 2 + 2?" --max-new-tokens 64 --no-thinking
delta run "Explain your solution to 17 * 23." --max-new-tokens 512 --json
```

`run` defaults to 256 output tokens. Thinking behavior otherwise comes from `evaluation.json`. A short budget may end before the final answer; the command reports that. Text goes to stdout, progress to stderr. `--json` adds provenance and token statistics. `--seed` defaults to 1000. Use `--workspace PATH` to work from another directory.

## Run any leaderboard entry

```sh
delta leaderboard --model qwen35-08b-v1
# Copy the desired entry's version_id, not its submission ID.
delta run "What is 2 + 2?" --version VERSION_ID --no-thinking
delta evaluate --version VERSION_ID --limit 2 --max-new-tokens 256 --output entry-quick.json
```

`--version` downloads the immutable resulting checkpoint, verifies its hashes, and caches it under `.delta/models/VERSION_ID` inside the workspace. It works for the original model and submitted model versions, regardless of whether the submission used a LoRA, diff, or full weights. The version must belong to the workspace's model. Downloads need network access but not login. The inference runs locally; no private test data is downloaded. Completed cached files are reused on subsequent commands.

For a fully offline workspace based on that entry, run `delta competition init MODEL_ID --parent VERSION_ID --output another-experiment` while online, then run/evaluate from that workspace without `--version`.

## Try your own edit or patch

```sh
delta run "What is 2 + 2?" --weights edited.safetensors --no-thinking
delta evaluate --weights edited.safetensors --limit 2 --max-new-tokens 256 --output edit-quick.json

delta evaluate --patch submission --limit 2 --max-new-tokens 256 --output patch-quick.json
```

`--weights` takes a full modified safetensors checkpoint, using the workspace parent's tokenizer and configuration. `--patch` takes a submission directory with `manifest.json`; it checks the parent and artifact hashes, then reconstructs supported LoRA/sparse/dense/full artifacts into a temporary checkpoint. Neither changes the parent. You can combine either with `--version` when editing that leaderboard checkpoint; a patch must reference that exact version. Temporary reconstructed files need disk space roughly equal to a full checkpoint and are removed on completion.

## Evaluate public validation

Start small:

```sh
delta evaluate --limit 2 --max-new-tokens 256 --output baseline-quick.json
```

`--limit` selects the first N problems **per benchmark**; `--benchmark gsm8k` selects one benchmark. These are development checks, not representative estimates of the full public score.

For the entire public set with its configured generation settings:

```sh
delta evaluate --output baseline-full.json
delta evaluate --weights edited.safetensors --output edit-full.json
```

Full evaluation can be expensive locally: the starter configuration permits 32,768 generated tokens per example. The CLI prints the number of generations and upper token budget before loading the model. Use Ctrl-C to interrupt. Completed generation records remain in the JSONL file; no complete score report is written for an interrupted run. Choose a fresh output filename to rerun; automatic resume is not implemented.

The JSON report contains per-benchmark scores, the equal-weight mean across selected benchmarks, per-example sample averaging, checkpoint hash, source version, validation hash, runtime versions, actual generation settings, coverage, and truncation statistics. A companion `*.generations.jsonl` contains answers and per-sample scores. Existing output files are never overwritten.

The local scorer shares the platform's answer-scoring implementation: reasoning traces before `</think>` are excluded, math uses math-verify, and multiple-choice answers follow the same extraction rules. Configured samples use the same per-sample seeds. `--max-new-tokens` and `--thinking/--no-thinking` overrides are recorded explicitly.

Scores remain **advisory local results**. Transformers, vLLM, device precision, and sampling implementations can differ even with the same seed. Compare candidates using the same local settings, then submit for the official validation and private test scores. Local evaluation does not upload results or assign a leaderboard rank.
