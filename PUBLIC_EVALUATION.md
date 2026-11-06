# Reproduce public validation

`delta public-eval` runs the frozen `delta-public-v1` protocol used for public
validation: 1,110 items and 18 equally weighted benchmark percentages. It uses
native lm-eval tasks, HumanEval/MBPP pass@1, and BFCL single/multi-turn checkers.
The bundled manifest is byte-for-byte the existing `qwen-small-public-v1` suite.
This extraction changes packaging and invocation, not selections or scoring.

Private competition tests, private grading, worker credentials, scheduling and
leaderboard publication are not part of this package. Reports produced locally
are advisory. A supplied content hash records which checkpoint you say you are
serving; an external inference endpoint does not attest its weights.

## Install the pinned evaluator

Use Python 3.12 on Linux and a separate environment. BFCL requires NumPy 1.x;
vLLM and the platform may need incompatible dependencies. The evaluator calls a
separate GPU inference server and uses CPU-only PyTorch itself.

From this repository at the reviewed commit (until a release includes it):

```sh
python3.12 -m venv .venv-public
.venv-public/bin/pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
.venv-public/bin/pip install '.[public-eval]'
.venv-public/bin/python -m delta_cli.public_eval.install_wasi "$PWD/.wasi"
export DELTA_WASI_ROOT="$PWD/.wasi"
.venv-public/bin/delta public-eval config
```

For an installed release, replace `'.[public-eval]'` with the exact
`'deltaleague[public-eval]==VERSION'`. Install the package version that matches the
suite/configuration; an unknown or modified suite fails before model inference.
The package includes `requirements.txt`, the public manifest, runtime settings,
and the protocol/coverage validator. Base CLI imports do not require lm-eval,
BFCL, vLLM or Wasmtime.

## Serve the checkpoint

Download the checkpoint and tokenizer with `delta model pull VERSION_ID --output
model`, or prepare your candidate checkpoint with the same tokenizer. Serve it
with the digest-pinned vLLM base recorded by `delta public-eval config`, in its own
GPU environment:

```sh
vllm serve /absolute/path/to/model --served-model-name candidate \
  --host 127.0.0.1 --port 8000 --language-model-only --dtype bfloat16 \
  --max-model-len 34816 --max-num-seqs 32 --gpu-memory-utilization .85 \
  --no-enable-log-requests
```

Both `/v1/completions` (including echoed prompt log probabilities) and
`/v1/chat/completions` must work. Text uses native few-shot prompts, metrics and
generation defaults from lm-eval 0.4.13. Generative tasks apply the checkpoint's
chat template with thinking disabled; likelihood tasks use completion prompts.
Coding and BFCL use greedy, non-thinking generation capped at 4,096 tokens.
The protocol seed is in the manifest. Private-test generation settings returned
by an evaluation configuration are not substituted for public task settings.

## Evaluate

Use the 64-character model content hash from `delta model pull` metadata for the
exact checkpoint being served. A changed candidate must use its own content
hash, not its parent's hash.

```sh
.venv-public/bin/delta public-eval run \
  --endpoint http://127.0.0.1:8000/v1 --model candidate \
  --tokenizer /absolute/path/to/model \
  --content-hash CHECKPOINT_CONTENT_HASH --output public-results
```

For an initialized competition workspace, the existing command recognizes the
current protocol and validates the downloaded manifest before dispatching:

```sh
.venv-public/bin/delta evaluate --workspace experiment \
  --endpoint http://127.0.0.1:8000/v1 --served-model candidate \
  --content-hash CHECKPOINT_CONTENT_HASH --output public-results
```

The tokenizer defaults to `experiment/model`; override it with `--tokenizer` if
needed. `--version`, `--weights` and `--patch` do not change an already-running
server: serve the desired checkpoint first. The pinned evaluator runs the full
suite and rejects example limits, benchmark subsets and generation overrides.
Legacy validation configurations continue using the existing local evaluator.

`--concurrency` defaults to 32 and is recorded in the report. Coding execution
has a separate four-sandbox limit. Generated code runs in hash-pinned CPython
3.12.10/WASI with Wasmtime 50.0.1, a read-only standard library, no network or host
environment, bounded memory/fuel/time/output, and a completion marker. An
infrastructure failure is an error, not a model zero.

Outputs must go to a fresh directory:

- `run.json`: protocol/source hashes, suite identity, runtime configuration,
  requested checkpoint/tokenizer, concurrency, dependency versions and status.
- `reports/`: native per-benchmark evidence and public responses.
- `result.json`: complete public item scores, evidence, benchmark percentages and
  aggregate score. It is written only after coverage and identity checks pass.

Interrupted or failed evaluations retain a failed `run.json` and no completed
result. For a retry, choose a new output directory. This is not an upload or an
API submission and never changes the leaderboard.

Compare two completed runs with:

```sh
delta compare baseline-results/result.json candidate-results/result.json \
  --output comparison.json
```

The comparison validates both reports against the bundled public suite and pairs
scores by benchmark and item ID, regardless of row order. It recomputes scores
from the items and keeps the equal-weight mean of 18 benchmarks. Different
checkpoint content hashes are expected; both runs must use the same pinned
protocol and complete item coverage. Reference-check reports and mixed
public/legacy reports are rejected. No generation sidecar is needed.

Confidence intervals resample paired items within each benchmark. They do not
measure variability across generation runs or correct for repeated candidate
selection, and assume independence of items within benchmarks. The `separable`
flag is an exploratory screen: the interval must exclude zero and the pooled
item sign test must have p < 0.05 and agree in direction. Pooled signs weight
items equally rather than benchmarks, so this is not a calibrated test of the
benchmark-weighted mean. A negative screen does not establish equivalence.
Use matching serving settings and retain each run's `run.json` for
runtime provenance: the result contract does not attest endpoint weights or
matching runtimes. Token/truncation summaries are unavailable (`null` in JSON,
`n/a` in text), and `generation_differs` is `null` rather than claiming settings
match. Legacy local report comparisons remain supported.

## Public image

`containers/public-evaluator.Dockerfile` builds the same public package plus the
pinned vLLM base, a separate `/opt/eval` CPU environment, cached public datasets,
and the WASI runtime. It copies an explicit public-source allowlist. The default
entry point is `/opt/eval/bin/delta public-eval`; GPU serving remains a separate
process. For local testing:

```sh
docker build -f containers/public-evaluator.Dockerfile -t delta-public-evaluator .
docker run --rm delta-public-evaluator config
```

The **Build public evaluator image** workflow can build and smoke-test any
reviewed branch without publishing. Publishing requires explicit dispatch on
`main`. The dedicated package is
`ghcr.io/ilblackdragon/deltaleague-public-evaluator`; a maintainer must set that
package's visibility to **public** on its first publication. The workflow checks
anonymous registry access and exports the immutable digest only after that
check passes. Until that release completes, build from source; this PR does not
claim that a public image is already available. Do not use the private platform
worker image for local reproduction.

With the published digest, one possible Linux GPU setup is:

```sh
# Replace IMAGE with ghcr.io/ilblackdragon/deltaleague-public-evaluator@sha256:DIGEST.
docker run --rm --gpus all --ipc=host -p 127.0.0.1:8000:8000 \
  -v /absolute/path/to/model:/model:ro --entrypoint python3 IMAGE \
  -m vllm.entrypoints.openai.api_server --model /model --served-model-name candidate \
  --host 0.0.0.0 --port 8000 --language-model-only --dtype bfloat16 \
  --max-model-len 34816 --max-num-seqs 32 --gpu-memory-utilization .85 --no-enable-log-requests

# Run in another terminal. Use a new subdirectory under the mounted results directory.
docker run --rm --network host -v /absolute/path/to/model:/model:ro \
  -v /absolute/path/to/results:/results IMAGE run \
  --endpoint http://127.0.0.1:8000/v1 --model candidate --tokenizer /model \
  --content-hash CHECKPOINT_CONTENT_HASH --output /results/run-1
```

## Platform adoption

Pin a reviewed commit/release of `deltaleague` and the resulting public image
digest. The platform can import the dependency-free protocol validator:

```python
from delta_cli.public_eval.suite import manifest, expected_items, score_public
```

Its trusted worker can invoke the same public evaluator:

```python
from delta_cli.public_eval.runner import evaluate_public

result = evaluate_public(
    endpoint="http://127.0.0.1:8000/v1", model="candidate", tokenizer=model_dir,
    content_hash=trusted_content_hash, output=fresh_public_output_dir,
    concurrency=32,
    progress=lambda completed, total: report_public_progress(completed, total),
)
public_validation = result["public_validation"]
public_evidence = result["public_evidence"]
```

The result's `public_validation` contract is unchanged: protocol, suite hash,
content hash, reference-check flag and complete per-item groups. Validate it on
the controller with `score_public(result, trusted_content_hash)` before accepting
scores. Keep private generation, private scoring, leases, signed URLs, GPU
identity and official publication in the platform wrapper. The platform worker
image may extend the public image and add only its orchestration code.

Adoption is a separate platform change: switch imports/calls, remove duplicate
public implementations, compare saved reports, pin the new runtime, and use the
platform's versioned rollout process. This PR neither changes production imports
nor activates a new official evaluation.

## Provenance and validation

The suite originated in Delta League's public-validation implementation. Dataset
repository revisions, selected row IDs/document hashes, and BFCL bundled-data
hashes are frozen in the manifest. Source datasets retain their upstream terms:
[HumanEval](https://github.com/openai/human-eval),
[MBPP](https://github.com/google-research/google-research/tree/master/mbpp),
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), and
[BFCL](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard).
The manifest contains public coding reference solutions and tests; those are
public benchmark material, not the competition's private test set.

Fast tests run with `python -m pytest -q tests`. Native scorer/reference checks
require the isolated evaluator dependencies and dataset access/cache:

```sh
.venv-public/bin/python -m delta_cli.public_eval.verify --output protocol-checks
```

Reference checks do not produce model scores. A full GPU run is needed to
establish end-to-end numerical/runtime parity on real checkpoints; unit fixtures
and source comparisons establish only the extraction and report contracts.
