# Delta League CLI

The CLI and submission tools for [Delta League](https://deltaleague.ai): the open competition to improve intelligence.

Submit a patch. Prove an improvement. Advance the model.

Repository: [ilblackdragon/deltaleague](https://github.com/ilblackdragon/deltaleague). Private during launch preparation; intended to become public. The platform is maintained separately. This package contains no private test data.

## Quick start

Requires Python 3.12+. Install from [PyPI](https://pypi.org/project/deltaleague/) in a virtual environment:

```sh
pip install deltaleague
```

For portable local inference, use `pip install 'deltaleague[local]'`. The current public suite uses the separate `[public-eval]` environment described in [PUBLIC_EVALUATION.md](PUBLIC_EVALUATION.md). CPU PyTorch is sufficient for downloading and building patches. Contributors can instead clone this repository and install `pip install .`.

```sh
delta competition list
delta competition init qwen35-08b-v1 --output experiment
cd experiment
```

This selects a model competition and downloads its original checkpoint into `model/`, public examples into `validation.json`, and scoring/generation settings into `evaluation.json`. `delta-workspace.json` records the API, immutable parent version, and evaluation version. Use `--parent VERSION_ID` to start from another submission in the same competition. Downloads verify hashes, reuse completed files on retry, and avoid overwriting modified files.

Have your agent edit `model/model.safetensors` into a **separate** `edited.safetensors` file. Preserve tensor names, shapes, and dtypes. Keep the parent unchanged. Then:

```sh
delta patch build --parent model --modified edited.safetensors --output submission
delta validate submission
delta login --no-browser
# Open the research PR first; replace PR_URL with its actual URL.
delta submit submission --title "My weight edit" --pr PR_URL
delta submission status SUBMISSION_ID --watch
delta submission results SUBMISSION_ID --json
delta leaderboard --model qwen35-08b-v1
```

Before submitting, open a research PR in [the public repository](https://github.com/ilblackdragon/deltaleague) containing your hypothesis, research trace, edit/reproduction code, parent and evaluation IDs, seeds, and public validation findings. Attach its actual URL with `--pr`. The CLI records the link; it does not create the PR. Source-free submissions remain supported by omitting `--pr`.

Open the authorization URL in a browser and sign in with GitHub. `delta login` opens the browser automatically; `--no-browser` suits remote shells and agents. Authentication is only required for submission/account actions. Credentials are stored with mode 0600 and scoped to the API origin. Put `--api URL` before a command for another deployment.

If a transfer fails, rerun the same `delta submit` command. Version 0.1.2 saves `.delta-upload.json` before uploading and retries unfinished files using the same session. Keep this file private: while a transfer is pending, it contains signed upload URLs. Unfinished sessions expire after one hour; a later retry creates a new session. The two-session upload limit is separate from the submission quota shown by `delta quota`.

## Pinned public evaluation

For the current `delta-public-v1` suite, use `delta public-eval run` with the
separate pinned evaluator runtime and a served checkpoint. `delta evaluate`
recognizes this protocol with `--endpoint` and `--content-hash`.
See [public evaluation](PUBLIC_EVALUATION.md) for source/image setup,
reproducible configuration, and the shared platform API.

## Run and evaluate on your computer

Install with `pip install 'deltaleague[local]'` for prompt-level checks inside an initialized competition workspace:

```sh
delta run "What is 2 + 2?" --no-thinking
delta run "What is 2 + 2?" --weights edited.safetensors --no-thinking
# Any entry from `delta leaderboard` can be run by version_id:
delta run "What is 2 + 2?" --version VERSION_ID --no-thinking
```

Macs use MPS where available; Linux uses CUDA where available; CPU is supported on both. Local scores are advisory. See [LOCAL.md](LOCAL.md) for installation, full public evaluation, applying submission patches, hardware limits, and reproducibility details.

See [installation channels](DISTRIBUTION.md) for npm/npx and Homebrew availability.

### Compare two evaluations

For completed public-evaluator runs:

```sh
delta compare baseline-results/result.json candidate-results/result.json --output compare.json
```

Public reports must cover the same complete pinned suite. Items are paired by
benchmark and ID, and each of the 18 benchmarks receives equal weight. No
`.generations.jsonl` files are needed. Checkpoint hashes identify the declared
models; matching serving runtimes are not attested. See
[PUBLIC_EVALUATION.md](PUBLIC_EVALUATION.md) for setup and report details.

Legacy local report pairs remain supported with their generation sidecars; see
[LOCAL.md](LOCAL.md#compare-two-runs). Public and legacy formats cannot be mixed.
The command reports paired confidence intervals and an exploratory `separable`
screen. Neither a positive screen nor a small interval establishes reproducibility
across generation runs; a negative screen does not establish equivalence.

## Contributions

Read [CONTRIBUTING.md](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md) before opening a PR. Research experiments, tooling, and competition changes have separate review paths. External PRs cannot change root dotfiles or dot-directories.

## Agent starting prompt

> Work on Delta League using its CLI. Run `delta competition list`, select `qwen35-08b-v1`, and initialize a fresh experiment with `delta competition init`. Read evaluation.json and validation.json. Use the CLI local extra and `delta run` for bounded prompt-level checks with --max-new-tokens; --version, --weights and --patch select local checkpoints. For protocol_version delta-public-v1, follow PUBLIC_EVALUATION.md: use the pinned public-eval environment, serve each checkpoint, and run the full suite with its own --content-hash. Do not pass legacy --limit, --weights or generation overrides to this evaluation. Compare the completed result.json files with `delta compare`. For legacy configurations only, use LOCAL.md for bounded `delta evaluate` runs. Keep the downloaded parent unchanged and save candidate weights separately. Record your hypothesis, changes, and public validation results. Build and validate a patch. Before submitting, read CONTRIBUTING.md, keep the experiment in `research/<github-login>-<experiment>/`, and open a research-only PR in ilblackdragon/deltaleague with the hypothesis, research trace, reproducible code, parent/evaluation IDs, seeds, and public validation findings. Exclude credentials, private test data, and weight artifacts. Ask me to complete browser authorization if needed, then submit with `delta submit submission --title "SHORT DESCRIPTION" --pr ACTUAL_PR_URL` and watch the result. The CLI does not create the PR automatically. If I explicitly choose a source-free submission, omit `--pr`. Update the research PR with the submission link and verified results when available. Never request private test examples or credentials. The server's official validation and private test scores are authoritative; local validation scores are advisory. Use paired differences and uncertainty to assess local evidence before submitting; the exploratory separable flag does not establish a reproducible gain.

See [the published-edit reproduction recipe](examples/blog-reproduction/README.md) for a concrete first experiment and [ARTIFACTS.md](ARTIFACTS.md) for LoRA, sparse/dense diff, and full-checkpoint formats. `delta validate` checks manifest/file integrity; the server also checks parent compatibility, numerical validity, and reconstruction. JSON output supports agent use. Public validation scores and private test scores share one leaderboard row, ranked by private test score.
