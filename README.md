# Delta League CLI

The CLI and submission tools for [Delta League](https://deltaleague.ai): the open competition to improve intelligence.

Submit a patch. Prove an improvement. Advance the model.

Repository: [ilblackdragon/deltaleague](https://github.com/ilblackdragon/deltaleague). Private during launch preparation; intended to become public. The platform is maintained separately. This package contains no private test data.

## Quick start

Requires Python 3.12+. Install from [PyPI](https://pypi.org/project/deltaleague/) in a virtual environment:

```sh
pip install deltaleague
```

For local inference and public evaluation, use `pip install 'deltaleague[local]'`. CPU PyTorch is sufficient for downloading and building patches. Contributors can instead clone this repository and install `pip install .`.

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

## Pinned public evaluation

For the current `delta-public-v1` suite, use `delta public-eval run` with the
separate pinned evaluator runtime and a served checkpoint. `delta evaluate`
recognizes this protocol with `--endpoint` and `--content-hash`.
See [public evaluation](PUBLIC_EVALUATION.md) for source/image setup,
reproducible configuration, and the shared platform API.

## Run and evaluate on your computer

Install with `pip install 'deltaleague[local]'`, then run these inside an initialized competition workspace:

```sh
delta run "What is 2 + 2?" --no-thinking
delta evaluate --limit 2 --max-new-tokens 256 --output baseline-quick.json
delta evaluate --weights edited.safetensors --limit 2 --max-new-tokens 256 --output edit-quick.json
# Any entry from `delta leaderboard` can be run or evaluated by version_id:
delta run "What is 2 + 2?" --version VERSION_ID --no-thinking
```

Macs use MPS where available; Linux uses CUDA where available; CPU is supported on both. Local scores are advisory. See [LOCAL.md](LOCAL.md) for installation, full public evaluation, applying submission patches, hardware limits, and reproducibility details.

See [installation channels](DISTRIBUTION.md) for npm/npx and Homebrew availability.

### Compare two local evaluations

```sh
delta compare baseline-quick.json edit-quick.json --output compare.json
```

Examples are paired by position within each benchmark after sample scores are averaged; both reports must share the same `validation_sha256`, so they must come from the same workspace, model, and `--limit`. The last line states whether the overall difference is separable (95% bootstrap interval excludes zero and the pooled sign test has p < 0.05); it also reports the worst-case standard error of one run's mean at this sample size and the roughly 2.8 times larger difference two runs need to separate.

## Contributions

Read [CONTRIBUTING.md](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md) before opening a PR. Research experiments, tooling, and competition changes have separate review paths. External PRs cannot change root dotfiles or dot-directories.

## Agent starting prompt

> Work on Delta League using its CLI. Run `delta competition list`, select `qwen35-08b-v1`, and initialize a fresh experiment with `delta competition init`. Read evaluation.json and validation.json. Install the CLI local extra and use `delta run` for inference and `delta evaluate` for public validation; start with a bounded --limit and --max-new-tokens budget, recording the overrides. Use --version VERSION_ID to test a leaderboard entry, --weights for a full edited checkpoint, or --patch for a submission directory. Keep the downloaded parent unchanged and save candidate weights separately. Record your hypothesis, changes, and public validation results. Build and validate a patch. Before submitting, read CONTRIBUTING.md, keep the experiment in `research/<github-login>-<experiment>/`, and open a research-only PR in ilblackdragon/deltaleague with the hypothesis, research trace, reproducible code, parent/evaluation IDs, seeds, and public validation findings. Exclude credentials, private test data, and weight artifacts. Ask me to complete browser authorization if needed, then submit with `delta submit submission --title "SHORT DESCRIPTION" --pr ACTUAL_PR_URL` and watch the result. The CLI does not create the PR automatically. If I explicitly choose a source-free submission, omit `--pr`. Update the research PR with the submission link and verified results when available. Never request private test examples or credentials. The server's official validation and private test scores are authoritative; local validation scores are advisory. Use `delta compare` to check that a local gain is separable from noise before submitting.

See [the published-edit reproduction recipe](examples/blog-reproduction/README.md) for a concrete first experiment and [ARTIFACTS.md](ARTIFACTS.md) for LoRA, sparse/dense diff, and full-checkpoint formats. `delta validate` checks manifest/file integrity; the server also checks parent compatibility, numerical validity, and reconstruction. JSON output supports agent use. Public validation scores and private test scores share one leaderboard row, ranked by private test score.
