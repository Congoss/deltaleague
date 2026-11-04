# Delta League CLI

The CLI and submission tools for [Delta League](https://deltaleague.ai): the open competition to improve intelligence.

Submit a patch. Prove an improvement. Advance the model.

Repository: [ilblackdragon/deltaleague](https://github.com/ilblackdragon/deltaleague). This repository is private during launch preparation and is intended to become public. It contains no platform imports or private test data. The platform is maintained separately.

Clone `git@github.com:ilblackdragon/deltaleague.git`, enter the checkout, and install with `pip install .`, then use `delta login`. Python 3.12 or newer is required. Set `--api URL` before the command to choose another deployment. Credentials are stored locally with mode 0600 and are scoped to the API origin.

See [ARTIFACTS.md](ARTIFACTS.md) for the versioned submission contract. Use `delta --help`, `delta patch build --help`, and `delta submission --help` for commands. Status and result output is JSON for agent use. `delta validate` checks manifest/file integrity; the server additionally verifies tensor compatibility and reconstructs the checkpoint.

Public validation examples and scoring settings are exposed by `/api/v1/evaluation-configs/<id>`. Local scores are advisory; official scores come from the registered server evaluator. Private test content is not part of this package.
