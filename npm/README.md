# Delta League

Run the [Delta League](https://deltaleague.ai) Python CLI through npm:

```sh
npx deltaleague competition list
npx deltaleague competition init qwen35-08b-v1 --output experiment
cd experiment
npx deltaleague run "What is 2 + 2?" --no-thinking
npx deltaleague evaluate --limit 2 --max-new-tokens 256 --output quick.json
```

Requires Node.js 20+ and Python 3.12+ on macOS or Linux. This small launcher installs the published `deltaleague==0.1.1` Python package into a cached, isolated virtual environment on first use. It does not modify system Python. The first run needs internet access and downloads PyTorch and other dependencies; later invocations reuse the environment. `run` and `evaluate` automatically install the matching `[local]` extra in a separate cached environment.

All arguments, input/output, and exit codes are forwarded to the Python CLI. Use `DELTA_PYTHON=/path/to/python3.12` to select Python. The environments live under `$XDG_CACHE_HOME/deltaleague/npx`, `~/Library/Caches/deltaleague/npx` on macOS, or `~/.cache/deltaleague/npx` on Linux. Removing that directory uninstalls the cached Python environments; it does not remove CLI credentials or experiments.

You can also install the launcher with `npm install -g deltaleague`, then use `deltaleague ...`, or install the Python package directly with `pip install 'deltaleague[local]'` and use `delta ...`.

[CLI guide](https://github.com/ilblackdragon/deltaleague) · [Local inference and public evaluation](https://github.com/ilblackdragon/deltaleague/blob/main/LOCAL.md)
