# Installation channels

All channels run the same Python CLI published as `deltaleague`. The Python executable is `delta`; the npm launcher is named `deltaleague`.

## PyPI — published

```sh
pip install deltaleague
pip install 'deltaleague[local]'  # add local inference and evaluation
```

Python 3.12+ is required. See [PUBLISHING.md](PUBLISHING.md) for releases.

## npm / npx — prepared, first publication pending npm authorization

After the npm package is published:

```sh
npx deltaleague competition list
npm install -g deltaleague
deltaleague competition list
```

Requires Node.js 20+ and Python 3.12+ on macOS or Linux. The launcher caches isolated Python environments, pins the Python package version, and automatically adds the local extra for `run` and `evaluate`. It neither bundles Python nor rewrites the CLI in JavaScript. See [npm/README.md](npm/README.md).

To release, run `npm test --prefix npm`, inspect `npm pack ./npm --dry-run`, then run `npm publish --access public` from `npm/` with an authorized npm account. The launcher version and its Python version pin must identify an existing PyPI release. Keep npm credentials outside source control. The first package publication registers the npm name.

## Homebrew — private tap during launch testing

```sh
brew install ilblackdragon/deltaleague/deltaleague
```

Alternatively:

```sh
brew tap ilblackdragon/deltaleague
brew install deltaleague
```

The tap lives in [ilblackdragon/homebrew-deltaleague](https://github.com/ilblackdragon/homebrew-deltaleague). It remains private until release and requires GitHub access for now. This is a third-party tap, not Homebrew/core; plain `brew install deltaleague` requires adding the tap first.

The formula installs the pinned PyPI source into a private Python environment, with local inference dependencies. The source checksum is verified. Transitive dependencies resolve at install time. Linux uses CPU PyTorch; use the pip installation path for a separately configured NVIDIA CUDA environment. Formula updates and macOS/Linux installation checks are maintained in the tap repository.
