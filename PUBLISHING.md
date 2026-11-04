# Publishing `deltaleague` on PyPI

The distribution is named `deltaleague`; its executable remains `delta`. Python 3.12+ is required.

```sh
pip install deltaleague
pip install 'deltaleague[local]'  # optional local inference and public validation
```

Version `0.1.0` was published on 2026-11-04 using Trusted Publishing. Version `0.1.1` adds the login confirmation code display required by the server. For development, install from this checkout with `pip install .` or `pip install '.[local]'`.

## One-time account setup

The PyPI account owner adds a pending GitHub Trusted Publisher at https://pypi.org/manage/account/publishing/:

| Field | Value |
| --- | --- |
| PyPI project name | `deltaleague` |
| Repository owner | `ilblackdragon` |
| Repository name | `deltaleague` |
| Workflow filename | `publish.yml` |
| Environment name | `pypi` |

The GitHub environment is also named `pypi`. No long-lived PyPI token is required. A pending publisher does not reserve the name; the first successful upload creates the project.

## Release

1. Set a new version in `pyproject.toml` and push the reviewed changes to `main`.
2. Run **Build and publish Python package** (`publish.yml`). Leave `publish` false for a build-only check or set it true to publish.
3. The build job runs CLI tests, builds wheel/sdist, and validates metadata. The separate publishing job obtains a short-lived PyPI credential through GitHub OIDC and publishes the exact build artifacts. Publication is allowed only from `main`.
4. Verify https://pypi.org/project/deltaleague/ and install the released version into a fresh environment. PyPI versions are immutable; corrections need a new version.

The public distribution contains the standalone CLI and public documentation. The GitHub repository can remain private during launch preparation; PyPI package contents are public.
