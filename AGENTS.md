# Delta League contribution boundaries

Read CONTRIBUTING.md before changing files or opening a PR.

- A research PR changes only one `research/<github-login>-<experiment>/` directory. Keep a README with the hypothesis, lineage, reproducible commands, public validation results, and honest limitations. Do not modify shared tooling, scoring, rules, or another author's experiment in the same PR.
- Tooling improvements are separate PRs. Changes to scoring, benchmark definitions, or competition rules are competition proposals and require explicit maintainer review; they do not automatically change official evaluation.
- External contributors (anyone without repository admin/maintain permission) must not add, edit, delete, or rename root dotfiles or anything in root dot-directories. This includes `.github/`, `.gitignore`, and future root paths starting with `.`. Propose needed infrastructure changes in the PR description for a maintainer to implement separately.
- Never commit weights, credentials, signed download URLs, private test material, or raw generation logs into research records. CI checks filenames, file types, and size limits; it cannot prove that arbitrary text contains no secrets.
- Open the research PR before submitting a weight artifact; attach its URL with `delta submit ... --pr URL`. Update the research record with the submission link and verified results. PR merging is not required to submit, and a PR-reported score is not an official result.
- Use local public validation for development. Do not try to access private test data. Never change the trusted scoring code to claim a model improvement.
- CI does not execute research code. Reproduction is reviewed separately in an appropriate environment. All PRs require maintainer review and passing required checks before merge, subject to the repository administrator's explicit override.
