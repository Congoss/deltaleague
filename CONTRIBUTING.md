# Contributing to Delta League

There are three contribution types. Checks infer the type from changed paths; a title or label cannot override the policy.

| Type | Editable scope | Checks and review |
| --- | --- | --- |
| Research | One `research/<github-login>-<experiment>/` directory | File-boundary and research-data checks; maintainer review. Research code is not run in CI. |
| Tooling | CLI, tests, npm launcher, examples, and general docs | Contribution-policy checks, software tests, and maintainer review. |
| Competition | Scoring rules, benchmark/dataset proposals, `ARTIFACTS.md`, `PROTOCOL.md`, contribution rules, or `research/README.md` | Software checks where applicable and explicit maintainer review. Merging a proposal does not activate an official evaluation version. |

## Protected files

**External PRs cannot change any root path whose first component starts with `.`.** The check covers additions, edits, deletions, and both the old and new names of renames. Examples: `.github/workflows/*`, `.github/CODEOWNERS`, `.gitignore`, `.env`, and any future `.something/` directory.

External means the PR author does not have `admin` or `maintain` permission on this repository. Write/triage access, collaborator association, branch names, and self-applied labels do not grant an exception. Maintainers can implement a necessary infrastructure change separately after reviewing a proposal.

The CLI, scorer, package configuration, and competition contracts are locked against **research PRs**, but may be changed in separately reviewed tooling/competition PRs. Research changes mixed with any shared-file change fail the check. Renaming a protected file into a research directory does not evade the rule.

## Research records

Create `research/<your-github-login>-<experiment>/README.md` with:

- Hypothesis and what the agent/human could access during research.
- Exact model/parent version, evaluation version, seeds, and generation settings.
- Code and commands to reproduce the edit, plus expected tensor/artifact hashes.
- Public validation comparisons with the parent; distinguish local estimates from verified results.
- Submission URL/ID once submitted, and limitations or negative findings.

Keep one experiment per PR. Do not edit another contributor's directory. Regular files are allowed; symlinks and submodules are rejected. Each file must be at most 1 MiB and the experiment directory at most 5 MiB. Upload model weights through the CLI instead of Git.

Research checks reject weight/archive artifact extensions (`.safetensors`, `.bin`, `.pt`, `.pth`, `.ckpt`, `.gguf`, `.npz`), credentials filenames, `.private.` filenames, and `*.generations.jsonl`. Do not evade these checks by renaming files. Aggregate public scores and reproduction manifests are welcome. A size/path check does not establish data provenance or detect every possible secret; maintainer review remains required.

Data placed in this repository or its PRs must be suitable for public release. Proposed public benchmark material belongs in a separate competition PR, preferably under `benchmarks/public/`. Genuine hidden evaluation examples belong in the private platform repository, not here. Existing research PRs may need restructuring to meet the new rules.

## PR workflow

Use a template from `.github/PULL_REQUEST_TEMPLATE/`: `research.md`, `tooling.md`, or `competition.md`. Split research from tooling changes. Linking a research PR to a submission does not require the PR to be merged.

Automation applies `type:research`, `type:tooling`, or `type:competition` and adds `policy:blocked` when boundaries fail. Labels are informational; the required checks enforce the result. Inspect their summaries for violations.

Before merging, `main` requires **Contribution policy**, **Research validation**, and **CLI tests**, an up-to-date branch, code-owner approval, and resolved review conversations. New commits dismiss stale approvals. Administrator override remains available for emergency/maintainer operations; external contributors cannot bypass these requirements.

The trusted policy workflow reads file metadata using base-branch code and never executes PR code. Software tests run on disposable GitHub-hosted runners with a read-only token, no repository secrets, no publishing environment, and no shared cache. Research-only PRs get metadata checks without executing participant scripts.

Run local tests with `python -m pytest -q tests`, `npm test --prefix npm`, and `node --test .github/scripts/policy.test.cjs` as applicable. A maintainer can rerun policy inspection via the workflow's `pr_number` input; adding `ci:run` to an open PR triggers its software checks.
