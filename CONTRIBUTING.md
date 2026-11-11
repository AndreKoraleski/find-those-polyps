# Contributing

## Development Environment

The project is managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync                    # create the environment and install dependencies
uv run pre-commit install  # install the pre-commit and commit-msg hooks
```

## Quality Checks

All checks run automatically on commit and in continuous integration. They can
also be run manually:

```bash
uv run pre-commit run --all-files  # lint, format and type-check
uv run pytest                      # run the test suite
```

Code is formatted and linted with [Ruff](https://docs.astral.sh/ruff/) and
type-checked with [mypy](https://mypy.readthedocs.io/) in strict mode. All
public functions must be fully annotated. Tests mirror the layout of
`src/find_those_polyps`.

Docstrings follow the
[Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
with `Args`, `Returns`, `Yields`, `Raises` and `Attributes` sections, which
Ruff checks, and use single backticks for code rather than Sphinx notation.

## Adding a Dataset

Only datasets that anyone can download in batches from files their authors
host, under a licence allowing research use, are registered. Never commit images,
masks or other dataset content. Split files hold sample identifiers only.

1. Add a dataset card to `src/find_those_polyps/registry/data/registry.json`
   with its licence, provenance, source family and every documented overlap
   with other datasets or with pretraining corpora, which need a display name.
2. List the files of an open dataset under `downloads`. Zip archives are
   streamed and pinned by their size and member index, which the following
   command prints. Plain files such as split tables need a checksum.

   ```bash
   uv run polyps-data index https://host/dataset.zip
   ```

3. Add one task entry per labelled problem, with its kind, classes, primary
   metric, label budgets and loader. The generic loaders cover most layouts
   through `loader_options` alone.
   - `folders`: classes given by folder names, optionally several levels up.
   - `table`: classes given by a label column or one hot columns of CSV files.
   - `masks`: images paired with binary masks by file stem.

   Each of them accepts `group`, a regular expression extracting the patient
   or video from the sample identifier, `official_test`, the published test
   partition, and `expand`, nested tarballs to unpack. Write a dedicated
   `Loader` in `src/find_those_polyps/tasks/loaders/` only when no generic
   loader fits.
4. Build the canonical splits, which also synchronizes the registry, bumps its
   version and regenerates `docs/datasets.md`.

   ```bash
   uv run polyps-split --missing
   ```

A test partition published by the authors is always used verbatim, so that
results stay comparable with the literature. Otherwise the benchmark holds out
a stratified fifth of the groups. Samples sharing a group with the test
partition and byte identical images with conflicting labels are excluded and
recorded in the split file.

Canonical splits are never rebuilt silently. `polyps-split` refuses to
overwrite an existing split unless `--force` is given, since every published
result on the task becomes incomparable. `uv run polyps-registry --check`
verifies that the registry hashes and `docs/datasets.md` are current, and
`uv run polyps-evaluate MODULE:FACTORY --dry-run` shows which tasks an
evaluation would run and how their data would be read.

## Dependencies

Add dependencies with uv so that `uv.lock` stays consistent:

```bash
uv add <package>          # runtime dependency
uv add --dev <package>    # development dependency
```

## Commit Messages

Commit messages must follow the
[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)
specification, which is enforced by the `commit-msg` hook:

```text
<type>(<optional scope>): <description>
```

Common types are `feat`, `fix`, `docs`, `refactor`, `test`, `chore` and `ci`.

## Pull Requests

1. Create a branch from `main`.
2. Keep each pull request focused on a single change.
3. Ensure all checks pass before requesting a review.
