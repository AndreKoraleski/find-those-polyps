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
type-checked with [mypy](https://mypy.readthedocs.io/) in strict mode, which
covers `src`, `tests` and `encoders`. All public functions must be fully
annotated. Tests mirror the layout of `src/find_those_polyps`.

Tests never touch the network. They download no dataset and no weights, so that
they run offline and in continuous integration. Build small synthetic inputs
instead, and replace a model factory by a tiny stand in, as the tests of the
DINOv2 and GastroNet encoders do with `monkeypatch`.

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

## Adding a Model

Models of the team live in [`encoders/`](encoders/README.md), one file per model
with a `build()` factory, which [`encoders/template.py`](encoders/template.py)
shows. Copy the template, set its constants and evaluate the file directly.

```bash
uv run polyps-evaluate encoders/my_model.py:build --limit 16  # smoke test
uv run polyps-evaluate encoders/my_model.py:build
```

- `pretrained_on` lists every dataset and corpus seen in pretraining, with the
  keys of [`docs/datasets.md`](docs/datasets.md). Tasks on those datasets and on
  datasets documented to overlap them are skipped. An empty tuple states that no
  data was seen, and leaving it undeclared marks the model on the leaderboard.
- `polyps-data check` searches a pretraining corpus for copies and near
  duplicates of the test images, which catches overlaps the registry does not
  document.
- Names are unique and carry the input size when it differs from 224 pixels,
  for instance `my-model-448`, since they key the leaderboard.

A model becomes a reference encoder in `src/find_those_polyps/baselines/` only
once it is stable and useful to everyone who compares against it. A reference
encoder subclasses `Encoder`, declares `pretrained_on`, is exported from
`find_those_polyps.baselines`, has offline tests in `tests/baselines/` and is
listed among the reference encoders of the README.

Weights never enter git. Each file records where its checkpoint comes from and
under which licence, in its module docstring. Some licences forbid
redistribution, as the model usage agreement of the GastroNet weights does, so
such checkpoints are never shared, even within the team, and every user obtains
them from their source. Results obtained with a model cite the works its
licence or its authors require.

## Adding a Protocol or a Metric

A protocol is a pydantic model subclassing `Protocol` in
`src/find_those_polyps/evaluation/protocols/`. It sets `name`, the task `kinds` it
applies to, `dense` when it consumes feature maps and `stochastic` when its
result depends on the seed, and implements `predict`. A protocol measuring the
representation itself subclasses `DirectProtocol` and implements `measure`
instead. Register the default instance in `PROTOCOLS` and describe it in the
protocol table of the README.

Hyperparameters are fields of the protocol and enter its hash. Changing a
default therefore makes every earlier row of that protocol stale, so change one
only on purpose and say so in the pull request.

A metric is described in `METRIC_INFO` of `src/find_those_polyps/metrics/` with
its direction and its source, computed in the function of its task family and
added to the names that `metric_names` reports for the task kinds it applies to.
Test it against a reference implementation, such as scikit-learn, whenever one
exists.

## Publishing Results

`results/leaderboard.csv`, `results/leaderboard.md` and
`results/leaderboard.xlsx` hold published results only.

- Publish full evaluations only, never smoke tests, which write to
  `results/smoke/`, nor runs restricted with `--limit`.
- Never edit the tables by hand. Every evaluation rewrites all three from the
  CSV file. After merging rows from another machine into the CSV file,
  regenerate the others with
  `uv run python -c "from pathlib import Path; from find_those_polyps.evaluation.leaderboard import Leaderboard; Leaderboard(Path('results/leaderboard.csv')).refresh()"`.
- An interrupted evaluation continues with `--resume`, which skips every setting
  already on the leaderboard. Running elsewhere, copy the current
  `results/leaderboard.csv` there first, so that its rows are kept and only the
  missing ones are added.
- Check before committing that the rows of every other model are unchanged, and
  commit the three tables together, with a message naming the model added.
- Per image records under `results/per_image/` are regenerated by every
  evaluation and stay out of git.

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

Changes reach `main` through pull requests, including those of members with
write access. GitHub enforces this on `main`. A pull request merges only once
another member approved it, both CI checks pass on a branch up to date with
`main` and every review conversation is resolved. New commits dismiss earlier
approvals, and force pushes to `main` and its deletion are refused.
Administrators may push small changes directly, such as documentation fixes and
leaderboard updates, after running every check locally. Larger changes go
through a pull request whoever makes them.

1. Create a branch from `main`.
2. Keep each pull request focused on a single change.
3. Ensure all checks pass before requesting a review.
4. Ask another member of the team to review, and merge only after their
   approval.
