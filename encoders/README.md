# Team encoders

This folder holds the encoders of the team, one Python file per model, kept apart from the reference
encoders of the package in `src/find_those_polyps/baselines/`. A model moves there only once it has
become a stable reference for others to compare against.

## Adding a model

1. Copy [`template.py`](template.py) under the name of the model, for instance `ssl_vits16.py`.
2. Set `NAME`, `DESCRIPTION`, `PRETRAINED_ON` and `WEIGHTS`, and adapt `build` to the architecture.
   The template wraps a timm ViT with `TorchEncoder`. Subclassing `Encoder`, as the reference
   encoders do, gives full control over every stage.
3. Put the checkpoint under `encoders/weights/`, which git ignores, or download it on first use.
4. Run a smoke test, then the full evaluation.

```bash
uv run polyps-evaluate encoders/ssl_vits16.py:build --limit 16
uv run polyps-evaluate encoders/ssl_vits16.py:build
```

Several models, including the references, can be evaluated in one pass, which reads the images of
every task once for all of them.

```bash
uv run polyps-evaluate encoders/ssl_vits16.py:build find_those_polyps.baselines:DINOv2
```

## Rules

- Names are unique across the team, since they key the leaderboard. Results at another resolution
  carry the size in the name, for instance `ssl-vits16-448`.
- `PRETRAINED_ON` lists every dataset seen in pretraining, with the keys of
  [`docs/datasets.md`](../docs/datasets.md). Tasks on those datasets and on datasets documented to
  overlap them are skipped. Declaring nothing would score the model on images it already saw.
- The factory raises when its checkpoint is missing, so that random weights are never evaluated
  under the name of a trained model.
- Checkpoints never enter git. Record where each one comes from and its licence in the docstring of
  its file.
