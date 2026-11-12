"""Template of a team encoder, a ViT-S/16 loaded from a local checkpoint.

Copy this file under a new name, set the constants below and adapt `build` to the model. Evaluate it
with `uv run polyps-evaluate encoders/<file>.py:build`, or run a smoke test first with `--limit 16`.
The factory takes no argument on the command line, so it reads the checkpoint from `WEIGHTS`.
"""

from pathlib import Path

import timm
import torch
from timm.models.vision_transformer import VisionTransformer

from find_those_polyps import TorchEncoder, normalize, patch_grid, resize

NAME = "template-vits16"
"""Identifier of the model on the leaderboard, unique across the team."""

DESCRIPTION = "Template ViT-S/16, to be replaced by one sentence on architecture, method and data."
"""One sentence summary shown on the leaderboard."""

PRETRAINED_ON: tuple[str, ...] = ("hyperkvasir-unlabeled",)
"""Keys of every dataset seen in pretraining, as listed in `docs/datasets.md`.

Tasks on these datasets, and on datasets documented to overlap them, are skipped, so that no score
is measured on images the model already saw. An empty tuple declares that no data was seen.
"""

IMAGE_SIZE = 224
"""Side in pixels of the square input."""

WEIGHTS = Path(__file__).parent / "weights" / f"{NAME}.pth"
"""Checkpoint read by `build`, a state dictionary saved with `torch.save`, kept out of git."""


def build(weights: Path = WEIGHTS) -> TorchEncoder:
    """Frozen ViT-S/16 with the class token as global and the patch tokens as dense features.

    Args:
        weights: Checkpoint holding the state dictionary of the network, `WEIGHTS` by default.

    Returns:
        The encoder, ready to evaluate.

    Raises:
        FileNotFoundError: If the checkpoint does not exist, so that a missing file never
            evaluates random weights under the name of the model.
        TypeError: If timm does not build a vision transformer.
    """
    if not weights.is_file():
        raise FileNotFoundError(f"missing checkpoint {weights}")
    model = timm.create_model(
        "vit_small_patch16_224", pretrained=False, num_classes=0, img_size=IMAGE_SIZE
    )
    if not isinstance(model, VisionTransformer):
        raise TypeError("expected a timm vision transformer")
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    model.eval().requires_grad_(False)
    prefix = model.num_prefix_tokens
    return TorchEncoder(
        model.forward_features,
        name=NAME,
        description=DESCRIPTION,
        image_size=IMAGE_SIZE,
        preprocess=lambda image: normalize(resize(image, IMAGE_SIZE)),
        postprocess=lambda tokens: tokens[:, 0],
        postprocess_dense=lambda tokens: patch_grid(tokens[:, prefix:]),
        pretrained_on=PRETRAINED_ON,
    )
