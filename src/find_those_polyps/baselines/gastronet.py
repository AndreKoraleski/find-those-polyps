"""GastroNet reference encoder, DINOv2 trained on endoscopy images by Jong et al. (2025).

The ViT-B/14 was pretrained with DINOv2 on GastroNet-5M, about 4.8 million unlabelled images from
some 500 thousand procedures of the upper and lower digestive tract in eight Dutch hospitals. With
the architecture and the method of `DINOv2`, only the pretraining data differs, which isolates the
effect of in domain pretraining.

The weights are served by Theta Vision Cortex under a model usage agreement, free for academic
research, which forbids redistributing them, so they never enter this repository. Each user requests
access at `PORTAL` and places the checkpoint `dinov2.pth` locally. Publications must cite the
papers listed there, Jong et al. (2025) first among them.
"""

import os
import re
from pathlib import Path

import timm
import torch
from timm.layers.pos_embed import resample_abs_pos_embed
from timm.models.vision_transformer import VisionTransformer

from find_those_polyps.baselines.dinov2 import PATCH, TokenEncoder, check_size
from find_those_polyps.baselines.resnet import DEFAULT_SIZE, sized
from find_those_polyps.data.paths import default_data_root

PORTAL = "https://cortex.thetavision.nl/dataset-provider/listing/2/"
"""Page where access to the GastroNet-5M pretrained weights is requested."""

WEIGHTS_ENV = "FIND_THOSE_POLYPS_GASTRONET"
"""Environment variable naming the checkpoint, read when no path is given."""

ARCHITECTURE = "vit_base_patch14_dinov2"
"""Timm architecture of the checkpoint."""

_CHUNKED = re.compile(r"^blocks\.\d+\.(\d+)\.")
"""Block names of DINOv2 training checkpoints, which group blocks in chunks by global index."""

_IGNORED = frozenset({"mask_token"})
"""Backbone entries used only in training, which timm networks do not hold."""


def default_weights() -> Path:
    """Checkpoint path used when none is given.

    Returns:
        The path named by `FIND_THOSE_POLYPS_GASTRONET`, or `weights/gastronet-dinov2.pth` under
        the data root when the variable is unset.
    """
    named = os.environ.get(WEIGHTS_ENV)
    return Path(named) if named else default_data_root() / "weights" / "gastronet-dinov2.pth"


def backbone_state(checkpoint: dict[str, object]) -> dict[str, torch.Tensor]:
    """Weights of the teacher backbone under the names of timm.

    The `backbone.` prefix is removed, entries outside the backbone such as the projection heads
    are dropped, and chunked block names `blocks.<chunk>.<index>.` become `blocks.<index>.`.

    Args:
        checkpoint: Content of a DINOv2 training checkpoint, holding the `teacher` state.

    Returns:
        The backbone weights by timm name.

    Raises:
        KeyError: If the checkpoint holds no teacher state.
    """
    teacher = checkpoint["teacher"]
    if not isinstance(teacher, dict):
        raise KeyError("the checkpoint holds no teacher state dictionary")
    state = {}
    for key, value in teacher.items():
        if not key.startswith("backbone."):
            continue
        name = _CHUNKED.sub(r"blocks.\1.", key.removeprefix("backbone."))
        if name not in _IGNORED:
            state[name] = value
    return state


def load_backbone(
    path: Path, image_size: int, architecture: str = ARCHITECTURE
) -> VisionTransformer:
    """Vision transformer holding the checkpoint weights, for inputs of `image_size` pixels.

    The position embeddings are resampled from the grid of the checkpoint to that of the input.
    Every weight of the network must come from the checkpoint, so that a renamed or truncated file
    fails here instead of evaluating random weights.

    Args:
        path: DINOv2 training checkpoint of the network.
        image_size: Side in pixels of the square input.
        architecture: Timm architecture of the checkpoint.

    Returns:
        The network in evaluation mode.

    Raises:
        TypeError: If timm does not build a vision transformer.
        ValueError: If the checkpoint misses weights of the network or holds unknown ones.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = backbone_state(checkpoint)
    network = timm.create_model(architecture, pretrained=False, num_classes=0, img_size=image_size)
    if not isinstance(network, VisionTransformer):
        raise TypeError(f"{architecture} is not a vision transformer")
    side = image_size // PATCH
    state["pos_embed"] = resample_abs_pos_embed(
        state["pos_embed"], new_size=[side, side], num_prefix_tokens=network.num_prefix_tokens
    )
    result = network.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(
            f"checkpoint does not match {architecture}, missing {result.missing_keys[:5]}, "
            f"unexpected {result.unexpected_keys[:5]}"
        )
    return network.eval()


class GastroNetDINOv2(TokenEncoder):
    """DINOv2 ViT-B/14 pretrained on GastroNet-5M, read from a local checkpoint.

    Images are standardized with the ImageNet statistics, the default of DINOv2 training, since
    the release states none. The name is `gastronet-dinov2-vitb14`, followed by the input size when
    it is not 224 pixels, the native resolution of the checkpoint being 336.

    Attributes:
        pretrained_on: The GastroNet-5M corpus, which shares no images with the registered datasets.
    """

    pretrained_on = ("gastronet-5m",)

    def __init__(self, weights: str | Path | None = None, image_size: int = DEFAULT_SIZE) -> None:
        """Load the checkpoint.

        Args:
            weights: Path of `dinov2.pth`, `default_weights()` when `None`.
            image_size: Side in pixels of the square input, 224 by default.

        Raises:
            FileNotFoundError: If the checkpoint does not exist.
        """
        check_size(image_size)
        path = Path(weights) if weights is not None else default_weights()
        if not path.is_file():
            raise FileNotFoundError(
                f"missing GastroNet checkpoint {path}, request access at {PORTAL}, then place "
                f"dinov2.pth there or name it with {WEIGHTS_ENV}"
            )
        super().__init__(
            load_backbone(path, image_size),
            name=sized("gastronet-dinov2-vitb14", image_size),
            description="Self supervised DINOv2 ViT-B/14 on GastroNet-5M endoscopy images.",
            image_size=image_size,
        )
