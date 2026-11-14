"""GastroNet ResNet-50 reference encoder, trained on endoscopy images by Boers et al. (2024).

The ResNet-50 was pretrained with DINO, the first version of Caron et al. (2021), on GastroNet-5M,
about 4.8 million unlabelled images from some 500 thousand procedures of the upper and lower
digestive tract in eight Dutch hospitals. With the architecture of `ImageNetResNet50`, it is the in
domain counterpart of that network, so that comparing the two isolates the effect of in domain
pretraining on a convolutional network, as `GastroNetDINOv2` does for a vision transformer.

The weights are served by Theta Vision Cortex under a model usage agreement, free for academic
research, which forbids redistributing them, so they never enter this repository. Each user requests
access at `PORTAL` and places the ResNet-50 DINO checkpoint locally. Publications must cite Jong et
al. (2025), which describes GastroNet-5M, and Boers et al. (2024), which trained the network.
"""

import os
import re
from collections.abc import Mapping
from pathlib import Path

import torch
from torchvision.models import ResNet, resnet50

from find_those_polyps.baselines.gastronet import PORTAL
from find_those_polyps.baselines.resnet import DEFAULT_SIZE, ResNet50Encoder, sized
from find_those_polyps.data.paths import default_data_root

WEIGHTS_ENV = "FIND_THOSE_POLYPS_GASTRONET_RESNET50"
"""Environment variable naming the checkpoint, read when no path is given."""

_NESTED = ("teacher", "state_dict", "model", "student")
"""Entries holding the network in training checkpoints, the DINO teacher preferred."""

_PREFIX = re.compile(r"^(module\.|backbone\.|encoder_q\.|encoder\.|model\.)+")
"""Wrappers of distributed training and self supervised frameworks around the backbone names."""


def default_weights() -> Path:
    """Checkpoint path used when none is given.

    Returns:
        The path named by `FIND_THOSE_POLYPS_GASTRONET_RESNET50`, or
        `weights/gastronet-resnet50.pth` under the data root when the variable is unset.
    """
    named = os.environ.get(WEIGHTS_ENV)
    return Path(named) if named else default_data_root() / "weights" / "gastronet-resnet50.pth"


def backbone_state(checkpoint: Mapping[str, object]) -> dict[str, torch.Tensor]:
    """Weights of the ResNet-50 backbone under the names of torchvision.

    The checkpoint is either a bare state dictionary or a training checkpoint holding it under
    `teacher`, `state_dict`, `model` or `student`, in that order of preference. Prefixes added by
    distributed training and by the self supervised frameworks are removed, and entries outside
    the backbone, such as projection heads and the classifier, are dropped by the caller.

    Args:
        checkpoint: Content of the checkpoint file.

    Returns:
        Every tensor of the network by torchvision name.

    Raises:
        KeyError: If the checkpoint holds no tensors.
    """
    state: Mapping[str, object] = checkpoint
    for key in _NESTED:
        nested = checkpoint.get(key)
        if isinstance(nested, Mapping):
            state = nested
            break
    weights = {
        _PREFIX.sub("", name): value
        for name, value in state.items()
        if isinstance(value, torch.Tensor)
    }
    if not weights:
        raise KeyError("the checkpoint holds no network weights")
    return weights


def load_backbone(path: Path) -> ResNet:
    """Torchvision ResNet-50 holding the checkpoint weights.

    Every weight of the network except the classifier must come from the checkpoint with its
    shape, so that a renamed, truncated or foreign file fails here instead of evaluating random
    weights.

    Args:
        path: Checkpoint of the network.

    Returns:
        The network with its classifier left untrained, since `ResNet50Encoder` replaces it.

    Raises:
        ValueError: If the checkpoint misses weights of the network or holds them with another
            shape.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = backbone_state(checkpoint)
    network = resnet50(weights=None)
    expected = {
        name: value.shape
        for name, value in network.state_dict().items()
        if not name.startswith("fc.")
    }
    missing = sorted(name for name in expected if name not in state)
    wrong = sorted(
        name for name in expected if name in state and state[name].shape != expected[name]
    )
    if missing or wrong:
        raise ValueError(
            f"checkpoint does not hold a ResNet-50, missing {missing[:5]}, wrong shape {wrong[:5]}"
        )
    network.load_state_dict({name: state[name] for name in expected}, strict=False)
    return network


class GastroNetResNet50(ResNet50Encoder):
    """ResNet-50 pretrained with DINO on GastroNet-5M, read from a local checkpoint.

    Images are standardized with the ImageNet statistics, the default of DINO training. The name
    is `gastronet-resnet50`, followed by the input size when it is not 224 pixels.

    Attributes:
        pretrained_on: The GastroNet-5M corpus, which shares no images with the registered datasets.
    """

    pretrained_on = ("gastronet-5m",)

    def __init__(self, weights: str | Path | None = None, image_size: int = DEFAULT_SIZE) -> None:
        """Load the checkpoint.

        Args:
            weights: Path of the checkpoint, `default_weights()` when `None`.
            image_size: Side in pixels of the square input, 224 by default.

        Raises:
            FileNotFoundError: If the checkpoint does not exist.
        """
        path = Path(weights) if weights is not None else default_weights()
        if not path.is_file():
            raise FileNotFoundError(
                f"missing GastroNet ResNet-50 checkpoint {path}, request access at {PORTAL}, then "
                f"place the ResNet-50 DINO weights there or name them with {WEIGHTS_ENV}"
            )
        super().__init__(
            load_backbone(path),
            name=sized("gastronet-resnet50", image_size),
            description="Self supervised DINO ResNet-50 on GastroNet-5M endoscopy images.",
            image_size=image_size,
        )
