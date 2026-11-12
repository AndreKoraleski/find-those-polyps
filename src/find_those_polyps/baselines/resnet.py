"""ResNet-50 reference encoders with global and dense features.

The global feature is the average pooled output of the last stage, with 2048 channels, and the
dense feature is the output of the third stage, with 1024 channels at a stride of sixteen. A
randomly initialized network is the customary lower reference of self supervised learning studies,
and the supervised ImageNet network the customary transfer learning reference. Names carry the
input size when it differs from the customary 224 pixels, for instance `imagenet-resnet50-448`, so
that results at different resolutions are kept apart on the leaderboard.
"""

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torchvision.models import ResNet, ResNet50_Weights, resnet50

from find_those_polyps.models.encoder import Encoder
from find_those_polyps.models.transforms import normalize, resize

DEFAULT_SIZE = 224


def sized(name: str, image_size: int) -> str:
    """Name followed by the input size when it differs from the default.

    Args:
        name: Base name of the encoder.
        image_size: Side in pixels of the square input.

    Returns:
        `name` unchanged at `DEFAULT_SIZE` pixels, and `name` followed by a hyphen and the size
        otherwise.
    """
    return name if image_size == DEFAULT_SIZE else f"{name}-{image_size}"


class ResNet50Encoder(Encoder):
    """Frozen torchvision ResNet-50 exposing pooled and third stage features.

    Images are resized to a square of `image_size` pixels and standardized with the ImageNet
    statistics, on `preprocess_device` when it is set. The classification head is replaced by the
    identity.

    Attributes:
        name: Identifier keying the leaderboard.
        description: One sentence summary of the encoder.
        image_size: Side in pixels of the square input.
        backbone: Torchvision network in evaluation mode with gradients disabled.
    """

    def __init__(
        self, backbone: ResNet, name: str, description: str, image_size: int = DEFAULT_SIZE
    ):
        """Freeze a torchvision ResNet-50 and drop its classification head.

        Args:
            backbone: Network to wrap, modified in place.
            name: Identifier keying the leaderboard.
            description: One sentence summary of the encoder.
            image_size: Side in pixels of the square input, 224 by default.
        """
        super().__init__()
        self.name = name
        self.description = description
        self.image_size = image_size
        backbone.fc = nn.Identity()
        self.backbone: ResNet = backbone.eval().requires_grad_(False)

    def preprocess(self, image: NDArray[np.uint8]) -> torch.Tensor:
        """Resize an RGB image and standardize it with the ImageNet statistics.

        Args:
            image: Image of shape (H, W, 3) with values in [0, 255].

        Returns:
            A float tensor of shape (3, `image_size`, `image_size`).
        """
        return normalize(resize(image, self.image_size, device=self.preprocess_device))

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Pooled last stage and raw third stage features of a batch.

        Args:
            images: Preprocessed images of shape (B, 3, S, S) with S equal to `image_size`.

        Returns:
            A mapping with `global` of shape (B, 2048) and `dense` of shape (B, 1024, S/16, S/16).
        """
        net = self.backbone
        x = net.maxpool(net.relu(net.bn1(net.conv1(images))))
        stage3 = net.layer3(net.layer2(net.layer1(x)))
        stage4 = net.layer4(stage3)
        return {"global": torch.flatten(net.avgpool(stage4), 1), "dense": stage3}

    def postprocess(self, output: dict[str, torch.Tensor]) -> torch.Tensor:
        """Global embedding of a batch.

        Args:
            output: Mapping returned by `forward`.

        Returns:
            The pooled last stage features of shape (B, 2048).
        """
        return output["global"]

    def postprocess_dense(self, output: dict[str, torch.Tensor]) -> torch.Tensor:
        """Dense feature map of a batch.

        Args:
            output: Mapping returned by `forward`.

        Returns:
            The third stage features of shape (B, 1024, h, w).
        """
        return output["dense"]


class RandomResNet50(ResNet50Encoder):
    """ResNet-50 with frozen random weights drawn from a fixed seed.

    The weights are drawn in a forked random state, so that the global generator is left
    untouched. The name is `random-resnet50` under seed zero and `random-resnet50-s<seed>`
    otherwise, followed by the input size when it is not the default.

    Attributes:
        pretrained_on: Empty, since the network saw no data.
    """

    pretrained_on = ()

    def __init__(self, seed: int = 0, image_size: int = DEFAULT_SIZE) -> None:
        """Draw the random weights.

        Args:
            seed: Seed of the He initialization, 0 by default.
            image_size: Side in pixels of the square input, 224 by default.
        """
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            backbone = resnet50(weights=None)
        super().__init__(
            backbone,
            name=sized("random-resnet50" if seed == 0 else f"random-resnet50-s{seed}", image_size),
            description=f"Untrained ResNet-50 with He initialization under seed {seed}.",
            image_size=image_size,
        )


class ImageNetResNet50(ResNet50Encoder):
    """ResNet-50 trained with labels on ImageNet-1k, torchvision weights version two.

    Attributes:
        pretrained_on: The ImageNet-1k corpus, so that overlapping tasks are skipped.
    """

    pretrained_on = ("imagenet-1k",)

    def __init__(self, image_size: int = DEFAULT_SIZE) -> None:
        """Load the `IMAGENET1K_V2` weights, downloading them on first use.

        Args:
            image_size: Side in pixels of the square input, 224 by default.
        """
        super().__init__(
            resnet50(weights=ResNet50_Weights.IMAGENET1K_V2),
            name=sized("imagenet-resnet50", image_size),
            description="Supervised ImageNet-1k ResNet-50, torchvision IMAGENET1K_V2 weights.",
            image_size=image_size,
        )
