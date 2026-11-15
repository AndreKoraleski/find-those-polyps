"""Supervised ImageNet vision transformers, DeiT III of Touvron et al. (2022).

The networks were trained with labels on ImageNet-1k alone, with the DeiT III recipe, a strong
supervised training of plain vision transformers without distillation. The ViT-B/16 has the width,
depth and heads of the DINOv2 ViT-B/14, so that comparing the two separates the training method
from the architecture, and it was trained on the data of the supervised ResNet-50, so that
comparing those two separates the architecture from the data. The patches are of sixteen pixels
rather than fourteen, so that a 224 pixel input gives a 14 by 14 grid where DINOv2 gives 16 by 16.

Weights come from timm, which resamples the position embeddings to the input size, under the
`fb_in1k` tag converted from the official release of facebookresearch/deit. The code and the
weights are released under the Apache License 2.0. The global feature is the class token and the
dense feature the patch tokens, both after the final normalization and before the classifier.
"""

from typing import ClassVar, Literal

import timm
import torch
from timm.models.vision_transformer import VisionTransformer

from find_those_polyps.baselines.dinov2 import TokenEncoder
from find_those_polyps.baselines.resnet import DEFAULT_SIZE, sized
from find_those_polyps.models.encoder import patch_grid

PATCH = 16
"""Side in pixels of the patches of every DeiT III network served here."""

Size = Literal["small", "base", "large"]


class ImageNetViT(TokenEncoder):
    """DeiT III vision transformer trained with labels on ImageNet-1k, with weights from timm.

    The name is `imagenet-deit3-vit<s|b|l>16`, followed by the input size when it is not 224
    pixels.

    Attributes:
        pretrained_on: The ImageNet-1k corpus, so that overlapping tasks are skipped.
    """

    pretrained_on = ("imagenet-1k",)
    letters: ClassVar[dict[str, str]] = {"small": "s", "base": "b", "large": "l"}

    def __init__(self, size: Size = "base", image_size: int = DEFAULT_SIZE) -> None:
        """Load the DeiT III weights from timm, downloading them on first use.

        Args:
            size: Width of the network, `base` by default.
            image_size: Side in pixels of the square input, 224 by default.

        Raises:
            ValueError: If the size is unknown or the input side is not a multiple of sixteen.
            TypeError: If timm does not build a vision transformer.
        """
        if size not in self.letters:
            raise ValueError(f"unknown DeiT III size {size!r}")
        if image_size % PATCH:
            raise ValueError(f"image_size must be a multiple of {PATCH}, got {image_size}")
        weights = f"deit3_{size}_patch16_224.fb_in1k"
        backbone = timm.create_model(weights, pretrained=True, img_size=image_size)
        if not isinstance(backbone, VisionTransformer):
            raise TypeError(f"{weights} is not a vision transformer")
        super().__init__(
            backbone,
            name=sized(f"imagenet-deit3-vit{self.letters[size]}16", image_size),
            description=f"Supervised DeiT III ViT on ImageNet-1k, timm weights {weights}.",
            image_size=image_size,
        )

    def postprocess_dense(self, output: torch.Tensor) -> torch.Tensor:
        """Dense feature map of a batch.

        Args:
            output: Tokens returned by `forward`.

        Returns:
            The patch tokens as a map of shape (B, D, S/16, S/16).
        """
        return patch_grid(output[:, self.backbone.num_prefix_tokens :], self.image_size // PATCH)
