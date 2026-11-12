"""DINOv2 reference encoders, self supervised vision transformers of Oquab et al. (2024).

The networks were trained without labels on LVD-142M, a corpus of 142 million web images curated
by retrieval against public datasets, and are a strong general reference for self supervised
learning. Weights come from timm, which resamples the position embeddings to the input size. The
global feature is the class token and the dense feature the patch tokens, both after the final
normalization, with patches of fourteen pixels, so that a 224 pixel input gives a 16 by 16 grid.
"""

from typing import ClassVar, Literal

import numpy as np
import timm
import torch
from numpy.typing import NDArray
from timm.models.vision_transformer import VisionTransformer

from find_those_polyps.baselines.resnet import DEFAULT_SIZE, sized
from find_those_polyps.models.encoder import Encoder, patch_grid
from find_those_polyps.models.transforms import normalize, resize

PATCH = 14
"""Side in pixels of the patches of every DINOv2 network."""

Size = Literal["small", "base", "large", "giant"]


class DINOv2(Encoder):
    """Frozen DINOv2 vision transformer exposing its class and patch tokens.

    Images are resized to a square of `image_size` pixels and standardized with the ImageNet
    statistics, on `preprocess_device` when it is set. The name is `dinov2-vit<s|b|l|g>14`,
    followed by `-reg4` with registers and by the input size when it is not 224 pixels.

    Attributes:
        name: Identifier keying the leaderboard.
        description: One sentence summary of the encoder.
        image_size: Side in pixels of the square input, a multiple of fourteen.
        pretrained_on: The LVD-142M corpus.
        backbone: Timm network in evaluation mode with gradients disabled.
    """

    pretrained_on = ("lvd-142m",)
    letters: ClassVar[dict[str, str]] = {"small": "s", "base": "b", "large": "l", "giant": "g"}

    def __init__(
        self, size: Size = "base", image_size: int = DEFAULT_SIZE, registers: bool = False
    ) -> None:
        """Load the DINOv2 weights from timm, downloading them on first use.

        Args:
            size: Width of the network, `base` by default.
            image_size: Side in pixels of the square input, 224 by default.
            registers: Whether to use the variant with four register tokens of Darcet et al.
                (2024).

        Raises:
            ValueError: If the size is unknown or the input side is not a multiple of fourteen.
        """
        if size not in self.letters:
            raise ValueError(f"unknown DINOv2 size {size!r}")
        if image_size % PATCH:
            raise ValueError(f"image_size must be a multiple of {PATCH}, got {image_size}")
        super().__init__()
        reg = "_reg4" if registers else ""
        weights = f"vit_{size}_patch14{reg}_dinov2.lvd142m"
        backbone = timm.create_model(weights, pretrained=True, img_size=image_size)
        if not isinstance(backbone, VisionTransformer):
            raise TypeError(f"{weights} is not a vision transformer")
        self.image_size = image_size
        self.name = sized(f"dinov2-vit{self.letters[size]}14{reg.replace('_', '-')}", image_size)
        self.description = f"Self supervised DINOv2 ViT on LVD-142M, timm weights {weights}."
        self.backbone: VisionTransformer = backbone.eval().requires_grad_(False)

    def preprocess(self, image: NDArray[np.uint8]) -> torch.Tensor:
        """Resize an RGB image and standardize it with the ImageNet statistics.

        Args:
            image: Image of shape (H, W, 3) with values in [0, 255].

        Returns:
            A float tensor of shape (3, `image_size`, `image_size`).
        """
        return normalize(resize(image, self.image_size, device=self.preprocess_device))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Normalized tokens of a batch.

        Args:
            images: Preprocessed images of shape (B, 3, S, S) with S equal to `image_size`.

        Returns:
            Tokens of shape (B, P + N, D), the class token and any registers first, then the
            N patch tokens in row major order.
        """
        tokens: torch.Tensor = self.backbone.forward_features(images)
        return tokens

    def postprocess(self, output: torch.Tensor) -> torch.Tensor:
        """Global embedding of a batch.

        Args:
            output: Tokens returned by `forward`.

        Returns:
            The class tokens of shape (B, D).
        """
        return output[:, 0]

    def postprocess_dense(self, output: torch.Tensor) -> torch.Tensor:
        """Dense feature map of a batch.

        Args:
            output: Tokens returned by `forward`.

        Returns:
            The patch tokens as a map of shape (B, D, S/14, S/14).
        """
        return patch_grid(output[:, self.backbone.num_prefix_tokens :], self.image_size // PATCH)
