"""Backbone of the pretraining recipes, a ViT-B/14 that accepts inputs of any size.

The network has the architecture of `find_those_polyps.baselines.DINOv2`, so a backbone trained
here is evaluated as that encoder is. It starts from the DINOv2 weights served by timm, for
continued pretraining, or from random weights. Its position embeddings are resampled to the input
at run time, which the crops of several sizes of some methods need.
"""

from typing import Literal

import timm
import torch
from timm.models.vision_transformer import VisionTransformer

ARCHITECTURE = "vit_base_patch14_dinov2.lvd142m"
"""Timm name of the backbone and of its DINOv2 weights."""

PATCH = 14
"""Side in pixels of the patches."""

IMAGE_SIZE = 224
"""Side in pixels of the full views, a 16 by 16 grid of patches."""

START_CORPUS = "lvd-142m"
"""Corpus the DINOv2 weights were trained on, declared by every backbone that continues them."""

Init = Literal["dinov2", "random"]


def backbone(init: Init = "dinov2", *, drop_path: float = 0.0) -> VisionTransformer:
    """ViT-B/14 without a classifier, for inputs whose sides are multiples of fourteen.

    Args:
        init: `dinov2` to continue the DINOv2 weights, downloaded by timm on first use, or
            `random` to start from random weights.
        drop_path: Largest rate of stochastic depth, zero by default.

    Returns:
        The network in training mode, with position embeddings for a 224 pixel input that are
        resampled to other sizes at run time.

    Raises:
        TypeError: If timm does not build a vision transformer.
    """
    network = timm.create_model(
        ARCHITECTURE,
        pretrained=init == "dinov2",
        num_classes=0,
        img_size=IMAGE_SIZE,
        dynamic_img_size=True,
        drop_path_rate=drop_path,
    )
    if not isinstance(network, VisionTransformer):
        raise TypeError(f"{ARCHITECTURE} is not a vision transformer")
    return network


def embed(network: VisionTransformer, images: torch.Tensor) -> torch.Tensor:
    """Patch and class tokens of a batch with their position embeddings, before any block.

    Args:
        network: Vision transformer built with position embeddings resampled at run time.
        images: Images of shape (B, 3, H, W) with sides that are multiples of the patch side.

    Returns:
        Tokens of shape (B, P + N, D), the P prefix tokens first, then the N patch tokens in row
        major order.
    """
    tokens: torch.Tensor = network._pos_embed(network.patch_embed(images))
    embedded: torch.Tensor = network.norm_pre(network.patch_drop(tokens))
    return embedded


def encode(network: VisionTransformer, tokens: torch.Tensor) -> torch.Tensor:
    """Run embedded tokens through the blocks and the final normalization.

    Args:
        network: Vision transformer the tokens were embedded by.
        tokens: Tokens of shape (B, T, D), which may be a subset of those of the image.

    Returns:
        The normalized tokens of shape (B, T, D).
    """
    encoded: torch.Tensor = network.norm(network.blocks(tokens))
    return encoded
