"""DeiT III ViT-B/16 at 448 pixels, the resolution of the 448 pixel ResNet-50 references.

The reference encoder takes its input size as an argument, which the command line cannot pass, so
this factory fixes it. A 448 pixel input gives a grid of 28 by 28 patches instead of 14 by 14,
which matters for the dense tasks. Evaluate it on them with
`uv run polyps-evaluate encoders/deit3_448.py:build --tasks "seg/*" "det/*"`.

The weights are those of `ImageNetViT`, released by Meta under Apache 2.0 and served by timm.
"""

from find_those_polyps.baselines import ImageNetViT


def build() -> ImageNetViT:
    """DeiT III ViT-B/16 for inputs of 448 pixels, named `imagenet-deit3-vitb16-448`.

    Returns:
        The encoder, ready to evaluate.
    """
    return ImageNetViT(image_size=448)
