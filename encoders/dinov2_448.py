"""DINOv2 ViT-B/14 at 448 pixels, the resolution of the 448 pixel ResNet-50 references.

The reference encoder takes its input size as an argument, which the command line cannot pass, so
this factory fixes it. A 448 pixel input gives a grid of 32 by 32 patches instead of 16 by 16,
which matters for the dense tasks. Evaluate it on them with
`uv run polyps-evaluate encoders/dinov2_448.py:build --tasks "seg/*" "det/*"`.

The weights are those of `DINOv2`, released by Meta under Apache 2.0 and served by timm.
"""

from find_those_polyps.baselines import DINOv2


def build() -> DINOv2:
    """DINOv2 ViT-B/14 for inputs of 448 pixels, named `dinov2-vitb14-448`.

    Returns:
        The encoder, ready to evaluate.
    """
    return DINOv2(image_size=448)
