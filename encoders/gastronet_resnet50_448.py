"""GastroNet ResNet-50 at 448 pixels, the resolution of the 448 pixel ResNet-50 references.

The reference encoder takes its input size as an argument, which the command line cannot pass, so
this factory fixes it. A 448 pixel input gives a dense map of 28 by 28 instead of 14 by 14, which
matters for the dense tasks. Evaluate it on them with
`uv run polyps-evaluate encoders/gastronet_resnet50_448.py:build --tasks "seg/*" "det/*"`.

The weights are those of `GastroNetResNet50`, served by Theta Vision Cortex under a model usage
agreement that forbids redistributing them. Each user requests access and places the checkpoint
locally, as that encoder describes, and the factory raises when it is missing.
"""

from find_those_polyps.baselines import GastroNetResNet50


def build() -> GastroNetResNet50:
    """GastroNet ResNet-50 for inputs of 448 pixels, named `gastronet-resnet50-448`.

    Returns:
        The encoder, ready to evaluate.
    """
    return GastroNetResNet50(image_size=448)
