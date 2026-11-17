"""Augmented views of a frame for self supervised pretraining.

Every pipeline takes a decoded RGB image and returns tensors standardized with the ImageNet
statistics, as the backbone expects. The pipelines are plain objects, so that the workers of a
data loader can receive them. Frames are flipped along both axes, since an endoscope has no
upright orientation.
"""

from collections.abc import Sequence

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode, v2

from find_those_polyps.models.transforms import IMAGENET_MEAN, IMAGENET_STD


def crop(
    size: int,
    scale: tuple[float, float],
    *,
    colour: bool = True,
    blur: float = 0.0,
    solarize: float = 0.0,
) -> v2.Compose:
    """Pipeline of one random crop with photometric distortions.

    Args:
        size: Side in pixels of the square output.
        scale: Smallest and largest share of the image area that the crop covers.
        colour: Whether brightness, contrast, saturation and hue are jittered and the image is
            sometimes turned to grey. Defaults to `True`.
        blur: Probability of a Gaussian blur. Defaults to zero.
        solarize: Probability of inverting the bright pixels. Defaults to zero.

    Returns:
        The pipeline, from an RGB image to a float tensor of shape (3, `size`, `size`).
    """
    steps: list[torch.nn.Module] = [
        v2.RandomResizedCrop(size, scale=scale, interpolation=InterpolationMode.BICUBIC),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
    ]
    if colour:
        steps += [
            v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            v2.RandomGrayscale(p=0.2),
        ]
    if blur:
        steps.append(v2.RandomApply([v2.GaussianBlur(9, sigma=(0.1, 2.0))], p=blur))
    if solarize:
        steps.append(v2.RandomSolarize(threshold=128, p=solarize))
    steps += [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return v2.Compose(steps)


class Views:
    """Several views of one image, each from its own pipeline.

    Args:
        pipelines: Pipeline of every view, in the order the method expects them.

    Attributes:
        pipelines: Pipeline of every view.
    """

    def __init__(self, pipelines: Sequence[v2.Compose]) -> None:
        self.pipelines = tuple(pipelines)

    def __call__(self, image: Image.Image) -> list[torch.Tensor]:
        """Apply every pipeline to the image.

        Args:
            image: Decoded RGB image.

        Returns:
            One tensor per pipeline.
        """
        return [pipeline(image) for pipeline in self.pipelines]
