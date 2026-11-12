"""Common preprocessing steps for encoders.

`IMAGENET_MEAN` and `IMAGENET_STD` are the per channel RGB statistics of ImageNet on the unit
scale, the customary normalization of networks pretrained on it.
"""

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from numpy.typing import NDArray

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resize(
    image: NDArray[np.uint8],
    size: int | tuple[int, int],
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Antialiased bilinear resize of an RGB image to a float tensor.

    The aspect ratio is not preserved, so the image is stretched to the requested size. With a
    `device` the image is moved there in one byte per channel before it is converted and resized,
    which spares the host the work on large images. The antialiasing of the accelerator differs
    slightly from that of the host, so the result is close to the host result but not equal.

    Args:
        image: Image of shape (H, W, 3) with values in [0, 255].
        size: Output side in pixels for a square image, or its height and width.
        device: Device on which to resize, the host when `None`.

    Returns:
        A float32 tensor of shape (3, H', W') with values in [0, 1], on `device` when given.
    """
    height, width = (size, size) if isinstance(size, int) else size
    tensor = torch.from_numpy(np.ascontiguousarray(image))
    if device is not None:
        tensor = tensor.to(device)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).float().div(255)
    resized = F.interpolate(
        tensor,
        size=(height, width),
        mode="bilinear",
        antialias=True,
        align_corners=False,
    )
    return resized.squeeze(0).clamp(0, 1)


def normalize(
    image: torch.Tensor,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> torch.Tensor:
    """Per channel standardization of an image tensor.

    Args:
        image: Tensor of shape (3, H, W), usually the output of `resize`.
        mean: Mean of each channel, the ImageNet statistics by default.
        std: Standard deviation of each channel, the ImageNet statistics by default.

    Returns:
        The tensor of shape (3, H, W) with `mean` subtracted and divided by `std` channel by
        channel, in the dtype of `image`.
    """
    mean_t = torch.tensor(mean, dtype=image.dtype, device=image.device).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=image.dtype, device=image.device).view(3, 1, 1)
    return (image - mean_t) / std_t
