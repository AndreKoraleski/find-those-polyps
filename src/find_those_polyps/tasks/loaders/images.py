"""Shared helpers for loaders of image files."""

import csv
import io

import numpy as np
from numpy.typing import NDArray
from PIL import Image

IMAGE_SUFFIXES = frozenset({".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"})


def read_rgb(data: bytes) -> NDArray[np.uint8]:
    """Decode an image file as an RGB array.

    Args:
        data: Encoded image in any format Pillow reads.

    Returns:
        Array of shape `(H, W, 3)` and dtype uint8.
    """
    with Image.open(io.BytesIO(data)) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8)


def read_mask(data: bytes, threshold: int = 128) -> NDArray[np.uint8]:
    """Decode a binary mask from its grey levels.

    Masks are kept in one byte per pixel, since a task holds the masks of all its images at their
    native resolution.

    Args:
        data: Encoded mask image, converted to grey levels before thresholding.
        threshold: Lowest grey level, between 0 and 255, mapped to one.

    Returns:
        Array of shape `(H, W)` and dtype uint8 holding zeros and ones.
    """
    with Image.open(io.BytesIO(data)) as image:
        return (np.asarray(image.convert("L")) >= threshold).astype(np.uint8)


def read_layers(data: bytes, layers: int, threshold: int = 128) -> NDArray[np.uint8]:
    """Decode a multi page image into one binary mask per page.

    Args:
        data: Encoded multi page image such as a TIFF stack.
        layers: Number of masks to return. Pages beyond it are ignored and missing pages are empty.
        threshold: Lowest grey level, between 0 and 255, mapped to one.

    Returns:
        Array of shape `(layers, H, W)` and dtype uint8 holding zeros and ones.
    """
    with Image.open(io.BytesIO(data)) as image:
        pages = []
        for index in range(min(layers, getattr(image, "n_frames", 1))):
            image.seek(index)
            pages.append(np.asarray(image.convert("L")) >= threshold)
    stack = np.zeros((layers, *pages[0].shape), dtype=np.uint8)
    stack[: len(pages)] = pages
    return stack


def read_rows(text: str, delimiter: str = ",") -> list[dict[str, str]]:
    """Rows of a delimited text table keyed by its header.

    Args:
        text: Table whose first line names the columns.
        delimiter: Column separator.

    Returns:
        One mapping from column name to cell text per row.
    """
    return list(csv.DictReader(io.StringIO(text), delimiter=delimiter))


def stem(name: str) -> str:
    """File name without directories and suffix.

    Args:
        name: Slash separated file name.

    Returns:
        The last component without its final suffix.
    """
    base = name.rsplit("/", 1)[-1]
    return base.rsplit(".", 1)[0] if "." in base else base


def parent(name: str) -> str:
    """Name of the directory directly containing a file.

    Args:
        name: Slash separated file name.

    Returns:
        The second to last component, or an empty string for a name without directories.
    """
    parts = name.split("/")
    return parts[-2] if len(parts) > 1 else ""
