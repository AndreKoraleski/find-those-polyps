"""Handcrafted colour and texture features, a reference that involves no learning at all.

The encoder describes an image by the distribution of its colours and by two classic texture
descriptors, the orientation of its gradients and its local binary patterns of Ojala et al.
(2002). Tasks it solves well reward low level image statistics rather than semantic content, so it
calibrates what every task measures, in particular where an untrained network beats trained ones.

Images are resized to `image_size` pixels with values in [0, 1]. Colour is described in RGB and
HSV, the hue as a vector scaled by the saturation so that it has no discontinuity. Gradients come
from Sobel filters on the luma, and local binary patterns compare every pixel with its eight
neighbours, merged into the ten rotation invariant uniform classes.

The global feature concatenates three blocks, colour statistics, the joint RGB histogram and the
texture histograms. Every block is centred on fixed reference values and scaled to unit norm, so
that cosine similarities are not dominated by offsets shared by all images and every block weighs
alike. The dense feature holds the same descriptors computed over cells of sixteen pixels, a grid
of 14 by 14 at 224 pixels like the third stage of a ResNet-50.
"""

import math

import torch
import torch.nn.functional as F  # noqa: N812

from find_those_polyps.baselines.resnet import DEFAULT_SIZE, sized
from find_those_polyps.models.encoder import Encoder

CELL = 16
"""Side in pixels of the cells of the dense feature map."""

ORIENTATIONS = 8
"""Number of bins of unsigned gradient orientation over a half turn."""

LBP_CLASSES = 10
"""Rotation invariant uniform patterns with zero to eight set bits, then every other pattern."""

QUANTILES = (0.1, 0.5, 0.9)
"""Quantiles of every colour channel in the global feature."""

RGB_BINS = 4
"""Bins per channel of the joint RGB histogram."""

_SOBEL = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 4
_NEIGHBOURS = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))


def colour_channels(images: torch.Tensor) -> torch.Tensor:
    """Colour and edge channels of a batch, each with a fixed reference value of zero.

    Args:
        images: RGB images of shape (B, 3, S, S) with values in [0, 1].

    Returns:
        Channels of shape (B, 8, S, S), R, G, B, value and saturation centred on one half, the
        gradient magnitude of the luma in [0, 1] and the hue as its cosine and sine scaled by the
        saturation.
    """
    value, _ = images.max(dim=1)
    minimum, _ = images.min(dim=1)
    chroma = value - minimum
    saturation = torch.where(value > 0, chroma / value.clamp(min=1e-8), torch.zeros_like(value))
    hue = _hue(images, value, chroma)
    magnitude, _ = gradients(images)
    centred = torch.cat([images, value[:, None], saturation[:, None]], dim=1) - 0.5
    hue_vector = torch.stack([saturation * hue.cos(), saturation * hue.sin()], dim=1)
    return torch.cat([centred, magnitude[:, None], hue_vector], dim=1)


def _hue(images: torch.Tensor, value: torch.Tensor, chroma: torch.Tensor) -> torch.Tensor:
    """Hue angle in radians of every pixel, zero for grey pixels.

    Args:
        images: RGB images of shape (B, 3, S, S).
        value: Largest channel of every pixel, of shape (B, S, S).
        chroma: Largest minus smallest channel, of shape (B, S, S).

    Returns:
        Angles of shape (B, S, S) in [0, 2 pi).
    """
    red, green, blue = images.unbind(dim=1)
    safe = chroma.clamp(min=1e-8)
    sector = torch.where(
        value == red,
        ((green - blue) / safe) % 6,
        torch.where(value == green, (blue - red) / safe + 2, (red - green) / safe + 4),
    )
    return torch.where(chroma > 0, sector * (math.pi / 3), torch.zeros_like(sector))


def gradients(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sobel gradient magnitude and unsigned orientation bin of the luma.

    Args:
        images: RGB images of shape (B, 3, S, S) with values in [0, 1].

    Returns:
        The magnitude of shape (B, S, S) in [0, 1] and the orientation bin of every pixel, of
        shape (B, S, S), among `ORIENTATIONS` bins over a half turn.
    """
    luma = _luma(images)
    kernel = _SOBEL.to(images)
    padded = F.pad(luma, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(padded, kernel[None, None])[:, 0]
    gy = F.conv2d(padded, kernel.T[None, None])[:, 0]
    magnitude = (gx.square() + gy.square()).sqrt().clamp(max=1.0)
    angle = torch.atan2(gy, gx) % math.pi
    bins = (angle / math.pi * ORIENTATIONS).long().clamp(max=ORIENTATIONS - 1)
    return magnitude, bins


def local_binary_patterns(images: torch.Tensor) -> torch.Tensor:
    """Rotation invariant uniform local binary pattern class of every pixel.

    A neighbour sets its bit when it is at least as bright as the centre. Patterns with at most two
    transitions around the circle take the number of set bits as class, from zero to eight, and
    every other pattern takes class nine.

    Args:
        images: RGB images of shape (B, 3, S, S) with values in [0, 1].

    Returns:
        Classes of shape (B, S, S) in [0, `LBP_CLASSES`).
    """
    luma = _luma(images)
    size = luma.shape[-1]
    padded = F.pad(luma, (1, 1, 1, 1), mode="replicate")[:, 0]
    bits = torch.stack(
        [
            padded[:, 1 + dy : 1 + dy + size, 1 + dx : 1 + dx + size] >= luma[:, 0]
            for dy, dx in _NEIGHBOURS
        ],
        dim=1,
    )
    ones = bits.sum(dim=1)
    transitions = (bits != bits.roll(1, dims=1)).sum(dim=1)
    return torch.where(transitions <= 2, ones, torch.full_like(ones, LBP_CLASSES - 1))


def _luma(images: torch.Tensor) -> torch.Tensor:
    """Luma of a batch of RGB images, of shape (B, 1, S, S)."""
    weights = torch.tensor([0.299, 0.587, 0.114]).to(images)
    return (images * weights[None, :, None, None]).sum(dim=1, keepdim=True)


def _histogram_map(
    classes: torch.Tensor, count: int, weights: torch.Tensor | None = None
) -> torch.Tensor:
    """Indicator maps of every class, weighted per pixel when weights are given.

    Args:
        classes: Class of every pixel, of shape (B, S, S).
        count: Number of classes.
        weights: Weight of every pixel, of shape (B, S, S), one when `None`.

    Returns:
        Maps of shape (B, count, S, S).
    """
    maps = F.one_hot(classes, count).permute(0, 3, 1, 2).to(torch.float32)
    return maps if weights is None else maps * weights[:, None]


def _hellinger(counts: torch.Tensor, dim: int) -> torch.Tensor:
    """Square roots of normalized counts minus those of the uniform histogram.

    Args:
        counts: Non negative counts along `dim`.
        dim: Dimension holding the bins.

    Returns:
        Centred Hellinger embedding of the histogram, zero for a uniform one.
    """
    bins = counts.shape[dim]
    total = counts.sum(dim=dim, keepdim=True).clamp(min=1e-8)
    return (counts / total).sqrt() - math.sqrt(1 / bins)


def _unit(block: torch.Tensor) -> torch.Tensor:
    """Rows of a block of shape (B, K) scaled to unit norm."""
    return F.normalize(block, dim=1)


class ColourTexture(Encoder):
    """Colour statistics and histograms with gradient orientation and local binary patterns.

    The network has no parameters and saw no data. The name is `colour-texture`, followed by the
    input size when it is not 224 pixels.

    Attributes:
        name: Identifier keying the leaderboard.
        description: One sentence summary of the encoder.
        image_size: Side in pixels of the square input, a multiple of `CELL`.
        pretrained_on: Empty, since no data was seen.
    """

    pretrained_on = ()

    def __init__(self, image_size: int = DEFAULT_SIZE) -> None:
        """Set the input size.

        Args:
            image_size: Side in pixels of the square input, 224 by default.

        Raises:
            ValueError: If the side is not a multiple of sixteen.
        """
        if image_size % CELL:
            raise ValueError(f"image_size must be a multiple of {CELL}, got {image_size}")
        super().__init__()
        self.image_size = image_size
        self.name = sized("colour-texture", image_size)
        self.description = (
            "Handcrafted colour statistics and histograms with gradient orientation and local "
            "binary pattern texture, without learning."
        )

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Global and dense descriptors of a batch.

        Args:
            images: RGB images of shape (B, 3, S, S) with values in [0, 1].

        Returns:
            A mapping with `global` of shape (B, 8 * (2 + Q) + RGB_BINS^3 + 8 + 10) for Q
            quantiles and `dense` of shape (B, 27, S/16, S/16).
        """
        images = images.float().clamp(0, 1)
        channels = colour_channels(images)
        magnitude, orientation = gradients(images)
        patterns = local_binary_patterns(images)
        oriented = _histogram_map(orientation, ORIENTATIONS, magnitude)
        textured = _histogram_map(patterns, LBP_CLASSES)
        flat = channels.flatten(2)
        quantiles = torch.quantile(flat, torch.tensor(QUANTILES).to(flat), dim=2).permute(1, 2, 0)
        statistics = torch.cat(
            [flat.mean(dim=2, keepdim=True), flat.std(dim=2, keepdim=True), quantiles], dim=2
        )
        levels = (images * RGB_BINS).long().clamp(max=RGB_BINS - 1)
        joint = (levels[:, 0] * RGB_BINS + levels[:, 1]) * RGB_BINS + levels[:, 2]
        colour_histogram = _histogram_map(joint, RGB_BINS**3).flatten(2).sum(dim=2)
        texture = torch.cat(
            [
                _hellinger(oriented.flatten(2).sum(dim=2), 1),
                _hellinger(textured.flatten(2).sum(dim=2), 1),
            ],
            dim=1,
        )
        global_features = torch.cat(
            [_unit(statistics.flatten(1)), _unit(_hellinger(colour_histogram, 1)), _unit(texture)],
            dim=1,
        )
        cell_means = F.avg_pool2d(channels, CELL)
        value = channels[:, 3:4]
        cell_spread = (
            (F.avg_pool2d(value.square(), CELL) - cell_means[:, 3:4].square()).clamp(min=0).sqrt()
        )
        dense = torch.cat(
            [
                cell_means,
                cell_spread,
                _hellinger(F.avg_pool2d(oriented, CELL), 1),
                _hellinger(F.avg_pool2d(textured, CELL), 1),
            ],
            dim=1,
        )
        return {"global": global_features, "dense": dense}

    def postprocess(self, output: dict[str, torch.Tensor]) -> torch.Tensor:
        """Global descriptor of a batch.

        Args:
            output: Mapping returned by `forward`.

        Returns:
            The concatenated unit norm blocks of shape (B, D).
        """
        return output["global"]

    def postprocess_dense(self, output: dict[str, torch.Tensor]) -> torch.Tensor:
        """Dense descriptor map of a batch.

        Args:
            output: Mapping returned by `forward`.

        Returns:
            The cell descriptors of shape (B, 27, S/16, S/16).
        """
        return output["dense"]
