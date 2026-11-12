import itertools
import math

import numpy as np
import pytest
import torch

from find_those_polyps.baselines import ColourTexture, handcrafted
from find_those_polyps.models import run

CPU = torch.device("cpu")


def _solid(red: float, green: float, blue: float, size: int = 32) -> torch.Tensor:
    return torch.tensor([red, green, blue])[None, :, None, None].expand(1, 3, size, size).clone()


@pytest.mark.parametrize(
    ("colour", "angle"),
    [
        ((1.0, 0.0, 0.0), 0.0),
        ((0.0, 1.0, 0.0), 2 * math.pi / 3),
        ((0.0, 0.0, 1.0), 4 * math.pi / 3),
    ],
)
def test_hue_is_a_saturation_scaled_vector(
    colour: tuple[float, float, float], angle: float
) -> None:
    channels = handcrafted.colour_channels(_solid(*colour))
    hue = channels[0, 6:8, 0, 0]
    assert torch.allclose(hue, torch.tensor([math.cos(angle), math.sin(angle)]), atol=1e-6)
    grey = handcrafted.colour_channels(_solid(0.3, 0.3, 0.3))
    assert torch.all(grey[:, 6:8] == 0)


def test_gradients_follow_the_edge_direction() -> None:
    vertical = torch.zeros(1, 3, 8, 8)
    vertical[..., 4:] = 1.0
    magnitude, bins = handcrafted.gradients(vertical)
    assert magnitude[0, 4, 3] > 0 and magnitude[0, 4, 0] == 0
    assert bins[0, 4, 3] == 0
    _, horizontal = handcrafted.gradients(vertical.transpose(2, 3))
    assert horizontal[0, 3, 4] == handcrafted.ORIENTATIONS // 2


def test_local_binary_patterns_separate_flat_and_alternating_texture() -> None:
    flat = handcrafted.local_binary_patterns(_solid(0.5, 0.5, 0.5, 6))
    assert torch.all(flat == 8)
    board = (torch.arange(6)[:, None] + torch.arange(6)[None]) % 2
    images = board.float()[None, None].expand(1, 3, 6, 6)
    classes = handcrafted.local_binary_patterns(images)
    assert classes[0, 2, 2] == 8
    assert classes[0, 2, 3] == 9


def test_descriptors_have_fixed_shapes_and_unit_blocks() -> None:
    encoder = ColourTexture()
    rng = np.random.default_rng(0)
    images = [rng.integers(0, 255, (60, 80, 3), dtype=np.uint8), np.zeros((20, 20, 3), np.uint8)]
    with torch.inference_mode():
        global_features = run(encoder, images, CPU)
        dense = run(encoder, images, CPU, dense=True)
    statistics = 8 * (2 + len(handcrafted.QUANTILES))
    colours = handcrafted.RGB_BINS**3
    assert global_features.shape == (2, statistics + colours + 18)
    assert dense.shape == (2, 27, 14, 14)
    assert torch.isfinite(global_features).all() and torch.isfinite(dense).all()
    first = global_features[0]
    edges = (0, statistics, statistics + colours, len(first))
    blocks = [first[start:end] for start, end in itertools.pairwise(edges)]
    assert all(torch.isclose(block.norm(), torch.tensor(1.0)) for block in blocks)
    channels = handcrafted.colour_channels(encoder.preprocess(images[0])[None])
    cell = channels[0, :, :16, :16].mean(dim=(1, 2))
    assert torch.allclose(dense[0, :8, 0, 0], cell, atol=1e-6)
    assert encoder.card.pretrained_on == ()
    assert encoder.name == "colour-texture"
    assert ColourTexture(448).name == "colour-texture-448"
    with pytest.raises(ValueError, match="multiple of 16"):
        ColourTexture(100)
