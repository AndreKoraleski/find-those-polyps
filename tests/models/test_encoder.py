from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from find_those_polyps.models import TorchEncoder, patch_grid, run
from find_those_polyps.models.encoder import _code
from find_those_polyps.models.transforms import normalize, resize


def test_pipeline_applies_every_stage() -> None:
    encoder = TorchEncoder(
        torch.nn.Identity(),
        name="identity",
        description="Identity map.",
        preprocess=lambda image: torch.full((3, 2, 2), float(image.mean())),
        postprocess=lambda output: output.flatten(1),
    )
    images = [np.full((5, 7, 3), 10, dtype=np.uint8)] * 2
    features = run(encoder, images, torch.device("cpu"))
    assert features.shape == (2, 12)
    assert torch.all(features == 10)


def test_default_preprocessing_resizes_to_unit_range() -> None:
    encoder = TorchEncoder(torch.nn.Flatten(), name="flat", description="Flatten.", image_size=4)
    features = run(encoder, [np.full((9, 6, 3), 255, dtype=np.uint8)], torch.device("cpu"))
    assert features.shape == (1, 48)
    assert torch.allclose(features, torch.ones(1, 48))


def test_wrong_output_shape_is_reported() -> None:
    encoder = TorchEncoder(torch.nn.Identity(), name="raw", description="Raw images.")
    with pytest.raises(ValueError, match="expected"):
        run(encoder, [np.zeros((4, 4, 3), dtype=np.uint8)], torch.device("cpu"))


@pytest.mark.parametrize(("name", "description"), [("", "x"), ("bad name", "x"), ("ok", "")])
def test_card_requires_metadata(name: str, description: str) -> None:
    encoder = TorchEncoder(torch.nn.Identity(), name=name, description=description)
    with pytest.raises(ValidationError):
        _ = encoder.card


def test_patch_grid() -> None:
    tokens = torch.arange(2 * 6 * 4, dtype=torch.float32).view(2, 6, 4)
    assert patch_grid(tokens, height=2).shape == (2, 4, 2, 3)


def test_fingerprint_tracks_weights_size_and_bound_modules() -> None:
    module = torch.nn.Linear(3, 2)

    def build(stage: Callable[[torch.Tensor], Any], size: int = 8) -> str:
        return TorchEncoder(
            stage, name="linear", description="Linear map.", image_size=size
        ).fingerprint()

    first, bound = build(module), build(module.forward)
    assert build(module) == first
    assert build(module, size=16) != first
    with torch.no_grad():
        module.weight += 1
    assert build(module) != first
    assert build(module.forward) != bound


PLAIN = """
class A:
    def f(self):
        return 1
"""

DOCUMENTED = '''
class A:
    """Doc."""

    def f(self):
        """Doc."""
        # note
        return 1
'''

CHANGED = """
class A:
    def f(self):
        return 2
"""

FRAGMENT = """    postprocess=lambda out: out[0],
)
"""


def test_fingerprint_ignores_documentation_but_not_code() -> None:
    assert _code(PLAIN) == _code(DOCUMENTED)
    assert _code(PLAIN) != _code(CHANGED)
    assert _code(FRAGMENT) == FRAGMENT


def test_resizing_on_a_device_matches_the_host_and_keeps_features_apart() -> None:
    image = np.random.default_rng(0).integers(0, 255, size=(30, 40, 3), dtype=np.uint8)
    assert torch.equal(resize(image, 16), resize(image, 16, device="cpu"))
    assert normalize(resize(image, 16, device="cpu")).device.type == "cpu"
    encoder = TorchEncoder(torch.nn.Identity(), name="identity", description="Identity.")
    before = encoder.fingerprint()
    encoder.preprocess_device = "cuda:0"
    assert encoder.fingerprint() != before
    encoder.preprocess_device = None
    assert encoder.fingerprint() == before
