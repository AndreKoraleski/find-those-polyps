from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import timm
import torch

from find_those_polyps.evaluation.__main__ import _module, load_encoder
from find_those_polyps.models import run

TEMPLATE = Path(__file__).parents[2] / "encoders" / "template.py"


def _template() -> ModuleType:
    module = _module(str(TEMPLATE))
    assert isinstance(module, ModuleType)
    return module


def test_template_refuses_to_run_without_its_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing checkpoint"):
        _template().build(tmp_path / "absent.pth")
    if not _template().WEIGHTS.exists():
        with pytest.raises(FileNotFoundError):
            load_encoder(f"{TEMPLATE}:build")


def test_template_loads_a_checkpoint_into_global_and_dense_features(tmp_path: Path) -> None:
    template = _template()
    weights = tmp_path / "model.pth"
    network = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0)
    torch.save(network.state_dict(), weights)
    encoder = template.build(weights)
    images = [np.random.default_rng(0).integers(0, 255, (50, 40, 3), dtype=np.uint8)]
    with torch.inference_mode():
        global_features = run(encoder, images, torch.device("cpu"))
        dense = run(encoder, images, torch.device("cpu"), dense=True)
    assert global_features.shape == (1, 384)
    assert dense.shape == (1, 384, 14, 14)
    assert encoder.name == template.NAME
    assert encoder.card.pretrained_on == template.PRETRAINED_ON
    assert encoder.owner is not None
    assert torch.equal(next(encoder.owner.parameters()), next(network.parameters()))
