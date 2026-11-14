import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from torchvision.models import resnet50

from find_those_polyps.baselines import GastroNetResNet50, gastronet
from find_those_polyps.baselines import gastronet_resnet as module
from find_those_polyps.models import run


def _reference() -> dict[str, torch.Tensor]:
    """Weights of a seeded random ResNet-50 without its classifier."""
    torch.manual_seed(0)
    state = resnet50(weights=None).state_dict()
    return {name: value for name, value in state.items() if not name.startswith("fc.")}


def _dino(reference: dict[str, torch.Tensor]) -> dict[str, object]:
    """DINO training checkpoint, the student wrapped for distributed training, with heads."""
    student = {f"module.backbone.{name}": value + 1 for name, value in reference.items()}
    teacher = {f"backbone.{name}": value for name, value in reference.items()}
    teacher["head.mlp.0.weight"] = torch.zeros(4, 2048)
    return {"student": student, "teacher": teacher, "epoch": 100}


@pytest.mark.parametrize(
    "layout",
    [
        lambda ref: ref,
        lambda ref: {**ref, "fc.weight": torch.zeros(10, 2048), "fc.bias": torch.zeros(10)},
        lambda ref: {"state_dict": {f"module.encoder_q.{n}": v for n, v in ref.items()}},
        _dino,
    ],
    ids=["bare", "with-classifier", "moco", "dino"],
)
def test_checkpoint_weights_reach_every_layer(
    tmp_path: Path, layout: Callable[[dict[str, torch.Tensor]], object]
) -> None:
    reference = _reference()
    path = tmp_path / "resnet50.pth"
    torch.save(layout(reference), path)
    loaded = module.load_backbone(path).state_dict()
    for name, value in reference.items():
        assert torch.equal(loaded[name], value), name


def test_incomplete_or_foreign_checkpoints_are_refused(tmp_path: Path) -> None:
    reference = _reference()
    path = tmp_path / "resnet50.pth"
    torch.save({n: v for n, v in reference.items() if n != "layer4.2.conv3.weight"}, path)
    with pytest.raises(ValueError, match=r"missing \['layer4.2.conv3.weight'\]"):
        module.load_backbone(path)
    torch.save({**reference, "conv1.weight": torch.zeros(64, 1, 7, 7)}, path)
    with pytest.raises(ValueError, match=r"wrong shape \['conv1.weight'\]"):
        module.load_backbone(path)
    torch.save({"epoch": 3}, path)
    with pytest.raises(KeyError, match="no network weights"):
        module.load_backbone(path)


def test_gastronet_resnet_reads_the_configured_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "resnet50.pth"
    torch.save(_dino(_reference()), path)
    monkeypatch.setenv(module.WEIGHTS_ENV, str(path))
    encoder = GastroNetResNet50(image_size=64)
    images = [np.random.default_rng(0).integers(0, 255, (30, 40, 3), dtype=np.uint8)] * 2
    with torch.inference_mode():
        global_features = run(encoder, images, torch.device("cpu"))
        dense = run(encoder, images, torch.device("cpu"), dense=True)
    assert global_features.shape == (2, 2048)
    assert dense.shape == (2, 1024, 4, 4)
    assert encoder.name == "gastronet-resnet50-64"
    assert encoder.card.pretrained_on == ("gastronet-5m",)
    assert not any(p.requires_grad for p in encoder.parameters())


def test_a_missing_checkpoint_points_to_the_portal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(module.WEIGHTS_ENV, raising=False)
    monkeypatch.setenv("FIND_THOSE_POLYPS_DATA", str(tmp_path))
    assert module.default_weights() == tmp_path / "weights" / "gastronet-resnet50.pth"
    with pytest.raises(FileNotFoundError, match=re.escape(gastronet.PORTAL)):
        GastroNetResNet50()
