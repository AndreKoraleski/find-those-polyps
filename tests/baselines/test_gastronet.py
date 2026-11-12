import re
from pathlib import Path

import numpy as np
import pytest
import timm
import torch

from find_those_polyps.baselines import GastroNetDINOv2
from find_those_polyps.baselines import gastronet as gastronet_module
from find_those_polyps.models import run

SMALL = "vit_small_patch14_dinov2"


def _checkpoint(path: Path, drop: str | None = None) -> dict[str, torch.Tensor]:
    """Save a small DINOv2 network as a training checkpoint, with chunked blocks and heads."""
    torch.manual_seed(0)
    network = timm.create_model(SMALL, pretrained=False, num_classes=0, img_size=56)
    reference = network.state_dict()
    teacher: dict[str, torch.Tensor] = {}
    for name, value in reference.items():
        if name == drop:
            continue
        found = re.match(r"blocks\.(\d+)\.", name)
        if found:
            index = int(found.group(1))
            name = f"blocks.{index // 4}.{index}." + name[found.end() :]
        teacher[f"backbone.{name}"] = value
    teacher["backbone.mask_token"] = torch.zeros(1, 384)
    teacher["dino_head.mlp.0.weight"] = torch.zeros(4, 384)
    torch.save({"teacher": teacher, "iteration": 7}, path)
    return reference


def test_checkpoint_weights_reach_every_layer(tmp_path: Path) -> None:
    path = tmp_path / "dinov2.pth"
    reference = _checkpoint(path)
    native = gastronet_module.load_backbone(path, 56, SMALL)
    for name, value in native.state_dict().items():
        assert torch.equal(value, reference[name]), name
    smaller = gastronet_module.load_backbone(path, 28, SMALL).state_dict()
    assert smaller["pos_embed"].shape == (1, 1 + 4, 384)
    assert torch.equal(smaller["blocks.11.mlp.fc2.weight"], reference["blocks.11.mlp.fc2.weight"])


def test_incomplete_checkpoints_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "dinov2.pth"
    _checkpoint(path, drop="blocks.3.attn.qkv.weight")
    with pytest.raises(ValueError, match=r"missing \['blocks.3.attn.qkv.weight'\]"):
        gastronet_module.load_backbone(path, 56, SMALL)


def test_gastronet_encoder_reads_the_configured_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "dinov2.pth"
    _checkpoint(path)
    load = gastronet_module.load_backbone
    monkeypatch.setattr(
        gastronet_module, "load_backbone", lambda where, size: load(where, size, SMALL)
    )
    monkeypatch.setenv(gastronet_module.WEIGHTS_ENV, str(path))
    encoder = GastroNetDINOv2(image_size=28)
    images = [np.random.default_rng(0).integers(0, 255, (30, 40, 3), dtype=np.uint8)] * 2
    with torch.inference_mode():
        global_features = run(encoder, images, torch.device("cpu"))
        dense = run(encoder, images, torch.device("cpu"), dense=True)
    assert global_features.shape == (2, 384)
    assert dense.shape == (2, 384, 2, 2)
    assert encoder.name == "gastronet-dinov2-vitb14-28"
    assert encoder.card.pretrained_on == ("gastronet-5m",)
    assert not any(p.requires_grad for p in encoder.parameters())


def test_a_missing_checkpoint_points_to_the_portal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(gastronet_module.WEIGHTS_ENV, raising=False)
    monkeypatch.setenv("FIND_THOSE_POLYPS_DATA", str(tmp_path))
    assert gastronet_module.default_weights() == tmp_path / "weights" / "gastronet-dinov2.pth"
    with pytest.raises(FileNotFoundError, match=re.escape(gastronet_module.PORTAL)):
        GastroNetDINOv2()
