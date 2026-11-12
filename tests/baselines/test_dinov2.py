import numpy as np
import pytest
import timm
import torch
from timm.models.vision_transformer import VisionTransformer

from find_those_polyps.baselines import DINOv2
from find_those_polyps.models import run


@pytest.fixture
def requested(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, bool, int]]:
    """Replace the timm factory by a small random transformer, recording what was requested."""
    calls: list[tuple[str, bool, int]] = []

    def fake(name: str, pretrained: bool = False, img_size: int = 224) -> VisionTransformer:
        calls.append((name, pretrained, img_size))
        registers = 4 if "reg4" in name else 0
        return VisionTransformer(
            img_size=img_size,
            patch_size=14,
            embed_dim=8,
            depth=1,
            num_heads=2,
            reg_tokens=registers,
        )

    monkeypatch.setattr(timm, "create_model", fake)
    return calls


@pytest.mark.parametrize("registers", [False, True])
def test_dinov2_exposes_class_and_patch_tokens(
    requested: list[tuple[str, bool, int]], registers: bool
) -> None:
    encoder = DINOv2("small", image_size=28, registers=registers)
    images = [np.random.default_rng(0).integers(0, 255, (40, 30, 3), dtype=np.uint8)] * 2
    device = torch.device("cpu")
    with torch.inference_mode():
        global_features = run(encoder, images, device)
        dense = run(encoder, images, device, dense=True)
        tokens = encoder(torch.stack([encoder.preprocess(image) for image in images]))
    prefix = 5 if registers else 1
    assert global_features.shape == (2, 8)
    assert dense.shape == (2, 8, 2, 2)
    assert torch.equal(global_features, tokens[:, 0])
    assert torch.equal(dense[:, :, 0, 1], tokens[:, prefix + 1])
    assert torch.equal(dense[:, :, 1, 0], tokens[:, prefix + 2])
    assert encoder.has_dense
    assert encoder.card.pretrained_on == ("lvd-142m",)
    assert not any(p.requires_grad for p in encoder.parameters())
    reg = "_reg4" if registers else ""
    assert requested == [(f"vit_small_patch14{reg}_dinov2.lvd142m", True, 28)]


def test_dinov2_names_follow_size_registers_and_resolution(
    requested: list[tuple[str, bool, int]],
) -> None:
    assert DINOv2().name == "dinov2-vitb14"
    assert DINOv2("large", registers=True).name == "dinov2-vitl14-reg4"
    assert DINOv2(image_size=448).name == "dinov2-vitb14-448"
    assert [name for name, _, _ in requested] == [
        "vit_base_patch14_dinov2.lvd142m",
        "vit_large_patch14_reg4_dinov2.lvd142m",
        "vit_base_patch14_dinov2.lvd142m",
    ]


def test_dinov2_rejects_sizes_it_cannot_serve(requested: list[tuple[str, bool, int]]) -> None:
    with pytest.raises(ValueError, match="multiple of 14"):
        DINOv2(image_size=225)
    with pytest.raises(ValueError, match="unknown"):
        DINOv2("tiny")  # type: ignore[arg-type]
    assert requested == []
