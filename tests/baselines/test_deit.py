import numpy as np
import pytest
import timm
import torch
from timm.models.vision_transformer import VisionTransformer

from find_those_polyps.baselines import ImageNetViT
from find_those_polyps.models import run


@pytest.fixture
def requested(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, bool, int]]:
    """Replace the timm factory by a small random transformer, recording what was requested."""
    calls: list[tuple[str, bool, int]] = []

    def fake(name: str, pretrained: bool = False, img_size: int = 224) -> VisionTransformer:
        calls.append((name, pretrained, img_size))
        return VisionTransformer(
            img_size=img_size, patch_size=16, embed_dim=8, depth=1, num_heads=2, init_values=1e-6
        )

    monkeypatch.setattr(timm, "create_model", fake)
    return calls


def test_imagenet_vit_exposes_class_and_patch_tokens(
    requested: list[tuple[str, bool, int]],
) -> None:
    encoder = ImageNetViT("small", image_size=32)
    images = [np.random.default_rng(0).integers(0, 255, (40, 30, 3), dtype=np.uint8)] * 2
    device = torch.device("cpu")
    with torch.inference_mode():
        global_features = run(encoder, images, device)
        dense = run(encoder, images, device, dense=True)
        tokens = encoder(torch.stack([encoder.preprocess(image) for image in images]))
    assert global_features.shape == (2, 8)
    assert dense.shape == (2, 8, 2, 2)
    assert torch.equal(global_features, tokens[:, 0])
    assert torch.equal(dense[:, :, 0, 1], tokens[:, 2])
    assert torch.equal(dense[:, :, 1, 0], tokens[:, 3])
    assert encoder.has_dense
    assert encoder.card.pretrained_on == ("imagenet-1k",)
    assert not any(p.requires_grad for p in encoder.parameters())
    assert requested == [("deit3_small_patch16_224.fb_in1k", True, 32)]


def test_imagenet_vit_names_follow_size_and_resolution(
    requested: list[tuple[str, bool, int]],
) -> None:
    assert ImageNetViT().name == "imagenet-deit3-vitb16"
    assert ImageNetViT("large").name == "imagenet-deit3-vitl16"
    assert ImageNetViT(image_size=256).name == "imagenet-deit3-vitb16-256"
    assert [name for name, _, _ in requested] == [
        "deit3_base_patch16_224.fb_in1k",
        "deit3_large_patch16_224.fb_in1k",
        "deit3_base_patch16_224.fb_in1k",
    ]


def test_imagenet_vit_rejects_sizes_it_cannot_serve(
    requested: list[tuple[str, bool, int]],
) -> None:
    with pytest.raises(ValueError, match="multiple of 16"):
        ImageNetViT(image_size=224 + 14)
    with pytest.raises(ValueError, match="unknown"):
        ImageNetViT("tiny")  # type: ignore[arg-type]
    assert requested == []
