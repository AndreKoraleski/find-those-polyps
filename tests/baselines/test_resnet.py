import numpy as np
import pytest
import torch
from torchvision.models import ResNet, ResNet50_Weights, resnet50

from find_those_polyps.baselines import ImageNetResNet50, RandomResNet50
from find_those_polyps.baselines import resnet as resnet_module
from find_those_polyps.models import run


def test_random_resnet_is_seeded_and_shaped() -> None:
    images = [np.full((40, 30, 3), 120, dtype=np.uint8)] * 2
    device = torch.device("cpu")
    first, second, other = RandomResNet50(0), RandomResNet50(0), RandomResNet50(1)
    with torch.inference_mode():
        global_features = run(first, images, device)
        dense = run(first, images, device, dense=True)
        assert torch.equal(global_features, run(second, images, device))
        assert not torch.equal(global_features, run(other, images, device))
    assert global_features.shape == (2, 2048)
    assert dense.shape == (2, 1024, 14, 14)
    assert first.has_dense
    assert first.card.name == "random-resnet50"
    assert other.card.name == "random-resnet50-s1"
    assert not any(p.requires_grad for p in first.parameters())


def test_imagenet_reference_uses_torchvision_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    requested: list[object] = []

    def fake(weights: object = None) -> ResNet:
        requested.append(weights)
        return resnet50(weights=None)

    monkeypatch.setattr(resnet_module, "resnet50", fake)
    encoder = ImageNetResNet50()
    assert requested == [ResNet50_Weights.IMAGENET1K_V2]
    assert encoder.card.name == "imagenet-resnet50"
    assert encoder.has_dense
    larger = ImageNetResNet50(image_size=448)
    assert (larger.name, larger.image_size) == ("imagenet-resnet50-448", 448)


def test_names_carry_the_input_size_unless_it_is_the_default() -> None:
    assert RandomResNet50(image_size=224).name == "random-resnet50"
    assert RandomResNet50(image_size=448).name == "random-resnet50-448"
    assert RandomResNet50(1, image_size=336).name == "random-resnet50-s1-336"
    assert resnet_module.sized("x", 224) == "x"
