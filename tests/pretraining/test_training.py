import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import save_file
from timm.models.vision_transformer import VisionTransformer

from find_those_polyps.models import run
from find_those_polyps.pretraining import CorpusCard, CorpusDataset, CorpusReader
from find_those_polyps.pretraining import encoder as encoder_module
from find_those_polyps.pretraining import train as command
from find_those_polyps.pretraining.encoder import PretrainedViT
from find_those_polyps.pretraining.methods import (
    MAE,
    MAESettings,
    Method,
    MoCo,
    MoCoSettings,
    Schedule,
)
from find_those_polyps.pretraining.trainer import (
    CHECKPOINT,
    export,
    learning_rate,
    parameter_groups,
    train,
)

WIDTH = 16
UNITS = ("v0", "v1")
FRAMES = 8
SMALL = MAESettings(
    image_size=28, mask_ratio=0.5, decoder_width=16, decoder_depth=1, decoder_heads=2
)


def _vit(image_size: int = 28, *, dynamic: bool = True) -> VisionTransformer:
    """Small vision transformer with the patches of the backbone."""
    return VisionTransformer(
        img_size=image_size,
        patch_size=14,
        embed_dim=WIDTH,
        depth=1,
        num_heads=2,
        num_classes=0,
        dynamic_img_size=dynamic,
    )


def _card() -> CorpusCard:
    archives = [
        {
            "unit": unit,
            "url": f"https://example.invalid/{unit}.tar.gz",
            "filename": f"{unit}.tar.gz",
            "checksum": "md5:" + "0" * 32,
            "size": 1,
            "frames": FRAMES,
            "fps": 1.0,
        }
        for unit in UNITS
    ]
    return CorpusCard.model_validate(
        {
            "key": "toy",
            "name": "Toy",
            "modality": "colonoscopy",
            "licence": "CC BY 4.0",
            "url": "https://example.invalid",
            "citation": "c",
            "provenance": "p",
            "family": "toy",
            "frame_pattern": r"v\d_(\d+)\.jpg",
            "archives": archives,
        }
    )


@pytest.fixture
def dataset(tmp_path: Path) -> CorpusDataset:
    """Dataset over a toy corpus whose units are already cached, so that no host is read."""
    card = _card()
    root = tmp_path / "data"
    generator = np.random.default_rng(0)
    for unit in UNITS:
        folder = root / "corpora" / "toy" / unit
        folder.mkdir(parents=True)
        for index in range(FRAMES):
            pixels = generator.integers(0, 255, (32, 32, 3), dtype=np.uint8)
            buffer = io.BytesIO()
            Image.fromarray(pixels).save(buffer, format="PNG")
            (folder / f"{unit}_{index}.jpg").write_bytes(buffer.getvalue())
        marker = root / "corpora" / "toy" / ".complete" / f"{unit}.json"
        marker.parent.mkdir(exist_ok=True)
        marker.write_text(json.dumps({"checksum": "md5:" + "0" * 32, "strides": [1]}))
    return CorpusDataset([CorpusReader(card, root, stream=False)])


def _schedule(**changes: Any) -> Schedule:
    settings = {"epochs": 2, "batch_size": 4, "base_lr": 1e-2, "weight_decay": 0.05}
    return Schedule.model_validate(settings | {"warmup_epochs": 1} | changes)


def test_learning_rate_warms_up_then_decays() -> None:
    schedule = _schedule(epochs=3, batch_size=256, base_lr=1.0, min_lr=0.1)
    rates = [learning_rate(schedule, step, 10) for step in range(30)]
    assert rates[0] == pytest.approx(0.1) and rates[9] == pytest.approx(1.0)
    assert rates[9:] == sorted(rates[9:], reverse=True)
    assert rates[-1] == pytest.approx(0.1, abs=0.01)
    assert learning_rate(_schedule(warmup_epochs=0, batch_size=512), 0, 10) == pytest.approx(2e-2)


def test_weight_decay_spares_biases_norms_and_tokens() -> None:
    method = MAE(_vit(), SMALL)
    method.backbone.patch_embed.requires_grad_(False)
    decayed, plain = parameter_groups(method, 0.05)
    assert decayed["weight_decay"] == 0.05 and plain["weight_decay"] == 0.0
    names = {id(p): n for n, p in method.named_parameters()}
    spared = {names[id(p)] for p in plain["params"]}
    assert {"backbone.cls_token", "backbone.pos_embed", "mask_token", "predict.bias"} <= spared
    assert "predict.weight" in {names[id(p)] for p in decayed["params"]}
    everything = {names[id(p)] for group in (decayed, plain) for p in group["params"]}
    assert not any(name.startswith("backbone.patch_embed") for name in everything)


def test_training_checkpoints_every_epoch_and_resumes(
    dataset: CorpusDataset, tmp_path: Path
) -> None:
    torch.manual_seed(0)
    output = tmp_path / "run"
    method = MAE(_vit(), SMALL)
    before = [p.clone() for p in method.backbone.parameters()]
    path = train(method, dataset, _schedule(), output)
    assert path == output / CHECKPOINT
    state = torch.load(path, weights_only=True)
    assert (state["epoch"], state["step"]) == (2, 8)
    after = list(method.backbone.parameters())
    assert any(not torch.equal(old, new) for old, new in zip(before, after, strict=True))
    assert not list(output.glob("*.part"))

    resumed = MAE(_vit(), SMALL)
    train(resumed, dataset, _schedule(epochs=3), output)
    state = torch.load(path, weights_only=True)
    assert (state["epoch"], state["step"]) == (3, 12)
    untouched = MAE(_vit(), SMALL)
    train(untouched, dataset, _schedule(epochs=3), output)
    for kept, loaded in zip(resumed.parameters(), untouched.parameters(), strict=True):
        assert torch.equal(kept, loaded)
    with pytest.raises(ValueError, match="holds a mae run"):
        train(
            MoCo(_vit(), MoCoSettings(image_size=28, width=8, hidden=16)),
            dataset,
            _schedule(),
            output,
        )


def test_epochs_can_be_capped_and_must_hold_a_batch(dataset: CorpusDataset, tmp_path: Path) -> None:
    method = MAE(_vit(), SMALL)
    path = train(method, dataset, _schedule(epochs=1), tmp_path / "run", steps_per_epoch=1)
    assert torch.load(path, weights_only=True)["step"] == 1
    with pytest.raises(ValueError, match="less than one batch"):
        train(method, dataset, _schedule(batch_size=64), tmp_path / "other")


class _Broken(MAE):
    """Method whose loss is not a number."""

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return super().forward(batch) * float("nan")


def test_a_loss_that_is_not_finite_stops_the_run(dataset: CorpusDataset, tmp_path: Path) -> None:
    with pytest.raises(FloatingPointError, match="step 0"):
        train(_Broken(_vit(), SMALL), dataset, _schedule(), tmp_path / "run")
    assert not (tmp_path / "run" / CHECKPOINT).exists()


@pytest.fixture
def small_architecture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the backbone of the exported encoder by the small stand in."""
    monkeypatch.setattr(
        encoder_module, "architecture", lambda image_size: _vit(image_size, dynamic=False)
    )


def test_exported_backbones_become_benchmark_encoders(
    small_architecture: None, tmp_path: Path
) -> None:
    torch.manual_seed(0)
    method = MAE(_vit(), SMALL)
    path = export(
        method,
        tmp_path / "out" / "toy.safetensors",
        name="vitb14-mae",
        pretrained_on=["lvd-142m", "real-colon-train"],
    )
    encoder = PretrainedViT(path)
    assert encoder.name == "vitb14-mae"
    assert encoder.card.pretrained_on == ("lvd-142m", "real-colon-train")
    assert encoder.method == "mae"
    assert not any(p.requires_grad for p in encoder.parameters())
    for name, value in method.backbone.state_dict().items():
        if name != "pos_embed":
            assert torch.equal(encoder.backbone.state_dict()[name], value), name
    larger = PretrainedViT(path, image_size=56)
    assert larger.name == "vitb14-mae-56"
    images = [np.random.default_rng(0).integers(0, 255, (30, 40, 3), dtype=np.uint8)] * 2
    with torch.inference_mode():
        assert run(larger, images, torch.device("cpu")).shape == (2, WIDTH)
        assert run(larger, images, torch.device("cpu"), dense=True).shape == (2, WIDTH, 4, 4)


def test_files_that_are_not_exports_are_refused(small_architecture: None, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing pretrained backbone"):
        PretrainedViT(tmp_path / "absent.safetensors")
    with pytest.raises(ValueError, match="multiple of 14"):
        PretrainedViT(tmp_path / "absent.safetensors", image_size=230)
    bare = tmp_path / "bare.safetensors"
    save_file(_vit().state_dict(), bare)
    with pytest.raises(ValueError, match="not written by the pretraining export"):
        PretrainedViT(bare)
    foreign = tmp_path / "foreign.safetensors"
    state = _vit().state_dict()
    state["extra"] = torch.zeros(1)
    save_file(state, foreign, metadata={"name": "n", "method": "mae", "pretrained_on": "[]"})
    with pytest.raises(ValueError, match="unexpected"):
        PretrainedViT(foreign)


def test_command_trains_and_exports(
    dataset: CorpusDataset,
    small_architecture: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    built: list[tuple[str, float]] = []

    def backbone(init: str, *, drop_path: float = 0.0) -> VisionTransformer:
        built.append((init, drop_path))
        return _vit()

    card = dataset.readers[0].card
    lookup: Callable[[str], CorpusCard] = lambda key: card  # noqa: E731
    monkeypatch.setattr(command, "backbone", backbone)
    monkeypatch.setattr(command, "corpus", lookup)
    output = tmp_path / "run"
    arguments = ["--root", str(tmp_path / "data"), "--output", str(output), "--no-stream"]
    arguments += ["--epochs", "1", "--batch-size", "2", "--steps-per-epoch", "2"]
    arguments += ["--workers", "0", "--device", "cpu", "--init", "random"]
    assert command.main(["moco", *arguments]) == 0
    assert built == [("random", 0.0)]
    exported = PretrainedViT(output / "vitb14-moco-scratch.safetensors")
    assert exported.card.pretrained_on == ("toy",)
    assert (output / CHECKPOINT).exists()
    assert command.main(["moco", "--name", "mine", *arguments[:-2]]) == 0
    assert PretrainedViT(output / "mine.safetensors").card.pretrained_on == ("lvd-142m", "toy")
    assert "resuming" in capsys.readouterr().out


def test_command_rejects_unknown_corpora(capsys: pytest.CaptureFixture[str]) -> None:
    assert command.main(["mae", "--corpora", "absent"]) == 1
    assert "unknown corpus" in capsys.readouterr().err


def test_methods_carry_their_progress() -> None:
    method: Method = MAE(_vit(), SMALL)
    assert method.progress == 0.0
    assert method.exported() is method.backbone
