import pickle
from collections.abc import Sequence

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image
from timm.models.vision_transformer import VisionTransformer

from find_those_polyps.pretraining.architecture import embed, encode
from find_those_polyps.pretraining.augment import Views, crop
from find_those_polyps.pretraining.methods import (
    DINO,
    MAE,
    METHODS,
    DINOSettings,
    MAESettings,
    Method,
    MoCo,
    MoCoSettings,
)
from find_those_polyps.pretraining.methods.base import cosine
from find_those_polyps.pretraining.methods.mae import sincos_positions

WIDTH = 32
DINO_SMALL = DINOSettings(
    prototypes=64,
    hidden=64,
    bottleneck=16,
    global_size=28,
    local_size=14,
    local_crops=2,
    momentum=0.99,
)
MAE_SMALL = MAESettings(
    image_size=56, mask_ratio=0.5, decoder_width=16, decoder_depth=1, decoder_heads=2
)
MOCO_SMALL = MoCoSettings(image_size=28, width=8, hidden=16)


def _vit() -> VisionTransformer:
    """Small vision transformer with the patches of the backbone, for inputs of any size."""
    return VisionTransformer(
        img_size=28,
        patch_size=14,
        embed_dim=WIDTH,
        depth=2,
        num_heads=2,
        num_classes=0,
        dynamic_img_size=True,
    )


def _frames(count: int, seed: int = 0) -> torch.Tensor:
    """Distinct smooth images of shape (count, 3, 28, 28)."""
    generator = torch.Generator().manual_seed(seed)
    coarse = torch.randn(count, 3, 4, 4, generator=generator)
    return F.interpolate(coarse, size=(28, 28), mode="bilinear")


def _views(frames: torch.Tensor, sizes: list[int]) -> list[torch.Tensor]:
    """Noisy views of the same frames at several sizes."""
    views = []
    for size in sizes:
        view = F.interpolate(frames, size=(size, size), mode="bilinear")
        views.append(view + 0.2 * torch.randn_like(view))
    return views


def _fit(method: Method, batches: Sequence[object], rate: float) -> list[float]:
    """Train a method on the given batches and return the loss of every step."""
    optimizer = torch.optim.AdamW([p for p in method.parameters() if p.requires_grad], lr=rate)
    losses = []
    method.train()
    for step, batch in enumerate(batches):
        method.progress = step / len(batches)
        loss = method(batch)
        optimizer.zero_grad()
        loss.backward()
        method.before_step(step / len(batches))
        optimizer.step()
        method.after_step((step + 1) / len(batches))
        losses.append(loss.item())
    return losses


def test_every_method_has_a_recipe_for_both_starts() -> None:
    assert sorted(METHODS) == ["dino", "mae", "moco"]
    for method in METHODS.values():
        continued, scratch = method.recipe("dinov2"), method.recipe("random")
        assert continued.base_lr < scratch.base_lr
        assert continued.epochs < scratch.epochs
        assert continued.lr == continued.base_lr * continued.batch_size / 256


@pytest.mark.parametrize(
    ("method", "sizes"),
    [
        (lambda: DINO(_vit(), DINO_SMALL), [28, 28, 14, 14]),
        (lambda: MAE(_vit(), MAE_SMALL), [56]),
        (lambda: MoCo(_vit(), MOCO_SMALL), [28, 28]),
    ],
    ids=["dino", "mae", "moco"],
)
def test_views_have_the_sizes_of_the_method_and_reach_workers(
    method: object, sizes: list[int]
) -> None:
    assert callable(method)
    views = pickle.loads(pickle.dumps(method().views()))
    image = Image.new("RGB", (60, 45), (150, 60, 40))
    made = views(image)
    tensors = made if isinstance(made, list) else [made]
    assert [tuple(view.shape) for view in tensors] == [(3, size, size) for size in sizes]
    assert all(view.dtype == torch.float32 for view in tensors)


def test_crops_are_standardized_and_views_follow_their_pipelines() -> None:
    plain = crop(14, (0.5, 1.0), colour=False)
    grey = plain(Image.new("RGB", (30, 30), (124, 116, 104)))
    assert grey.abs().max() < 0.05
    views = Views([plain, crop(28, (0.5, 1.0), blur=1.0, solarize=1.0)])
    assert [view.shape[-1] for view in views(Image.new("RGB", (30, 30)))] == [14, 28]


def test_tokens_can_be_encoded_in_part() -> None:
    torch.manual_seed(0)
    network = _vit().eval()
    images = torch.randn(2, 3, 42, 28)
    tokens = embed(network, images)
    assert tokens.shape == (2, 1 + 6, WIDTH)
    assert torch.allclose(encode(network, tokens), network.forward_features(images), atol=1e-6)
    assert encode(network, tokens[:, :3]).shape == (2, 3, WIDTH)


def test_mae_reconstructs_hidden_patches_only() -> None:
    torch.manual_seed(0)
    method = MAE(_vit(), MAE_SMALL)
    frames = F.interpolate(_frames(8), size=(56, 56), mode="bilinear")
    patches = method.patches(frames)
    assert patches.shape == (8, 16, 14 * 14 * 3)
    assert torch.equal(patches[0, 5].reshape(14, 14, 3)[:, :, 1], frames[0, 1, 14:28, 14:28])
    losses = _fit(method, [frames] * 40, 3e-3)
    assert losses[-1] < 0.8 * losses[0]
    assert all(p.grad is not None for p in method.backbone.blocks.parameters())
    assert "positions" not in method.state_dict()
    with pytest.raises(ValueError, match="multiple of 14"):
        MAE(_vit(), MAESettings(image_size=60))


def test_sincos_positions_tell_cells_apart() -> None:
    positions = sincos_positions(16, 3)
    assert positions.shape == (9, 16)
    assert len({tuple(row.tolist()) for row in positions}) == 9
    assert torch.equal(positions[0, :8], positions[1, :8])
    assert not torch.equal(positions[0, :8], positions[3, :8])


def test_moco_matches_views_of_the_same_frame() -> None:
    torch.manual_seed(0)
    backbone = _vit()
    method = MoCo(backbone, MOCO_SMALL)
    assert not any(p.requires_grad for p in backbone.patch_embed.parameters())
    assert not any(p.requires_grad for p in method.momentum_backbone.parameters())
    before = [p.clone() for p in method.momentum_backbone.blocks.parameters()]
    frames = _frames(16)
    losses = _fit(method, [_views(frames, [28, 28]) for _ in range(40)], 3e-3)
    assert losses[-1] < 0.6 * losses[0]
    after = list(method.momentum_backbone.blocks.parameters())
    assert any(not torch.equal(old, new) for old, new in zip(before, after, strict=True))
    assert method.exported() is backbone
    trainable = MoCo(_vit(), MOCO_SMALL.model_copy(update={"freeze_patch_projection": False}))
    assert all(p.requires_grad for p in trainable.backbone.patch_embed.parameters())


def test_dino_spreads_frames_over_its_prototypes() -> None:
    torch.manual_seed(0)
    method = DINO(_vit(), DINO_SMALL)
    assert not any(p.requires_grad for p in method.teacher.parameters())
    assert not method.teacher.training
    pool = _frames(64)

    def prototypes() -> int:
        with torch.no_grad():
            scores = method.teacher_head(method.teacher.forward_features(pool)[:, 0])
        return int((scores - method.centre).argmax(dim=-1).unique().numel())

    assert prototypes() <= 3
    batches = []
    for _ in range(150):
        chosen = pool[torch.randint(0, len(pool), (32,))]
        batches.append(_views(chosen, [28, 28, 14, 14]))
    losses = _fit(method, batches, 1e-3)
    assert sum(losses[-10:]) / 10 < max(losses) - 0.2
    assert prototypes() >= 6
    assert not method.teacher.training
    assert method.exported() is method.teacher
    assert method.centre.abs().sum() > 0


def test_dino_freezes_its_prototypes_early_and_warms_its_teacher() -> None:
    torch.manual_seed(0)
    method = DINO(_vit(), DINO_SMALL.model_copy(update={"freeze_last_layer": 0.5}))
    batch = _views(_frames(8), [28, 28, 14, 14])
    method(batch).backward()
    method.before_step(0.6)
    assert method.head.last.parametrizations.weight.original1.grad is not None
    method.before_step(0.1)
    assert all(p.grad is None for p in method.head.last.parameters())
    method.progress = 0.0
    cold = method(batch)
    method.progress = 1.0
    assert not torch.isclose(method(batch), cold)


def test_cosine_goes_from_start_to_end() -> None:
    assert cosine(0.99, 1.0, 0.0) == pytest.approx(0.99)
    assert cosine(0.99, 1.0, 1.0) == pytest.approx(1.0)
    assert cosine(0.0, 1.0, 0.5) == pytest.approx(0.5)
    assert cosine(0.0, 1.0, 2.0) == pytest.approx(1.0)
