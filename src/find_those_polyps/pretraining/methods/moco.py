"""Contrastive learning with a momentum encoder, MoCo v3 of Chen et al. (2021).

Two views of a frame are encoded by the backbone and by a momentum copy of it. The backbone,
through a projector and a predictor, must pick the momentum embedding of its own frame among those
of the batch, in both directions. The patch projection can be frozen, which the paper found to
stabilize vision transformers.
"""

import copy
import itertools
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from timm.models.vision_transformer import VisionTransformer
from torch import nn

from find_those_polyps.pretraining.architecture import IMAGE_SIZE
from find_those_polyps.pretraining.augment import Views, crop
from find_those_polyps.pretraining.methods.base import Method, Schedule, cosine, follow


class MoCoSettings(BaseModel):
    """Settings of MoCo v3.

    Attributes:
        image_size: Side in pixels of both views.
        width: Width of the embeddings that are compared.
        hidden: Width of the hidden layers of the projector and the predictor.
        temperature: Temperature of the contrastive loss.
        momentum: Momentum of the momentum encoder at the start, which a cosine brings to one.
        scale: Share of the frame area that a crop covers.
        freeze_patch_projection: Whether the patch projection of the backbone is left untrained.
    """

    model_config = ConfigDict(frozen=True)

    image_size: int = IMAGE_SIZE
    width: int = Field(default=256, gt=0)
    hidden: int = Field(default=4096, gt=0)
    temperature: float = Field(default=0.2, gt=0)
    momentum: float = Field(default=0.99, ge=0, le=1)
    scale: tuple[float, float] = (0.08, 1.0)
    freeze_patch_projection: bool = True


def mlp(sizes: list[int], *, last_norm: bool) -> nn.Sequential:
    """Multilayer perceptron with batch normalization, as the heads of MoCo v3.

    Args:
        sizes: Width of the input, of every hidden layer and of the output.
        last_norm: Whether the output is batch normalized, without a learned scale or shift.

    Returns:
        The layers in order.
    """
    layers: list[nn.Module] = []
    for index, (wide, narrow) in enumerate(itertools.pairwise(sizes)):
        final = index == len(sizes) - 2
        layers.append(nn.Linear(wide, narrow, bias=False))
        if not final:
            layers += [nn.BatchNorm1d(narrow), nn.ReLU(inplace=True)]
        elif last_norm:
            layers.append(nn.BatchNorm1d(narrow, affine=False))
    return nn.Sequential(*layers)


class MoCo(Method):
    """MoCo v3 around a vision transformer.

    Args:
        backbone: Vision transformer that is trained and kept.
        settings: Settings of the method, those of the paper for vision transformers when `None`.

    Attributes:
        settings: Settings of the method.
        projector: Projection of the class token of the backbone.
        predictor: Prediction of the momentum embedding from the projection.
        momentum_backbone: Momentum copy of the backbone.
        momentum_projector: Momentum copy of the projector.
    """

    name = "moco"
    scratch = Schedule(epochs=100, base_lr=1.5e-4, weight_decay=0.1, warmup_epochs=10)
    continued = Schedule(epochs=20, base_lr=1.5e-5, weight_decay=0.1, warmup_epochs=2)

    def __init__(self, backbone: VisionTransformer, settings: MoCoSettings | None = None) -> None:
        super().__init__(backbone)
        self.settings = s = settings or MoCoSettings()
        if s.freeze_patch_projection:
            backbone.patch_embed.requires_grad_(False)
        width = backbone.embed_dim
        self.projector = mlp([width, s.hidden, s.hidden, s.width], last_norm=True)
        self.predictor = mlp([s.width, s.hidden, s.width], last_norm=False)
        self.momentum_backbone = copy.deepcopy(backbone).requires_grad_(False)
        self.momentum_projector = copy.deepcopy(self.projector).requires_grad_(False)

    def views(self) -> Callable[[Image.Image], Any]:
        """Two crops with asymmetric blur and solarization.

        Returns:
            The augmentation, from a frame to two tensors.
        """
        s = self.settings
        first = crop(s.image_size, s.scale, blur=1.0)
        second = crop(s.image_size, s.scale, blur=0.1, solarize=0.2)
        return Views([first, second])

    def contrast(self, queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """Cross entropy of matching each query with the key of the same frame.

        Args:
            queries: Predictions of shape (B, D).
            keys: Momentum embeddings of shape (B, D), in the same frame order.

        Returns:
            The loss, scaled by twice the temperature as in the paper.
        """
        temperature = self.settings.temperature
        queries = F.normalize(queries.float(), dim=1)
        keys = F.normalize(keys.float(), dim=1)
        logits = queries @ keys.T / temperature
        labels = torch.arange(len(queries), device=queries.device)
        return F.cross_entropy(logits, labels) * 2 * temperature

    def forward(self, batch: list[torch.Tensor]) -> torch.Tensor:
        """Symmetric contrastive loss of two views.

        Args:
            batch: The two batches of views, each of shape (B, 3, S, S).

        Returns:
            The sum of the losses in both directions.
        """
        first, second = batch
        queries = [
            self.predictor(self.projector(self.backbone.forward_features(view)[:, 0]))
            for view in (first, second)
        ]
        with torch.no_grad():
            keys = [
                self.momentum_projector(self.momentum_backbone.forward_features(view)[:, 0])
                for view in (first, second)
            ]
        return self.contrast(queries[0], keys[1]) + self.contrast(queries[1], keys[0])

    def after_step(self, progress: float) -> None:
        """Move the momentum encoder towards the backbone.

        Args:
            progress: Share of the training steps taken.
        """
        momentum = cosine(self.settings.momentum, 1.0, progress)
        follow(self.momentum_backbone, self.backbone, momentum)
        follow(self.momentum_projector, self.projector, momentum)
