"""Self distillation with no labels, DINO of Caron et al. (2021).

A student sees every crop of a frame, two large and several small, and a teacher sees only the
large ones. The student learns to predict the output of the teacher on a different crop, as a
distribution over prototypes. The teacher is a momentum average of the student, its outputs are
centred and sharpened so that they do not collapse, and its backbone is the one kept.
"""

import copy
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from timm.models.vision_transformer import VisionTransformer
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from find_those_polyps.pretraining.architecture import IMAGE_SIZE
from find_those_polyps.pretraining.augment import Views, crop
from find_those_polyps.pretraining.methods.base import Method, Schedule, cosine, follow


class DINOSettings(BaseModel):
    """Settings of DINO.

    Attributes:
        prototypes: Size of the output distribution.
        hidden: Width of the hidden layers of the head.
        bottleneck: Width of the layer before the prototypes.
        global_size: Side in pixels of the two large crops.
        local_size: Side in pixels of the small crops.
        local_crops: Number of small crops.
        global_scale: Share of the frame area that a large crop covers.
        local_scale: Share of the frame area that a small crop covers.
        student_temperature: Temperature of the student distribution.
        teacher_temperature: Temperature of the teacher at the start and after its warm up.
        temperature_warmup: Share of the training over which the teacher temperature rises.
        momentum: Momentum of the teacher at the start, which a cosine brings to one.
        centre_momentum: Momentum of the running centre of the teacher outputs.
        freeze_last_layer: Share of the training during which the prototypes are not updated.
    """

    model_config = ConfigDict(frozen=True)

    prototypes: int = Field(default=65536, gt=1)
    hidden: int = Field(default=2048, gt=0)
    bottleneck: int = Field(default=256, gt=0)
    global_size: int = IMAGE_SIZE
    local_size: int = 98
    local_crops: int = Field(default=8, ge=0)
    global_scale: tuple[float, float] = (0.32, 1.0)
    local_scale: tuple[float, float] = (0.05, 0.32)
    student_temperature: float = Field(default=0.1, gt=0)
    teacher_temperature: tuple[float, float] = (0.04, 0.07)
    temperature_warmup: float = Field(default=0.3, ge=0, le=1)
    momentum: float = Field(default=0.996, ge=0, le=1)
    centre_momentum: float = Field(default=0.9, ge=0, le=1)
    freeze_last_layer: float = Field(default=0.01, ge=0, le=1)


class Head(nn.Module):
    """Projection of the class token to scores over the prototypes.

    Args:
        width: Width of the class token.
        settings: Sizes of the head.
    """

    def __init__(self, width: int, settings: DINOSettings) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(width, settings.hidden),
            nn.GELU(),
            nn.Linear(settings.hidden, settings.hidden),
            nn.GELU(),
            nn.Linear(settings.hidden, settings.bottleneck),
        )
        self.last = weight_norm(nn.Linear(settings.bottleneck, settings.prototypes, bias=False))
        self.last.parametrizations.weight.original0.data.fill_(1)
        self.last.parametrizations.weight.original0.requires_grad = False

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Scores of a batch of class tokens.

        Args:
            tokens: Class tokens of shape (B, D).

        Returns:
            Scores of shape (B, prototypes).
        """
        scores: torch.Tensor = self.last(F.normalize(self.mlp(tokens), dim=-1))
        return scores


class DINO(Method):
    """DINO around a vision transformer.

    Args:
        backbone: Vision transformer of the student.
        settings: Settings of the method, the defaults of the paper adapted to patches of
            fourteen pixels when `None`.

    Attributes:
        settings: Settings of the method.
        head: Head of the student.
        teacher: Momentum copy of the backbone, the network that is kept.
        teacher_head: Momentum copy of the head.
        centre: Running mean of the teacher scores.
    """

    name = "dino"
    scratch = Schedule(
        epochs=100, base_lr=5e-4, weight_decay=0.04, warmup_epochs=10, clip_grad=3.0, drop_path=0.1
    )
    continued = Schedule(
        epochs=20, base_lr=5e-5, weight_decay=0.04, warmup_epochs=2, clip_grad=3.0, drop_path=0.1
    )

    centre: torch.Tensor

    def __init__(self, backbone: VisionTransformer, settings: DINOSettings | None = None) -> None:
        super().__init__(backbone)
        self.settings = settings or DINOSettings()
        self.head = Head(backbone.embed_dim, self.settings)
        self.teacher = copy.deepcopy(backbone).requires_grad_(False).eval()
        self.teacher_head = copy.deepcopy(self.head).requires_grad_(False).eval()
        self.register_buffer("centre", torch.zeros(1, self.settings.prototypes))

    def views(self) -> Callable[[Image.Image], Any]:
        """Two large crops followed by the small ones.

        Returns:
            The augmentation, whose first two views are the large crops.
        """
        s = self.settings
        first = crop(s.global_size, s.global_scale, blur=1.0)
        second = crop(s.global_size, s.global_scale, blur=0.1, solarize=0.2)
        small = crop(s.local_size, s.local_scale, blur=0.5)
        return Views([first, second, *[small] * s.local_crops])

    def train(self, mode: bool = True) -> "DINO":
        """Set the mode of the student, keeping the teacher in evaluation mode.

        Args:
            mode: Whether the student trains.

        Returns:
            The method.
        """
        super().train(mode)
        self.teacher.eval()
        self.teacher_head.eval()
        return self

    def forward(self, batch: list[torch.Tensor]) -> torch.Tensor:
        """Cross entropy between the teacher and the student on different crops.

        Args:
            batch: The two batches of large crops, then the batches of small crops.

        Returns:
            The loss averaged over every pair of a teacher crop and a different student crop.
        """
        s = self.settings
        with torch.no_grad():
            temperature = s.teacher_temperature[1]
            if s.temperature_warmup and self.progress < s.temperature_warmup:
                share = self.progress / s.temperature_warmup
                low, high = s.teacher_temperature
                temperature = low + (high - low) * share
            large = torch.cat(batch[:2])
            scores = self.teacher_head(self.teacher.forward_features(large)[:, 0]).float()
            targets = F.softmax((scores - self.centre) / temperature, dim=-1).chunk(2)
            self.centre.mul_(s.centre_momentum).add_(
                scores.mean(dim=0, keepdim=True), alpha=1 - s.centre_momentum
            )
        tokens = [self.backbone.forward_features(large)[:, 0]]
        if len(batch) > 2:
            tokens.append(self.backbone.forward_features(torch.cat(batch[2:]))[:, 0])
        outputs = self.head(torch.cat(tokens)).float() / s.student_temperature
        predictions = F.log_softmax(outputs, dim=-1).chunk(len(batch))
        losses = [
            -(target * prediction).sum(dim=-1).mean()
            for seen, target in enumerate(targets)
            for shown, prediction in enumerate(predictions)
            if shown != seen
        ]
        return torch.stack(losses).mean()

    def before_step(self, progress: float) -> None:
        """Drop the gradient of the prototypes early in training.

        Args:
            progress: Share of the training steps already taken.
        """
        if progress < self.settings.freeze_last_layer:
            for parameter in self.head.last.parameters():
                parameter.grad = None

    def after_step(self, progress: float) -> None:
        """Move the teacher towards the student.

        Args:
            progress: Share of the training steps taken.
        """
        momentum = cosine(self.settings.momentum, 1.0, progress)
        follow(self.teacher, self.backbone, momentum)
        follow(self.teacher_head, self.head, momentum)

    def exported(self) -> VisionTransformer:
        """Backbone of the teacher.

        Returns:
            The momentum copy of the backbone.
        """
        return self.teacher
