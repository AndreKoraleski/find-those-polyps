"""Common ground of the pretraining methods.

A method wraps the backbone with whatever its objective needs, heads, a momentum copy or a
decoder, and carries its recipe, the schedule that trains it. The trainer only asks a method for
the views of a frame and for the loss of a batch of views.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar

import torch
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from timm.models.vision_transformer import VisionTransformer
from torch import nn

from find_those_polyps.pretraining.architecture import Init


class Schedule(BaseModel):
    """Optimization settings of a recipe, for AdamW with a warm up and a cosine decay.

    Attributes:
        epochs: Number of passes over the corpora.
        batch_size: Number of frames per step.
        base_lr: Peak learning rate for a batch of 256 frames, scaled linearly with the batch.
        min_lr: Learning rate at the end of the cosine decay.
        weight_decay: Weight decay of the weights, never applied to biases, normalization and
            tokens.
        warmup_epochs: Epochs over which the learning rate rises linearly from zero.
        betas: Coefficients of the running averages of AdamW.
        clip_grad: Largest norm of the gradient, or `None` to leave it unclipped.
        drop_path: Largest rate of stochastic depth of the backbone.
    """

    model_config = ConfigDict(frozen=True)

    epochs: int = Field(gt=0)
    batch_size: int = Field(default=256, gt=0)
    base_lr: float = Field(gt=0)
    min_lr: float = Field(default=1e-6, ge=0)
    weight_decay: float = Field(ge=0)
    warmup_epochs: float = Field(ge=0)
    betas: tuple[float, float] = (0.9, 0.999)
    clip_grad: float | None = None
    drop_path: float = Field(default=0.0, ge=0, lt=1)

    @property
    def lr(self) -> float:
        """Peak learning rate for the batch size of the schedule."""
        return self.base_lr * self.batch_size / 256


class Method(nn.Module, ABC):
    """Self supervised objective around a backbone.

    Subclasses set `name`, `scratch` and `continued`, build their views in `views` and return
    their loss from `forward`.

    Args:
        backbone: Vision transformer to train, whose inputs may have any size its patches tile.

    Attributes:
        name: Identifier of the method on the command line.
        scratch: Recipe that trains the backbone from random weights.
        continued: Recipe that continues pretrained weights, with a lower learning rate.
        backbone: Vision transformer being trained.
        progress: Share of the training steps already taken, in [0, 1], set by the trainer
            before every step.
    """

    name: ClassVar[str]
    scratch: ClassVar[Schedule]
    continued: ClassVar[Schedule]

    def __init__(self, backbone: VisionTransformer) -> None:
        super().__init__()
        self.backbone = backbone
        self.progress = 0.0

    @classmethod
    def recipe(cls, init: Init) -> Schedule:
        """Schedule of the method for a starting point.

        Args:
            init: `dinov2` when the backbone continues pretrained weights, `random` otherwise.

        Returns:
            `continued` for pretrained weights and `scratch` for random ones.
        """
        return cls.continued if init == "dinov2" else cls.scratch

    @abstractmethod
    def views(self) -> Callable[[Image.Image], Any]:
        """Augmentation that turns one frame into the input of the method.

        Returns:
            A function from a decoded RGB image to a tensor or a list of tensors, which the data
            loader batches position by position.
        """

    @abstractmethod
    def forward(self, batch: Any) -> torch.Tensor:
        """Loss of a batch of views.

        Args:
            batch: Views returned by `views`, batched and moved to the device of the method.

        Returns:
            The scalar loss.
        """

    def before_step(self, progress: float) -> None:
        """Adjust the gradients before the optimizer applies them.

        Args:
            progress: Share of the training steps already taken, in [0, 1].
        """

    def after_step(self, progress: float) -> None:
        """Update what the optimizer does not, such as a momentum copy.

        Args:
            progress: Share of the training steps taken, this one included, in [0, 1].
        """

    def exported(self) -> VisionTransformer:
        """Network whose weights the trained encoder keeps.

        Returns:
            The backbone, or the copy of it that the method evaluates.
        """
        return self.backbone


def cosine(start: float, end: float, progress: float) -> float:
    """Value of a half cosine from `start` to `end`.

    Args:
        start: Value at progress zero.
        end: Value at progress one.
        progress: Position in [0, 1], clipped to it.

    Returns:
        The interpolated value.
    """
    position = min(max(progress, 0.0), 1.0)
    return end + (start - end) * (1 + math.cos(math.pi * position)) / 2


@torch.no_grad()
def follow(target: nn.Module, source: nn.Module, momentum: float) -> None:
    """Move the weights of a momentum copy towards those of the network it follows.

    Args:
        target: Momentum copy, updated in place.
        source: Network being followed, with the same parameters.
        momentum: Share of its own weights that the copy keeps.
    """
    for kept, new in zip(target.parameters(), source.parameters(), strict=True):
        kept.mul_(momentum).add_(new.detach(), alpha=1 - momentum)
    for buffer, value in zip(target.buffers(), source.buffers(), strict=True):
        buffer.copy_(value)
