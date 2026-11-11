"""Linear segmentation on frozen dense features.

A pointwise linear head maps every cell of the feature map to class logits, which are upsampled to
the resolution of the labels. Only the head is trained, so the score reflects the frozen features.
"""

import math
from collections.abc import Iterator, Sequence
from typing import ClassVar

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from numpy.typing import NDArray
from pydantic import NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt
from torch import nn

from find_those_polyps.evaluation.features import Features
from find_those_polyps.evaluation.protocols.base import Protocol, better
from find_those_polyps.logging import get_logger
from find_those_polyps.metrics import score
from find_those_polyps.tasks.schema import Target, TaskKind, TaskSpec

log = get_logger(__name__)

_EVAL_BATCH = 32


def _nearest(length: int, size: int) -> NDArray[np.int64]:
    """Source indices of nearest neighbour resampling from `length` to `size` samples.

    Args:
        length: Number of source samples along one axis.
        size: Number of resampled samples along that axis.

    Returns:
        Index into the source axis of each of the `size` resampled samples, as chosen by the
        default nearest mode of `F.interpolate`.
    """
    positions = torch.arange(length, dtype=torch.float32)[None, None]
    return F.interpolate(positions, size=size)[0, 0].long().numpy()


def _resize_masks(features: Features, size: int) -> torch.Tensor:
    """Nearest neighbour resampling of every mask, or stack of class masks, to side `size`.

    Rows and columns are gathered at the source indices of `F.interpolate`, which gives the same
    masks without resampling every pixel in floating point.

    Args:
        features: Features whose targets are masks of shape `(H, W)` or stacks of class masks of
            shape `(C, H, W)`.
        size: Side of the square resampled masks.

    Returns:
        Masks of shape `(N, size, size)` or `(N, C, size, size)` in uint8 on the CPU.
    """
    masks = []
    for target in features.targets:
        mask = np.asarray(target)
        height, width = mask.shape[-2:]
        resized = mask[..., _nearest(height, size), :][..., _nearest(width, size)]
        masks.append(torch.as_tensor(resized.astype(np.uint8)))
    return torch.stack(masks)


def resident(values: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Values moved to `device` when they take at most a quarter of its free memory.

    Keeping the training features on the accelerator avoids a host transfer at every step, while
    the limit leaves room for the head and its activations. Devices other than CUDA keep the values
    where they are.

    Args:
        values: Tensor to place, usually features or labels on the CPU.
        device: Device on which the computation runs.

    Returns:
        The values on `device` when they fit, and the unchanged values otherwise.
    """
    if device.type != "cuda":
        return values
    free, _ = torch.cuda.mem_get_info(device)
    return values.to(device) if values.nbytes <= free // 4 else values


def _probability_maps(
    head: nn.Module,
    values: torch.Tensor,
    targets: Sequence[Target],
    device: torch.device,
    *,
    multilabel: bool = False,
) -> Iterator[NDArray[np.float32]]:
    """Class probabilities at the native resolution of each target, lazily.

    Classes compete through a softmax, or are independent through sigmoids with `multilabel`.
    Logits are upsampled bilinearly before the nonlinearity. Each map of shape `(C, H, W)` stores
    the classes innermost in memory, which speeds up the reductions over classes in the metrics.

    Args:
        head: Segmentation head, switched to evaluation mode.
        values: Dense features of shape `(N, D, h, w)` on any device.
        targets: Masks whose last two dimensions give the native resolution `(H, W)`.
        device: Device on which the head runs.
        multilabel: Whether classes may overlap.

    Yields:
        Probability map of shape `(C, H, W)` in float32 for each target in order.
    """
    head.eval()
    for start in range(0, len(targets), _EVAL_BATCH):
        with torch.inference_mode():
            logits = head(values[start : start + _EVAL_BATCH].to(device).float())
        for logit, target in zip(logits, targets[start : start + _EVAL_BATCH], strict=True):
            height, width = np.asarray(target).shape[-2:]
            upsampled = F.interpolate(
                logit[None], size=(height, width), mode="bilinear", align_corners=False
            )[0]
            probability = upsampled.sigmoid() if multilabel else upsampled.softmax(dim=0)
            interleaved = probability.permute(1, 2, 0).contiguous().cpu().numpy()
            yield interleaved.transpose(2, 0, 1).astype(np.float32, copy=False)


class LinearSegmentation(Protocol):
    """Batch normalization and a pointwise convolution on frozen dense features.

    Mirrors the linear segmentation head of Oquab et al. (2024). Logits are bilinearly upsampled and
    trained with cross entropy at `label_size` under AdamW with cosine decay over `steps`, or with a
    binary cross entropy per class when classes may overlap. Masks are resampled to `label_size` by
    nearest neighbour. The validation score is computed at native mask resolution every
    `eval_every` steps and after the last step, and the checkpoint with the best score is kept.
    Training stops early once `patience` successive evaluations bring no improvement, but never
    before `min_steps`.

    Attributes:
        steps: Maximal number of optimization steps, also the period of the cosine decay.
            Defaults to 2000.
        batch_size: Number of images per step, drawn uniformly with replacement. Defaults to 16.
        lr: Initial learning rate of AdamW. Defaults to 1e-3.
        weight_decay: Decoupled weight decay of AdamW. Defaults to 1e-4.
        eval_every: Number of steps between validation evaluations. Defaults to 200.
        patience: Number of successive evaluations without improvement that stops training.
            Defaults to 5.
        min_steps: Number of steps before which training never stops early. Defaults to 600.
        label_size: Side in pixels of the square masks on which the loss is computed.
            Defaults to 224.
    """

    name: ClassVar[str] = "linear_seg"
    kinds: ClassVar[frozenset[TaskKind]] = frozenset(
        {TaskKind.SEGMENTATION, TaskKind.MULTILABEL_SEGMENTATION}
    )
    dense: ClassVar[bool] = True
    stochastic: ClassVar[bool] = True

    steps: PositiveInt = 2000
    batch_size: PositiveInt = 16
    lr: PositiveFloat = 1e-3
    weight_decay: NonNegativeFloat = 1e-4
    eval_every: PositiveInt = 200
    patience: PositiveInt = 5
    min_steps: NonNegativeInt = 600
    label_size: PositiveInt = 224

    def predict(
        self,
        train: Features,
        val: Features,
        test: Features,
        spec: TaskSpec,
        *,
        seed: int,
        device: torch.device,
    ) -> Iterator[NDArray[np.float32]]:
        """Train the head on `train`, select its checkpoint on `val` and predict `test`.

        Args:
            train: Dense training features of shape `(N, D, h, w)` with their masks.
            val: Dense validation features with their masks, used to select the checkpoint.
            test: Dense test features with their masks, which give the output resolution.
            spec: Task specification with the classes, the kind and the primary metric.
            seed: Seed of the head initialization and of the batch sampling.
            device: Device on which the head is trained.

        Returns:
            Lazy iterator over the probability maps of shape `(C, H, W)` of the test images, at
            the native resolution of their masks.
        """
        n_classes = len(spec.classes)
        multilabel = spec.kind is TaskKind.MULTILABEL_SEGMENTATION
        dim = train.values.shape[1]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            head = nn.Sequential(nn.BatchNorm2d(dim), nn.Conv2d(dim, n_classes, 1))
        head.to(device)
        generator = torch.Generator().manual_seed(seed)
        labels = resident(_resize_masks(train, self.label_size), device)
        values = resident(train.values, device)
        val_values = resident(val.values, device)
        batches = torch.randint(len(labels), (self.steps, self.batch_size), generator=generator)
        value_batches, label_batches = batches.to(values.device), batches.to(labels.device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        schedule = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: 0.5 * (1 + math.cos(math.pi * step / self.steps))
        )
        primary = spec.primary_metric

        best_state = {k: v.clone() for k, v in head.state_dict().items()}
        best_score = float("nan")
        stalled = 0
        for step in range(1, self.steps + 1):
            head.train()
            logits = head(values[value_batches[step - 1]].to(device).float())
            logits = F.interpolate(
                logits,
                size=(self.label_size, self.label_size),
                mode="bilinear",
                align_corners=False,
            )
            target = labels[label_batches[step - 1]].to(device)
            loss = (
                F.binary_cross_entropy_with_logits(logits, target.float())
                if multilabel
                else F.cross_entropy(logits, target.long())
            )
            optimizer.zero_grad(set_to_none=True)
            torch.autograd.backward(loss)
            optimizer.step()
            schedule.step()
            if step % self.eval_every == 0 or step == self.steps:
                maps = _probability_maps(
                    head, val_values, val.targets, device, multilabel=multilabel
                )
                current = score(spec.kind, maps, val.targets, n_classes, [primary])
                stalled += 1
                if better(current[primary], best_score, primary):
                    best_score, stalled = current[primary], 0
                    best_state = {k: v.clone() for k, v in head.state_dict().items()}
                log.info(
                    "probe validated",
                    protocol=self.name,
                    step=step,
                    steps=self.steps,
                    score=round(current[primary], 4),
                    best=round(best_score, 4),
                )
                if stalled >= self.patience and step >= self.min_steps:
                    break
        head.load_state_dict(best_state)
        return _probability_maps(head, test.values, test.targets, device, multilabel=multilabel)
