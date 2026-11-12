"""Linear segmentation on frozen dense features.

A pointwise linear head maps every cell of the feature map to class logits, which are upsampled to
the resolution of the labels. Only the head is trained, so the score reflects the frozen features.
"""

import math
import time
from collections.abc import Iterator, Sequence
from typing import ClassVar

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from numpy.typing import NDArray
from pydantic import NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt
from torch import nn

from find_those_polyps.evaluation.features import Features
from find_those_polyps.evaluation.protocols.base import Draw, Protocol, better
from find_those_polyps.evaluation.protocols.heads import StackedHeads
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


def _device_dice(
    head: nn.Module,
    values: torch.Tensor,
    masks: Sequence[torch.Tensor],
    device: torch.device,
    *,
    n_classes: int,
    multilabel: bool = False,
) -> float:
    """Mean Dice over images, computed on the device without moving probability maps to the host.

    Predictions follow `_probability_maps` and the masks, classes and averaging follow
    `per_image`, so the value equals the Dice that `score` gives for the same head. Only the
    counts of true and false positives and false negatives leave the device, and the quotients
    and means are taken on the host as there.

    Args:
        head: Segmentation head, switched to evaluation mode.
        values: Dense features of shape `(N, D, h, w)` on any device.
        masks: Target of every image at native resolution, a class mask of shape `(H, W)` or a
            stack of class masks of shape `(C, H, W)` with `multilabel`.
        device: Device on which the head runs.
        n_classes: Number of classes `C`.
        multilabel: Whether classes may overlap.

    Returns:
        The mean over images of the Dice of each image, NaN without images.
    """
    head.eval()
    counts = []
    with torch.inference_mode():
        for start in range(0, len(masks), _EVAL_BATCH):
            logits = head(values[start : start + _EVAL_BATCH].to(device).float())
            for logit, mask in zip(logits, masks[start : start + _EVAL_BATCH], strict=True):
                target = mask.to(device)
                upsampled = F.interpolate(
                    logit[None], size=tuple(target.shape[-2:]), mode="bilinear", align_corners=False
                )[0]
                probability = upsampled.sigmoid() if multilabel else upsampled.softmax(dim=0)
                if multilabel:
                    predicted, truth = probability > 0.5, target > 0
                elif n_classes == 2:
                    predicted = (probability[1] > probability[0])[None]
                    truth = (target == 1)[None]
                else:
                    classes = torch.arange(1, n_classes, device=device)[:, None, None]
                    predicted = probability.argmax(dim=0)[None] == classes
                    truth = target[None] == classes
                counts.append(
                    torch.stack(
                        [
                            (predicted & truth).sum(dim=(1, 2)),
                            (predicted & ~truth).sum(dim=(1, 2)),
                            (~predicted & truth).sum(dim=(1, 2)),
                            (predicted.flatten(1).any(1) | truth.flatten(1).any(1)).long(),
                        ]
                    )
                )
    if not counts:
        return float("nan")
    scores = []
    for tp, fp, fn, present in torch.stack(counts).cpu().tolist():
        dice = [_safe(2.0 * a, 2.0 * a + b + c) for a, b, c in zip(tp, fp, fn, strict=True)]
        chosen = [d for d, keep in zip(dice, present, strict=True) if keep]
        scores.append(float(np.mean(chosen if len(dice) > 1 and chosen else dice)))
    return float(np.mean(scores))


def _safe(numerator: float, denominator: float) -> float:
    """Dice quotient, one when both masks are empty, as in `overlap`."""
    return numerator / denominator if denominator else 1.0


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
        draw = Draw(seed, np.arange(len(train.targets)))
        return next(self.predict_runs(train, val, test, spec, [draw], device=device))

    def predict_runs(
        self,
        train: Features,
        val: Features,
        test: Features,
        spec: TaskSpec,
        draws: Sequence[Draw],
        *,
        device: torch.device,
    ) -> Iterator[Iterator[NDArray[np.float32]]]:
        """Train the heads of every repeat at once and predict `test` with each.

        The heads are stacked by `StackedHeads`, so every step trains all of them in the same
        kernels, each on its own batch drawn from its own subset with its own seed, as `predict`
        trains one. Each repeat is validated on its own subset and stops on its own, after which
        its checkpoint no longer changes, and training ends once every repeat has stopped.

        Args:
            train: Whole dense training partition of shape `(N, D, h, w)` with its masks.
            val: Whole dense validation partition with its masks.
            test: Dense test features with their masks, which give the output resolution.
            spec: Task specification with the classes, the kind and the primary metric.
            draws: Seed and samples of every repeat.
            device: Device on which the heads are trained.

        Returns:
            One lazy iterator over the test probability maps per repeat, in the order of
            `draws`, every head being trained before this returns.
        """
        began = time.monotonic()
        n_classes = len(spec.classes)
        multilabel = spec.kind is TaskKind.MULTILABEL_SEGMENTATION
        dim = train.values.shape[1]
        heads = []
        for draw in draws:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(draw.seed)
                heads.append(nn.Sequential(nn.BatchNorm2d(dim), nn.Conv2d(dim, n_classes, 1)))
        stacked = StackedHeads(heads).to(device)
        labels = resident(_resize_masks(train, self.label_size), device)
        values = resident(train.values, device)
        validation = _validation(val, draws, device)
        batches = batch_positions(draws, self.steps, self.batch_size)
        value_batches, label_batches = batches.to(values.device), batches.to(labels.device)
        optimizer = torch.optim.AdamW(
            stacked.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        schedule = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: 0.5 * (1 + math.cos(math.pi * step / self.steps))
        )
        primary = spec.primary_metric
        size = (self.label_size, self.label_size)

        repeats = len(draws)
        best_states = [stacked.member(index).state_dict() for index in range(repeats)]
        best_scores = [float("nan")] * repeats
        stalled = [0] * repeats
        active = [True] * repeats
        step = 0
        training = torch.arange(repeats)
        for step in range(1, self.steps + 1):
            stacked.train()
            logits = stacked(values[value_batches[step - 1]].to(device).float())
            # Repeats that stopped keep their checkpoint, so only those still training pay for
            # the loss at label resolution.
            logits = F.interpolate(
                logits[:, training.to(device)].flatten(1, 2),
                size=size,
                mode="bilinear",
                align_corners=False,
            ).view(-1, len(training), n_classes, *size)
            rows = label_batches[step - 1][:, training.to(label_batches.device)]
            target = labels[rows].to(device)
            # Every repeat averages over the same number of pixels, so the sum of their means is
            # the mean over all of them times the number of repeats.
            if multilabel:
                loss = F.binary_cross_entropy_with_logits(logits, target.float())
            else:
                loss = F.cross_entropy(logits.flatten(0, 1), target.flatten(0, 1).long())
            optimizer.zero_grad(set_to_none=True)
            torch.autograd.backward(loss * len(training))
            optimizer.step()
            schedule.step()
            if step % self.eval_every and step != self.steps:
                continue
            for index, draw in enumerate(draws):
                if not active[index]:
                    continue
                head = stacked.member(index)
                held_out, val_values, val_masks = validation[index]
                if primary == "dice":
                    dice = _device_dice(
                        head,
                        val_values,
                        val_masks,
                        device,
                        n_classes=n_classes,
                        multilabel=multilabel,
                    )
                    current = {primary: dice}
                else:
                    maps = _probability_maps(
                        head, val_values, held_out.targets, device, multilabel=multilabel
                    )
                    current = score(spec.kind, maps, held_out.targets, n_classes, [primary])
                stalled[index] += 1
                if better(current[primary], best_scores[index], primary):
                    best_scores[index], stalled[index] = current[primary], 0
                    best_states[index] = head.state_dict()
                log.info(
                    "probe validated",
                    protocol=self.name,
                    seed=draw.seed,
                    step=step,
                    steps=self.steps,
                    score=round(current[primary], 4),
                    best=round(best_scores[index], 4),
                )
                if stalled[index] >= self.patience and step >= self.min_steps:
                    active[index] = False
            if not any(active):
                break
            training = torch.tensor([index for index in range(repeats) if active[index]])
        chosen = []
        for index, state in enumerate(best_states):
            head = stacked.member(index)
            head.load_state_dict(state)
            chosen.append(head)
        log.info(
            "probes trained",
            protocol=self.name,
            runs=repeats,
            steps=step,
            seconds=round(time.monotonic() - began, 1),
        )
        return iter(
            [
                _probability_maps(head, test.values, test.targets, device, multilabel=multilabel)
                for head in chosen
            ]
        )


def batch_positions(draws: Sequence[Draw], steps: int, batch_size: int) -> torch.Tensor:
    """Training positions of every step and repeat, as each repeat would draw them alone.

    Every repeat draws its batches uniformly with replacement from its own subset, with a
    generator seeded by its seed, and its positions in the subset are mapped back to positions in
    the whole partition.

    Args:
        draws: Seed and samples of every repeat.
        steps: Number of optimization steps.
        batch_size: Number of samples per step and repeat.

    Returns:
        Positions in the training partition of shape `(steps, batch_size, S)` for `S` repeats.
    """
    columns = []
    for draw in draws:
        generator = torch.Generator().manual_seed(draw.seed)
        subset = torch.as_tensor(draw.train, dtype=torch.long)
        columns.append(subset[torch.randint(len(subset), (steps, batch_size), generator=generator)])
    return torch.stack(columns, dim=2)


def _validation(
    val: Features, draws: Sequence[Draw], device: torch.device
) -> list[tuple[Features, torch.Tensor, list[torch.Tensor]]]:
    """Validation features of every repeat, with their values placed and their masks as tensors.

    Repeats validated on the whole partition share one copy of it on the device.

    Args:
        val: Whole validation partition.
        draws: Seed and samples of every repeat.
        device: Device on which the heads are validated.

    Returns:
        For every repeat its validation features, their values placed by `resident` and their
        masks as tensors on the host.
    """
    whole: tuple[Features, torch.Tensor, list[torch.Tensor]] | None = None
    result = []
    for draw in draws:
        if draw.val is None and whole is not None:
            result.append(whole)
            continue
        held_out = val if draw.val is None else val.subset(draw.val)
        masks = [torch.from_numpy(np.ascontiguousarray(t)) for t in held_out.targets]
        entry = (held_out, resident(held_out.values, device), masks)
        if draw.val is None:
            whole = entry
        result.append(entry)
    return result
