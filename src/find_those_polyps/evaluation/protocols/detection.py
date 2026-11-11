"""Linear object detection on frozen dense features.

Every cell of the feature map predicts the score of each object class, the log distances from its
centre to the four sides of a box in units of cells and a centerness score, through batch
normalization and one pointwise convolution. This is the fully convolutional one stage detector of
Tian et al. (2019) reduced to a linear head on a single feature map, so that the score reflects the
frozen features rather than a trained neck. Box rows hold the class counted from one, then the
corners x0, y0, x1, y1 normalized by the image size, which leaves their overlaps unchanged.
Predicted rows insert the score after the class.
"""

import math
from typing import ClassVar

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from numpy.typing import NDArray
from pydantic import NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt, field_validator
from torch import nn
from torchvision.ops import batched_nms, sigmoid_focal_loss

from find_those_polyps.evaluation.features import Features
from find_those_polyps.evaluation.protocols.base import Protocol, better
from find_those_polyps.evaluation.protocols.segmentation import resident
from find_those_polyps.logging import get_logger
from find_those_polyps.metrics import score
from find_those_polyps.tasks.schema import TaskKind, TaskSpec

log = get_logger(__name__)

_EVAL_BATCH = 32
_MAX_LOG_DISTANCE = 8.0


def _centres(
    height: int, width: int, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalized coordinates of the cell centres as two `(h, w)` grids on `device`.

    Args:
        height: Number of rows `h` of the feature map.
        width: Number of columns `w` of the feature map.
        device: Device of the grids, the default device when None.

    Returns:
        Vertical and horizontal coordinates of every cell centre in `(0, 1)`.
    """
    rows = (torch.arange(height, dtype=torch.float32, device=device) + 0.5) / height
    columns = (torch.arange(width, dtype=torch.float32, device=device) + 0.5) / width
    row_grid, column_grid = torch.meshgrid(rows, columns, indexing="ij")
    return row_grid, column_grid


def assign(
    boxes: NDArray[np.float64], n_objects: int, height: int, width: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Training targets of every cell of an `(h, w)` feature map for the boxes of one image.

    A cell is positive when the centre of the cell lies strictly inside at least one box, and it
    takes the box of smallest area among those. A box containing no centre takes the cell under its
    own centre, unless a smaller box holds it.

    Args:
        boxes: Rows class, x0, y0, x1, y1 of shape `(M, 5)`, with classes counted from one and
            corners normalized by the image size.
        n_objects: Number of object classes `K`, excluding the background.
        height: Number of rows `h` of the feature map.
        width: Number of columns `w` of the feature map.

    Returns:
        Class indicators of shape `(K, h, w)` with a one at the class of the assigned box, the
        assigned corners of shape `(4, h, w)`, zero at negative cells, and the positive cells of
        shape `(h, w)`.
    """
    classes = torch.zeros(n_objects, height, width)
    assigned = torch.zeros(4, height, width)
    positive = torch.zeros(height, width, dtype=torch.bool)
    if not len(boxes):
        return classes, assigned, positive
    table = torch.as_tensor(boxes, dtype=torch.float32)
    labels, corners = table[:, 0].long() - 1, table[:, 1:]
    cy, cx = _centres(height, width)
    inside = (
        (cx > corners[:, 0, None, None])
        & (cx < corners[:, 2, None, None])
        & (cy > corners[:, 1, None, None])
        & (cy < corners[:, 3, None, None])
    )
    for index in torch.nonzero(inside.flatten(1).sum(1) == 0).flatten().tolist():
        x = (corners[index, 0] + corners[index, 2]) / 2
        y = (corners[index, 1] + corners[index, 3]) / 2
        row = min(max(int(y * height), 0), height - 1)
        column = min(max(int(x * width), 0), width - 1)
        inside[index, row, column] = True
    area = (corners[:, 2] - corners[:, 0]) * (corners[:, 3] - corners[:, 1])
    cost = torch.where(inside, area[:, None, None], torch.tensor(math.inf))
    best = cost.argmin(0)
    positive = torch.isfinite(cost.min(0).values)
    assigned = corners[best].permute(2, 0, 1) * positive
    classes[labels[best][positive], positive] = 1.0
    return classes, assigned, positive


def _decode(regression: torch.Tensor) -> torch.Tensor:
    """Boxes predicted at every cell from log distances to the sides in units of cells.

    Log distances are clamped at eight before the exponential, which bounds the boxes and keeps
    the overlap loss finite.

    Args:
        regression: Log distances to the left, top, right and bottom sides of shape
            `(B, 4, h, w)`.

    Returns:
        Corners x0, y0, x1, y1 of shape `(B, 4, h, w)`, normalized by the image size and not
        clipped to the image.
    """
    _, _, height, width = regression.shape
    cy, cx = _centres(height, width, regression.device)
    distance = regression.clamp(max=_MAX_LOG_DISTANCE).exp()
    return torch.stack(
        [
            cx - distance[:, 0] / width,
            cy - distance[:, 1] / height,
            cx + distance[:, 2] / width,
            cy + distance[:, 3] / height,
        ],
        dim=1,
    )


class LinearDetection(Protocol):
    """Batch normalization and a pointwise convolution predicting boxes on frozen dense features.

    Classification uses the focal loss of Lin et al. (2017), regression the generalized overlap loss
    of Rezatofighi et al. (2019) and centerness a binary cross entropy, under AdamW with cosine
    decay over `steps`. The class biases start at the focal loss prior of 0.01. A detection scores
    the geometric mean of its class probability and its centerness, and detections scoring above
    `min_score` are kept after class wise non maximum suppression at `nms_iou`. The validation
    score is computed every `eval_every` steps and after the last step, and the checkpoint with the
    best score is kept. Training stops early once `patience` successive evaluations bring no
    improvement, but never before `min_steps`. A `kernel` above one lets every cell read its
    neighbours, which keeps the head linear in the features.

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
        min_score: Score a detection must exceed to be kept. Defaults to 0.05.
        nms_iou: Overlap above which non maximum suppression discards the lower scoring box of a
            class. Defaults to 0.6.
        max_detections: Budget of detections per object class. The highest scoring detections of
            an image are kept up to `max_detections` times the number of object classes, whatever
            their classes. Defaults to 100.
        kernel: Side in cells of the convolution kernel, padded by half its side, which must be
            odd so that the output keeps the size of the feature map. Defaults to 1.
    """

    name: ClassVar[str] = "linear_det"
    kinds: ClassVar[frozenset[TaskKind]] = frozenset({TaskKind.DETECTION})
    dense: ClassVar[bool] = True
    stochastic: ClassVar[bool] = True

    steps: PositiveInt = 2000
    batch_size: PositiveInt = 16
    lr: PositiveFloat = 1e-3
    weight_decay: NonNegativeFloat = 1e-4
    eval_every: PositiveInt = 200
    patience: PositiveInt = 5
    min_steps: NonNegativeInt = 600
    min_score: NonNegativeFloat = 0.05
    nms_iou: PositiveFloat = 0.6
    max_detections: PositiveInt = 100
    kernel: PositiveInt = 1

    @field_validator("kernel")
    @classmethod
    def _odd_kernel(cls, kernel: int) -> int:
        """Accept only odd kernel sides, which keep the size of the feature map.

        Args:
            kernel: Side in cells of the convolution kernel.

        Returns:
            The unchanged side.

        Raises:
            ValueError: If the side is even.
        """
        if kernel % 2 == 0:
            raise ValueError(f"kernel must be odd, got {kernel}")
        return kernel

    def _targets(self, features: Features, n_objects: int) -> tuple[torch.Tensor, ...]:
        """Stacked cell targets of every image, as `assign` gives them for one image.

        Args:
            features: Dense features of shape `(N, D, h, w)` whose targets are box tables.
            n_objects: Number of object classes `K`, excluding the background.

        Returns:
            Class indicators of shape `(N, K, h, w)`, assigned corners of shape `(N, 4, h, w)` and
            positive cells of shape `(N, h, w)`, on the CPU.
        """
        _, _, height, width = features.values.shape
        parts = [
            assign(np.asarray(t, dtype=np.float64).reshape(-1, 5), n_objects, height, width)
            for t in features.targets
        ]
        return tuple(torch.stack(part) for part in zip(*parts, strict=True))

    def _detections(
        self, head: nn.Module, values: torch.Tensor, device: torch.device
    ) -> list[NDArray[np.float64]]:
        """Scored boxes of every image after thresholding and class wise suppression.

        Args:
            head: Detection head, switched to evaluation mode.
            values: Dense features of shape `(N, D, h, w)` on any device.
            device: Device on which the head runs.

        Returns:
            One table of shape `(M, 6)` in float64 per image, with rows class, score, x0, y0, x1,
            y1 in decreasing order of score, classes counted from one and corners clipped to the
            image and normalized by its size.
        """
        head.eval()
        results: list[NDArray[np.float64]] = []
        for start in range(0, len(values), _EVAL_BATCH):
            with torch.inference_mode():
                output = head(values[start : start + _EVAL_BATCH].to(device).float())
            n_objects = output.shape[1] - 5
            boxes = _decode(output[:, n_objects : n_objects + 4]).clamp(0, 1)
            scores = (output[:, :n_objects].sigmoid() * output[:, -1:].sigmoid()).sqrt()
            for image_scores, image_boxes in zip(scores, boxes, strict=True):
                flat = image_scores.flatten(1)
                classes, cells = torch.nonzero(flat > self.min_score, as_tuple=True)
                kept_scores = flat[classes, cells]
                kept_boxes = image_boxes.flatten(1)[:, cells].T
                order = batched_nms(kept_boxes, kept_scores, classes, self.nms_iou)
                order = order[: self.max_detections * n_objects]
                table = torch.cat(
                    [
                        (classes[order] + 1)[:, None].float(),
                        kept_scores[order, None],
                        kept_boxes[order],
                    ],
                    dim=1,
                )
                results.append(table.cpu().numpy().astype(np.float64))
        return results

    def predict(
        self,
        train: Features,
        val: Features,
        test: Features,
        spec: TaskSpec,
        *,
        seed: int,
        device: torch.device,
    ) -> list[NDArray[np.float64]]:
        """Train the head on `train`, select its checkpoint on `val` and detect on `test`.

        Cells without an object regress towards the whole image, a placeholder that keeps their
        discarded overlap and centerness terms finite.

        Args:
            train: Dense training features of shape `(N, D, h, w)` with their box tables.
            val: Dense validation features with their box tables, used to select the checkpoint.
            test: Dense test features of shape `(M, D, h, w)`.
            spec: Task specification whose classes include the background first.
            seed: Seed of the head initialization and of the batch sampling.
            device: Device on which the head is trained.

        Returns:
            One table of scored boxes of shape `(R, 6)` per test image, with rows class, score,
            x0, y0, x1, y1 and corners normalized by the image size.
        """
        n_classes = len(spec.classes)
        n_objects = n_classes - 1
        dim = train.values.shape[1]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            layer = nn.Conv2d(dim, n_objects + 5, self.kernel, padding=self.kernel // 2)
            head = nn.Sequential(nn.BatchNorm2d(dim), layer)
            prior = -math.log((1 - 0.01) / 0.01)
            assert layer.bias is not None
            with torch.no_grad():
                layer.bias[:n_objects] = prior
        head.to(device)
        generator = torch.Generator().manual_seed(seed)
        classes, assigned, positive = (part.to(device) for part in self._targets(train, n_objects))
        whole = torch.tensor([0.0, 0.0, 1.0, 1.0], device=device)[:, None, None]
        boxes = torch.where(positive[:, None], assigned, whole)
        values = resident(train.values, device)
        val_values = resident(val.values, device)
        batches = torch.randint(len(values), (self.steps, self.batch_size), generator=generator)
        value_batches, target_batches = batches.to(values.device), batches.to(device)
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
            index = target_batches[step - 1]
            output = head(values[value_batches[step - 1]].to(device).float())
            loss = _loss(output, classes[index], boxes[index], positive[index])
            optimizer.zero_grad(set_to_none=True)
            torch.autograd.backward(loss)
            optimizer.step()
            schedule.step()
            if step % self.eval_every == 0 or step == self.steps:
                detections = self._detections(head, val_values, device)
                current = score(spec.kind, detections, val.targets, n_classes, [primary])
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
        return self._detections(head, test.values, device)


def _loss(
    output: torch.Tensor, classes: torch.Tensor, boxes: torch.Tensor, positive: torch.Tensor
) -> torch.Tensor:
    """Focal, overlap and centerness losses of a batch, each summed and divided by its positives.

    The focal loss is summed over every cell and class. The overlap and centerness terms are
    computed at every cell and weighted by `positive`, which equals their sum over positive cells
    without indexing by a mask that the host must read. The number of positives is clamped at one.
    Cells without an object hold the whole image in `boxes`, which keeps their discarded terms
    finite.

    Args:
        output: Head output of shape `(B, K + 5, h, w)`, holding the class logits, the log
            distances to the four sides and the centerness logit.
        classes: Class indicators of shape `(B, K, h, w)`.
        boxes: Assigned corners of shape `(B, 4, h, w)`, normalized by the image size.
        positive: Positive cells of shape `(B, h, w)`.

    Returns:
        Scalar loss of the batch.
    """
    n_objects = classes.shape[1]
    weight = positive.float()
    n_positive = weight.sum().clamp(min=1.0)
    loss: torch.Tensor = sigmoid_focal_loss(output[:, :n_objects], classes, reduction="sum")
    loss = loss / n_positive
    overlap = _overlap_loss(_decode(output[:, n_objects : n_objects + 4]), boxes)
    centre = F.binary_cross_entropy_with_logits(
        output[:, -1], _centre_targets(boxes), reduction="none"
    )
    return loss + ((overlap + centre) * weight).sum() / n_positive


def _overlap_loss(predicted: torch.Tensor, truth: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Generalized overlap loss of Rezatofighi et al. (2019) between box maps `(B, 4, h, w)`.

    It equals `generalized_box_iou_loss` of torchvision, with the intersection clamped at zero
    instead of assigned through a mask, so that no step waits for the host.

    Args:
        predicted: Predicted corners x0, y0, x1, y1 of shape `(B, 4, h, w)`.
        truth: Target corners of the same shape.
        eps: Constant added to the union and the enclosing area against division by zero.

    Returns:
        One minus the generalized overlap at every cell, of shape `(B, h, w)` and in `[0, 2]`.
    """
    x0, y0, x1, y1 = predicted.unbind(1)
    u0, v0, u1, v1 = truth.unbind(1)
    width = (torch.min(x1, u1) - torch.max(x0, u0)).clamp(min=0)
    height = (torch.min(y1, v1) - torch.max(y0, v0)).clamp(min=0)
    intersection = width * height
    union = (x1 - x0) * (y1 - y0) + (u1 - u0) * (v1 - v0) - intersection
    enclosing = (torch.max(x1, u1) - torch.min(x0, u0)) * (torch.max(y1, v1) - torch.min(y0, v0))
    overlap = intersection / (union + eps)
    return 1 - (overlap - (enclosing - union) / (enclosing + eps))


def _centre_targets(boxes: torch.Tensor) -> torch.Tensor:
    """FCOS centerness of every cell with respect to its assigned box.

    The centerness is the square root of the product of the ratios of the smaller to the larger
    distance to the opposite sides, horizontally and vertically. Distances are clamped at 1e-6,
    which gives a centerness near zero for a cell outside its box.

    Args:
        boxes: Assigned corners x0, y0, x1, y1 of shape `(B, 4, h, w)`, normalized by the image
            size.

    Returns:
        Centerness targets of shape `(B, h, w)` in `(0, 1]`.
    """
    _, _, height, width = boxes.shape
    y, x = _centres(height, width, boxes.device)
    left, right = (x - boxes[:, 0]).clamp(min=1e-6), (boxes[:, 2] - x).clamp(min=1e-6)
    top, bottom = (y - boxes[:, 1]).clamp(min=1e-6), (boxes[:, 3] - y).clamp(min=1e-6)
    ratio = (torch.minimum(left, right) / torch.maximum(left, right)) * (
        torch.minimum(top, bottom) / torch.maximum(top, bottom)
    )
    return ratio.sqrt()
