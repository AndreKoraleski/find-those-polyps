"""Per image segmentation metrics, averaged over the test set.

Region and boundary overlap are measured on the predicted masks, the most probable class at every
pixel for single label tasks and the probabilities above one half for multilabel tasks. For binary
tasks the salient object measures popularized for polyps by PraNet (Fan et al. 2020) are computed
on the foreground probability after per image min max normalization, following the evaluation
toolbox of that work. Masks are compared at native resolution, and boundary distances are reported
both in pixels and relative to the image diagonal.
"""

from collections.abc import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

EPS = np.finfo(np.float64).eps
"""Machine epsilon of float64, which guards the divisions of the salient object measures."""

REGION = ("dice", "iou", "precision", "recall", "specificity", "f2")
"""Pixel counting metrics of a predicted mask against its target."""
BOUNDARY = ("nsd", "nsd_strict", "nhd95", "nassd")
"""Boundary metrics relative to the image diagonal, comparable across image sizes."""
PIXEL = ("hd95", "assd")
"""Boundary distances in pixels, computed only on request for per image analysis."""
OVERLAP = REGION + BOUNDARY
"""Region and boundary metrics of every segmentation task."""
NSD_TOLERANCE = 0.02
"""Tolerance of the normalized surface Dice as a fraction of the image diagonal."""
NSD_STRICT_TOLERANCE = 0.01
"""Tolerance of the strict normalized surface Dice as a fraction of the image diagonal."""
SALIENCY = (
    "mae",
    "s_measure",
    "mean_e_measure",
    "max_e_measure",
    "weighted_f_measure",
)
"""Salient object measures of binary segmentation tasks."""


def _safe(numerator: float, denominator: float, empty: float) -> float:
    """Quotient of the two counts, or `empty` when the denominator is zero."""
    return numerator / denominator if denominator else empty


def _crop(
    prediction: NDArray[np.bool_], target: NDArray[np.bool_]
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Both masks cropped to the bounding box of their union with a margin of one pixel.

    Every boundary pixel lies inside the box, so distances between boundaries are unchanged.

    Args:
        prediction: Predicted mask of shape `(H, W)`.
        target: Target mask of shape `(H, W)`, whose union with `prediction` is not empty.

    Returns:
        The two masks restricted to the same window.
    """
    union = prediction | target
    rows = np.flatnonzero(union.any(axis=1))
    columns = np.flatnonzero(union.any(axis=0))
    window = (
        slice(max(rows[0] - 1, 0), rows[-1] + 2),
        slice(max(columns[0] - 1, 0), columns[-1] + 2),
    )
    return prediction[window], target[window]


def _surface_distances(
    prediction: NDArray[np.bool_], target: NDArray[np.bool_]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Distances from each boundary pixel of one mask to the boundary of the other.

    The boundary of a mask is the set of its pixels removed by one binary erosion, so that pixels
    on the edge of the image belong to it.

    Args:
        prediction: Predicted mask of shape `(H, W)`, not empty.
        target: Target mask of shape `(H, W)`, not empty.

    Returns:
        Euclidean distances in pixels from the boundary of `prediction` to that of `target`, and
        from the boundary of `target` to that of `prediction`.
    """
    border_p = prediction ^ ndimage.binary_erosion(prediction)
    border_t = target ^ ndimage.binary_erosion(target)
    to_t = ndimage.distance_transform_edt(~border_t)[border_p]
    to_p = ndimage.distance_transform_edt(~border_p)[border_t]
    return np.asarray(to_t, dtype=np.float64), np.asarray(to_p, dtype=np.float64)


def boundary(prediction: NDArray[np.bool_], target: NDArray[np.bool_]) -> dict[str, float]:
    """Boundary agreement of two masks, relative to the image diagonal and in pixels.

    `nsd` is the normalized surface Dice of Nikolov et al. (2021) at a tolerance of two percent of
    the image diagonal, the fraction of boundary pixels of both masks within the tolerance of the
    other boundary, which the Boundary IoU of Cheng et al. (2021) closely resembles. `nsd_strict`
    uses a tolerance of one percent. `nhd95` is the larger of the two directed 95th percentile
    Hausdorff distances and `nassd` the average symmetric surface distance, both divided by the
    diagonal so that images of any size compare, while `hd95` and `assd` keep them in pixels for
    per image analysis. Two empty masks agree perfectly, and when exactly one is empty the
    distances are the whole diagonal and the surface Dice is zero.

    Args:
        prediction: Predicted mask of shape `(H, W)`.
        target: Target mask of shape `(H, W)`.

    Returns:
        `nsd` and `nsd_strict` in `[0, 1]`, `nhd95` and `nassd` in `[0, 1]` relative to the
        diagonal, and `hd95` and `assd` in pixels.
    """
    diagonal = float(np.hypot(*target.shape))
    if not prediction.any() and not target.any():
        return {"nsd": 1.0, "nsd_strict": 1.0, "nhd95": 0.0, "nassd": 0.0, "hd95": 0.0, "assd": 0.0}
    if not prediction.any() or not target.any():
        return {
            "nsd": 0.0,
            "nsd_strict": 0.0,
            "nhd95": 1.0,
            "nassd": 1.0,
            "hd95": diagonal,
            "assd": diagonal,
        }
    to_t, to_p = _surface_distances(*_crop(prediction, target))
    count = len(to_t) + len(to_p)

    def surface_dice(tolerance: float) -> float:
        limit = tolerance * diagonal
        return float((np.sum(to_t <= limit) + np.sum(to_p <= limit)) / count)

    hausdorff = float(max(np.percentile(to_t, 95), np.percentile(to_p, 95)))
    average = float((to_t.sum() + to_p.sum()) / count)
    return {
        "nsd": surface_dice(NSD_TOLERANCE),
        "nsd_strict": surface_dice(NSD_STRICT_TOLERANCE),
        "nhd95": hausdorff / diagonal,
        "nassd": average / diagonal,
        "hd95": hausdorff,
        "assd": average,
    }


def hd95(prediction: NDArray[np.bool_], target: NDArray[np.bool_]) -> float:
    """95th percentile symmetric Hausdorff distance between boundaries, in pixels.

    Args:
        prediction: Predicted mask of shape `(H, W)`.
        target: Target mask of shape `(H, W)`.

    Returns:
        Distance in pixels, zero for two empty masks and the image diagonal when exactly one is
        empty.
    """
    return boundary(prediction, target)["hd95"]


def overlap(
    prediction: NDArray[np.bool_], target: NDArray[np.bool_], *, boundaries: bool = True
) -> dict[str, float]:
    """Region agreement of two binary masks, and boundary agreement unless `boundaries` is off.

    Dice, IoU, recall and specificity equal one when their denominator is zero. Precision and F2
    equal one when both masks are empty and zero when they are otherwise undefined, so that an
    empty prediction of a nonempty target is not rewarded. F2 weighs recall four times as much as
    precision.

    Args:
        prediction: Predicted mask of shape `(H, W)`.
        target: Target mask of shape `(H, W)`.
        boundaries: Whether to add the metrics of `boundary`.

    Returns:
        Region metrics in `[0, 1]`, with the boundary metrics when requested.
    """
    tp = float(np.sum(prediction & target))
    fp = float(np.sum(prediction & ~target))
    fn = float(np.sum(~prediction & target))
    tn = float(prediction.size) - tp - fp - fn
    empty = 1.0 if tp + fp + fn == 0 else 0.0
    precision = _safe(tp, tp + fp, empty)
    recall = _safe(tp, tp + fn, 1.0)
    return {
        "dice": _safe(2 * tp, 2 * tp + fp + fn, 1.0),
        "iou": _safe(tp, tp + fp + fn, 1.0),
        "precision": precision,
        "recall": recall,
        "specificity": _safe(tn, tn + fp, 1.0),
        "f2": _safe(5 * precision * recall, 4 * precision + recall, empty),
    } | (boundary(prediction, target) if boundaries else {})


def _object_score(values: NDArray[np.float64]) -> float:
    """Object score of Fan et al. (2017) of the probabilities in one region, zero below two pixels.

    Args:
        values: Probabilities of the region, flattened, of the foreground inside the target and
            of the background outside it.

    Returns:
        Twice the mean over its square plus one plus the sample standard deviation.
    """
    if values.size < 2:
        return 0.0
    mean = values.mean()
    return float(2 * mean / (mean**2 + 1 + values.std(ddof=1) + EPS))


def _ssim(prediction: NDArray[np.float64], target: NDArray[np.float64]) -> float:
    """Structural similarity of one block, as in the toolbox of Fan et al. (2017).

    Args:
        prediction: Foreground probabilities of the block.
        target: Target of the block as zeros and ones of the same shape.

    Returns:
        Similarity without stabilizing constants. It is zero below two pixels, and when its
        numerator vanishes it is one if the denominator vanishes too and zero otherwise.
    """
    n = prediction.size
    if n < 2:
        return 0.0
    x, y = prediction.mean(), target.mean()
    sigma_x = np.sum((prediction - x) ** 2) / (n - 1)
    sigma_y = np.sum((target - y) ** 2) / (n - 1)
    sigma_xy = np.sum((prediction - x) * (target - y)) / (n - 1)
    alpha = 4 * x * y * sigma_xy
    beta = (x**2 + y**2) * (sigma_x + sigma_y)
    if alpha != 0:
        return float(alpha / (beta + EPS))
    return 1.0 if beta == 0 else 0.0


def s_measure(prediction: NDArray[np.float64], target: NDArray[np.bool_]) -> float:
    """Structure measure with equal object and region weights (Fan et al. 2017).

    The object term weighs the foreground and background object scores by the fraction of each in
    the target. The region term splits the image into four blocks just after the rounded centroid
    of the target, as the one based indexing of the toolbox does, and weighs their structural
    similarity by their area. An empty target scores one minus the mean
    prediction and a full target the mean prediction.

    Args:
        prediction: Foreground probabilities of shape `(H, W)` in `[0, 1]`.
        target: Target mask of shape `(H, W)`.

    Returns:
        Structure measure in `[0, 1]`.
    """
    y = target.mean()
    if y == 0:
        return float(1 - prediction.mean())
    if y == 1:
        return float(prediction.mean())
    gt = target.astype(np.float64)
    foreground = _object_score(prediction[target])
    background = _object_score(1 - prediction[~target])
    object_score = y * foreground + (1 - y) * background

    height, width = target.shape
    rows, cols = np.nonzero(target)
    cy, cx = int(np.round(rows.mean())) + 1, int(np.round(cols.mean())) + 1
    region_score = 0.0
    for ys, xs in (
        (slice(0, cy), slice(0, cx)),
        (slice(0, cy), slice(cx, width)),
        (slice(cy, height), slice(0, cx)),
        (slice(cy, height), slice(cx, width)),
    ):
        block = prediction[ys, xs]
        weight = block.size / prediction.size
        if weight:
            region_score += weight * _ssim(block, gt[ys, xs])
    return float(max(0.0, 0.5 * object_score + 0.5 * region_score))


def e_measure(prediction: NDArray[np.float64], target: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Enhanced alignment measure at 256 thresholds (Fan et al. 2018).

    Probabilities are quantized to 256 levels, and the foreground at a threshold holds the levels at
    or above it. Counts of the four agreement cases are accumulated from histograms, so all
    thresholds cost a single pass over the image. An empty target scores the fraction of predicted
    background and a full target the fraction of predicted foreground. As in the toolbox the sum is
    divided by the number of pixels minus one.

    Args:
        prediction: Foreground probabilities of shape `(H, W)` in `[0, 1]`.
        target: Target mask of shape `(H, W)`.

    Returns:
        Measure of shape `(256,)` from the highest threshold to the lowest, nonnegative and at
        most `n / (n - 1)` for `n` pixels.
    """
    n = prediction.size
    quantized = np.round(prediction * 255).astype(np.int64)
    bins = np.arange(257)
    hist_fg, _ = np.histogram(quantized[target], bins=bins)
    hist_bg, _ = np.histogram(quantized[~target], bins=bins)
    fg_fg = np.cumsum(hist_fg[::-1]).astype(np.float64)
    fg_bg = np.cumsum(hist_bg[::-1]).astype(np.float64)
    predicted = fg_fg + fg_bg
    gt_fg = float(target.sum())
    if gt_fg == 0:
        total = n - predicted
    elif gt_fg == n:
        total = predicted
    else:
        bg_fg = gt_fg - fg_fg
        bg_bg = (n - gt_fg) - fg_bg
        mean_pred = predicted / n
        mean_gt = gt_fg / n
        total = np.zeros(256)
        for count, p, g in (
            (fg_fg, 1 - mean_pred, 1 - mean_gt),
            (fg_bg, 1 - mean_pred, -mean_gt),
            (bg_fg, -mean_pred, 1 - mean_gt),
            (bg_bg, -mean_pred, -mean_gt),
        ):
            align = 2 * p * g / (p**2 + g**2 + EPS)
            total += count * (align + 1) ** 2 / 4
    return np.asarray(total / (n - 1 + EPS), dtype=np.float64)


def _gaussian_kernel(size: int = 7, sigma: float = 5.0) -> NDArray[np.float64]:
    """Normalized Gaussian kernel equal to MATLAB `fspecial('gaussian')`.

    Args:
        size: Side of the square kernel in pixels.
        sigma: Standard deviation in pixels.

    Returns:
        Kernel of shape `(size, size)` that sums to one, with negligible entries set to zero.
    """
    half = (size - 1) / 2
    y, x = np.ogrid[-half : half + 1, -half : half + 1]
    kernel = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    kernel[kernel < np.finfo(kernel.dtype).eps * kernel.max()] = 0
    return np.asarray(kernel / kernel.sum(), dtype=np.float64)


def weighted_f_measure(prediction: NDArray[np.float64], target: NDArray[np.bool_]) -> float:
    """Weighted F measure with beta squared equal to one (Margolin et al. 2014).

    Inside the target, an error is replaced by its Gaussian smoothing when that is smaller, where
    the smoothing sees at every pixel outside the target the error of the nearest target pixel.
    Errors outside the target grow in importance with their distance to it. This follows the
    toolbox of the authors.

    Args:
        prediction: Foreground probabilities of shape `(H, W)` in `[0, 1]`.
        target: Target mask of shape `(H, W)`.

    Returns:
        Weighted F measure in `[0, 1]`, zero for an empty target.
    """
    if not target.any():
        return 0.0
    distance, indices = ndimage.distance_transform_edt(~target, return_indices=True)
    error = np.abs(prediction - target)
    spread = error.copy()
    outside = ~target
    spread[outside] = error[indices[0][outside], indices[1][outside]]
    smoothed = ndimage.convolve(spread, _gaussian_kernel(), mode="constant", cval=0)
    minimum = np.where(target & (smoothed < error), smoothed, error)
    importance = np.where(outside, 2 - np.exp(np.log(0.5) / 5 * distance), 1.0)
    weighted = minimum * importance
    tp = target.sum() - weighted[target].sum()
    fp = weighted[outside].sum()
    recall = 1 - weighted[target].mean()
    precision = tp / (tp + fp + EPS)
    return float(2 * recall * precision / (recall + precision + EPS))


def saliency(probability: NDArray[np.float64], target: NDArray[np.bool_]) -> dict[str, float]:
    """Salient object measures of a foreground probability map.

    The map is min max normalized first unless it is constant.

    Args:
        probability: Foreground probabilities of shape `(H, W)`.
        target: Target mask of shape `(H, W)`.

    Returns:
        Mean absolute error, structure measure, mean and maximal enhanced alignment and weighted
        F measure. All lie in `[0, 1]`, except that the enhanced alignment may exceed one by the
        factor `n / (n - 1)` of the toolbox for `n` pixels.
    """
    low, high = probability.min(), probability.max()
    normalized = (probability - low) / (high - low) if high > low else probability
    curve = e_measure(normalized, target)
    return {
        "mae": float(np.abs(normalized - target).mean()),
        "s_measure": s_measure(normalized, target),
        "mean_e_measure": float(curve.mean()),
        "max_e_measure": float(curve.max()),
        "weighted_f_measure": weighted_f_measure(normalized, target),
    }


def per_image(
    probabilities: Iterable[NDArray[np.float32]],
    targets: Sequence[NDArray[np.int64]],
    n_classes: int,
    names: Sequence[str] | None = None,
    *,
    multilabel: bool = False,
) -> list[dict[str, float]]:
    """Requested metrics of every image in a single pass over the probability maps.

    Binary tasks predict the foreground where its probability exceeds that of the background.
    Multiclass overlap metrics are averaged over the foreground classes present in the target or
    the prediction, so that absent classes do not count as perfect agreement, and over all
    foreground classes when none is present. Saliency measures apply to binary tasks. With
    `multilabel` every class, the first included, is a separate binary mask in the target,
    predicted where its probability exceeds one half, and classes are averaged the same way.

    Args:
        probabilities: Probability maps of shape `(C, H, W)`, one per image.
        targets: Class masks of shape `(H, W)`, or stacks of class masks of shape `(C, H, W)`
            with `multilabel`.
        n_classes: Number of classes `C`, which for single label tasks includes the background.
        names: Metrics to compute, by default every overlap, saliency and pixel metric.
        multilabel: Whether classes may overlap.

    Returns:
        One record per image with the requested metrics that apply to the task.
    """
    wanted = set(names) if names is not None else set(OVERLAP) | set(SALIENCY) | set(PIXEL)
    records = []
    for probability, target in zip(probabilities, targets, strict=True):
        if multilabel:
            masks = [(probability[c] > 0.5, target[c] > 0) for c in range(n_classes)]
        elif n_classes == 2:
            masks = [(probability[1] > probability[0], target == 1)]
        else:
            labels = probability.argmax(axis=0)
            masks = [(labels == c, target == c) for c in range(1, n_classes)]
        values: dict[str, float] = {}
        if wanted & set(OVERLAP):
            present = [pair for pair in masks if pair[0].any() or pair[1].any()]
            chosen = present if len(masks) > 1 and present else masks
            distances = bool(wanted & (set(BOUNDARY) | set(PIXEL)))
            scores = [overlap(p, g, boundaries=distances) for p, g in chosen]
            values |= {k: float(np.mean([s[k] for s in scores])) for k in scores[0]}
        if n_classes == 2 and not multilabel and wanted & set(SALIENCY):
            values |= saliency(probability[1].astype(np.float64), target == 1)
        records.append({k: v for k, v in values.items() if k in wanted})
    return records


def segmentation(
    probabilities: Iterable[NDArray[np.float32]],
    targets: Sequence[NDArray[np.int64]],
    n_classes: int,
    names: Sequence[str] | None = None,
    *,
    multilabel: bool = False,
) -> dict[str, float]:
    """Mean over images of the requested metrics, by default those of the leaderboard.

    Args:
        probabilities: Probability maps of shape `(C, H, W)`, one per image.
        targets: Class masks of shape `(H, W)`, or stacks of class masks of shape `(C, H, W)`
            with `multilabel`.
        n_classes: Number of classes `C`, which for single label tasks includes the background.
        names: Metrics to average, by default the overlap metrics and, for binary single label
            tasks, the salient object measures.
        multilabel: Whether classes may overlap.

    Returns:
        Mean over images of every requested metric that applies to the task, NaN for each of them
        when there are no images.
    """
    if names is None:
        names = OVERLAP + (SALIENCY if n_classes == 2 and not multilabel else ())
    records = per_image(probabilities, targets, n_classes, names, multilabel=multilabel)
    if not records:
        return dict.fromkeys(_applicable(names, n_classes, multilabel), float("nan"))
    return {key: float(np.mean([r[key] for r in records])) for key in records[0]}


def _applicable(names: Sequence[str], n_classes: int, multilabel: bool) -> list[str]:
    """Requested metrics that `per_image` reports for the task, in the order it reports them.

    Args:
        names: Requested metrics.
        n_classes: Number of classes `C`, which for single label tasks includes the background.
        multilabel: Whether classes may overlap.

    Returns:
        The overlap metrics, with the boundary and pixel distances when one of them is requested,
        then the salient object measures of binary single label tasks, restricted to `names`.
    """
    wanted = set(names)
    reported: list[str] = []
    if wanted & set(OVERLAP):
        reported += REGION
        if wanted & (set(BOUNDARY) | set(PIXEL)):
            reported += BOUNDARY + PIXEL
    if n_classes == 2 and not multilabel:
        reported += SALIENCY
    return [name for name in reported if name in wanted]
