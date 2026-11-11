"""Linear probing of frozen global features with logistic regression."""

from collections.abc import Callable
from typing import ClassVar, cast

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from numpy.typing import NDArray
from pydantic import PositiveFloat, PositiveInt

from find_those_polyps.evaluation.features import Features
from find_those_polyps.evaluation.protocols.base import Protocol, better
from find_those_polyps.metrics import score
from find_those_polyps.tasks.schema import TaskKind, TaskSpec

Predictor = Callable[[torch.Tensor], NDArray[np.float64]]
"""Function mapping standardized features of shape `(M, D)` to probabilities of shape `(M, C)`."""


def standardize(train: torch.Tensor, *others: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Standardize features with the mean and deviation of the training features.

    Constant features keep a unit scale, as in scikit-learn's `StandardScaler`.

    Args:
        train: Training features of shape `(N, D)`.
        *others: Further features of shape `(M, D)` transformed the same way.

    Returns:
        The standardized training features followed by the other ones.
    """
    mean = train.mean(dim=0)
    deviation = train.std(dim=0, correction=0)
    deviation = torch.where(deviation > 0, deviation, torch.ones_like(deviation))
    return tuple((x - mean) / deviation for x in (train, *others))


def fit_logistic(
    x: torch.Tensor,
    y: torch.Tensor,
    c: float,
    *,
    multilabel: bool,
    max_iter: int,
    tolerance: float,
    start: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weights of an L2 regularized logistic regression fitted by L-BFGS.

    The objective is the mean log loss plus the squared norm of the weights, bias excluded, divided
    by `2 C N`, which has the minimizer of scikit-learn's `LogisticRegression` with `C=c`. The loss
    is multinomial for class indices and a sum of independent binary losses for indicators.

    Args:
        x: Standardized features of shape `(N, D)`.
        y: Class indices of shape `(N,)`, or indicators of shape `(N, K)` when `multilabel`.
        c: Inverse regularization strength.
        multilabel: Whether `y` holds indicators of independent labels.
        max_iter: Iteration limit of L-BFGS.
        tolerance: Largest absolute gradient entry at which L-BFGS stops.
        start: Weights of shape `(D + 1, K)` to start from, zero when `None`.

    Returns:
        Weights of shape `(D + 1, K)`, the last row holding the biases.
    """
    n, d = x.shape
    k = y.shape[1] if multilabel else int(y.max()) + 1
    initial = start if start is not None else x.new_zeros(d + 1, k)
    weights = initial.clone().requires_grad_(True)
    penalty = 1.0 / (2.0 * c * n)
    optimizer = torch.optim.LBFGS(
        [weights],
        lr=1.0,
        max_iter=max_iter,
        tolerance_grad=tolerance,
        tolerance_change=0.0,
        history_size=10,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        """Regularized loss of the current weights, with its gradient."""
        optimizer.zero_grad()
        logits = x @ weights[:-1] + weights[-1]
        if multilabel:
            loss = F.binary_cross_entropy_with_logits(logits, y, reduction="sum") / n
        else:
            loss = F.cross_entropy(logits, y)
        loss = loss + penalty * weights[:-1].pow(2).sum()
        torch.autograd.backward(loss)
        return loss

    step = cast(Callable[[Callable[[], torch.Tensor]], object], optimizer.step)
    step(closure)
    return weights.detach()


class LinearProbe(Protocol):
    """L2 regularized logistic regression on standardized global features.

    The inverse regularization strength is chosen from a logarithmic grid by the primary metric on
    the validation partition, after Radford et al. (2021). Features are standardized with the mean
    and deviation of the training partition. The objective is that of scikit-learn's
    `LogisticRegression`, multinomial for single label tasks and one binary model per label for
    multilabel tasks, and it is minimized by full batch L-BFGS in double precision on the device.
    Each strength starts from the solution of the previous one, which shortens the fits of weakly
    regularized models without moving the optimum of a convex objective. The protocol is
    deterministic.

    Attributes:
        c_grid: Inverse regularization strengths tried in order, by default the powers of ten from
            1e-4 to 1e4. Ties on the validation partition keep the earliest value.
        max_iter: Iteration limit of L-BFGS for each strength. Defaults to 1000.
        tolerance: Largest absolute gradient entry at which L-BFGS stops. Defaults to 1e-6.
    """

    name: ClassVar[str] = "linear"
    kinds: ClassVar[frozenset[TaskKind]] = frozenset(
        {TaskKind.CLASSIFICATION, TaskKind.ORDINAL, TaskKind.MULTILABEL}
    )

    c_grid: tuple[PositiveFloat, ...] = tuple(10.0**e for e in range(-4, 5))
    max_iter: PositiveInt = 1000
    tolerance: PositiveFloat = 1e-6

    def predict(
        self,
        train: Features,
        val: Features,
        test: Features,
        spec: TaskSpec,
        *,
        seed: int,
        device: torch.device,
    ) -> NDArray[np.float64]:
        """Fit one model per value of `c_grid` and predict `test` with the best on `val`.

        Args:
            train: Global training features of shape `(N, D)` with their targets.
            val: Global validation features with their targets, used to choose the strength.
            test: Global test features of shape `(M, D)`.
            spec: Task specification with the classes, the kind and the primary metric.
            seed: Seed, unused since the protocol is deterministic.
            device: Device on which the models are fitted.

        Returns:
            Class or label probabilities of shape `(M, C)`.
        """
        n_classes = len(spec.classes)
        x_train, x_val, x_test = standardize(
            *(f.values.to(device=device, dtype=torch.float64) for f in (train, val, test))
        )
        multilabel = spec.kind is TaskKind.MULTILABEL
        if multilabel:
            fit = self._multilabel(x_train, train.indicators(n_classes))
        else:
            fit = self._single_label(x_train, train.labels(), n_classes)
        best: Predictor | None = None
        best_score = float("nan")
        start: torch.Tensor | None = None
        for c in self.c_grid:
            predictor, start = fit(c, start)
            metrics = score(
                spec.kind, predictor(x_val), val.targets, n_classes, names=[spec.primary_metric]
            )
            candidate = metrics[spec.primary_metric]
            if best is None or better(candidate, best_score, spec.primary_metric):
                best, best_score = predictor, candidate
        assert best is not None
        return best(x_test)

    def _single_label(
        self, x: torch.Tensor, labels: NDArray[np.int64], n_classes: int
    ) -> Callable[[float, torch.Tensor | None], tuple[Predictor, torch.Tensor | None]]:
        """Fitting function of a multinomial model over the classes present in `labels`.

        A single present class gives a constant one hot prediction. Absent classes receive zero
        probability.

        Args:
            x: Standardized training features of shape `(N, D)`.
            labels: Class indices of shape `(N,)`.
            n_classes: Number of classes, including those absent from `labels`.

        Returns:
            A function of the strength and the starting weights returning the predictor and the
            fitted weights.
        """
        present = np.unique(labels)
        if len(present) == 1:
            constant = np.eye(n_classes)[present[0]]
            return lambda c, start: (lambda z: np.tile(constant, (len(z), 1)), start)
        index = np.searchsorted(present, labels)
        y = torch.as_tensor(index, dtype=torch.long, device=x.device)

        def fit(c: float, start: torch.Tensor | None) -> tuple[Predictor, torch.Tensor]:
            weights = fit_logistic(
                x,
                y,
                c,
                multilabel=False,
                max_iter=self.max_iter,
                tolerance=self.tolerance,
                start=start,
            )

            def predict(z: torch.Tensor) -> NDArray[np.float64]:
                probabilities = np.zeros((len(z), n_classes))
                logits = z @ weights[:-1] + weights[-1]
                probabilities[:, present] = torch.softmax(logits, dim=1).cpu().numpy()
                return probabilities

            return predict, weights

        return fit

    def _multilabel(
        self, x: torch.Tensor, indicators: NDArray[np.float64]
    ) -> Callable[[float, torch.Tensor | None], tuple[Predictor, torch.Tensor | None]]:
        """Fitting function of one binary model per label, constant labels keeping their value.

        Args:
            x: Standardized training features of shape `(N, D)`.
            indicators: Label indicators of shape `(N, C)`.

        Returns:
            A function of the strength and the starting weights returning the predictor and the
            fitted weights of the varying labels.
        """
        varying = np.flatnonzero(indicators.min(axis=0) != indicators.max(axis=0))
        constants = indicators[0].astype(np.float64)
        y = torch.as_tensor(indicators[:, varying], dtype=torch.float64, device=x.device)

        def fit(c: float, start: torch.Tensor | None) -> tuple[Predictor, torch.Tensor | None]:
            weights = None
            if len(varying):
                weights = fit_logistic(
                    x,
                    y,
                    c,
                    multilabel=True,
                    max_iter=self.max_iter,
                    tolerance=self.tolerance,
                    start=start,
                )

            def predict(z: torch.Tensor) -> NDArray[np.float64]:
                probabilities = np.tile(constants, (len(z), 1))
                if weights is not None:
                    logits = z @ weights[:-1] + weights[-1]
                    probabilities[:, varying] = torch.sigmoid(logits).cpu().numpy()
                return probabilities

            return predict, weights

        return fit
