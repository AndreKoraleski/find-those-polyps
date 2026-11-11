"""Linear probing of frozen global features with logistic regression."""

import warnings
from collections.abc import Callable
from typing import ClassVar

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import PositiveFloat, PositiveInt
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from find_those_polyps.evaluation.features import Features
from find_those_polyps.evaluation.protocols.base import Protocol, better
from find_those_polyps.metrics import score
from find_those_polyps.tasks.schema import TaskKind, TaskSpec

Predictor = Callable[[NDArray[np.float64]], NDArray[np.float64]]
"""Function mapping standardized features of shape `(M, D)` to probabilities of shape `(M, C)`."""


def _fit_single_label(
    x: NDArray[np.float64],
    y: NDArray[np.int64],
    n_classes: int,
    c: float,
    max_iter: int,
) -> Predictor:
    """Multinomial logistic regression, or a constant one hot prediction when `y` is constant.

    Args:
        x: Standardized training features of shape `(N, D)`.
        y: Class indices of shape `(N,)`.
        n_classes: Number of classes, including those absent from `y`.
        c: Inverse regularization strength.
        max_iter: Iteration limit of L-BFGS.

    Returns:
        Predictor mapping features of shape `(M, D)` to probabilities of shape `(M, C)`, zero for
        the classes absent from `y`.
    """
    if len(np.unique(y)) == 1:
        constant = np.eye(n_classes)[y[0]]
        return lambda z: np.tile(constant, (len(z), 1))
    model = LogisticRegression(C=c, max_iter=max_iter)
    model.fit(x, y)

    def predict(z: NDArray[np.float64]) -> NDArray[np.float64]:
        probabilities = np.zeros((len(z), n_classes))
        probabilities[:, model.classes_] = model.predict_proba(z)
        return probabilities

    return predict


def _fit_multilabel(
    x: NDArray[np.float64], y: NDArray[np.float64], c: float, max_iter: int
) -> Predictor:
    """One binary logistic regression per label, or its constant value when the label is constant.

    Args:
        x: Standardized training features of shape `(N, D)`.
        y: Label indicators of shape `(N, C)`.
        c: Inverse regularization strength.
        max_iter: Iteration limit of L-BFGS.

    Returns:
        Predictor mapping features of shape `(M, D)` to label probabilities of shape `(M, C)`.
    """
    models: list[LogisticRegression | float] = []
    for column in y.T:
        if column.min() == column.max():
            models.append(float(column[0]))
        else:
            models.append(LogisticRegression(C=c, max_iter=max_iter).fit(x, column))

    def predict(z: NDArray[np.float64]) -> NDArray[np.float64]:
        columns = [
            np.full(len(z), m) if isinstance(m, float) else m.predict_proba(z)[:, 1] for m in models
        ]
        return np.stack(columns, axis=1)

    return predict


class LinearProbe(Protocol):
    """L2 regularized logistic regression on standardized global features.

    The inverse regularization strength is chosen from a logarithmic grid by the primary metric on
    the validation partition, after Radford et al. (2021). Features are standardized with the mean
    and deviation of the training partition. Multilabel tasks fit one binary model per label.
    Optimization by L-BFGS makes the protocol deterministic.

    Attributes:
        c_grid: Inverse regularization strengths tried in order, by default the powers of ten from
            1e-4 to 1e4. Ties on the validation partition keep the earliest value.
        max_iter: Iteration limit of L-BFGS. Defaults to 1000, and convergence warnings are
            silenced.
    """

    name: ClassVar[str] = "linear"
    kinds: ClassVar[frozenset[TaskKind]] = frozenset(
        {TaskKind.CLASSIFICATION, TaskKind.ORDINAL, TaskKind.MULTILABEL}
    )

    c_grid: tuple[PositiveFloat, ...] = tuple(10.0**e for e in range(-4, 5))
    max_iter: PositiveInt = 1000

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
            device: Device, unused since fitting runs on the CPU.

        Returns:
            Class or label probabilities of shape `(M, C)`.
        """
        n_classes = len(spec.classes)
        scaler = StandardScaler().fit(train.values.double().numpy())
        x_train, x_val, x_test = (
            scaler.transform(f.values.double().numpy()) for f in (train, val, test)
        )
        best: Predictor | None = None
        best_score = float("nan")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for c in self.c_grid:
                if spec.kind is TaskKind.MULTILABEL:
                    predictor = _fit_multilabel(
                        x_train, train.indicators(n_classes), c, self.max_iter
                    )
                else:
                    predictor = _fit_single_label(
                        x_train, train.labels(), n_classes, c, self.max_iter
                    )
                metrics = score(
                    spec.kind,
                    predictor(x_val),
                    val.targets,
                    n_classes,
                    names=[spec.primary_metric],
                )
                candidate = metrics[spec.primary_metric]
                if best is None or better(candidate, best_score, spec.primary_metric):
                    best, best_score = predictor, candidate
        assert best is not None
        return best(x_test)
