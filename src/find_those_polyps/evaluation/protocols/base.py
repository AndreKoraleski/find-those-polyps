"""Common interface of the evaluation protocols.

A protocol either fits a predictor on frozen training features and returns its test predictions,
or measures the representation directly and returns its own metrics. Hyperparameters are pydantic
fields, so that two protocols with equal settings share one hash.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import ClassVar, NamedTuple

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from find_those_polyps.evaluation.features import Features
from find_those_polyps.metrics import Predictions, higher_is_better
from find_those_polyps.metrics.bootstrap import Statistic, mean_statistic
from find_those_polyps.tasks.schema import TaskKind, TaskSpec
from find_those_polyps.utils.hashing import canonical_sha256


class Draw(NamedTuple):
    """Seed and samples of one repeat of a protocol.

    Attributes:
        seed: Seed of every random choice of the repeat.
        train: Positions of the training samples of the repeat in the training partition.
        val: Positions of its validation samples in the validation partition, or None for the
            whole partition.
    """

    seed: int
    train: NDArray[np.intp]
    val: NDArray[np.intp] | None = None

    def partitions(self, train: Features, val: Features) -> tuple[Features, Features]:
        """Training and validation features of the repeat.

        Args:
            train: Whole training partition.
            val: Whole validation partition.

        Returns:
            The training subset and the validation subset, or the whole validation partition
            itself when `val` is None.
        """
        return train.subset(self.train), val if self.val is None else val.subset(self.val)


class Protocol(BaseModel, ABC):
    """Procedure fitting a predictor on frozen training features.

    Hyperparameters are model fields and enter the protocol hash, so results obtained under
    different settings are never ranked together. Validation data serve model selection only and the
    test partition is used once.

    Attributes:
        name: Identifier of the protocol in configurations and result tables.
        kinds: Task kinds the protocol applies to.
        dense: Whether the protocol consumes dense feature maps of shape `(N, D, h, w)` instead of
            global features of shape `(N, D)`.
        stochastic: Whether the result depends on the seed, so that it is repeated over seeds even
            on the full label budget.
    """

    model_config = ConfigDict(frozen=True)

    name: ClassVar[str]
    kinds: ClassVar[frozenset[TaskKind]]
    dense: ClassVar[bool] = False
    stochastic: ClassVar[bool] = False

    @property
    def hash(self) -> str:
        """First twelve hexadecimal digits of the hash of the name, hyperparameters and metric.

        The primary metric enters only for protocols that define their own, such as
        `DirectProtocol` subclasses.
        """
        parts = [self.name, self.model_dump(mode="json")]
        primary = getattr(self, "primary", None)
        return canonical_sha256(parts if primary is None else [*parts, primary])[:12]

    @abstractmethod
    def predict(
        self,
        train: Features,
        val: Features,
        test: Features,
        spec: TaskSpec,
        *,
        seed: int,
        device: torch.device,
    ) -> Predictions:
        """Fit on `train`, select on `val` and predict `test`.

        Args:
            train: Features and targets of the labelled training partition.
            val: Features and targets used for model selection only.
            test: Features of the partition to predict.
            spec: Task specification with the classes and the primary metric.
            seed: Seed of every random choice of the protocol.
            device: Device on which the predictor is fitted.

        Returns:
            Class probabilities of shape `(N, C)`, an iterable of probability maps of shape
            `(C, H, W)` or one table of scored boxes per test image, as `score` expects for the
            task kind.
        """

    def predict_runs(
        self,
        train: Features,
        val: Features,
        test: Features,
        spec: TaskSpec,
        draws: Sequence[Draw],
        *,
        device: torch.device,
    ) -> Iterator[Predictions]:
        """Predictions of every repeat, each fitted on its own subsets and seed.

        Repeats are fitted one after another by `predict`, each when its predictions are
        requested. Protocols whose predictors are too small to occupy the device override this
        to fit every repeat at once.

        Args:
            train: Whole training partition, from which every repeat takes its subset.
            val: Whole validation partition, from which every repeat takes its subset.
            test: Features of the partition to predict.
            spec: Task specification with the classes and the primary metric.
            draws: Seed and samples of every repeat.
            device: Device on which the predictors are fitted.

        Yields:
            The predictions of every repeat in the order of `draws`, as `predict` returns them.
        """
        for draw in draws:
            subset, held_out = draw.partitions(train, val)
            yield self.predict(subset, held_out, test, spec, seed=draw.seed, device=device)


class DirectProtocol(Protocol):
    """Protocol that measures the representation itself instead of a fitted predictor.

    It reports its own `metrics` and ranks by `primary`. Label budgets limit the labelled training
    data it may use when `budgeted` holds, and it runs on the full budget only otherwise.

    Attributes:
        primary: Metric by which the protocol ranks encoders.
        metrics: Metrics the protocol reports.
        budgeted: Whether label budgets apply to the protocol.
    """

    primary: ClassVar[str]
    metrics: ClassVar[tuple[str, ...]]
    budgeted: ClassVar[bool] = True

    def predict(
        self,
        train: Features,
        val: Features,
        test: Features,
        spec: TaskSpec,
        *,
        seed: int,
        device: torch.device,
    ) -> Predictions:
        """Refuse to predict, since the protocol yields metrics and not predictions.

        Args:
            train: Features and targets of the labelled training partition.
            val: Features and targets of the validation partition.
            test: Features and targets of the test partition.
            spec: Task specification.
            seed: Seed of the protocol.
            device: Device of the computation.

        Raises:
            NotImplementedError: Always, since `measure` replaces this method.
        """
        raise NotImplementedError(f"{self.name} measures representations directly")

    @abstractmethod
    def measure(
        self,
        train: Features,
        val: Features,
        test: Features,
        spec: TaskSpec,
        *,
        seed: int,
        device: torch.device,
    ) -> tuple[dict[str, float], pd.DataFrame]:
        """Test metrics of the representation and the per image records behind them.

        Args:
            train: Features and targets of the labelled training partition.
            val: Features and targets of the validation partition.
            test: Features and targets of the test partition.
            spec: Task specification with the classes.
            seed: Seed of every random choice of the protocol.
            device: Device of the computation.

        Returns:
            The value of every name in `metrics` on the test partition, and one record per test
            image from which `statistic` recomputes the primary metric.
        """

    def statistic(self, records: pd.DataFrame) -> Statistic:
        """Primary metric of resampled test records, by default the mean of a per image column.

        Args:
            records: Per image records returned by `measure`.

        Returns:
            Statistic mapping resampled row indices to the primary metric, ignoring undefined
            values.
        """
        return mean_statistic(records[self.primary].to_numpy(dtype=float))


def better(candidate: float, incumbent: float, metric: str) -> bool:
    """Whether `candidate` strictly improves on `incumbent`, treating NaN as worst.

    Ties keep the incumbent, so that model selection retains the earliest of equal candidates.

    Args:
        candidate: Value of `metric` for the new candidate.
        incumbent: Value of `metric` for the best candidate so far.
        metric: Name of the metric, which sets the direction of improvement.

    Returns:
        False when `candidate` is NaN, True when only `incumbent` is NaN, and otherwise whether
        `candidate` is strictly better in the direction of `metric`.
    """
    if candidate != candidate:
        return False
    if incumbent != incumbent:
        return True
    if higher_is_better(metric):
        return candidate > incumbent
    return candidate < incumbent
