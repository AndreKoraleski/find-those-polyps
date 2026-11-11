"""Weighted nearest neighbour classification on frozen global features."""

from typing import ClassVar

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from pydantic import PositiveFloat, PositiveInt

from find_those_polyps.evaluation.features import Features
from find_those_polyps.evaluation.protocols.base import Protocol
from find_those_polyps.tasks.schema import TaskKind, TaskSpec

_CHUNK = 1024


class KNN(Protocol):
    """Temperature weighted vote of the k most cosine similar training samples.

    Follows the protocol of Wu et al. (2018) as used by DINO. The weights are the softmax of the
    similarities divided by the temperature over the k neighbours, which rescales the exponential
    weights of DINO without changing the vote. The protocol has no fitted parameters, ignores the
    validation partition and is deterministic.

    Attributes:
        k: Number of neighbours, reduced to the size of the training set when it is smaller.
            Defaults to 20.
        temperature: Temperature dividing the cosine similarities. Defaults to 0.07.
    """

    name: ClassVar[str] = "knn"
    kinds: ClassVar[frozenset[TaskKind]] = frozenset(
        {TaskKind.CLASSIFICATION, TaskKind.ORDINAL, TaskKind.MULTILABEL}
    )

    k: PositiveInt = 20
    temperature: PositiveFloat = 0.07

    def predict(
        self,
        train: Features,
        val: Features,
        test: Features,
        spec: TaskSpec,
        *,
        seed: int,
        device: torch.device,
    ) -> np.ndarray:
        """Vote of the training neighbours of every test sample.

        Args:
            train: Global training features of shape `(N, D)`, which form the memory bank.
            val: Validation features, unused.
            test: Global test features of shape `(M, D)`, which are the queries.
            spec: Task specification with the classes.
            seed: Seed, unused since the protocol is deterministic.
            device: Device of the similarity computation.

        Returns:
            Scores of shape `(M, C)` in float64. Each row is the weighted mean of the indicator
            vectors of the neighbours, a probability distribution for single label tasks and a
            weighted label frequency for multilabel tasks.
        """
        bank = F.normalize(train.values.float().to(device), dim=1)
        queries = F.normalize(test.values.float().to(device), dim=1)
        targets = torch.from_numpy(train.indicators(len(spec.classes))).float()
        targets = targets.to(device)
        k = min(self.k, len(bank))
        scores = []
        for start in range(0, len(queries), _CHUNK):
            similarity = queries[start : start + _CHUNK] @ bank.T
            top, index = similarity.topk(k, dim=1)
            weights = (top / self.temperature).softmax(dim=1)
            scores.append((weights.unsqueeze(-1) * targets[index]).sum(dim=1))
        return torch.cat(scores).double().cpu().numpy()
