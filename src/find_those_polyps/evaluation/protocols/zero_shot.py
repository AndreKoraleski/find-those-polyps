"""Protocols that measure frozen representations without fitting a predictor.

Retrieval ranks the labelled training images by cosine similarity to each test image, so the label
budget limits the gallery. Clustering partitions the test images with k-means into as many clusters
as there are classes and uses no labels at all, so it is evaluated on the full budget only.
"""

from typing import ClassVar

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812
from numpy.typing import NDArray
from pydantic import PositiveInt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score

from find_those_polyps.evaluation.features import Features
from find_those_polyps.evaluation.protocols.base import DirectProtocol
from find_those_polyps.metrics.bootstrap import Statistic
from find_those_polyps.metrics.retrieval import (
    CLUSTERING,
    RETRIEVAL,
    clustering,
    matching,
    retrieval,
)
from find_those_polyps.tasks.schema import TaskKind, TaskSpec

SINGLE_LABEL = frozenset({TaskKind.CLASSIFICATION, TaskKind.ORDINAL})
"""Task kinds with one class per image, to which retrieval and clustering apply."""


class Retrieval(DirectProtocol):
    """Class consistent nearest neighbour retrieval from the training gallery.

    Every test image queries the labelled training images by cosine similarity, and a training
    image is relevant when it shares the class of the query. The protocol has no hyperparameters
    and ranks by MAP@R of Musgrave et al. (2020).
    """

    name: ClassVar[str] = "retrieval"
    kinds: ClassVar[frozenset[TaskKind]] = SINGLE_LABEL
    primary: ClassVar[str] = "map_at_r"
    metrics: ClassVar[tuple[str, ...]] = RETRIEVAL

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
        """Retrieval scores of every test query against the training gallery.

        Args:
            train: Global features and labels of the gallery.
            val: Validation features, unused.
            test: Global features and labels of the queries.
            spec: Task specification with the class names.
            seed: Seed, unused since retrieval is deterministic.
            device: Device of the similarity computation.

        Returns:
            The mean over queries of every retrieval metric, ignoring queries whose class is
            absent from the gallery, and one record per query with its class, its nearest
            gallery image and the per query value of every metric.
        """
        labels = test.labels()
        scores = retrieval(test.values.to(device), labels, train.values.to(device), train.labels())
        nearest = np.asarray(scores["nearest"], dtype=np.int64)
        records = pd.DataFrame(
            {
                "target": [spec.classes[i] for i in labels],
                "nearest_id": [train.ids[i] if train.ids else i for i in nearest],
                "nearest_path": [train.paths[i] if train.paths else "" for i in nearest],
                "nearest_target": [spec.classes[train.labels()[i]] for i in nearest],
            }
            | {name: scores[name] for name in RETRIEVAL}
        )
        metrics = {name: float(np.nanmean(scores[name])) for name in RETRIEVAL}
        return metrics, records


class KMeansClustering(DirectProtocol):
    """K-means on the normalized test features with one cluster per class.

    It ranks by adjusted mutual information, which unlike normalized mutual information corrects
    for chance agreement and depends little on the number of classes and test images. A test set
    smaller than the number of classes, as in smoke tests, gets one cluster per image. The training
    and validation partitions are unused.

    Attributes:
        n_init: Number of k-means restarts, of which the one with the lowest inertia is kept.
            Defaults to 10.
    """

    name: ClassVar[str] = "kmeans"
    kinds: ClassVar[frozenset[TaskKind]] = SINGLE_LABEL
    stochastic: ClassVar[bool] = True
    budgeted: ClassVar[bool] = False
    primary: ClassVar[str] = "ami"
    metrics: ClassVar[tuple[str, ...]] = CLUSTERING

    n_init: PositiveInt = 10

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
        """Cluster the test images and compare the clusters with their classes.

        Args:
            train: Training features, unused.
            val: Validation features, unused.
            test: Global features and labels of the images to cluster.
            spec: Task specification with the class names, whose number sets the clusters.
            seed: Seed of the k-means initialization.
            device: Device, unused since k-means runs on the CPU.

        Returns:
            The clustering metrics, and one record per test image with its class, its cluster,
            the class matched to that cluster and whether the two agree.
        """
        values = F.normalize(test.values.float(), dim=1).numpy()
        n_clusters = min(len(spec.classes), len(values))
        model = KMeans(n_clusters=n_clusters, n_init=self.n_init, random_state=seed)
        clusters = model.fit_predict(values).astype(np.int64)
        labels = test.labels()
        mapping = matching(labels, clusters)
        matched = np.array([mapping.get(int(c), -1) for c in clusters])
        records = pd.DataFrame(
            {
                "target": [spec.classes[i] for i in labels],
                "cluster": clusters,
                "matched_class": [spec.classes[i] if i >= 0 else "" for i in matched],
                "correct": matched == labels,
            }
        )
        return clustering(labels, clusters), records

    def statistic(self, records: pd.DataFrame) -> Statistic:
        """Adjusted mutual information of resampled records, marked as centred.

        Duplicated images agree with themselves and inflate the agreement on every resample, so
        the bootstrap interval is centred on the estimate on all images.

        Args:
            records: Per image records returned by `measure`.

        Returns:
            Statistic mapping resampled row indices to the adjusted mutual information between
            classes and the fixed clusters.
        """
        labels = pd.factorize(records["target"])[0]
        clusters = records["cluster"].to_numpy()

        def ami(index: NDArray[np.intp]) -> float:
            return float(adjusted_mutual_info_score(labels[index], clusters[index]))

        vars(ami)["centred"] = True
        return ami
