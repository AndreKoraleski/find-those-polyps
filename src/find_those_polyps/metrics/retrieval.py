"""Retrieval and clustering metrics of frozen representations.

Retrieval ranks a labelled gallery by cosine similarity to each query, and a gallery item is
relevant when it shares the query's class. MAP@R and R precision follow Musgrave et al. (2020),
where R is the number of relevant gallery items. Clustering compares k-means assignments with the
classes through mutual information, the adjusted Rand index and the accuracy of the best one to one
matching of clusters to classes.
"""

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

RETRIEVAL = ("precision_at_1", "precision_at_5", "r_precision", "map_at_r", "map")
"""Per query retrieval metrics."""
CLUSTERING = ("nmi", "ami", "ari", "cluster_accuracy")
"""Agreement metrics between clusters and classes."""
CHUNK = 256
"""Number of queries ranked at once, which bounds the memory of the similarity matrix."""


def retrieval(
    queries: torch.Tensor,
    query_labels: NDArray[np.int64],
    gallery: torch.Tensor,
    gallery_labels: NDArray[np.int64],
) -> dict[str, NDArray[np.float64] | NDArray[np.int64]]:
    """Per query retrieval scores and the index of the nearest gallery item.

    The gallery is ranked by decreasing cosine similarity to each query. With R the number of
    gallery items of the class of the query, R precision is the fraction of relevant items among
    the first R, MAP@R the sum of the precision at every relevant rank up to R divided by R, and
    MAP the same sum over the whole ranking divided by R. Precision at five averages over fewer
    items when the gallery is smaller. Queries whose class is absent from the gallery have
    undefined scores, reported as NaN.

    Args:
        queries: Query features of shape `(M, D)`, moved to the device of `gallery`.
        query_labels: Class indices of the queries, of shape `(M,)`.
        gallery: Gallery features of shape `(N, D)`.
        gallery_labels: Class indices of the gallery, of shape `(N,)`.

    Returns:
        Every metric of `RETRIEVAL` as an array of shape `(M,)` in `[0, 1]`, and under `nearest`
        the gallery index of the most similar item of every query.
    """
    gallery_n = F.normalize(gallery.float(), dim=1)
    labels = torch.as_tensor(gallery_labels, device=gallery_n.device)
    relevant_counts = np.bincount(gallery_labels, minlength=int(query_labels.max(initial=0)) + 1)
    columns: dict[str, list[NDArray[np.float64]]] = {name: [] for name in RETRIEVAL}
    nearest: list[NDArray[np.int64]] = []
    positions = torch.arange(1, len(gallery_n) + 1, device=gallery_n.device, dtype=torch.float64)
    for start in range(0, len(queries), CHUNK):
        batch = F.normalize(queries[start : start + CHUNK].float().to(gallery_n.device), dim=1)
        order = (batch @ gallery_n.T).argsort(dim=1, descending=True)
        targets = torch.as_tensor(query_labels[start : start + CHUNK], device=gallery_n.device)
        hits = (labels[order] == targets[:, None]).double()
        cumulative = hits.cumsum(dim=1)
        precision = cumulative / positions
        r = torch.as_tensor(
            relevant_counts[query_labels[start : start + CHUNK]], device=gallery_n.device
        ).double()
        within_r = positions[None, :] <= r[:, None]
        safe_r = r.clamp(min=1)
        columns["precision_at_1"].append(hits[:, 0].cpu().numpy())
        columns["precision_at_5"].append(hits[:, :5].mean(dim=1).cpu().numpy())
        columns["r_precision"].append(((hits * within_r).sum(dim=1) / safe_r).cpu().numpy())
        columns["map_at_r"].append(
            ((precision * hits * within_r).sum(dim=1) / safe_r).cpu().numpy()
        )
        columns["map"].append(((precision * hits).sum(dim=1) / safe_r).cpu().numpy())
        nearest.append(order[:, 0].cpu().numpy())
    result: dict[str, NDArray[np.float64] | NDArray[np.int64]] = {}
    absent = relevant_counts[query_labels] == 0
    for name, parts in columns.items():
        values = np.concatenate(parts) if parts else np.zeros(0)
        values[absent] = np.nan
        result[name] = values
    result["nearest"] = np.concatenate(nearest) if nearest else np.zeros(0, dtype=np.int64)
    return result


def matching(labels: NDArray[np.int64], clusters: NDArray[np.int64]) -> dict[int, int]:
    """Class matched to each cluster by the Hungarian algorithm.

    The matching is one to one and maximizes the number of images whose cluster is matched to
    their class.

    Args:
        labels: Class indices of shape `(N,)`.
        clusters: Cluster indices of shape `(N,)`.

    Returns:
        Matched class of every cluster index up to the largest class or cluster index.
    """
    size = int(max(labels.max(initial=0), clusters.max(initial=0))) + 1
    counts = np.zeros((size, size), dtype=np.int64)
    np.add.at(counts, (clusters, labels), 1)
    rows, cols = linear_sum_assignment(counts, maximize=True)
    return {int(r): int(c) for r, c in zip(rows, cols, strict=True)}


def cluster_accuracy(labels: NDArray[np.int64], clusters: NDArray[np.int64]) -> float:
    """Accuracy of the one to one cluster to class matching that maximizes agreement.

    Args:
        labels: Class indices of shape `(N,)`.
        clusters: Cluster indices of shape `(N,)`.

    Returns:
        Fraction of images whose cluster is matched to their class, in `[0, 1]`, NaN without
        images.
    """
    mapping = matching(labels, clusters)
    matched = np.array([mapping.get(int(c), -1) for c in clusters])
    return float(np.mean(matched == labels)) if len(labels) else float("nan")


def clustering(labels: NDArray[np.int64], clusters: NDArray[np.int64]) -> dict[str, float]:
    """Agreement between cluster assignments and classes.

    Normalized mutual information uses the arithmetic mean of the entropies. Adjusted mutual
    information (Vinh et al. 2010) and the adjusted Rand index (Hubert and Arabie 1985) are zero in
    expectation under random assignments.

    Args:
        labels: Class indices of shape `(N,)`.
        clusters: Cluster indices of shape `(N,)`.

    Returns:
        `nmi` and `cluster_accuracy` in `[0, 1]`, and `ami` and `ari` at most one and negative
        below chance.
    """
    return {
        "nmi": float(normalized_mutual_info_score(labels, clusters)),
        "ami": float(adjusted_mutual_info_score(labels, clusters)),
        "ari": float(adjusted_rand_score(labels, clusters)),
        "cluster_accuracy": cluster_accuracy(labels, clusters),
    }
