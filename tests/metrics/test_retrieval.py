import numpy as np
import pytest
import torch

from find_those_polyps.metrics import METRIC_INFO, metric_names
from find_those_polyps.metrics.retrieval import (
    CLUSTERING,
    RETRIEVAL,
    cluster_accuracy,
    clustering,
    retrieval,
)
from find_those_polyps.tasks import TaskKind


def test_separable_classes_retrieve_perfectly() -> None:
    gallery = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    queries = torch.tensor([[1.0, 0.05], [0.05, 1.0]])
    scores = retrieval(queries, np.array([0, 1]), gallery, np.array([0, 0, 1, 1]))
    for name in RETRIEVAL:
        assert np.allclose(scores[name], 1.0) or name == "precision_at_5", name
    assert list(scores["nearest"]) == [0, 2]
    assert np.allclose(scores["precision_at_5"], 0.5)


def test_map_at_r_matches_a_hand_computed_ranking() -> None:
    gallery = torch.tensor([[1.0, 0.0], [0.8, 0.6], [0.6, 0.8], [0.0, 1.0]])
    query = torch.tensor([[1.0, 0.0]])
    scores = retrieval(query, np.array([0]), gallery, np.array([0, 1, 0, 1]))
    assert scores["r_precision"][0] == pytest.approx(0.5)
    assert scores["map_at_r"][0] == pytest.approx(0.5)
    assert scores["map"][0] == pytest.approx((1 + 2 / 3) / 2)


def test_absent_classes_are_undefined() -> None:
    gallery = torch.eye(2)
    scores = retrieval(torch.eye(2), np.array([0, 2]), gallery, np.array([0, 1]))
    assert np.isnan(scores["map_at_r"][1])
    assert scores["map_at_r"][0] == 1.0


def test_clustering_is_invariant_to_relabelling() -> None:
    labels = np.array([0, 0, 1, 1, 2, 2])
    perfect = clustering(labels, np.array([2, 2, 0, 0, 1, 1]))
    assert all(perfect[name] == pytest.approx(1.0) for name in CLUSTERING)
    assert cluster_accuracy(labels, np.array([0, 1, 0, 1, 0, 1])) == pytest.approx(2 / 6)


def test_every_metric_is_documented_and_top_k_follows_classes() -> None:
    for name in RETRIEVAL + CLUSTERING:
        assert name in METRIC_INFO
    assert "top5_accuracy" in metric_names(TaskKind.CLASSIFICATION, 6)
    assert "top5_accuracy" not in metric_names(TaskKind.CLASSIFICATION, 5)
    assert "top3_accuracy" in metric_names(TaskKind.ORDINAL, 4)
    for kind in TaskKind:
        assert set(metric_names(kind, 6)) <= set(METRIC_INFO)
