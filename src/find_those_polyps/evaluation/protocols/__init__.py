"""Evaluation protocols applied to frozen features.

Probing protocols fit a nearest neighbour, linear, segmentation or detection head on the training
features and are scored on the test predictions. Direct protocols measure retrieval and clustering
quality of the representation itself.
"""

from find_those_polyps.evaluation.protocols.base import DirectProtocol, Protocol
from find_those_polyps.evaluation.protocols.detection import LinearDetection
from find_those_polyps.evaluation.protocols.knn import KNN
from find_those_polyps.evaluation.protocols.linear import LinearProbe
from find_those_polyps.evaluation.protocols.segmentation import LinearSegmentation
from find_those_polyps.evaluation.protocols.zero_shot import KMeansClustering, Retrieval

PROTOCOLS: dict[str, Protocol] = {
    protocol.name: protocol
    for protocol in (
        KNN(),
        LinearProbe(),
        LinearSegmentation(),
        LinearDetection(),
        Retrieval(),
        KMeansClustering(),
    )
}
"""Protocol instances with their default hyperparameters, keyed by protocol name."""

__all__ = [
    "KNN",
    "PROTOCOLS",
    "DirectProtocol",
    "KMeansClustering",
    "LinearDetection",
    "LinearProbe",
    "LinearSegmentation",
    "Protocol",
    "Retrieval",
]
