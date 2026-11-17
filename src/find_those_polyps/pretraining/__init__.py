"""Self supervised pretraining on curated corpora.

The package holds the cards of the corpora that pretraining may use, a reader that takes their
frames from a local cache, a whole archive or the official host, and a torch dataset that reads
them while training.
"""

from find_those_polyps.pretraining.corpora import (
    Archive,
    CorpusCard,
    HeldOut,
    corpus,
    load_corpora,
)
from find_those_polyps.pretraining.dataset import CorpusDataset
from find_those_polyps.pretraining.reader import CorpusReader, CorpusUnavailableError, Frame, fetch

__all__ = [
    "Archive",
    "CorpusCard",
    "CorpusDataset",
    "CorpusReader",
    "CorpusUnavailableError",
    "Frame",
    "HeldOut",
    "corpus",
    "fetch",
    "load_corpora",
]
