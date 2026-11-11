"""Retrieval of benchmark datasets.

The package reads the files of a registered dataset either from a local directory or straight from
the zip archives on its official host through HTTP range requests, makes complete local copies of
openly downloadable datasets, and checks that the hosts still serve the files pinned on the
dataset cards.
"""

from find_those_polyps.data.download import ManualAccessError, extract, fetch
from find_those_polyps.data.files import DatasetFiles, LocalFiles, RemoteFiles, open_files
from find_those_polyps.data.remote import ArchiveIndex, fetch_file, read_index

__all__ = [
    "ArchiveIndex",
    "DatasetFiles",
    "LocalFiles",
    "ManualAccessError",
    "RemoteFiles",
    "extract",
    "fetch",
    "fetch_file",
    "open_files",
    "read_index",
]
