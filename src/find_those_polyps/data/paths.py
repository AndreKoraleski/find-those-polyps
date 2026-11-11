"""Location of local dataset copies."""

import os
from pathlib import Path

DATA_ENV = "FIND_THOSE_POLYPS_DATA"


def default_data_root() -> Path:
    """Directory that holds the local copies of the datasets.

    Returns:
        The path named by the `FIND_THOSE_POLYPS_DATA` environment variable, or the relative path
            `data`, resolved against the working directory, when the variable is unset.
    """
    return Path(os.environ.get(DATA_ENV, "data"))
