"""Benchmark of unsupervised and self supervised encoders for colonoscopy imaging.

Public names are imported on first access, so that light commands such as listing datasets do not
load PyTorch. The encoder interface and its helpers come from `find_those_polyps.models`, the
evaluation and the leaderboard from `find_those_polyps.evaluation`, the logging setup from
`find_those_polyps.logging` and the registry from `find_those_polyps.registry`.
"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from find_those_polyps.evaluation import (
        EvaluationConfig,
        Leaderboard,
        LeaderboardRow,
        evaluate,
    )
    from find_those_polyps.logging import configure_logging, get_logger
    from find_those_polyps.models import (
        Encoder,
        TorchEncoder,
        normalize,
        patch_grid,
        resize,
    )
    from find_those_polyps.registry import load_registry

_SOURCES = {
    "Encoder": "find_those_polyps.models",
    "EvaluationConfig": "find_those_polyps.evaluation",
    "Leaderboard": "find_those_polyps.evaluation",
    "LeaderboardRow": "find_those_polyps.evaluation",
    "TorchEncoder": "find_those_polyps.models",
    "configure_logging": "find_those_polyps.logging",
    "evaluate": "find_those_polyps.evaluation",
    "get_logger": "find_those_polyps.logging",
    "load_registry": "find_those_polyps.registry",
    "normalize": "find_those_polyps.models",
    "patch_grid": "find_those_polyps.models",
    "resize": "find_those_polyps.models",
}

__all__ = [
    "Encoder",
    "EvaluationConfig",
    "Leaderboard",
    "LeaderboardRow",
    "TorchEncoder",
    "configure_logging",
    "evaluate",
    "get_logger",
    "load_registry",
    "normalize",
    "patch_grid",
    "resize",
]


def __getattr__(name: str) -> Any:
    """Import a public name from its source module on first access.

    Args:
        name: Attribute requested from the package.

    Returns:
        The object of that name in the module that defines it.

    Raises:
        AttributeError: If `name` is not one of the public names of the package.
    """
    if name not in _SOURCES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(_SOURCES[name]), name)
