"""Downstream task definitions, canonical splits and label budgets.

A task pairs a declarative `TaskSpec` with a dataset `Loader` and a canonical `SplitFile`, and label
budgets select nested subsets of its training partition.
"""

from find_those_polyps.tasks.budgets import budget_indices
from find_those_polyps.tasks.loaders import LOADERS, Loader, register_loader
from find_those_polyps.tasks.schema import (
    FULL,
    Budget,
    Sample,
    Split,
    TaskKind,
    TaskSpec,
)
from find_those_polyps.tasks.splits import SplitFile, make_split
from find_those_polyps.tasks.task import LazySamples, Task

__all__ = [
    "FULL",
    "LOADERS",
    "Budget",
    "LazySamples",
    "Loader",
    "Sample",
    "Split",
    "SplitFile",
    "Task",
    "TaskKind",
    "TaskSpec",
    "budget_indices",
    "make_split",
    "register_loader",
]
