"""Frozen feature evaluation, metrics and the leaderboard."""

from find_those_polyps.evaluation.features import Features, extract, extract_many
from find_those_polyps.evaluation.leaderboard import Leaderboard, LeaderboardRow
from find_those_polyps.evaluation.protocols import (
    KNN,
    PROTOCOLS,
    LinearProbe,
    LinearSegmentation,
    Protocol,
)
from find_those_polyps.evaluation.runner import (
    EvaluationConfig,
    TaskFeatures,
    evaluate,
    evaluate_task,
    select_tasks,
)
from find_those_polyps.metrics import METRIC_INFO, metric_names, score

__all__ = [
    "KNN",
    "METRIC_INFO",
    "PROTOCOLS",
    "EvaluationConfig",
    "Features",
    "Leaderboard",
    "LeaderboardRow",
    "LinearProbe",
    "LinearSegmentation",
    "Protocol",
    "TaskFeatures",
    "evaluate",
    "evaluate_task",
    "extract",
    "extract_many",
    "metric_names",
    "score",
    "select_tasks",
]
