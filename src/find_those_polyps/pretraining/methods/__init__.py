"""Self supervised methods of the pretraining recipes, each with its schedule."""

from find_those_polyps.pretraining.methods.base import Method, Schedule
from find_those_polyps.pretraining.methods.dino import DINO, DINOSettings
from find_those_polyps.pretraining.methods.mae import MAE, MAESettings
from find_those_polyps.pretraining.methods.moco import MoCo, MoCoSettings

METHODS: dict[str, type[Method]] = {method.name: method for method in (DINO, MAE, MoCo)}
"""Every method by its name on the command line."""

__all__ = [
    "DINO",
    "MAE",
    "METHODS",
    "DINOSettings",
    "MAESettings",
    "Method",
    "MoCo",
    "MoCoSettings",
    "Schedule",
]
