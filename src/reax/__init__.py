"""REAX: A simple training framework for JAX-based projects"""

from . import data, listeners, metrics, modules, optimizers, saving, stages, strategies, training
from .data import DataLoader, DataModule, ReaxDataLoader
from .loggers import Logger
from .metrics import Metric
from .modules import *
from .optimizers import *
from .saving import *
from .strategies import *
from .training import *
from .utils.rngs import seed_everything

__all__ = (
    modules.__all__
    + optimizers.__all__
    + saving.__all__
    + strategies.__all__
    + training.__all__
    + data.__all__
    # Modules
    + (
        "data",
        "stages",
        "listeners",
        "metrics",
        "strategies",
    )
    # Classes/functions/variables
    + (
        "Metric",
        "DataLoader",
        "DataModule",
        "ReaxDataLoader",
        "Logger",
        "seed_everything",
    )
)

__version__ = "0.2.0"
