"""REAX: A simple training framework for JAX-based projects"""

from . import (
    data,
    exceptions,
    hooks,
    listeners,
    metrics,
    modules,
    optimizers,
    random,
    saving,
    stages,
    strategies,
    training,
)
from .data import DataLoader, DataModule, ReaxDataLoader
from .hooks import *
from .loggers import Logger
from .metrics import Metric
from .modules import *
from .optimizers import *
from .random import *
from .saving import *
from .stages import Stage
from .strategies import *
from .training import *

__all__ = (
    modules.__all__
    + hooks.__all__
    + optimizers.__all__
    + random.__all__
    + saving.__all__
    + strategies.__all__
    + training.__all__
    + data.__all__
    # Modules
    + (
        "data",
        "exceptions",
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
        "Stage",
    ),
)

__version__ = "0.5.0"
