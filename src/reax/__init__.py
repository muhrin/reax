"""REAX: A simple training framework for JAX-based projects"""

from . import data, listeners, metrics, modules, stages, training
from .data import DataLoader, DataModule, ReaxDataLoader
from .metrics import Metric
from .modules import *
from .training import *

__all__ = (
    modules.__all__
    + training.__all__
    + data.__all__
    # Modules
    + (
        "data",
        "stages",
        "listeners",
        "metrics",
    )
    # Classes/functions/variables
    + (
        "Metric",
        "DataLoader",
        "DataModule",
        "ReaxDataLoader",
    )
)


__version__ = "0.1.0"
