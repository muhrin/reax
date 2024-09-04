"""REAX: A simple training framework for JAX-based projects"""

# Set default device to CPU so that things don't get loaded onto the device until we want them to be
# flake8: noqa
# pylint: disable=wrong-import-position
# import jax
#
# jax.config.update("jax_platform_name", "cpu")

from . import data, listeners, metrics, modules, optimizers, stages, strategies, training
from .data import DataLoader, DataModule, ReaxDataLoader
from .metrics import Metric
from .modules import *
from .optimizers import *
from .strategies import *
from .training import *

__all__ = (
    modules.__all__
    + optimizers.__all__
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
    )
)

__version__ = "0.1.0"
