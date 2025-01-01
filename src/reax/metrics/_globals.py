import functools
from typing import TYPE_CHECKING

from . import _metric, _registry

if TYPE_CHECKING:
    import reax

__all__ = ("get",)


@functools.singledispatch
def get(metric) -> "reax.Metric":
    """Get function."""
    try:
        if issubclass(metric, _metric.Metric):
            return metric()
    except TypeError:
        pass

    raise TypeError(f"Cannot get metric, got {type(metric).__name__}")


@get.register
def get_str(metric: str):
    """Get str."""
    return _registry.get(metric)


@get.register(_metric.Metric)
def get_metric(metric: "reax.Metric"):
    """Get metric."""
    return metric
