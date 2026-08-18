import pathlib
from typing import Literal, Protocol, TypeVar, runtime_checkable

import jax.typing
import jaxtyping as jt

_OutT = TypeVar("_OutT")


Path = str | bytes | pathlib.Path
ArrayMask = jt.Int[jt.ArrayLike, "..."] | jt.Bool[jt.ArrayLike, "..."]
MetricsDict = dict[str, jax.typing.ArrayLike]

#: How a raw (i.e. non-metric) value has already been reduced over its batch by the caller.
#: ``"sum"`` means the value is the total over the batch, ``"mean"`` that it is the average over the
#: ``batch_size`` samples it was computed from.  In both cases the epoch value is the mean over all
#: the samples seen, but we can only work that out if we are told which of the two we were given.
ReduceFx = Literal["sum", "mean"]


@runtime_checkable
class MetricType(Protocol[_OutT]):
    """A protocol that defines metric types

    For something to be a metric type it needs to be able to create metric instances either in
    empty form, or using data supplied by the user.  In this sense, a metric type can be seen as
    being like a factory for metrics.
    """

    def empty(self) -> "MetricInstance[_OutT]":
        """Return an empty version of this metric"""

    def create(self, *args, **kwargs) -> "MetricInstance[_OutT]":
        """Create a new metric instance from the passed data."""


@runtime_checkable
class MetricInstance(MetricType[_OutT], Protocol[_OutT]):
    """A protocol that defines a metric instance that can be updated with new data,
    merged with another instance of the same type or be used to compute the value of the metric."""

    def update(self, *args, **kwargs) -> "MetricInstance[_OutT]":
        """Update the metric from new data and return a new instance."""

    def merge(self, other: "MetricInstance[_OutT]") -> "MetricInstance[_OutT]":
        """Merge the metric with data from another metric instance of the same type."""

    def compute(self) -> _OutT:
        """Compute the metric."""
