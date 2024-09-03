import abc
from typing import Optional, TypeVar

import jax
import jax.numpy as jnp

from . import _registry, utils
from .metric import Metric

__all__ = ("Average", "Std", "Min", "Max", "Unique", "NumUnique")


class Aggregation(Metric[jax.Array], metaclass=abc.ABCMeta):
    """
    Interface that defines an aggregation metric.  This takes a single array and possibly as ask
    and calculates a metric.
    """

    Self = TypeVar("Self", bound="Aggregation")

    @abc.abstractmethod
    def create(  # pylint: disable=arguments-differ
        self,
        values: jax.typing.ArrayLike,
        mask: Optional[jax.typing.ArrayLike] = None,
    ) -> Self:
        """Create the metric from data"""

    @abc.abstractmethod
    def update(  # pylint: disable=arguments-differ
        self,
        values: jax.typing.ArrayLike,
        mask: Optional[jax.typing.ArrayLike] = None,
    ) -> Self:
        """Update this metric and return an updated instance"""


class Sum(utils.WithAccumulator, Aggregation):
    reduce_fn = jnp.sum


class Min(utils.WithAccumulator, Aggregation):
    reduce_fn = jnp.min


class Max(utils.WithAccumulator, Aggregation):
    reduce_fn = jnp.max


class Unique(utils.WithAccumulator, Aggregation):
    """
    Get the set of unique values.

    warning: this cannot be used with JAX jit because it relies on dynamically-sized arrays
    """

    @staticmethod
    def reduce_fn(values, where=None):
        return jnp.unique(values[where] if where is not None else values)


class NumUnique(utils.WithAccumulator, Aggregation):
    """Count the number of unique values

    warning: this cannot be used with JAX jit because it relies on dynamically-sized arrays
    """

    @staticmethod
    def reduce_fn(values, where=None):
        return jnp.unique(values[where] if where is not None else values)

    def compute(self) -> jax.Array:
        return jnp.asarray(jnp.size(self.accumulator))


class Average(utils.WithAccumulatorAndCount, Aggregation):
    reduce_fn = jnp.sum


class Std(Aggregation):
    """
    Calculate standard deviation
    """

    total: jax.Array
    sum_of_squares: jax.Array
    count: jax.Array

    def __init__(
        self, total: jax.Array = None, sum_of_squares: jax.Array = None, count: jax.Array = None
    ):
        super().__init__()
        self.total = total or jnp.array(0, jnp.float32)
        self.sum_of_squares = sum_of_squares or jnp.array(0, jnp.float32)
        self.count = count or jnp.array(0, jnp.int32)

    def create(
        self,
        values: jax.typing.ArrayLike,
        mask: Optional[jax.typing.ArrayLike] = None,
    ) -> "Std":
        """Create the metric from data"""
        if values.ndim == 0:
            values = values[None]
        if mask is None:
            mask = jnp.ones(values.shape[0], dtype=jnp.bool)

        mask = utils.prepare_mask(values, mask)
        return type(self)(
            total=values.sum(),
            sum_of_squares=jnp.where(mask, values**2, jnp.zeros_like(values)).sum(),
            count=mask.sum(),
        )

    def update(
        # pylint: disable=arguments-differ
        self,
        values: jax.Array,
        mask: Optional[jax.Array] = None,
    ) -> "Std":
        if values.ndim == 0:
            values = values[None]
        if mask is None:
            mask = jnp.ones(values.shape[0], dtype=jnp.bool)

        mask = utils.prepare_mask(values, mask)
        return type(self)(
            total=self.total + values.sum(),
            sum_of_squares=self.sum_of_squares
            + jnp.where(mask, values**2, jnp.zeros_like(values)).sum(),
            count=self.count + mask.sum(),
        )

    def merge(self, other: "Std") -> "Std":
        return type(self)(
            total=self.total + other.total,
            sum_of_squares=self.sum_of_squares + other.sum_of_squares,
            count=self.count + other.count,
        )

    def compute(self) -> jax.Array:
        # var(X) = 1/N \sum_i (x_i - mean)^2
        #        = 1/N \sum_i (x_i^2 - 2 x_i mean + mean^2)
        #        = 1/N ( \sum_i x_i^2 - 2 mean \sum_i x_i + N * mean^2 )
        #        = 1/N ( \sum_i x_i^2 - 2 mean N mean + N * mean^2 )
        #        = 1/N ( \sum_i x_i^2 - N * mean^2 )
        #        = \sum_i x_i^2 / N - mean^2
        mean = self.total / self.count
        variance = self.sum_of_squares / self.count - mean**2
        # Mathematically variance can never be negative but in reality we may run
        # into such issues due to numeric reasons.
        variance = jnp.clip(variance, min=0.0)
        return variance**0.5


_registry.get_registry().register_many(
    {
        "mean": Average,
        "min": Min,
        "max": Max,
        "num_unique": NumUnique,
        "unique": Unique,
        "std": Std,
        "sum": Sum,
    }
)
