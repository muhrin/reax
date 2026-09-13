import abc
from typing import Any, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from typing_extensions import override

from reax.utils import arrays

from . import utils
from ._metric import Metric

__all__ = ("Average", "Std", "Min", "Max", "Unique", "NumUnique", "Sum")


class Aggregation(Metric[jax.Array], abc.ABC):
    """Interface that defines an aggregation metric, i.e. one that take raw array-like data and an
    optional mask.
    """

    Self = TypeVar("Self", bound="Aggregation")

    @abc.abstractmethod
    def create(  # pylint: disable=arguments-differ
        self,
        values: jax.typing.ArrayLike,
        mask: jax.typing.ArrayLike | None = None,
    ) -> Self:
        """Create the metric from data."""

    @abc.abstractmethod
    def update(  # pylint: disable=arguments-differ
        self,
        values: jax.typing.ArrayLike,
        mask: jax.typing.ArrayLike | None = None,
    ) -> Self:
        """Update this metric and return an updated instance."""


class Sum(utils.WithAccumulator, Aggregation):
    reduce_fn = jnp.sum


class Min(utils.WithAccumulator, Aggregation):
    reduce_fn = jnp.min


class Max(utils.WithAccumulator, Aggregation):
    reduce_fn = jnp.max


SET_ACCUMULATION_MAX_SIZE = 512
SET_ACCUMULATION_FILL_VALUE = -1e30


class SetAccumulation(Metric[jax.Array]):
    """Get the set of unique values with a fixed memory footprint."""

    # We keep these static so JAX knows the shape of the accumulator at compile time
    max_size: int = eqx.field(static=True, default=SET_ACCUMULATION_MAX_SIZE)
    fill_value: Any = eqx.field(static=True, default=-1)
    _accumulator: jax.Array | None = None

    @property
    def accumulator(self) -> jax.Array:
        if self._accumulator is None:
            raise RuntimeError("Metric is empty!")

        return self._accumulator

    @classmethod
    def empty(cls, max_size=SET_ACCUMULATION_MAX_SIZE, fill_value=None) -> "SetAccumulation":
        # Leave fill_value as None so 'create' can infer it on the first update
        return cls(_accumulator=None, max_size=max_size, fill_value=fill_value)

    @classmethod
    def create(
        cls,
        values: jax.typing.ArrayLike,
        mask: jax.typing.ArrayLike | None = None,
        max_size: int = SET_ACCUMULATION_MAX_SIZE,
        fill_value: Any = None,
    ) -> "SetAccumulation":
        np_ = arrays.infer_backend([values])
        val_arr = np_.array(values)

        if fill_value is None:
            if jnp.issubdtype(val_arr.dtype, jnp.floating):
                # Use a large finite negative number to avoid NaN gradient poisoning
                fill_value = SET_ACCUMULATION_FILL_VALUE
            else:
                fill_value = jnp.iinfo(val_arr.dtype).max

        acc = cls._get_unique_fixed(np_, val_arr, max_size, fill_value, where=mask)
        return cls(_accumulator=acc, max_size=max_size, fill_value=fill_value)

    @override
    def update(self, values, mask=None) -> "SetAccumulation":
        if self._accumulator is None:
            return self.create(
                values, mask=mask, max_size=self.max_size, fill_value=self.fill_value
            )

        # 1. Align dtypes
        values = values.astype(self.accumulator.dtype)
        fv_cast = self._cast_sentinel(self.accumulator)

        if mask is not None:
            values = jnp.where(mask, values, fv_cast)

        combined = jnp.concatenate([self.accumulator, values])

        # 3. Re-uniquify the combined buffer
        new_acc = self._get_unique_fixed(jnp, combined, self.max_size, self.fill_value)

        return eqx.tree_at(
            lambda m: m._accumulator, self, new_acc  # pylint: disable=protected-access
        )

    @override
    def merge(self, other: "SetAccumulation") -> "SetAccumulation":
        if self._accumulator is None:
            # Ensure we return an instance with OUR config, even if the other one is None
            return self
        if other._accumulator is None:  # pylint: disable=protected-access
            return self

        np_ = arrays.infer_backend([self.accumulator, other.accumulator])
        combined = np_.concatenate([self.accumulator, other.accumulator])
        new_acc = self._get_unique_fixed(np_, combined, self.max_size, self.fill_value)
        return eqx.tree_at(lambda m: m.accumulator, self, new_acc)

    @override
    def compute(self) -> jax.Array:
        # Returns the actual set (ignoring the sentinel padding)
        sentinel = self._cast_sentinel(self.accumulator)
        if jnp.issubdtype(self.accumulator.dtype, jnp.floating):
            mask = jnp.logical_not(jnp.isclose(self.accumulator, sentinel))
        else:
            mask = self.accumulator != sentinel

        return self.accumulator[mask]  # pylint: disable=unsubscriptable-object

    @override
    def reduce(self, axis: int = 0) -> "SetAccumulation":
        """Collapses a vectorized (vmapped) metric into a single instance using scan."""
        data = jnp.moveaxis(self.accumulator, axis, 0)
        init_val = jnp.full((self.max_size,), self.fill_value, dtype=data.dtype)

        def _scan_op(acc, next_row):
            combined = jnp.concatenate([acc, next_row])
            res = self._get_unique_fixed(jnp, combined, self.max_size, self.fill_value)
            return res, None

        final_acc, _ = jax.lax.scan(_scan_op, init_val, data)
        return type(self)(
            _accumulator=final_acc, max_size=self.max_size, fill_value=self.fill_value
        )

    def saturation(self) -> jax.Array:
        # 5. FIX: Ensure saturation uses the same robust mask as compute
        sentinel = self._cast_sentinel(self.accumulator)
        if jnp.issubdtype(self.accumulator.dtype, jnp.floating):
            mask = jnp.logical_not(jnp.isclose(self.accumulator, sentinel))
        else:
            mask = self.accumulator != sentinel
        return jnp.sum(mask).astype(jnp.float32) / self.max_size

    @staticmethod
    def _get_unique_fixed(np_, values, max_size, fill_value, where=None):
        # Cast fill_value to match data exactly to prevent comparison drift
        fv_cast = np_.array(fill_value, dtype=values.dtype)

        if where is not None:
            where = utils.prepare_mask(values, where)
            values = np_.where(where, values, fv_cast)

        if np_ is jnp:
            # size=max_size forces the output to be a static-shaped (max_size,) array
            return jnp.unique(values, size=max_size, fill_value=fv_cast)

        # Fallback for standard NumPy
        uniques = np_.unique(values)
        res = np_.full((max_size,), fill_value, dtype=values.dtype)
        actual = uniques[uniques != fill_value]
        limit = min(len(actual), max_size)
        res[:limit] = actual[:limit]
        return res

    def _cast_sentinel(self, array: jax.Array) -> jax.Array:
        return jnp.array(self.fill_value, dtype=array.dtype)


class Unique(SetAccumulation):
    """Get the set of unique values."""


class NumUnique(SetAccumulation):
    """Count the number of unique values."""

    @override
    def compute(self) -> jax.Array:
        sentinel = self._cast_sentinel(self.accumulator)

        # Use the robust masking logic
        if jnp.issubdtype(self.accumulator.dtype, jnp.floating):
            # For float32, -1e30 != -1.2799...e+34 without isclose
            mask = jnp.logical_not(jnp.isclose(self.accumulator, sentinel))
        else:
            mask = self.accumulator != sentinel

        # We return an integer for a count
        return jnp.sum(mask).astype(jnp.int32)


class Average(utils.WithAccumulatorAndCount, Aggregation):
    """Compute an average"""

    reduce_fn = jnp.sum


class Std(Aggregation):
    """Calculate standard deviation."""

    total: jax.Array
    sum_of_squares: jax.Array
    count: jax.Array

    def __init__(
        self, total: jax.Array = None, sum_of_squares: jax.Array = None, count: jax.Array = None
    ):
        super().__init__()
        # State
        # Use explicit 'is None' checks to keep JAX happy
        self.total = total if total is not None else jnp.array(0, jnp.float32)
        self.sum_of_squares = (
            sum_of_squares if sum_of_squares is not None else jnp.array(0, jnp.float32)
        )
        self.count = count if count is not None else jnp.array(0, jnp.int32)

    @override
    @classmethod
    def empty(cls) -> "Metric[jax.Array]":
        return cls()

    @override
    @classmethod
    def create(  # pylint: disable=arguments-differ
        cls,
        values: jax.typing.ArrayLike,
        mask: jax.typing.ArrayLike | None = None,
    ) -> "Std":
        """Create the metric from data."""
        if values.ndim == 0:
            values = values[None]
        if mask is None:
            mask = jnp.ones(values.shape[0], dtype=jnp.bool)

        mask, num_elements = utils.prepare_mask(values, mask, return_count=True)
        return cls(
            total=values.sum(),
            sum_of_squares=jnp.where(mask, values**2, jnp.zeros_like(values)).sum(),
            count=num_elements,
        )

    def update(
        # pylint: disable=arguments-differ
        self,
        values: jax.Array,
        mask: jax.Array | None = None,
    ) -> "Std":
        """Update function."""
        if values.ndim == 0:
            values = values[None]
        if mask is None:
            mask = jnp.ones(values.shape[0], dtype=jnp.bool)

        mask, num_elements = utils.prepare_mask(values, mask, return_count=True)
        return type(self)(
            total=self.total + values.sum(),
            sum_of_squares=self.sum_of_squares
            + jnp.where(mask, values**2, jnp.zeros_like(values)).sum(),
            count=self.count + num_elements,
        )

    def merge(self, other: "Std") -> "Std":
        """Merge function."""
        return type(self)(
            total=self.total + other.total,
            sum_of_squares=self.sum_of_squares + other.sum_of_squares,
            count=self.count + other.count,
        )

    def compute(self) -> jax.Array:
        """Compute function."""
        # var(X) = 1/N \sum_i (x_i - mean)^2
        #        = 1/N \sum_i (x_i^2 - 2 x_i mean + mean^2)
        #        = 1/N ( \sum_i x_i^2 - 2 mean \sum_i x_i + N * mean^2 )
        #        = 1/N ( \sum_i x_i^2 - 2 mean N mean + N * mean^2 )
        #        = 1/N ( \sum_i x_i^2 - N * mean^2 )
        #        = \sum_i x_i^2 / N - mean^2
        mean = self.total / self.count
        variance = self.sum_of_squares / self.count - mean**2
        # Mathematically, variance can never be negative but in reality we may run
        # into such issues for numerical reasons.
        variance = jnp.clip(variance, min=0.0)
        return variance**0.5
