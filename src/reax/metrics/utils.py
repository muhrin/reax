from collections.abc import Callable
import math
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, TypeVar

import beartype
import equinox
import jax
import jax.numpy as jnp
import jaxtyping as jt
from jaxtyping import ArrayLike
from typing_extensions import override

from ._metric import Metric

if TYPE_CHECKING:
    import reax

    OptionalMask = reax.types.ArrayMask | None
else:
    OptionalMask = Any

__all__ = tuple()

M = TypeVar("M", bound=Metric)


class ReduceFn(Protocol):
    def __call__(self, values: ArrayLike, where: ArrayLike | None = None) -> ArrayLike:
        """Perform reduction on the passed values."""


@jt.jaxtyped(typechecker=beartype.beartype)
def _prepare_mask(
    mask: jt.Bool[ArrayLike, "N"], array: jt.Shaped[ArrayLike, "N ..."]
) -> jt.Bool[ArrayLike, "N ..."]:
    """Prepare a 1D entity mask by expanding its dimensions to broadcast against `array`.

    This relies on zero-copy reshaping. Appending size-1 dimensions allows JAX/NumPy
    to automatically broadcast the mask across feature dimensions in `jnp.where`
    without allocating memory for a dense N-dimensional boolean tensor.

    Args:
        mask: The 1D entity mask to prepare.
        array: The array the mask will be applied to.

    Returns:
        The prepared mask, padded with extra trailing dimensions of size 1.
    """
    return mask.reshape(mask.shape[0], *(1,) * (len(array.shape) - 1))


@jt.jaxtyped(typechecker=beartype.beartype)
def prepare_mask(
    values: jt.Shaped[ArrayLike, "N ..."], mask: OptionalMask = None, *, return_count: bool = False
) -> "OptionalMask | tuple[OptionalMask, int | jt.Int[ArrayLike, '']]":
    """Prepare a mask for use with jnp.where(mask, values, ...).

    Implements a dual-behavior convention:
      - If `mask` is 1D [N]: Treated as an entity mask. It is verified to match the
        leading dimension of `values` and broadcasted up to [N, *values_dims].
      - If `mask` is N-D [N, ...]: Treated as a data-specific mask. Its shape is
        strictly verified to match `values.shape` exactly.

    Args:
        values: the array the mask will be applied to
        mask: the mask to prepare
        return_count: if ``True``, returns a tuple where the second
            element is the number of valid (True) elements across all dimensions.

    Returns:
        the prepared mask, and optionally the count of masked elements.
    """
    if mask is None:
        if return_count:
            return None, values.size
        return None

    # Cast early to ensure boolean sums and properties are well-behaved downstream
    mask = mask.astype(bool)

    if mask.ndim == 1:
        # 1. Entity Mask Behavior: Shape must be [N]
        if mask.shape[0] != values.shape[0]:
            raise ValueError(
                f"1D entity mask must have the same leading dimension as `values`. "
                f"Received mask shape {mask.shape} and values shape {values.shape}."
            )

        mask = _prepare_mask(mask, values)

        if return_count:
            # math.prod computes at trace-time natively. This avoids the overhead
            # of allocating an intermediate JAX/NumPy array just to compute the product.
            feature_size = math.prod(values.shape[1:])
            count = mask.sum() * feature_size
            return mask, count

    else:
        # 2. Data-Specific Mask Behavior: Shape must be [N, *shape]
        if mask.shape != values.shape:
            raise ValueError(
                f"Multidimensional mask must exactly match `values` shape. "
                f"Received mask shape {mask.shape} and values shape {values.shape}."
            )

        if return_count:
            return mask, mask.sum()

    return mask


def concat(tensors: tuple[jax.Array]) -> jax.Array:
    """Concat function."""
    return jnp.concatenate(tuple(map(jnp.atleast_1d, tensors)))


class WithAccumulator(equinox.Module):
    """Abstract aggregation metric."""

    Self = TypeVar("Self", bound="WithAccumulator")

    reduce_fn: ClassVar[ReduceFn]
    merge_fn: ClassVar[Callable] = concat

    accumulator: jax.Array | None = None

    @classmethod
    def empty(cls):
        return cls(accumulator=None)

    @classmethod
    def create(
        cls,
        values: jax.typing.ArrayLike,
        mask: OptionalMask = None,
    ) -> Self:
        """Create function."""
        mask = prepare_mask(values, mask)
        return cls(accumulator=cls.reduce_fn(values, where=mask))

    @jt.jaxtyped(typechecker=beartype.beartype)
    def merge(self, other: Self) -> Self:
        """Merge function."""
        if self.accumulator is None:
            return other

        cls = type(self)
        return cls(accumulator=cls.reduce_fn(cls.merge_fn((self.accumulator, other.accumulator))))

    @jt.jaxtyped(typechecker=beartype.beartype)
    def update(
        self,
        values: jax.typing.ArrayLike,
        mask: jax.typing.ArrayLike | None = None,
    ) -> Self:
        """Update function."""
        cls = type(self)
        if self.accumulator is None:
            return cls.create(values, mask=mask)

        mask = prepare_mask(values, mask)
        reduced = cls.reduce_fn(values, where=mask)
        return cls(accumulator=cls.reduce_fn(cls.merge_fn((self.accumulator, reduced))))

    @jt.jaxtyped(typechecker=beartype.beartype)
    def compute(self) -> jax.Array:
        """Compute function."""
        return self.accumulator


class WithAccumulatorAndCount(WithAccumulator):
    """Helper class to group common functionality for metrics that can keep track using a total and
    count accumulators.
    """

    Self = TypeVar("Self", bound="WithAccumulator")

    count: jax.Array = 0

    @override
    @classmethod
    def create(
        cls,
        values: jt.ArrayLike,
        mask: OptionalMask = None,
    ) -> Self:
        """Create function."""
        values = jnp.atleast_1d(values)
        mask, num_elements = prepare_mask(values, mask, return_count=True)
        return cls(accumulator=cls.reduce_fn(values, where=mask), count=num_elements)

    @override
    @jt.jaxtyped(typechecker=beartype.beartype)
    def merge(self, other: Self) -> Self:
        """Merge function."""
        cls = type(self)
        return cls(
            accumulator=cls.reduce_fn(cls.merge_fn((self.accumulator, other.accumulator))),
            count=self.count + other.count,
        )

    @override
    @jt.jaxtyped(typechecker=beartype.beartype)
    def update(
        self,
        values: jax.typing.ArrayLike,
        mask: jax.typing.ArrayLike | None = None,
    ) -> Self:
        """Update function."""
        cls = type(self)
        if self.accumulator is None:
            # This metric is empty
            return cls.create(values, mask=mask)

        mask, num_elements = prepare_mask(values, mask, return_count=True)
        reduced = cls.reduce_fn(values, where=mask)
        return cls(
            accumulator=cls.reduce_fn(cls.merge_fn((self.accumulator, reduced))),
            count=self.count + num_elements,
        )

    @override
    @jt.jaxtyped(typechecker=beartype.beartype)
    def compute(self) -> jax.Array:
        """Compute function."""
        return self.accumulator / self.count
