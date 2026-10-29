from collections.abc import Callable
import math
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, TypeVar

import beartype
import jax
import jax.numpy as jnp
import jaxtyping as jt
from jaxtyping import ArrayLike
from typing_extensions import override

from .. import pytree
from ._metric import Metric

if TYPE_CHECKING:
    import reax

    Mask = reax.types.ArrayMask
    OptionalMask = reax.types.ArrayMask | None
else:
    Mask = Any
    OptionalMask = Any

__all__ = tuple()

M = TypeVar("M", bound=Metric)
_WithAccT = TypeVar("_WithAccT", bound="WithAccumulator")
_WithAccCountT = TypeVar("_WithAccCountT", bound="WithAccumulatorAndCount")


class ReduceFn(Protocol):
    def __call__(
        self, values: ArrayLike, /, axis: Any = None, *, where: ArrayLike | None = None
    ) -> jax.Array:
        """Perform reduction on the passed values (e.g. :func:`jax.numpy.sum`)."""


@jt.jaxtyped(typechecker=beartype.beartype)
def _prepare_mask(
    mask: jt.Bool[ArrayLike, "N"], array: jt.Shaped[ArrayLike, "N ..."]
) -> jt.Bool[jax.Array, "N ..."]:
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
    return jnp.reshape(mask, (jnp.shape(mask)[0], *(1,) * (jnp.ndim(array) - 1)))


@jt.jaxtyped(typechecker=beartype.beartype)
def prepare_mask(values: jt.Shaped[ArrayLike, "N ..."], mask: Mask) -> jt.Bool[jax.Array, "N ..."]:
    """Prepare a mask for use with jnp.where(mask, values, ...).

    Implements a dual-behavior convention:
      - If `mask` is 1D [N]: Treated as an entity mask. It is verified to match the
        leading dimension of `values` and reshaped to [N, 1, ...] so it broadcasts against
        `values`.
      - If `mask` is N-D [N, ...]: Treated as a data-specific mask. Its shape is
        strictly verified to match `values.shape` exactly.

    Callers with an optional mask should check for ``None`` first, e.g.
    ``where = None if mask is None else prepare_mask(values, mask)``.

    Args:
        values: the array the mask will be applied to
        mask: the mask to prepare

    Returns:
        the prepared boolean mask
    """
    values_shape = jnp.shape(values)
    # Cast early to ensure boolean sums and properties are well-behaved downstream
    mask = jnp.asarray(mask, dtype=bool)

    if mask.ndim == 1:
        # 1. Entity Mask Behavior: Shape must be [N]
        if mask.shape[0] != values_shape[0]:
            raise ValueError(
                f"1D entity mask must have the same leading dimension as `values`. "
                f"Received mask shape {mask.shape} and values shape {values_shape}."
            )
        return _prepare_mask(mask, values)

    # 2. Data-Specific Mask Behavior: Shape must be [N, *shape]
    if mask.shape != values_shape:
        raise ValueError(
            f"Multidimensional mask must exactly match `values` shape. "
            f"Received mask shape {mask.shape} and values shape {values_shape}."
        )
    return mask


def count_valid(
    values: jt.ArrayLike, mask: jt.Bool[jax.Array, "N ..."] | None
) -> "int | jt.Int[jax.Array, '']":
    """Count the elements of `values` selected by a mask returned from :func:`prepare_mask`.

    A ``None`` mask selects every element. A broadcast entity mask (shape [N, 1, ...]) counts
    each selected entity once per feature, without materialising the full mask.
    """
    num_values = math.prod(jnp.shape(values))
    if mask is None:
        return num_values
    # Computed at trace time: 1 for a full-shape mask, the feature size for an entity mask
    return mask.sum() * (num_values // mask.size)


def concat(tensors: tuple[jax.Array]) -> jax.Array:
    """Concat function."""
    return jnp.concatenate(tuple(map(jnp.atleast_1d, tensors)))


class WithAccumulator(pytree.PyTree):
    """Abstract aggregation metric."""

    reduce_fn: ClassVar[ReduceFn]
    merge_fn: ClassVar[Callable] = concat

    accumulator: jax.Array | None = None

    @classmethod
    def empty(cls):
        return cls(accumulator=None)

    @classmethod
    def create(
        cls: type[_WithAccT],
        values: jax.typing.ArrayLike,
        mask: OptionalMask = None,
    ) -> _WithAccT:
        """Create function."""
        where = None if mask is None else prepare_mask(values, mask)
        return cls(accumulator=cls.reduce_fn(values, where=where))

    @jt.jaxtyped(typechecker=beartype.beartype)
    def merge(self: _WithAccT, other: _WithAccT) -> _WithAccT:
        """Merge function."""
        if self.accumulator is None:
            return other
        if other.accumulator is None:
            return self

        cls = type(self)
        return cls(accumulator=cls.reduce_fn(cls.merge_fn((self.accumulator, other.accumulator))))

    @jt.jaxtyped(typechecker=beartype.beartype)
    def update(
        self: _WithAccT,
        values: jax.typing.ArrayLike,
        mask: jax.typing.ArrayLike | None = None,
    ) -> _WithAccT:
        """Update function."""
        cls = type(self)
        if self.accumulator is None:
            return cls.create(values, mask=mask)

        where = None if mask is None else prepare_mask(values, mask)
        reduced = cls.reduce_fn(values, where=where)
        return cls(accumulator=cls.reduce_fn(cls.merge_fn((self.accumulator, reduced))))

    @jt.jaxtyped(typechecker=beartype.beartype)
    def compute(self) -> jax.Array:
        """Compute function."""
        if self.accumulator is None:
            raise RuntimeError("Nothing to compute, metric is empty!")
        return self.accumulator


class WithAccumulatorAndCount(WithAccumulator):
    """Helper class to group common functionality for metrics that can keep track using a total and
    count accumulators.
    """

    count: jax.typing.ArrayLike = 0

    @override
    @classmethod
    def create(
        cls: type[_WithAccCountT],
        values: jt.ArrayLike,
        mask: OptionalMask = None,
    ) -> _WithAccCountT:
        """Create function."""
        values = jnp.atleast_1d(values)
        where = None if mask is None else prepare_mask(values, mask)
        return cls(accumulator=cls.reduce_fn(values, where=where), count=count_valid(values, where))

    @override
    @jt.jaxtyped(typechecker=beartype.beartype)
    def merge(self: _WithAccCountT, other: _WithAccCountT) -> _WithAccCountT:
        """Merge function."""
        if self.accumulator is None:
            return other
        if other.accumulator is None:
            return self

        cls = type(self)
        return cls(
            accumulator=cls.reduce_fn(cls.merge_fn((self.accumulator, other.accumulator))),
            count=self.count + other.count,
        )

    @override
    @jt.jaxtyped(typechecker=beartype.beartype)
    def update(
        self: _WithAccCountT,
        values: jax.typing.ArrayLike,
        mask: jax.typing.ArrayLike | None = None,
    ) -> _WithAccCountT:
        """Update function."""
        cls = type(self)
        if self.accumulator is None:
            # This metric is empty
            return cls.create(values, mask=mask)

        where = None if mask is None else prepare_mask(values, mask)
        reduced = cls.reduce_fn(values, where=where)
        return cls(
            accumulator=cls.reduce_fn(cls.merge_fn((self.accumulator, reduced))),
            count=self.count + count_valid(values, where),
        )

    @override
    @jt.jaxtyped(typechecker=beartype.beartype)
    def compute(self) -> jax.Array:
        """Compute function."""
        if self.accumulator is None:
            raise RuntimeError("Nothing to compute, metric is empty!")
        return self.accumulator / self.count
