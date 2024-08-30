from typing import ClassVar, Optional, Protocol, TypeVar

import beartype
import clu.internal.utils
import equinox
import jax
import jax.numpy as jnp
import jaxtyping as jt

from .metric import Metric

__all__ = tuple()

M = TypeVar("M", bound=Metric)


class ReduceFn(Protocol):
    def __call__(self, values: jax.Array, where: jax.Array = None) -> jax.Array:
        """Perform reduction on the passed values"""


def _prepare_mask(
    mask: jt.Bool[jax.Array, "n_elements"], array: jt.Float[jax.Array, "..."]
) -> jt.Float[jax.Array, "n_elements ..."]:
    """
    Prepare a mask for use with jnp.where(mask, array, ...).  This needs to be done to make sure the
    mask is of the right shape to be compatible with such an operation.  The other alternative is

        ``jnp.where(mask, array.T, ...).T``

    but this sometimes leads to creating a copy when doing one or both of the transposes.  I'm not
    sure why, but this approach seems to avoid the problem.

    :param mask: the mask to prepare
    :param array: the array the mask will be applied to
    :return: the prepared mask, typically this is just padded with extra dimensions (or reduced)
    """
    return mask.reshape(-1, *(1,) * len(array.shape[1:]))


@jt.jaxtyped(typechecker=beartype.beartype)
def prepare_mask(values: jax.typing.ArrayLike, mask: Optional[jax.typing.ArrayLike] = None):
    if mask is None:
        return None

    mask = _prepare_mask(mask, values)

    # Leading dimensions of mask and predictions must match.
    if mask.shape[0] != values.shape[0]:
        raise ValueError(
            f"Argument `mask` must have the same leading dimension as `values`. "
            f"Received mask of dimension {mask.shape} "
            f"and values of dimension {values.shape}."
        )

    clu.internal.utils.check_param(mask, dtype=bool, ndim=values.ndim)
    return mask


def concat(tensors: tuple[jax.Array]) -> jax.Array:
    return jnp.concatenate(tuple(map(jnp.atleast_1d, tensors)))


class WithAccumulator(equinox.Module):
    """Abstract aggregation metric."""

    Self = TypeVar("Self", bound="WithAccumulator")

    reduce_fn: ClassVar[ReduceFn]
    merge_fn = concat
    accumulator: jax.Array = None

    @classmethod
    def create(
        cls,
        values: jax.typing.ArrayLike,
        mask: Optional[jax.typing.ArrayLike] = None,
    ) -> Self:
        mask = prepare_mask(values, mask)
        return cls(accumulator=cls.reduce_fn(values, where=mask))

    @jt.jaxtyped(typechecker=beartype.beartype)
    def merge(self, other: Self) -> Self:
        if self.accumulator is None:
            return other

        cls = type(self)
        return cls(accumulator=cls.reduce_fn(cls.merge_fn((self.accumulator, other.accumulator))))

    @jt.jaxtyped(typechecker=beartype.beartype)
    def update(
        self,
        values: jax.typing.ArrayLike,
        mask: Optional[jax.typing.ArrayLike] = None,
    ) -> Self:
        cls = type(self)
        if self.accumulator is None:
            return cls.create(values, mask=mask)

        mask = prepare_mask(values, mask)
        reduced = cls.reduce_fn(values, where=mask)
        return cls(accumulator=cls.reduce_fn(cls.merge_fn((self.accumulator, reduced))))

    @jt.jaxtyped(typechecker=beartype.beartype)
    def compute(self) -> jax.Array:
        return self.accumulator


class WithAccumulatorAndCount(WithAccumulator):
    """
    Helper class to group common functionality for metrics that can keep track using a total and
    count accumulators
    """

    Self = TypeVar("Self", bound="WithAccumulator")

    count: jax.Array = 0

    @classmethod
    def create(
        cls,
        values: jax.typing.ArrayLike,
        mask: Optional[jax.typing.ArrayLike] = None,
    ) -> Self:
        mask = prepare_mask(values, mask)
        count = jnp.sum(mask) if mask is not None else values.size
        return cls(accumulator=cls.reduce_fn(values, where=mask), count=count)

    @jt.jaxtyped(typechecker=beartype.beartype)
    def merge(self, other: Self) -> Self:
        if self.count == 0:
            return other

        cls = type(self)
        return cls(
            accumulator=cls.reduce_fn(cls.merge_fn((self.accumulator, other.accumulator))),
            count=self.count + other.count,
        )

    @jt.jaxtyped(typechecker=beartype.beartype)
    def update(
        self,
        values: jax.typing.ArrayLike,
        mask: Optional[jax.typing.ArrayLike] = None,
    ) -> Self:
        cls = type(self)
        if self.count == 0:
            # This metric is empty
            return cls.create(values, mask=mask)

        mask = prepare_mask(values, mask)
        reduced = cls.reduce_fn(values, where=mask)
        count = jnp.sum(mask) if mask is not None else values.size
        return cls(
            accumulator=cls.reduce_fn(cls.merge_fn((self.accumulator, reduced))),
            count=self.count + count,
        )

    @jt.jaxtyped(typechecker=beartype.beartype)
    def compute(self) -> jax.Array:
        return self.accumulator / self.count
