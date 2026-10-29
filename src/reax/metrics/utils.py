from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, TypeVar

import beartype
import jax
import jax.numpy as jnp
import jaxtyping as jt
from jaxtyping import ArrayLike
from typing_extensions import override

from reax.utils import masking

from .. import pytree
from ._metric import Metric

if TYPE_CHECKING:
    import reax

    OptionalMask = reax.types.ArrayMask | None
else:
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
        where = None if mask is None else masking.prepare_mask(values, mask)
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

        where = None if mask is None else masking.prepare_mask(values, mask)
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
        where = None if mask is None else masking.prepare_mask(values, mask)
        return cls(
            accumulator=cls.reduce_fn(values, where=where), count=masking.count_valid(values, where)
        )

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

        where = None if mask is None else masking.prepare_mask(values, mask)
        reduced = cls.reduce_fn(values, where=where)
        return cls(
            accumulator=cls.reduce_fn(cls.merge_fn((self.accumulator, reduced))),
            count=self.count + masking.count_valid(values, where),
        )

    @override
    @jt.jaxtyped(typechecker=beartype.beartype)
    def compute(self) -> jax.Array:
        """Compute function."""
        if self.accumulator is None:
            raise RuntimeError("Nothing to compute, metric is empty!")
        return self.accumulator / self.count
