from typing import Any

import jax
import jax.numpy as jnp
from typing_extensions import override

from . import _metric, aggregation

__all__ = (
    "MeanSquaredError",
    "RootMeanSquareError",
    "MeanAbsoluteError",
    "LeastSquaresEstimate",
)

MeanSquaredError = aggregation.Average.from_fun(
    lambda values, targets, mask=None: (jnp.square(values - targets), mask),
    name="SquaredError",
)


MeanAbsoluteError = aggregation.Average.from_fun(
    lambda values, targets, mask=None: (jnp.abs(values - targets), mask),
    name="AbsoluteError",
)


class RootMeanSquareError(_metric.Metric[jax.Array]):
    # A `MeanSquaredError` instance, or `None` if empty
    mse: _metric.Metric[jax.Array] | None

    def __init__(self, mse: _metric.Metric[jax.Array] | None = None):
        """Init function."""
        super().__init__()
        # State
        self.mse = mse

    @property
    def is_empty(self) -> bool:
        """Is empty."""
        return self.mse is None

    @classmethod
    @override
    def create(
        cls,
        values: jax.typing.ArrayLike,
        targets: jax.typing.ArrayLike,
        mask: jax.typing.ArrayLike | None = None,
    ) -> "RootMeanSquareError":
        """Create function."""
        return cls(mse=MeanSquaredError.create(values, targets, mask))

    @override
    def update(
        self,
        values: jax.typing.ArrayLike,
        targets: jax.typing.ArrayLike,
        mask: jax.typing.ArrayLike | None = None,
    ) -> "RootMeanSquareError":
        """Update function."""
        if self.mse is None:
            return self.create(values, targets, mask)

        return type(self)(mse=self.mse.update(values, targets, mask=mask))

    @override
    def merge(self, other: "RootMeanSquareError") -> "RootMeanSquareError":
        """Merge function."""
        if self.mse is None:
            return other
        if other.mse is None:
            return self

        return type(self)(mse=self.mse.merge(other.mse))

    @override
    def compute(self) -> jax.Array:
        """Compute function."""
        if self.mse is None:
            raise RuntimeError("Nothing to compute, metric is empty!")
        return jnp.sqrt(self.mse.compute())


def _concat_reduce(
    x: Any, /, axis: int = 0, *, where: jax.typing.ArrayLike | None = None
) -> jax.Array:
    # Flatten the batch dimension into the data dimension: (Batch, N, D) -> (Batch * N, D)
    del where  # accepted to match the ReduceFn protocol, masking isn't supported here
    return jnp.concatenate(x, axis=axis)


class LeastSquaresEstimate(_metric.Metric[jax.Array]):
    values: jax.Array | None
    targets: jax.Array | None

    def __init__(self, values: jax.Array | None = None, targets: jax.Array | None = None):
        """Init function."""
        super().__init__()
        self.values = values
        self.targets = targets

    @property
    def is_empty(self) -> bool:
        """Is empty."""
        return self.values is None

    @classmethod
    @override
    def create(cls, inputs: jax.Array, outputs: jax.Array) -> "LeastSquaresEstimate":
        """Create function."""
        return cls(inputs, outputs)

    @override
    def update(self, inputs: jax.Array, outputs: jax.Array) -> "LeastSquaresEstimate":
        """Update function."""
        if self.values is None or self.targets is None:
            return self.create(inputs, outputs)

        return type(self)(
            values=jnp.concatenate((self.values, inputs)),
            targets=jnp.concatenate((self.targets, outputs)),
        )

    @override
    def merge(self, other: "LeastSquaresEstimate") -> "LeastSquaresEstimate":
        """Merge function."""
        if self.values is None or self.targets is None:
            return other
        if other.values is None or other.targets is None:
            return self

        return type(self)(
            values=jnp.concatenate((self.values, other.values)),
            targets=jnp.concatenate((self.targets, other.targets)),
        )

    @override
    def compute(self) -> jax.Array:
        """Compute function."""
        if self.values is None or self.targets is None:
            raise RuntimeError("Nothing to compute, metric is empty!")
        return jnp.linalg.lstsq(self.values, self.targets)[0]

    reduce_fn = staticmethod(_concat_reduce)
