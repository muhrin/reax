import jax
import jax.numpy as jnp

from . import _metric, aggregation

__all__ = "MeanSquaredError", "RootMeanSquareError", "MeanAbsoluteError", "LeastSquaresEstimate"


class MeanSquaredError(_metric.FromFun):
    metric = aggregation.Average()

    def func(self, values, targets, mask=None):
        return jnp.square(values - targets), mask


class MeanAbsoluteError(_metric.FromFun):
    metric = aggregation.Average()

    def func(self, values, targets, mask=None):
        return jnp.abs(values - targets), mask


class RootMeanSquareError(_metric.Metric):
    mse: MeanSquaredError

    def __init__(self, mse: MeanSquaredError = None):
        """Init function."""
        super().__init__()
        self.mse = mse

    @property
    def is_empty(self) -> bool:
        """Is empty."""
        return self.mse is None

    def create(  # pylint: disable=arguments-differ
        self,
        values: jax.Array,
        targets: jax.Array,
        mask: jax.Array | None = None,
    ) -> "RootMeanSquareError":
        """Create function."""
        return type(self)(mse=MeanSquaredError().create(values, targets, mask))

    def update(  # pylint: disable=arguments-differ
        self,
        values: jax.Array,
        targets: jax.Array,
        mask: jax.Array | None = None,
    ) -> "RootMeanSquareError":
        """Update function."""
        if self.is_empty:
            return self.create(values, targets, mask)

        return type(self)(mse=self.mse.update(values, targets, mask=mask))

    def merge(self, other: "RootMeanSquareError") -> "RootMeanSquareError":
        """Merge function."""
        if self.is_empty:
            return other
        if other.is_empty:
            return self

        return type(self)(mse=self.mse.merge(other.mse))

    def compute(self) -> jax.Array:
        """Compute function."""
        return jnp.sqrt(self.mse.compute())


class LeastSquaresEstimate(_metric.Metric):
    values: jax.Array | None
    targets: jax.Array | None

    def __init__(self, values: jax.Array = None, targets: jax.Array = None):
        """Init function."""
        super().__init__()
        self.values = values
        self.targets = targets

    @property
    def is_empty(self) -> bool:
        """Is empty."""
        return self.values is None

    def create(  # pylint: disable=arguments-differ
        self, inputs: jax.Array, outputs: jax.Array
    ) -> "LeastSquaresEstimate":
        """Create function."""
        return LeastSquaresEstimate(inputs, outputs)

    def update(  # pylint: disable=arguments-differ
        self, inputs: jax.Array, outputs: jax.Array
    ) -> "LeastSquaresEstimate":
        """Update function."""
        if self.is_empty:
            return self.create(inputs, outputs)

        return type(self)(values=inputs, targets=outputs)

    def merge(self, other: "LeastSquaresEstimate") -> "LeastSquaresEstimate":
        """Merge function."""
        if self.is_empty:
            return other
        if other.is_empty:
            return self

        return LeastSquaresEstimate(
            values=jnp.concatenate((self.values, other.values)),
            targets=jnp.concatenate((self.targets, other.targets)),
        )

    def compute(self) -> jax.Array:
        """Compute function."""
        return jnp.linalg.lstsq(self.values, self.targets)[0]
