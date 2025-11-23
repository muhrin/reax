import jax
import jax.numpy as jnp
from typing_extensions import override

from . import _metric, aggregation

__all__ = "MeanSquaredError", "RootMeanSquareError", "MeanAbsoluteError", "LeastSquaresEstimate"

MeanSquaredError = aggregation.Average.from_fun(
    lambda values, targets, mask=None: (jnp.square(values - targets), mask),
    name="SquaredError",
)


MeanAbsoluteError = aggregation.Average.from_fun(
    lambda values, targets, mask=None: (jnp.abs(values - targets), mask),
    name="AbsoluteError",
)


class RootMeanSquareError(_metric.Metric):
    mse: MeanSquaredError

    def __init__(self, mse: MeanSquaredError = None):
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
    def create(  # pylint: disable=arguments-differ
        cls, values: jax.Array, targets: jax.Array, mask: jax.Array | None = None, /
    ) -> "RootMeanSquareError":
        """Create function."""
        return cls(mse=MeanSquaredError().create(values, targets, mask))

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


class LeastSquaresEstimate(_metric.Metric[jax.Array]):
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

    @classmethod
    @override
    def empty(cls) -> "LeastSquaresEstimate":
        return cls()

    @classmethod
    @override
    def create(  # pylint: disable=arguments-differ
        cls, inputs: jax.Array, outputs: jax.Array
    ) -> "LeastSquaresEstimate":
        """Create function."""
        return cls(inputs, outputs)

    @override
    def update(  # pylint: disable=arguments-differ
        self, inputs: jax.Array, outputs: jax.Array
    ) -> "LeastSquaresEstimate":
        """Update function."""
        if self.is_empty:
            return self.create(inputs, outputs)

        return type(self)(
            values=jnp.concatenate((self.values, inputs)),
            targets=jnp.concatenate((self.targets, outputs)),
        )

    @override
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

    @override
    def compute(self) -> jax.Array:
        """Compute function."""
        return jnp.linalg.lstsq(self.values, self.targets)[0]
