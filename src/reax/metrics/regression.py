from typing import Optional

import jax
import jax.numpy as jnp

from . import aggregation
from ._metric import Metric

__all__ = "MeanSquaredError", "RootMeanSquareError", "MeanAbsoluteError", "LeastSquaresEstimate"

MeanSquaredError = aggregation.Average.from_fun(
    lambda values, targets, mask=None: (jnp.square(values - targets), mask)
)

MeanAbsoluteError = aggregation.Average.from_fun(
    lambda values, targets, mask=None: (jnp.abs(values - targets), mask)
)


class RootMeanSquareError(Metric):
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
        mask: Optional[jax.Array] = None,
    ) -> "RootMeanSquareError":
        """Create function."""
        return type(self)(mse=MeanSquaredError().create(values, targets, mask))

    def update(  # pylint: disable=arguments-differ
        self,
        values: jax.Array,
        targets: jax.Array,
        mask: Optional[jax.Array] = None,
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


class LeastSquaresEstimate(Metric):
    values: Optional[jax.Array]
    targets: Optional[jax.Array]

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


# _registry.get_registry().register_many(
#     {
#         "mse": MeanSquaredError,
#         "rmse": RootMeanSquareError,
#         "mae": MeanAbsoluteError,
#     }
# )
