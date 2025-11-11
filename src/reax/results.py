import dataclasses
from typing import TYPE_CHECKING, Generic, TypeVar

import jax
import jax.numpy as jnp
import jaxtyping as jt
from typing_extensions import override

from . import metrics as _metric
from . import types

if TYPE_CHECKING:
    import reax


_OutT = TypeVar("_OutT")


@dataclasses.dataclass
class Metadata:
    fx: str
    name: str
    batch_idx: int
    prog_bar: bool
    logger: bool
    on_step: bool
    on_epoch: bool


class ArrayResultMetric(_metric.Metric[jax.Array]):
    value: jax.Array = 0  # Redefine here so the typing hinting works
    cumulated_batch_size: int = 0

    @classmethod
    def create(
        # pylint: disable=arguments-differ
        cls,
        value: jt.ArrayLike,
        batch_size: int,
    ) -> "ArrayResultMetric":
        """Create function."""
        return ArrayResultMetric(value=jnp.asarray(value), cumulated_batch_size=batch_size)

    @override
    def update(
        # pylint: disable=arguments-differ
        self,
        value: jt.ArrayLike,
        batch_size: int,
    ) -> "ArrayResultMetric":
        """Update function."""
        return ArrayResultMetric(
            value=self.value + jnp.asarray(value),
            cumulated_batch_size=self.cumulated_batch_size + batch_size,
        )

    @override
    def merge(self, other: "ArrayResultMetric") -> "ArrayResultMetric":
        """Merge function."""
        return ArrayResultMetric(
            value=self.value + other.value,
            cumulated_batch_size=self.cumulated_batch_size + other.cumulated_batch_size,
        )

    @override
    def compute(self) -> jax.Array:
        """Compute function."""
        return self.value / self.cumulated_batch_size


class ResultEntry(Generic[_OutT]):
    def __init__(
        self,
        meta: Metadata,
        metric: "reax.typing.MetricInstance[_OutT]",
        last_value: "reax.types.MetricInstance[_OutT] | None" = None,
    ):
        """Init function."""
        self._meta = meta  # Readonly
        self.metric = metric
        self._last_value: "reax.types.MetricInstance[_OutT] | _OutT | None" = last_value

    @property
    def meta(self) -> Metadata:
        """Meta function."""
        return self._meta

    @property
    def last_value(self) -> _OutT | None:
        """Last value."""
        if isinstance(self._last_value, types.MetricInstance):
            # Lazily compute the metric as it has now been requested
            self._last_value = self._last_value.compute()

        return self._last_value


class ResultCollection(dict[str, ResultEntry]):
    """A dictionary holding model metrics."""

    def __str__(self) -> str:
        """Str function."""
        my_str = str(self)
        return f"{type(self)}.__name__({my_str})"

    def log(
        self,
        fx: str,
        name: str,
        value: "jt.ArrayLike | reax.typing.MetricInstance[_OutT]",
        batch_idx: int,
        *,
        prog_bar: bool = False,
        logger: bool = False,
        on_step: bool = False,
        on_epoch: bool = True,
        batch_size: int | None = None,
    ):
        """Log function."""
        key = f"{fx}.{name}"

        if isinstance(value, types.MetricInstance):
            metric = value
        else:
            try:
                metric = ArrayResultMetric.create(value, batch_size)
            except TypeError:
                raise TypeError(
                    f"Value must be a `reax.Metric` or a raw value, got {type(value).__name__}"
                ) from None

        meta = Metadata(
            fx=fx,
            name=name,
            batch_idx=batch_idx,
            prog_bar=prog_bar,
            logger=logger,
            on_step=on_step,
            on_epoch=on_epoch,
        )
        last_value = metric

        if key in self:
            # Merge with existing metric to propagate results
            metric = self[key].metric.merge(metric)

        self[key] = ResultEntry(meta, metric, last_value=last_value)
