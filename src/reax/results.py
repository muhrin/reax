import dataclasses
from typing import TYPE_CHECKING, Generic, TypeVar

import equinox
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
    """Accumulate raw logged values, reducing them over the batches of an epoch.

    With ``reduce_fn="sum"`` the values are simply added up.  With ``"mean"`` each is weighted by
    the ``batch_size`` it was logged with, so the result is the average over all the samples seen
    rather than over the batches, and a short final batch counts for the samples it actually holds.
    Where the batch size isn't known every value counts once, giving the mean over the batches.
    """

    value: jax.Array = 0  # Redefine here so the typing hinting works
    weight: jax.Array = 0
    reduce_fn: "reax.types.ReduceFn" = equinox.field(static=True, default="mean")

    @classmethod
    def create(
        # pylint: disable=arguments-differ
        cls,
        value: jt.ArrayLike,
        batch_size: int | None = None,
        reduce_fn: "reax.types.ReduceFn" = "mean",
    ) -> "ArrayResultMetric":
        """Create function."""
        reduce_fn = _check_reduce_fn(reduce_fn)
        weight = _weight(batch_size, reduce_fn)

        return ArrayResultMetric(
            value=jnp.asarray(value) * weight, weight=weight, reduce_fn=reduce_fn
        )

    @override
    def update(
        # pylint: disable=arguments-differ
        self,
        value: jt.ArrayLike,
        batch_size: int | None = None,
    ) -> "ArrayResultMetric":
        """Update function."""
        weight = _weight(batch_size, self.reduce_fn)

        return ArrayResultMetric(
            value=self.value + jnp.asarray(value) * weight,
            weight=self.weight + weight,
            reduce_fn=self.reduce_fn,
        )

    @override
    def merge(self, other: "ArrayResultMetric") -> "ArrayResultMetric":
        """Merge function."""
        if other.reduce_fn != self.reduce_fn:
            raise ValueError(
                f"Cannot merge results that are reduced differently, got '{self.reduce_fn}' and "
                f"'{other.reduce_fn}'"
            )

        return ArrayResultMetric(
            value=self.value + other.value,
            weight=self.weight + other.weight,
            reduce_fn=self.reduce_fn,
        )

    @override
    def compute(self) -> jax.Array:
        """Compute function."""
        if self.reduce_fn == "sum":
            return self.value

        return self.value / self.weight


def _weight(batch_size: int | None, reduce_fn: "reax.types.ReduceFn") -> int:
    """How much a value logged with ``batch_size`` counts for.

    Totals are added up as they are, so they carry no weight.  Averages are weighted by the number
    of samples behind them, falling back to counting once each when that isn't known.
    """
    if reduce_fn == "sum" or batch_size is None:
        return 1

    return batch_size


def _check_reduce_fn(reduce_fn: "reax.types.ReduceFn") -> "reax.types.ReduceFn":
    if reduce_fn not in ("sum", "mean"):
        raise ValueError(f"`reduce_fn` must be one of 'sum' or 'mean', got {reduce_fn!r}")

    return reduce_fn


class ResultEntry(Generic[_OutT]):
    def __init__(
        self,
        meta: Metadata,
        metric: "reax.types.MetricInstance[_OutT]",
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
        value: "jt.ArrayLike | reax.types.MetricInstance[_OutT]",
        batch_idx: int,
        *,
        prog_bar: bool = False,
        logger: bool = False,
        on_step: bool = False,
        on_epoch: bool = True,
        batch_size: int | None = None,
        reduce_fn: "reax.types.ReduceFn" = "mean",
    ):
        """Log function.

        ``reduce_fn`` says how raw values logged over an epoch should be reduced to a single value,
        with ``batch_size`` weighting them when taking their mean.  Both are ignored for metric
        instances, which know how to combine themselves.
        """
        key = f"{fx}.{name}"

        if isinstance(value, types.MetricInstance):
            metric = value
        else:
            try:
                metric = ArrayResultMetric.create(value, batch_size, reduce_fn)
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
