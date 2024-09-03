import dataclasses
from typing import TYPE_CHECKING, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np

from .metrics import metric as metric_

if TYPE_CHECKING:
    import reax


@dataclasses.dataclass
class Metadata:
    fx: str
    name: str
    batch_idx: int
    prog_bar: bool
    on_step: bool
    on_epoch: bool
    last_value: Optional["reax.Metric"] = None


# @dataclasses.dataclass
class ArrayResultMetric(metric_.Metric[jax.Array]):
    value: jax.Array = 0  # Redefine here so the typing hinting works
    cumulated_batch_size: int = 0

    @classmethod
    def create(
        # pylint: disable=arguments-differ
        cls,
        value: jax.typing.ArrayLike,
        batch_size: int,
    ) -> "ArrayResultMetric":
        return ArrayResultMetric(value=value, cumulated_batch_size=batch_size)

    def update(
        # pylint: disable=arguments-differ
        self,
        value: jax.typing.ArrayLike,
        batch_size: int,
    ) -> "ArrayResultMetric":
        return ArrayResultMetric(
            value=self.value + jnp.asarray(value),
            cumulated_batch_size=self.cumulated_batch_size + batch_size,
        )

    def merge(self, other: "ArrayResultMetric") -> "ArrayResultMetric":
        return ArrayResultMetric(
            value=self.value + other.value,
            cumulated_batch_size=self.cumulated_batch_size + other.cumulated_batch_size,
        )

    def compute(self) -> jax.Array:
        return self.value / self.cumulated_batch_size


class ResultEntry:
    def __init__(self, meta: Metadata, metric: "reax.Metric", last_value=None):
        self._meta = meta  # Readonly
        self.metric = metric
        self.last_value = last_value

    @property
    def meta(self) -> Metadata:
        return self._meta


class ResultCollection(dict[str, ResultEntry]):
    """A dictionary holding model metrics"""

    def __str__(self) -> str:
        my_str = str(self)
        return f"{type(self)}.__name__({my_str})"

    def log(
        self,
        fx: str,
        name: str,
        value: Union[jax.typing.ArrayLike, "reax.Metric"],
        batch_idx: int,
        prog_bar: bool = False,
        on_step: bool = False,
        on_epoch: bool = True,
        batch_size: Optional[int] = None,
    ):
        key = f"{fx}.{name}"

        if isinstance(value, (jax.Array, np.ndarray)):
            metric = ArrayResultMetric.create(value, batch_size)
        elif isinstance(value, metric_.Metric):
            metric = value
        else:
            raise TypeError(
                f"Value must be a `reax.Metric` or a raw value, got {type(value).__name__}"
            )

        meta = Metadata(
            fx=fx,
            name=name,
            batch_idx=batch_idx,
            prog_bar=prog_bar,
            on_step=on_step,
            on_epoch=on_epoch,
        )
        last_value = metric.compute() if on_step else None

        if key in self:
            metric = self[key].metric.merge(metric)

        self[key] = ResultEntry(meta, metric, last_value=last_value)
