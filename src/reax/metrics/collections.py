from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Union

import beartype
import equinox
import flax.core
import jaxtyping as jt

from . import metric as metric_

if TYPE_CHECKING:
    import reax

__all__ = ("MetricCollection", "combine")

MetricType = Union[type[metric_.Metric], metric_.Metric]


class MetricCollection(equinox.Module):
    """A collection of metrics that can be created/updated/merged using a single call"""

    _metrics: flax.core.FrozenDict[str, "reax.Metric"]

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self, metrics: Union["reax.Metric", Sequence["reax.Metric"], dict[str, "reax.Metric"]]
    ):
        super().__init__()
        self._metrics = flax.core.FrozenDict(_metrics_dict(metrics))

    def items(self):
        return self._metrics.items()

    def empty(self) -> "MetricCollection":
        """Create a new empty instance.

        By default, this will call the constructor with no arguments, if needed, subclasses can
        overwrite this with custom behaviour.
        """
        return MetricCollection({name: metric.empty() for name, metric in self._metrics.items()})

    def create(self, *args, **kwargs) -> "MetricCollection":
        return MetricCollection(
            {name: metric.create(*args, **kwargs) for name, metric in self._metrics.items()}
        )

    def update(self, *args, **kwargs) -> "MetricCollection":
        return MetricCollection(
            {name: metric.update(*args, **kwargs) for name, metric in self._metrics.items()}
        )

    @jt.jaxtyped(typechecker=beartype.beartype)
    def merge(self, other: "MetricCollection") -> "MetricCollection":
        merged = {}
        for name, metric in self._metrics.items():
            try:
                merged[name] = metric.merge(
                    other._metrics[name]  # pylint: disable=protected-access
                )
            except KeyError:
                merged[name] = metric  # Nothing to merge

        # Now put in all the ones that only appear in 'other'
        for name in set(other._metrics.keys()) - set(  # pylint: disable=protected-access
            self._metrics.keys()
        ):
            merged[name] = other._metrics[name]  # pylint: disable=protected-access

        return MetricCollection(merged)

    def compute(self) -> dict[str, Any]:
        return {name: metric.compute() for name, metric in self._metrics.items()}


def _metrics_dict(
    metrics: Union["reax.Metric", Sequence["reax.Metric"], dict[str, "reax.Metric"]]
) -> dict[str, metric_.Metric]:
    if isinstance(metrics, dict):
        return {name: _ensure_metric(metric) for name, metric in metrics.items()}

    if isinstance(metrics, Sequence):
        return {type(metric).__name__: metric for metric in map(_ensure_metric, metrics)}

    # Assume we have a single metric
    metric = _ensure_metric(metrics)
    return {type(metric).__name__: metric}


def _ensure_metric(metric: "reax.Metric") -> metric_.Metric:
    return metric


def combine(*metric: metric_.Metric) -> MetricCollection:
    """Combine multiple metrics with the same signature into a collection that can be used to
    calculate multiple metrics at once"""
    return MetricCollection(metric)
