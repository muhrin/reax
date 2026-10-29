"""Metrics evaluators"""

import abc
from typing import TYPE_CHECKING, TypeVar

import jax
from typing_extensions import override

from .. import pytree

if TYPE_CHECKING:
    import reax

__all__ = "MetricEvaluator", "DefaultEvaluator", "VmapEvaluator"

T = TypeVar("T", bound="reax.Metric")


class MetricEvaluator(abc.ABC, pytree.PyTree):
    @abc.abstractmethod
    def create(self, metric: T, *args, **kwargs) -> T:
        """Evaluate the metric creating a new instance"""


class DefaultEvaluator(MetricEvaluator):
    @override
    def create(self, metric: T, *args, **kwargs) -> T:
        return metric.create(*args, **kwargs)


class VmapEvaluator(MetricEvaluator):
    @override
    def create(self, metric: T, *args, **kwargs) -> T:
        # 1. Compute the metric for the whole batch in parallel
        batched_instance = jax.vmap(metric.create)(*args, **kwargs)

        # 2. Re-use the metric's own reduction logic to collapse the batch dim
        return batched_instance.reduce(axis=0)
