from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Union

from reax import stages

from . import collections, metric

if TYPE_CHECKING:
    import reax

__all__ = ("StatsEvaluator",)


class StatsEvaluator(stages.EpochStage):
    def __init__(
        self,
        stats: Union[metric.Metric, Sequence[metric.Metric], dict[str, metric.Metric]],
        dataloader: "reax.DataLoader",
    ):
        super().__init__("metrics-evaluator", dataloader)
        self._stats = collections.MetricCollection(stats)

    def run(self) -> dict[str, Any]:
        super().run()
        return {result.meta.name: result.value for result in self.results.values()}

    def _next(self) -> Any:
        # Calculate the log all the stats
        for name, stat in self._stats.items():
            self.log(name, stat.create(self.batch))
