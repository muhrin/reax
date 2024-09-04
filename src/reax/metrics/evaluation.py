from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, Union

import jax

from reax import stages, strategies

from . import collections, metric

if TYPE_CHECKING:
    import reax

__all__ = ("StatsEvaluator", "evaluate_stats")


class StatsEvaluator(stages.EpochStage):
    def __init__(
        self,
        stats: Union["reax.Metric", Sequence["reax.Metric"], dict[str, "reax.Metric"]],
        dataloader: "reax.DataLoader",
        accelerator: Literal["auto", "cpu", "gpu"] = "auto",
        strategy: "reax.Strategy" = None,
    ):
        accelerator = jax.devices()[0] if accelerator == "auto" else jax.devices(accelerator)[0]
        strategy = strategy or strategies.SingleDevice(accelerator)
        super().__init__("metrics-evaluator", dataloader, strategy)
        self._stats = collections.MetricCollection(stats)

    def run(self) -> dict[str, Any]:
        super().run()
        return {result.meta.name: result.value for result in self.results.values()}

    def _next(self) -> Any:
        # Calculate the log all the stats
        for name, stat in self._stats.items():
            if isinstance(self.batch, tuple):
                self.log(name, stat.create(*self.batch), on_step=False, on_epoch=True)
            else:
                self.log(name, stat.create(self.batch), on_step=False, on_epoch=True)


def evaluate_stats(
    stats: Union[metric.Metric, Sequence[metric.Metric], dict[str, metric.Metric]],
    dataloader: "reax.DataLoader",
):
    return StatsEvaluator(stats, dataloader).run()
