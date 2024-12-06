from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, Union

import jax
from typing_extensions import override

from reax import keys, metrics, stages, strategies

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
        super().__init__("Metrics evaluator", dataloader, strategy)
        self._stats = metrics.MetricCollection(stats)

    @override
    def _step(self) -> Any:
        # Calculate and log all the stats
        for name, stat in self._stats.items():
            if isinstance(self.batch, tuple):
                self.log(name, stat.create(*self.batch), on_step=False, on_epoch=True, logger=True)
            else:
                self.log(name, stat.create(self.batch), on_step=False, on_epoch=True, logger=True)


def evaluate_stats(
    stats: Union["reax.Metric", Sequence["reax.Metric"], dict[str, "reax.Metric"]],
    dataloader: "reax.DataLoader",
):
    evaluator = StatsEvaluator(stats, dataloader)
    evaluator.run()
    return evaluator.logged_metrics
