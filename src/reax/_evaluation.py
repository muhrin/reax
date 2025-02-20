from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, Optional, TypeVar, Union

import beartype
import jax
import jaxtyping as jt
from typing_extensions import override

from reax import metrics, stages, strategies, typing

if TYPE_CHECKING:
    import reax

__all__ = ("evaluate_stats",)


OutT = TypeVar("OutT")


class StatsEvaluator(stages.EpochStage):
    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        stats: Union["reax.Metric", Sequence["reax.Metric"], dict[str, "reax.Metric"]],
        dataloader: "reax.DataLoader",
        accelerator: Literal["auto", "cpu", "gpu"] = "auto",
        strategy: Optional["reax.Strategy"] = None,
    ):
        """Init function."""
        accelerator = jax.devices()[0] if accelerator == "auto" else jax.devices(accelerator)[0]
        strategy = strategy or strategies.SingleDevice(accelerator)
        super().__init__("Metrics evaluator", None, dataloader, strategy)
        self._stats = metrics.MetricCollection(stats)

    @override
    def _step(self) -> Any:
        """Step function."""
        # Calculate and log all the stats
        for name, stat in self._stats.items():
            if isinstance(self.batch, tuple):
                self.log(name, stat.create(*self.batch), on_step=False, on_epoch=True, logger=True)
            else:
                self.log(name, stat.create(self.batch), on_step=False, on_epoch=True, logger=True)


def evaluate_stats(
    stats: Union["reax.Metric[OutT]", Sequence["reax.Metric"], dict[str, "reax.Metric"]],
    dataloader: "reax.DataLoader",
) -> Union[OutT, typing.MetricsDict]:
    """Evaluate stats."""
    single_metric = isinstance(stats, metrics.Metric)
    evaluator = StatsEvaluator(stats, dataloader)
    evaluator.run()
    if single_metric:
        res: OutT = list(evaluator.logged_metrics.values())[0]
        return res

    return evaluator.logged_metrics
