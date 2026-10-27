"""Stage for evaluating dataset statistics"""

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import beartype
import jaxtyping as jt
from typing_extensions import override

from reax import metrics

from . import _loops, stages

if TYPE_CHECKING:
    from collections.abc import Sequence

    import reax


__all__ = ("EvaluateStats", "StatsEpoch", "StatsStage")


@runtime_checkable
class StatsStage(_loops.HostWithData, Protocol):
    """Structural interface a stage must provide to run a :class:`StatsEpoch`.

    ``StatsEpoch`` is generic over this protocol rather than a concrete stage, so
    statistics can be computed on any stage that exposes the data/loop interface (for
    example a custom stage such as ``FromData``), without forcing it to inherit from
    :class:`EvaluateStats`.
    """


class EvaluateStats(stages.Stage):
    """A stage that can be used to evaluate statistics (in the form of metrics) on a dataset"""

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        stats: "reax.Metric | Sequence[reax.Metric] | dict[str, reax.Metric]",
        datamanager: "reax.data.DataSourceManager",
        engine: "reax.Engine",
        *,
        dataset_name: str = "train",
        fast_dev_run: bool | int = False,
        limit_batches: int | float | None = None,
        ignore_missing=False,
        evaluator: "reax.metrics.MetricEvaluator | None" = None,
    ):
        """Init function."""
        super().__init__(
            "stats",
            engine,
            datamanager=datamanager,
            enable_checkpointing=False,
            fast_dev_run=fast_dev_run,
        )

        self._loop = StatsEpoch(
            self,
            stats=stats,
            fast_dev_run=fast_dev_run,
            limit_batches=limit_batches,
            dataloader_name=dataset_name,
            ignore_missing=ignore_missing,
            evaluator=evaluator,
        )

    @property
    def batch_idx(self) -> int:
        return self._loop.batch_idx

    @override
    def _run(self) -> "reax.stages.MetricResults":
        return self.run_loop(self._loop)


class StatsEpoch(_loops.LoopOverBatches[StatsStage]):
    def __init__(
        self,
        stage: StatsStage,
        *,
        stats: "reax.Metric | Sequence[reax.Metric] | dict[str, reax.Metric]",
        fast_dev_run: bool | int = False,
        limit_batches: int | float | None = None,
        dataloader_name: str | None = "train",
        ignore_missing=False,
        evaluator: "reax.metrics.MetricEvaluator | None" = None,
    ):
        super().__init__(
            "stats",
            stage,
            fast_dev_run=fast_dev_run,
            limit_batches=limit_batches,
            dataloader_name=dataloader_name,
        )

        self._stats = stats
        self._ignore_missing = ignore_missing
        self._evaluator = evaluator

    @override
    def _step(self) -> Any:
        evaluator = (
            self._evaluator if self._evaluator is not None else self._stage.engine.metric_evaluator
        )
        collection = metrics.MetricCollection(self._stats, evaluator)

        # Calculate...
        if isinstance(self.batch, tuple):
            calculated = collection.create(*self.batch, ignore_missing=self._ignore_missing)
        else:
            calculated = collection.create(self.batch, ignore_missing=self._ignore_missing)

        # ...and log all the stats
        for name, stat in calculated.items():
            self.log(name, stat, on_step=False, on_epoch=True, logger=True)
