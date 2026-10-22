import abc
from typing import TYPE_CHECKING, Any, Final, Generic, Protocol, TypeVar, runtime_checkable
import weakref

import beartype
import jaxtyping as jt
from lightning_utilities.core import enums
from typing_extensions import override

from .. import data, keys
from .. import results as results_
from ..lightning import rank_zero
from . import common

if TYPE_CHECKING:
    import reax
    import reax.results


__all__ = "LoopState", "Loop", "LoopOverBatches"


@runtime_checkable
class LoopHost(Protocol):
    """Minimum a host object must provide for a ``Loop`` to run.

    Captures only the members that ``Loop`` itself accesses via ``self._stage``.
    Concrete stages satisfy this structurally; nothing here forces a particular
    inheritance chain.
    """

    @property
    def name(self) -> str: ...

    @property
    def engine(self) -> "reax.Engine": ...

    @property
    def stop_requested(self) -> bool: ...

    @property
    def can_stop(self) -> bool: ...

    @property
    def events(self) -> "common.StageEvents": ...

    def stop(self, reason: str) -> None: ...

    def run_loop(self, loop: "reax.stages.Loop") -> Any: ...


@runtime_checkable
class HasDataManager(Protocol):
    """Minimum a host object must provide for a ``LoopOverBatches`` to run.

    Captures only the members that ``LoopOverBatches`` itself accesses via ``self._stage``,
    i.e. the extra members beyond those of :class:`LoopHost`.
    Concrete stages satisfy this structurally; nothing here forces a particular
    inheritance chain.
    """

    @property
    def data_manager(self) -> "reax.data.DataSourceManager | None": ...

    def publish_metrics(self, results: common.MetricResults) -> None: ...


@runtime_checkable
class HasModule(Protocol):
    """Minimum a host object must provide for a ``LoopOverBatches`` that drives a module.

    Captures the ``ModuleStage`` members that batch loops (train, validation, test,
    predict) access via ``self._stage``.
    """

    @property
    def module(self) -> "reax.Module": ...

    def configure_module(self, example_batch: Any) -> None: ...


@runtime_checkable
class HostWithData(LoopHost, HasDataManager, Protocol):
    """Minimum a host object must provide for a ``LoopOverBatches`` to run.

    Combines the stage interface consumed by :class:`Loop` with the data-manager
    interface consumed by :class:`LoopOverBatches`.
    """


@runtime_checkable
class ModuleBatchHost(HostWithData, HasModule, Protocol):
    """Minimum a host object must provide for a batch loop that drives a module.

    Combines :class:`HostWithData` with the ``ModuleStage`` interface consumed by
    training/validation/testing/prediction loops.
    """


_StageT = TypeVar("_StageT", bound=LoopHost)
_DataStageT = TypeVar("_DataStageT", bound=HostWithData)


class LoopState(enums.StrEnum):
    """Enum for the status of the :class:`~reax.stages.stages.Stage`"""

    INITIALIZING = "initializing"  # trainer creation
    RUNNING = "running"
    FINISHED = "finished"
    INTERRUPTED = "interrupted"

    @property
    def finished(self) -> bool:
        return self == self.FINISHED

    @property
    def interrupted(self) -> bool:
        return self == self.INTERRUPTED

    @property
    def stopped(self) -> bool:
        return self in (self.FINISHED, self.INTERRUPTED)


class Loop(Generic[_StageT], abc.ABC):
    """Abstract class for loops.  This could be a loop over batches (i.e. an epoch) or a loop over
    epochs, which itself contains a loop over batches.  Or something else.
    """

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        name: str,
        stage: _StageT,
        *,
        max_iters: int | None = None,
        min_iters: int = 0,
    ):
        # Params
        self._stage: _StageT = stage
        self._name: Final[str] = name
        self._min_iters: Final[int] = min_iters
        self._max_iters: Final[int | None] = max_iters

        # State
        self._state: LoopState = LoopState.INITIALIZING
        self._warning_cache = rank_zero.WarningCache()
        self._iter = -1
        self._num_completed = 0  # Iterations completed in the current execution of this loop
        self._total_iters = 0
        self._stopper = common.Stopper()
        self._stopper.add_condition(lambda: self._num_completed >= self._min_iters)
        self._stop_reason: str = ""
        self._run_count = 0

    def __str__(self) -> str:
        """Str function."""
        return self.name

    @property
    def state(self) -> "reax.stages.LoopState":
        return self._state

    @property
    def name(self) -> str:
        """Name function."""
        return self._name

    @property
    def iteration(self) -> int:
        """Get the current stage iteration."""
        return self._iter

    @property
    def min_iters(self) -> int:
        """Minimum number of iterations."""
        return self._min_iters

    @property
    def max_iters(self) -> int | None:
        """Max iters."""
        return self._max_iters

    @property
    def run_count(self) -> int:
        """Run count."""
        return self._run_count

    @property
    def can_stop(self) -> bool:
        """True iff this loop's own conditions are satisfied.

        Consulted by the stage when deciding whether the tree as a whole can
        honour a stop request.
        """
        return self._stopper.can_stop

    @property
    def stop_requested(self) -> bool:
        """True once the stage has been asked to stop, regardless of whether we
        can honour it yet.
        """
        return self._stage.stop_requested

    @property
    def should_stop(self) -> bool:
        """True when the stage has requested a stop and every active loop
        (including this one) can honour it.
        """
        return self._stage.stop_requested and self._stage.can_stop

    def stop(self, reason: str):
        """Request a stage-wide stop.

        Delegates to the stage so there is a single source of truth.  This loop
        will keep running until the stage's :attr:`should_stop` becomes true,
        which requires every active loop (including this one, via ``min_iters``)
        to have satisfied its own conditions.
        """
        self._stage.stop(reason)

    @property
    def stop_reason(self) -> str:
        """Stop reason."""
        return self._stop_reason

    def run(self) -> None:
        """Run the loop until the end or ``max_iters``."""
        while True:
            try:
                self.step()
            except StopIteration:
                break

    def step(self) -> None:
        """Advance the loop by one iteration."""
        events = self._stage.events
        try:
            if self.state != LoopState.RUNNING:
                # Starting
                self._on_starting()
                events.fire_event(common.StageListener.on_loop_starting, weakref.proxy(self))

                # Started
                self._on_started()
                events.fire_event(
                    common.StageListener.on_loop_started, weakref.proxy(self), self.max_iters
                )

            if self._stopping():
                raise StopIteration

            # Iteration
            self._on_iteration_starting()
            events.fire_event(common.StageListener.on_loop_iter_starting, weakref.proxy(self))

            self._on_iteration_started()
            events.fire_event(common.StageListener.on_loop_iter_started, weakref.proxy(self))

            res = self._step()

            # Iteration finishing
            self._on_iteration_finishing(res)
            events.fire_event(common.StageListener.on_loop_iter_finishing, weakref.proxy(self), res)

            # Iteration finished
            self._on_iteration_finished(res)
            events.fire_event(common.StageListener.on_loop_iter_finished, weakref.proxy(self), res)

        except StopIteration as exc:
            self._stop_reason = str(exc)
            # Stopping
            self._on_stopping()
            events.fire_event(common.StageListener.on_loop_finishing, weakref.proxy(self))

            # Stopped
            self._on_stopped()
            events.fire_event(common.StageListener.on_loop_finished, weakref.proxy(self))

            raise
        except BaseException:
            self._state = LoopState.INTERRUPTED
            raise

    def _on_starting(self):
        """Stage is starting."""
        self._state = LoopState.RUNNING
        self._iter = -1
        self._num_completed = 0

    def _on_started(self):
        """On started."""
        return None

    def _on_iteration_starting(self):
        """Loop iteration starting"""
        # Set ourselves up for this iteration
        self._iter += 1

    def _on_iteration_started(self):
        """Loop iteration starting"""
        return None

    @abc.abstractmethod
    def _step(self):
        """Perform a single iteration of the loop."""

    def _on_iteration_finishing(self, _outputs: Any, /):
        """Loop iteration finishing"""
        return None

    def _on_iteration_finished(self, _outputs: Any, /):
        """The iteration has finished.

        Set ourselves up for the next iteration (if there is one).
        """
        self._num_completed += 1
        # Keep track of the total number of iterations, even across multiple executions of this loop
        self._total_iters += 1
        # if self.should_stop and self._min_iters is not None and self._iter < self._min_iters:
        #     message = "%s `min_iters=%i` has not been met. Stage will continue"
        #     _LOGGER.info(message, self.name, self._min_iters)
        #     self.cancel_stop()

    def _on_stopping(self):
        """The loop is stopping."""
        return None

    def _on_stopped(self):
        """The loop has stopped."""
        self._state = LoopState.FINISHED
        self._run_count += 1

    def _done(self) -> bool:
        """True when the loop has naturally completed, i.e. ``max_iters`` has
        been reached.  This is *not* a stop request and is not subject to the
        tree-wide deferral that :attr:`should_stop` applies.
        """
        if self.max_iters is not None and self._iter + 1 >= self.max_iters:
            return True

        return False

    def _stopping(self) -> bool:
        """True when the loop should stop at the next iteration boundary.

        Stop is honoured when either the loop has completed naturally, or the
        stage's tree-wide :attr:`should_stop` is true (which already accounts for
        this loop's own conditions via :attr:`can_stop`).
        """
        return self._done() or self.should_stop

    def _on_exception(self, exception: BaseException) -> None:  # noqa: B027
        """Hook to deal with an exception"""

    def _fire_event(self, *args, **kwargs) -> None:
        """Fire an event."""
        self._stage.events.fire_event(*args, **kwargs)


class LoopOverBatches(Loop[_DataStageT]):
    """A loop that iterates over batches."""

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        name: str,
        stage: _DataStageT,
        *,
        fast_dev_run: bool | int = False,
        min_batches: int = 0,
        limit_batches: int | float | None = None,
        dataloader_name: str | None = None,
    ):
        if dataloader_name is None:
            dataloader_name = name
        if stage.data_manager is None:
            raise ValueError("stage.data_manager must be provided")
        if not stage.data_manager.has_dataloader(dataloader_name):
            raise ValueError(f"Stage {stage.name} has no dataloader named {name}")
        if fast_dev_run:
            limit_batches = 1 if fast_dev_run is True else fast_dev_run
        super().__init__(
            name,
            stage,
            min_iters=min_batches,
            max_iters=limit_batches if isinstance(limit_batches, int) else None,
        )

        # Params
        self._limit_batches: Final[int | float | None] = limit_batches
        self._dataloader_name: Final[str] = dataloader_name

        # State
        self._iterator = None
        self._batch: Any | None = None
        self._next_batch: Any | None = None
        self._metrics: reax.results.ResultCollection = results_.ResultCollection()
        self._metric_results: reax.stages.MetricResults = common.empty_metric_results()
        self._outputs = None

    @property
    def dataloader(self) -> "reax.DataLoader":
        """Dataloader function."""
        assert self._stage.data_manager is not None, (
            "Stage must have a data manager to access dataloaders"
        )
        return self._stage.data_manager.get_dataloader(self._dataloader_name)

    @property
    def dataloaders(self) -> "reax.DataLoader":
        """Dataloader function."""
        return self.dataloader

    @property
    def batch(self) -> Any | None:
        """Get the current batch."""
        return self._batch

    @property
    def batch_idx(self) -> int:
        """Get the current batch index."""
        return self._iter

    @property
    def total_batch_idx(self) -> int:
        """Get the current batch index when counting over all executions of this loop."""
        return self._total_iters

    @property
    def epoch(self) -> int:
        """Get the current epoch."""
        return self._run_count

    @property
    def limit_batches(self) -> int | float | None:
        """The limit on the number of batches specified by the user"""
        return self._limit_batches

    @property
    def max_iters(self) -> int | None:
        return self.max_batches

    @property
    def max_batches(self) -> int | None:
        """Get the current maximum number of batches."""
        return common.batches_limit(self._limit_batches, self.dataloader)

    @property
    def num_batches(self) -> int | float | None:
        """The number of batches that will be used in this epoch"""
        return self.max_batches

    @property
    def metrics(self) -> "reax.results.ResultCollection":
        """Metrics function."""
        return self._metrics

    @property
    def results(self) -> "reax.stages.MetricResults":
        """The results for this epoch (if any)."""
        return self._metric_results

    @property
    def listener_metrics(self) -> "reax.types.MetricsDict":
        """Get the metrics available to listeners."""
        metrics = {}
        if self.metrics:
            # Note: any identical keys will be overwritten by the child (i.e. us)
            metrics.update(self.results[keys.LISTENER])

        return metrics

    @property
    def logged_metrics(self) -> "reax.types.MetricsDict":
        """Get the metrics available to loggers."""
        return self.results[keys.LOG]

    @property
    def progress_bar_metrics(self) -> dict[str, float]:
        """Get the metrics available to progress indicators."""
        return self.results[keys.PBAR]

    @property
    def outputs(self) -> Any:
        """Outputs function."""
        return self._outputs

    def log(
        self,
        name: str,
        value: "jt.ArrayLike | reax.types.MetricInstance",
        batch_size: int | None = None,
        prog_bar: bool = False,
        logger: bool = False,
        on_step=False,
        on_epoch=True,
        reduce_fn: "reax.types.ReduceFn" = "mean",
    ) -> None:
        """Log metrics during the current epoch."""
        assert self._batch is not None
        if batch_size is None:
            batch_size = data.extract_batch_size(self._batch)

        self._metrics.log(
            self._name,
            name,
            value,
            prog_bar=prog_bar,
            logger=logger,
            batch_size=batch_size,
            batch_idx=self._iter,
            on_step=on_step,
            on_epoch=on_epoch,
            reduce_fn=reduce_fn,
        )

    @override
    def _on_starting(self):
        """On starting."""
        super()._on_starting()
        # Get the next bach ready for starting the loop
        self._fetch_next_batch()
        self._metrics.clear()
        common.clear_metric_results(self._metric_results)
        self._stage.publish_metrics(self._metric_results)

    @override
    def _on_iteration_starting(self):
        """On iteration starting."""
        if self._next_batch is None:
            # Exhausted all the data
            raise StopIteration

        # Move the next into the current
        self._batch = self._stage.engine.to_device(self._next_batch)
        self._next_batch = None
        super()._on_iteration_starting()

    @override
    def _on_iteration_finished(self, outputs: Any, /) -> None:
        """On iteration finished."""
        super()._on_iteration_finished(outputs)

        # If this is not the last batch then get the next batch ready for
        # starting the stage.
        max_batches = self.max_batches
        if (max_batches is None or self.batch_idx + 1 < max_batches) and not self.should_stop:
            # Prefetch the next batch (if there is one).  This allows for
            # pipelining i.e. the GPU keeps working on the model, while we (the
            # CPU), fetch and process the next batch.  We then process the
            # metrics and other things that may require a synchronisation.
            self._fetch_next_batch()
        else:
            self._next_batch = None  # Done

        metrics = common.empty_metric_results()
        for _name, entry in self.metrics.items():
            if entry.meta.on_step:  # Here we are stepping
                value = entry.last_value
                metrics[keys.LISTENER][entry.meta.name] = value
                if entry.meta.logger:
                    metrics[keys.LOG][entry.meta.name] = value
                if entry.meta.prog_bar:
                    metrics[keys.PBAR][entry.meta.name] = value

        # Convert tensors to python scalars
        self._metric_results = metrics
        self._publish_metric_results()

    @override
    def _on_stopping(self) -> None:
        super()._on_stopping()
        # Notify the module that the epoch has finished.  This may log additional
        # metrics, so it must run before the on-epoch metrics are gathered below.
        self._on_epoch_end()

        # Gather the on-epoch metrics now, before ``on_loop_finishing`` is fired, so
        # that listeners observing the end of the epoch can see them.
        metrics: common.MetricResults = common.empty_metric_results()
        for _name, entry in self.metrics.items():
            if entry.meta.on_epoch:
                value = self._stage.engine.compute(entry.metric)
                metrics[keys.LISTENER][entry.meta.name] = value
                if entry.meta.logger:
                    metrics[keys.LOG][entry.meta.name] = value
                if entry.meta.prog_bar:
                    metrics[keys.PBAR][entry.meta.name] = value

        self._metric_results = metrics
        self._publish_metric_results()

    def _on_epoch_end(self) -> None:
        """Called when the loop finishes an epoch (before on-epoch metrics are gathered)."""

    @override
    def _on_stopped(self):
        """Reset any per-execution state in case the loop is run again."""
        super()._on_stopped()
        self._iterator = None
        self._batch = None
        self._next_batch = None

    def _fetch_next_batch(self) -> None:
        if self._iterator is None:
            self._iterator = iter(self.dataloader)

        try:
            batch = next(self._iterator)
        except StopIteration:
            self._iterator = None
            batch = None

        self._next_batch = batch

    def _publish_metric_results(self):
        self._stage.publish_metrics(self._metric_results)
