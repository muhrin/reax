"""Stages that perform the actions expected over the lifetime of a model e.g. training, testing,
predicting etc."""

import abc
import logging
from typing import TYPE_CHECKING, Any, Final, Optional, TypeVar, Union
import weakref

import beartype
import jax
import jaxtyping as jt
from typing_extensions import override

from . import common
from .. import data, keys, results, typing
from ..lightning import rank_zero
from ..utils import arrays

# Note: We do not import the trainer here, the relationship is deliberately one way i.e. `Trainer`
# knows about stages, but stages don't know about the trainer.  This helps to reduce coupling.

if TYPE_CHECKING:
    import reax

__all__ = "Stage", "EpochStage"


_LOGGER = logging.getLogger(__name__)
_T_co = TypeVar("_T_co", covariant=True)


class Stage(abc.ABC):
    """Interface for loops."""

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        name: str,
        module: Optional["reax.Module"],
        strategy: "reax.Strategy",
        rng: Optional["reax.Generator"],
        *,
        max_iters: Optional[int] = None,
        min_iters: int = 0,
        parent: Optional["reax.Stage"] = None,
    ):
        # Params
        self._name = name
        self._strategy = strategy
        self._min_iters: Final[int] = min_iters
        self._max_iters: Final[Optional[int]] = max_iters

        # State
        self._module: Optional["reax.Module"] = module
        self._rng = rng
        self._warning_cache = rank_zero.WarningCache()
        self._iter = -1
        self._stopper = common.Stopper()
        self._stopper.add_condition(lambda: self._iter >= self._min_iters)
        self._stop_reason: str = ""
        self._run_count = 0
        self._parent = parent
        self._events = common.StageEvents() if parent is None else parent.events
        self._child: Optional["reax.Stage"] = None

    def __str__(self) -> str:
        """Str function."""
        return self.name

    @property
    def name(self) -> str:
        """Name function."""
        return self._name

    @property
    def is_root(self) -> bool:
        """Returns ``True`` if this is the root stage, ``False`` otherwise."""
        return self._parent is None

    @property
    def module(self) -> Optional["reax.Module"]:
        """Module function."""
        return self._module

    @property
    def rng(self) -> "Optional[reax.Generator]":
        return self._rng

    @property
    def iteration(self) -> int:
        """Get the current stage iteration."""
        return self._iter

    @property
    def min_iters(self) -> int:
        """Minimum number of iterations."""
        return self._min_iters

    @property
    def max_iters(self) -> Optional[int]:
        """Max iters."""
        return self._max_iters

    @property
    def run_count(self) -> int:
        """Run count."""
        return self._run_count

    @property
    def events(self) -> common.StageEvents:
        """Events function."""
        return self._events

    @property
    def should_stop(self) -> bool:
        """Should stop."""
        return self._stopper.get()

    @property
    def parent(self) -> Optional["reax.Stage"]:
        """Get the parent stage (if exists)."""
        return self._parent

    def stop(self, reason: str):
        """Stop this stage."""
        self._stopper.set()
        self._stop_reason = reason

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
        if self._iter == -1:
            # Starting
            self._on_starting()
            self.events.fire_event(common.StageListener.on_stage_starting, weakref.proxy(self))
            # Started
            self._on_started()
            self.events.fire_event(common.StageListener.on_stage_started, weakref.proxy(self))

        try:
            if self._done():
                raise StopIteration

            # Iteration
            self._on_iteration_starting()
            self.events.fire_event(
                common.StageListener.on_stage_iter_starting, weakref.proxy(self), self._iter
            )

            res = self._step()

            # Iteration finishing
            self._on_iteration_finishing(res)
            self.events.fire_event(
                common.StageListener.on_stage_iter_ending, weakref.proxy(self), self._iter, res
            )

            # Iteration finished
            self._on_iteration_finished(res)
            self.events.fire_event(
                common.StageListener.on_stage_iter_ended, weakref.proxy(self), self._iter, res
            )

        except StopIteration as exc:
            self._stop_reason = str(exc)
            # Stopping
            self._on_stopping()
            self.events.fire_event(common.StageListener.on_stage_ending, weakref.proxy(self))

            # Stopped
            self._on_stopped()
            self.events.fire_event(common.StageListener.on_stage_ended, weakref.proxy(self))

            raise

    @abc.abstractmethod
    def log(
        self,
        name: str,
        value,
        batch_size: Optional[int] = None,
        prog_bar: bool = False,
        logger: bool = False,
        on_step=False,
        on_epoch=True,
    ) -> None:
        """Log a result while the stage is running."""

    def _on_starting(self):
        """Stage is starting."""
        self._iter = 0

        # Events
        if self._module is not None:
            self._module.on_stage_starting(weakref.proxy(self))

    def _on_started(self):
        """On started."""
        if self._module is not None:
            self._module.on_stage_started(weakref.proxy(self))

    def _on_iteration_starting(self):
        """On iteration starting."""
        if self._module is not None:
            self._module.on_stage_iter_starting(weakref.proxy(self), self._iter)

    @abc.abstractmethod
    def _step(self) -> Any:
        """The advance logic that should be implemented by subclasses."""

    def _on_iteration_finishing(self, outputs: Any, /):
        """The iteration is about to finish."""
        if self._module is not None:
            self._module.on_stage_iter_ending(weakref.proxy(self), self._iter, outputs)

    def _on_iteration_finished(self, _outputs: Any, /):
        """The iteration has finished.

        Set ourselves up for the next iteration (if there is one).
        """
        # Set ourselves up for the next iteration
        self._iter += 1
        # if self.should_stop and self._min_iters is not None and self._iter < self._min_iters:
        #     message = "%s `min_iters=%i` has not been met. Stage will continue"
        #     _LOGGER.info(message, self.name, self._min_iters)
        #     self.cancel_stop()

    def _on_stopping(self):
        """The stage is stopping."""
        self._iter = -1
        self._run_count += 1

    def _on_stopped(self):
        """On stopped."""

    def _done(self) -> bool:
        """Done function."""
        if self.max_iters is not None and self._iter >= self.max_iters:
            return True

        if self._stopper.do_stop():
            return True

        return False

    def _run_child(self, stage: "reax.Stage"):
        """Run child."""
        self._child = stage
        try:
            self._child.run()
        finally:
            self._child = None


class EpochStage(Stage, abc.ABC):
    """Stage that represents a loop over batches i.e. a single epoch."""

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        name: str,
        module: Optional["reax.Module"],
        dataloader: "reax.DataLoader[_T_co]",
        strategy: "reax.Strategy",
        rng: Optional["reax.Generator"],
        *,
        min_batches: int = 0,
        max_batches: Optional[Union[int, float]] = None,
        parent: Optional["reax.Stage"] = None,
        datamanager: Optional = None,
    ):
        self._max_batches = max_batches
        super().__init__(
            name,
            module,
            strategy,
            rng,
            min_iters=min_batches,
            max_iters=None,
            parent=parent,
        )

        # State
        self._dataloader: "reax.DataLoader[_T_co]" = dataloader
        self._datamanager: Optional = datamanager
        self._iterator = None
        self._batch: Optional[Any] = None
        self._total_batch_idx: int = 0
        self._metrics: Optional["reax.results.ResultCollection"] = None
        self._metrics_results: Optional["reax.stages.MetricResults"] = None
        self._outputs = None

    @property
    def dataloader(self) -> "reax.DataLoader":
        """Dataloader function."""
        return self._dataloader

    @property
    def batch(self) -> Optional[Any]:
        """Get the current batch."""
        return self._batch

    @property
    def batch_idx(self) -> int:
        """Get the current batch index."""
        return self._iter

    @property
    def total_batch_idx(self) -> int:
        """Get the current batch index when counting over all executions of this loop."""
        return self._total_batch_idx

    @property
    def epoch(self) -> int:
        """Get the current epoch."""
        return self._run_count

    @property
    def max_iters(self) -> Optional[int]:
        """Get the current maximum number of iterations."""
        return common.batches_limit(self._max_batches, self.dataloader)

    @property
    def max_batches(self) -> Optional[int]:
        """Max batches."""
        return self.max_iters

    @property
    def metrics(self) -> "reax.results.ResultCollection":
        """Metrics function."""
        return self._metrics

    @property
    def results(self) -> Optional[dict]:
        """The results for this epoch (if any)."""
        return self._metrics_results

    @property
    def callback_metrics(self) -> typing.MetricsDict:
        """Get the metrics available to callbacks."""
        if not self._metrics_results:
            return dict()

        return self._metrics_results[keys.LISTENER]

    @property
    def listener_metrics(self) -> typing.MetricsDict:
        """Get the metrics available to callbacks."""
        if not self._metrics_results:
            return dict()

        return self._metrics_results[keys.LISTENER]

    @property
    def logged_metrics(self) -> typing.MetricsDict:
        """Get the metrics available to loggers."""
        if not self._metrics_results:
            return dict()

        return self._metrics_results[keys.LOG]

    @property
    def progress_bar_metrics(self) -> typing.MetricsDict:
        """Get the metrics available to progress indicators."""
        if not self._metrics_results:
            return dict()

        return self._metrics_results[keys.PBAR]

    @property
    def outputs(self) -> Any:
        """Outputs function."""
        return self._outputs

    def log(
        self,
        name: str,
        value: Union[jax.typing.ArrayLike, "reax.Metric"],
        batch_size: Optional[int] = None,
        prog_bar: bool = False,
        logger: bool = False,
        on_step=False,
        on_epoch=True,
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
        )

    @override
    def _on_starting(self):
        """On starting."""
        super()._on_starting()

        if self._datamanager is not None:
            # Make sure the data is ready
            self._datamanager.prepare_and_setup(self)

        if self._module is not None:
            was_uninitialised = self._module.parameters() is None
            example_batch = next(iter(self.dataloader))
            self._module.configure_model(self, example_batch)

            # Only the root stage does setup as this only needs to be done once per stage tree
            if was_uninitialised and self._module.parameters() is not None:
                params = self._strategy.to_device(self._module.parameters())
                self._module.set_parameters(params)

        self._metrics = results.ResultCollection()
        self._iterator = iter(self.dataloader)
        self._metrics_results = None

    @override
    def _on_iteration_starting(self):
        """On iteration starting."""
        batch = next(self._iterator)
        self._strategy.to_device(batch)
        self._batch = batch
        super()._on_iteration_starting()

    @override
    def _on_iteration_finishing(self, outputs: Any, /) -> None:
        """Look through the logged metrics and extract those where a user logged a results during
        this step and used the `on_step=True` option."""
        metrics = {keys.PBAR: {}, keys.LOG: {}, keys.LISTENER: {}}
        for _name, entry in self.metrics.items():
            if entry.meta.on_step:  # Here we are stepping
                value = entry.last_value
                if entry.meta.logger:
                    metrics[keys.LOG][entry.meta.name] = value
                if entry.meta.prog_bar:
                    metrics[keys.PBAR][entry.meta.name] = value

        # Convert tensors to python scalars
        self._metrics_results = jax.tree_map(arrays.to_base, metrics)

        super()._on_iteration_finishing(outputs)

    @override
    def _on_iteration_finished(self, outputs: Any, /) -> None:
        """On iteration finished."""
        super()._on_iteration_finished(outputs)
        # Keep track of the total number of batches, even across multiple executions of this loop
        self._total_batch_idx += 1

    @override
    def _on_stopping(self) -> None:
        """Look through the logged metrics and extract those where a user logged a results during
        this step and used the `on_epoch=True` option."""
        metrics = {keys.PBAR: {}, keys.LOG: {}, keys.LISTENER: {}}
        for _name, entry in self.metrics.items():
            if entry.meta.on_epoch:  # Here we are completing an epoch
                # Ask the metric to compute itself using results accumulated during the epoch
                value = entry.metric.compute()
                metrics[keys.LISTENER][entry.meta.name] = value
                if entry.meta.logger:
                    metrics[keys.LOG][entry.meta.name] = value
                if entry.meta.prog_bar:
                    metrics[keys.PBAR][entry.meta.name] = value

        # Convert tensors to python scalars
        self._metrics_results = jax.tree_map(arrays.to_base, metrics)
        super()._on_stopping()
