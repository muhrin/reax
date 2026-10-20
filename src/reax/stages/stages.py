"""Stages that perform the actions expected over the lifetime of a model e.g. training, testing,
predicting etc.
"""

import abc
import logging
from typing import TYPE_CHECKING, Any, Final, TypeVar
import weakref

import beartype
import jaxtyping as jt
from lightning_utilities.core import enums
from typing_extensions import override

import reax.modules

from .. import keys
from ..lightning import rank_zero
from . import _loops, common

# Note: We do not import the trainer here, the relationship is deliberately one way i.e. `Trainer`
# knows about stages, but stages don't know about the trainer.  This helps to reduce coupling.

if TYPE_CHECKING:
    import reax

__all__ = "Stage", "StageState", "ModuleStage"

_LOGGER = logging.getLogger(__name__)
_T_co = TypeVar("_T_co", covariant=True)
_StageT = TypeVar("_StageT", bound="Stage")


class StageState(enums.StrEnum):
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


class Stage(abc.ABC):
    """Abstract class for loops.  This could be a loop over batches (i.e. an epoch) or a loop over
    epochs, which itself contains a loop over batches.  Or something else.
    """

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        name: str,
        engine: "reax.Engine",
        *,
        datamanager: "reax.data.DataSourceManager | None" = None,
        enable_checkpointing: bool = False,
        fast_dev_run: bool | int = False,
    ):
        # Params
        self._name: Final[str] = name
        self._enable_checkpointing: Final[bool] = enable_checkpointing
        self._fast_dev_run: Final[bool | int] = fast_dev_run

        # State
        self._engine = engine
        self._state: StageState = StageState.INITIALIZING
        self._datamanager = datamanager
        self._warning_cache = rank_zero.WarningCache()
        self._stopper = common.Stopper()
        self._stop_reason: str = ""
        self._events = common.StageEvents([weakref.proxy(self)])
        self._loop_stack: list[reax.stages.Loop] = []
        self._metric_results: common.MetricResults = common.empty_metric_results()

    def __str__(self) -> str:
        """Str function."""
        return self.name

    @property
    def state(self) -> "reax.stages.StageState":
        return self._state

    @property
    def name(self) -> str:
        """Name function."""
        return self._name

    @property
    def engine(self) -> "reax.Engine":
        """The engine executing this stage."""
        return self._engine

    @property
    def data_manager(self) -> "reax.data.DataSourceManager | None":
        """Data manager function."""
        return self._datamanager

    @property
    def enable_checkpointing(self) -> bool:
        """`True` if this stage would like checkpointing to be enabled, `False` otherwise"""
        return self._enable_checkpointing

    @property
    def fast_dev_run(self) -> bool | int:
        """Whether the stage is running in fast development mode."""
        return self._fast_dev_run

    @property
    def events(self) -> common.StageEvents:
        """Events function."""
        return self._events

    @property
    def stop_requested(self) -> bool:
        """True once a stop has been requested, regardless of whether the loops
        currently executing can honour it yet.
        """
        return self._stopper.get()

    @property
    def can_stop(self) -> bool:
        """True iff a stop can be honoured right now.

        This requires the stage's own conditions (held by its stopper) to be
        met *and* every loop currently on the loop stack to report that its
        own conditions are satisfied.  Because :meth:`run_loop` pushes nested
        loops onto the stack, this single check covers loops at every depth.
        """
        return self._stopper.can_stop and all(loop.can_stop for loop in self._loop_stack)

    @property
    def should_stop(self) -> bool:
        """True once a stop has been requested, regardless of whether the running
        loops can honour it yet.
        """
        return self.stop_requested

    def stop(self, reason: str):
        """Request that this stage stop.

        The request is recorded immediately and is visible via
        :attr:`stop_requested`.  The stage will only actually stop once
        :attr:`can_stop` is true, i.e. once every loop currently on the loop
        stack has satisfied its own minimum conditions (``min_iters``,
        ``min_epochs``, ...).

        The first reason supplied wins, so an outer, user-visible reason is
        not overwritten by a later, more local request.
        """
        if not self._stopper.get():
            self._stop_reason = reason
        self._stopper.set()

    @property
    def stop_reason(self) -> str:
        """Stop reason."""
        return self._stop_reason

    # region metrics

    @property
    def results(self) -> "reax.stages.common.MetricResults":
        # todo: decide if this is a good name for this
        return self._metric_results

    def publish_metrics(self, results: common.MetricResults):
        self._metric_results = results

    @property
    def listener_metrics(self) -> "reax.types.MetricsDict":
        """Get the metrics available to listeners."""
        return self._metric_results[keys.LISTENER]

    @property
    def logged_metrics(self) -> "reax.types.MetricsDict":
        """Get the metrics available to loggers."""
        return self._metric_results[keys.LOG]

    @property
    def progress_bar_metrics(self) -> dict[str, float]:
        """Get the metrics available to progress indicators."""
        return self._metric_results[keys.PBAR]

    # endregion

    def run(self) -> None:
        """Run the loop until the end or ``max_iters``."""
        try:
            self._on_starting()
            self.events.fire_event(common.StageListener.on_stage_start)

            # Started
            self._on_started()
            self.events.fire_event(common.StageListener.on_stage_started)

            res = self._run()
        except BaseException as exc:
            self._on_exception(exc)
            self._state = StageState.INTERRUPTED
            raise

        # Stopping
        self._on_stopping()
        self.events.fire_event(common.StageListener.on_stage_finishing)

        # Stopped
        self._on_stopped()
        self.events.fire_event(common.StageListener.on_stage_finished)

        return res

    def log(
        self,
        name: str,
        value,
        batch_size: int | None = None,
        prog_bar: bool = False,
        logger: bool = False,
        on_step=False,
        on_epoch=True,
        reduce_fn: "reax.types.ReduceFn" = "mean",
    ) -> None:
        """Log a result while the stage is running."""
        if self._loop_stack:
            loop = self._loop_stack[-1]
            if isinstance(loop, _loops.LoopOverBatches):
                loop.log(
                    name,
                    value,
                    batch_size,
                    prog_bar,
                    logger,
                    on_step,
                    on_epoch,
                    reduce_fn,
                )

    def run_loop(self, loop: "reax.stages.Loop") -> Any:
        self._loop_stack.append(loop)
        try:
            return loop.run()
        finally:
            assert self._loop_stack[-1] is loop
            self._loop_stack.pop()

    def _on_starting(self):
        """Stage is starting."""
        self._state = StageState.RUNNING
        if self._datamanager is not None:
            self._datamanager.prepare_data()
            if self._datamanager.source is not self._datamanager:
                self._datamanager.setup(stage=self.name)

    def _on_started(self):  # noqa: B027
        """On started."""
        return None

    @abc.abstractmethod
    def _run(self) -> Any:
        """The advance logic that should be implemented by subclasses."""

    def _on_stopping(self):
        """The stage is stopping."""
        self._state = StageState.FINISHED

    def _on_stopped(self):
        """The stage has stopped."""
        if self._datamanager is not None and self._datamanager.source is not self._datamanager:
            self._datamanager.teardown(stage=self.name)

    def _on_exception(self, exception: BaseException) -> None:
        """Hook to deal with an exception"""
        if self._datamanager is not None:
            self._datamanager.on_exception(exception)


class ModuleStage(Stage, abc.ABC):
    """Stage that represents a loop over batches i.e. a single epoch."""

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        name: str,
        module: reax.modules.Module,
        datamanager: "reax.data.DataSourceManager",
        engine: "reax.Engine",
        *,
        enable_checkpointing: bool = False,
        fast_dev_run: bool | int = False,
    ):
        super().__init__(
            name,
            engine,
            datamanager=datamanager,
            enable_checkpointing=enable_checkpointing,
            fast_dev_run=fast_dev_run,
        )

        # Params
        self._module: Final[reax.Module] = module

        # State
        self._outputs = None

    @property
    def module(self) -> "reax.Module":
        """Module function."""
        return self._module

    @property
    def outputs(self) -> Any:
        """Outputs function."""
        return self._outputs

    @override
    def _on_starting(self):
        """Stage is starting."""
        super()._on_starting()
        self._module.setup(self._engine, stage=self.name)

    @override
    def _on_stopped(self):
        super()._on_stopped()
        self._module.teardown(self._engine, stage=self.name)

    def configure_module(self, example_batch: Any):
        """Configure the module with the example batch."""
        was_uninitialised = self._module.parameters() is None
        self._module.configure_model(self, example_batch)

        if was_uninitialised and self._module.parameters() is not None:
            _LOGGER.debug(
                "Module '%s' has been configured with the example batch from the '%s' stage.",
                type(self._module).__name__,
                self.name,
            )
            params = self._engine.to_device(self._module.parameters())
            self._module.set_parameters(params)
