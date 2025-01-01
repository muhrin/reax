"""
Stages that perform the actions expected over the lifetime of a model e.g. training, testing,
predicting etc.
"""

import abc
import logging
from typing import TYPE_CHECKING, Any, Callable, Optional, TypedDict, Union, cast
import weakref

import beartype
import jax
import jaxtyping as jt
from lightning_utilities.core import overrides
import optax
from typing_extensions import override

from .. import data, exceptions, keys, modules
from .. import optimizers as optimizers_
from .. import results, typing
from ..lightning import rank_zero
from ..utils import arrays, events

# Note: We do not import the trainer here, the relationship is deliberately one way i.e. `Trainer`
# knows about stages, but stages don't know about the trainer.  This helps to reduce coupling.

if TYPE_CHECKING:
    import reax

__all__ = (
    "StageListener",
    "Stage",
    "EpochStage",
    "Train",
    "Validate",
    "Test",
    "Fit",
    "FitEpoch",
    "Predict",
)

_LOGGER = logging.getLogger(__name__)


class MetricResults(TypedDict):
    callback: dict[str, jax.Array]
    log: dict[str, jax.Array]
    pbar: dict[str, float]


class StageListener:
    def on_stage_starting(self, stage: "reax.Stage", /):
        """The stage is about to start."""

    def on_stage_started(self, stage: "reax.Stage", /):
        """The stage has started, all initialisation if complete."""

    def on_stage_iter_starting(self, stage: "reax.Stage", step: int, /):
        """The stage is about to start an iteration."""

    def on_stage_iter_ending(self, stage: "reax.Stage", step: int, outputs: Any, /):
        """The stage just finished processing an iteration."""

    def on_stage_ending(self, stage: "reax.Stage", /):
        """The stage is about to finish."""

    def on_stage_ended(self, stage: "reax.Stage", /):
        """The stage has ended.

        It will not be mutated after this point until it is starting again.
        """


class _Stopper:
    def __init__(self):
        """Init function."""
        self._stop_requested = False
        self._conditions: list[Callable[[], bool]] = []

    @property
    def stop_requested(self) -> bool:
        """Stop requested."""
        return self._stop_requested

    @property
    def can_stop(self):
        """Can stop."""
        return all(condition() for condition in self._conditions)

    def do_stop(self) -> bool:
        """Do stop."""
        return self._stop_requested and self.can_stop

    def set(self):
        """Set function."""
        self._stop_requested = True

    def get(self) -> bool:
        """Get function."""
        return self._stop_requested

    def add_condition(self, condition: Callable[[], bool]) -> None:
        """Add condition."""
        self._conditions.append(condition)


class Stage(abc.ABC):
    """Interface for loops."""

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        name: str,
        module: Optional["reax.Module"],
        strategy: "reax.Strategy",
        max_iters: Union[int, float] = float("inf"),
        min_iters: int = -1,
        parent: Optional["reax.Stage"] = None,
    ):
        # Params
        self._name = name
        self._strategy = strategy
        self._min_iters = min_iters
        self._max_iters = max_iters

        # State
        self._module = module
        self._warning_cache = rank_zero.WarningCache()
        self._iter = -1
        self._stopper = _Stopper()
        self._stopper.add_condition(lambda: self._iter >= self._min_iters)
        self._stop_reason: str = ""
        self._run_count = 0
        self._parent = parent
        self._events = events.EventGenerator[StageListener]() if parent is None else parent.events
        self._child: Optional["reax.Stage"] = None

    @property
    def name(self) -> str:
        """Name function."""
        return self._name

    @property
    def iteration(self) -> int:
        """Get the current stage iteration."""
        return self._iter

    @property
    def max_iters(self) -> int:
        """Max iters."""
        return self._max_iters

    @property
    def run_count(self) -> int:
        """Run count."""
        return self._run_count

    @property
    def events(self) -> events.EventGenerator[StageListener]:
        """Events function."""
        return self._events

    @property
    def should_stop(self) -> bool:
        """Should stop."""
        return self._stopper.get()

    @property
    def parent(self) -> Optional["reax.Stage"]:
        """Parent function."""
        return self._parent

    def stop(self, reason: str):
        """Stop function."""
        self._stopper.set()
        self._stop_reason = reason

    @property
    def stop_reason(self) -> str:
        """Stop reason."""
        return self._stop_reason

    def run(self) -> None:
        """Run the loop until the end or `max_iters`."""
        while True:
            try:
                self.step()
            except StopIteration:
                break

    def step(self) -> None:
        """Advance the loop by one iteration."""
        if self._iter == -1:
            self._on_starting()
            self._on_started()

        try:
            if self._done():
                raise StopIteration

            self._on_iteration_starting()
            result = self._step()
            self._on_iteration_finishing(result)
            self._on_iteration_finished(result)
        except StopIteration as exc:
            self._stop_reason = str(exc)
            self._on_stopping()
            self._on_stopped()
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
        if self._module is not None:
            self._module.on_stage_starting(weakref.proxy(self))
        self.events.fire_event(StageListener.on_stage_starting, weakref.proxy(self))

    def _on_started(self):
        """On started."""
        if self._module is not None:
            self._module.on_stage_started(weakref.proxy(self))
        self.events.fire_event(StageListener.on_stage_started, weakref.proxy(self))

    def _on_iteration_starting(self):
        """On iteration starting."""
        if self._module is not None:
            self._module.on_stage_iter_starting(weakref.proxy(self), self._iter)
        self.events.fire_event(
            StageListener.on_stage_iter_starting, weakref.proxy(self), self._iter
        )

    @abc.abstractmethod
    def _step(self) -> Any:
        """The advance logic that should be implemented by subclasses."""

    def _on_iteration_finishing(self, outputs: Any):
        """The iteration is about to finish."""
        if self._module is not None:
            self._module.on_stage_iter_ending(weakref.proxy(self), self._iter, outputs)
        self.events.fire_event(
            StageListener.on_stage_iter_ending, weakref.proxy(self), self._iter, outputs
        )

    def _on_iteration_finished(self, outputs: Any):
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
        self.events.fire_event(StageListener.on_stage_ending, weakref.proxy(self))

    def _on_stopped(self):
        """On stopped."""
        self.events.fire_event(StageListener.on_stage_ended, weakref.proxy(self))

    def _done(self) -> bool:
        """Done function."""
        if self._iter >= self.max_iters:
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
        dataloader: "reax.DataLoader",
        strategy: "reax.Strategy",
        min_batches: int = -1,
        max_batches: Union[int, float] = float("inf"),
        parent: Optional["reax.Stage"] = None,
    ):
        max_batches = _batches_limit(max_batches, dataloader)
        super().__init__(
            name, module, strategy, min_iters=min_batches, max_iters=max_batches, parent=parent
        )
        if dataloader is None:
            raise ValueError(f"Stage {name} requires a data loader, got `None`")

        # Params
        self._dataloader = dataloader

        # State
        self._iterator = None
        self._batch: Optional[Any] = None
        self._total_batch_idx: int = 0
        self._metrics: Optional["reax.results.ResultCollection"] = None
        self._metrics_results: Optional[MetricResults] = None
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
    def max_batches(self) -> Optional[int]:
        """Max batches."""
        return self._max_iters

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

        return self._metrics_results[keys.CALLBACK]

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
        self._metrics = results.ResultCollection()
        self._iterator = iter(self._dataloader)
        self._metrics_results = None

    @override
    def _on_iteration_starting(self):
        """On iteration starting."""
        batch = next(self._iterator)
        self._strategy.to_device(batch)
        self._batch = batch
        super()._on_iteration_starting()

    @override
    def _on_iteration_finishing(self, outputs: Any) -> None:
        """Look through the logged metrics and extract those where a user logged a results during this
        step and used the `on_step=True` option
        """
        metrics = {keys.PBAR: {}, keys.LOG: {}, keys.CALLBACK: {}}
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
    def _on_iteration_finished(self, outputs: Any) -> None:
        """On iteration finished."""
        super()._on_iteration_finished(outputs)
        # Keep track of the total number of batches, even across multiple executions of this loop
        self._total_batch_idx += 1

    @override
    def _on_stopping(self) -> None:
        """Look through the logged metrics and extract those where a user logged a results during this
        step and used the `on_epoch=True` option
        """
        metrics = {keys.PBAR: {}, keys.LOG: {}, keys.CALLBACK: {}}
        for _name, entry in self.metrics.items():
            if entry.meta.on_epoch:  # Here we are completing an epoch
                # Ask the metric to compute itself using results accumulated during the epoch
                value = entry.metric.compute()
                metrics[keys.CALLBACK][entry.meta.name] = value
                if entry.meta.logger:
                    metrics[keys.LOG][entry.meta.name] = value
                if entry.meta.prog_bar:
                    metrics[keys.PBAR][entry.meta.name] = value

        # Convert tensors to python scalars
        self._metrics_results = jax.tree_map(arrays.to_base, metrics)
        super()._on_stopping()


class Train(EpochStage):
    """One training epoch."""

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        module: "reax.Module",
        dataloader: "reax.DataLoader",
        strategy: "reax.Strategy",
        optimizers: "list[reax.Optimizer]",
        min_updates: int = -1,
        max_updates: Union[int, float] = float("inf"),
        max_batches: Union[int, float] = float("inf"),
        accumulate_grad_batches: int = 1,
        parent: Optional["reax.Stage"] = None,
        stopper: Optional[_Stopper] = None,
    ):
        super().__init__(
            "Training epoch", module, dataloader, strategy, max_batches=max_batches, parent=parent
        )
        # Params
        self._min_updates = min_updates
        self._max_updates = max_updates
        self._accumulate_grad_batches = accumulate_grad_batches

        # State
        self._optimizers = optimizers
        self._stopper = stopper
        self._stopper.add_condition(lambda: self.updates >= self._min_updates)

    @property
    def updates(self) -> int:
        """Get the number of gradient updates that have been applied."""
        return sum(opt.update_count for opt in self._optimizers)

    @property
    def optimizers(self) -> Optional[list["reax.Optimizer"]]:
        """Optimizers function."""
        return self._optimizers

    @override
    def run(self) -> list["reax.Optimizer"]:
        """Run function."""
        super().run()
        return self._optimizers

    @override
    def _on_starting(self):
        """On starting."""
        super()._on_starting()
        self._module.setup(self)
        params = self._strategy.to_device(self._module.parameters())
        self._module.set_parameters(params)

        if not self._optimizers:
            opts = self._module.configure_optimizers()
            if not isinstance(opts, list):
                opts = [opts]

            optimizers: list[optimizers_.Optimizer] = []
            for opt, state in opts:
                # Move optimizer parameters to device
                state = self._strategy.to_device(state)
                if self._accumulate_grad_batches > 1:
                    stepper = optax.MultiSteps(opt, every_k_schedule=self._accumulate_grad_batches)
                    state = stepper.init(params)
                    opt = stepper.gradient_transformation()

                optimizers.append(optimizers_.Optimizer(opt, state))

            # Create the `Optimizer` instances
            self._optimizers = optimizers

        self._module.on_train_start(weakref.proxy(self))

    @override
    def _on_started(self):
        """On started."""
        super()._on_started()
        self._module.on_train_epoch_start(weakref.proxy(self))

    @override
    def _step(self) -> Any:
        """Step function."""
        res = self._module.training_step(self.batch, self._iter)
        if self._module.automatic_optimization:
            if isinstance(res, dict):
                grad = res["grad"]
            else:
                _loss, grad = res
            opt = self._optimizers[0]
            opt = opt.update_module(self._module, grad)
            self._optimizers = [opt]

        if (self._min_updates is None or self.updates >= self._min_updates) and (
            self.updates >= self._max_updates
        ):
            self.stop("Max updates reached")

        return res

    @override
    def _on_stopping(self) -> None:
        """On stopping."""
        self._module.on_train_epoch_end(weakref.proxy(self))
        super()._on_stopping()

    @override
    def _done(self) -> bool:
        """Done function."""
        if self.batch_idx >= self.max_batches:
            rank_zero.rank_zero_debug(
                f"`{type(self).__name__}` done: max_batches.{self.max_batches!r}` reached."
            )
            return True

        if self.updates >= self._max_updates:
            rank_zero.rank_zero_debug(
                f"`{type(self).__name__}` done: `max_updates={self._max_updates!r}` reached."
            )
            return True

        if self._stopper.stop_requested:
            if self._stopper.can_stop:
                rank_zero.rank_zero_debug(
                    f"`{type(self).__name__}` stopped: `{type(self).__name__}.should_stop` was set."
                )
            else:
                self._warning_cache.info(
                    f"Trainer was signaled to stop but the required "
                    f"`min_epochs={self.parent._min_iters!r}` or"
                    f" `min_steps={self._min_updates!r}` has not been met. "
                    f"Training will continue..."
                )
            return self._stopper.can_stop

        return False


class Validate(EpochStage):
    def __init__(
        self,
        module: "reax.Module",
        dataloader: "reax.DataLoader",
        strategy: "reax.Strategy",
        max_batches: Union[int, float] = -1,
        parent: Optional["reax.Stage"] = None,
    ):
        """Init function."""
        super().__init__(
            "Validation", module, dataloader, strategy, max_batches=max_batches, parent=parent
        )

    @override
    def _on_starting(self):
        """On starting."""
        super()._on_starting()
        self._module.setup(self)
        params = self._strategy.to_device(self._module.parameters())
        self._module.set_parameters(params)
        self._module.on_validation_start(weakref.proxy(self))

    @override
    def _on_started(self):
        """On started."""
        super()._on_started()
        self._module.on_validation_epoch_start(weakref.proxy(self))

    def _step(self) -> MetricResults:
        """Step function."""
        return self._module.validation_step(self.batch, self._iter)

    @override
    def _on_stopping(self) -> None:
        """On stopping."""
        self._module.on_validation_epoch_end(weakref.proxy(self))
        super()._on_stopping()

    @override
    def _on_stopped(self):
        """On stopped."""
        super()._on_stopped()
        self._module.on_validation_end(weakref.proxy(self))


class Test(EpochStage):
    def __init__(
        self,
        module: "reax.Module",
        dataloader,
        strategy: "reax.Strategy",
        max_batches: Union[int, float] = -1,
        parent: Optional["reax.Stage"] = None,
    ):
        """Init function."""
        super().__init__(
            "test", module, dataloader, strategy, max_batches=max_batches, parent=parent
        )

    @override
    def _on_starting(self):
        """On starting."""
        self._module.setup(self)
        params = self._strategy.to_device(self._module.parameters())
        self._module.set_parameters(params)
        super()._on_starting()

    @override
    def _on_started(self):
        """On started."""
        super()._on_started()
        self._module.on_test_epoch_start(weakref.proxy(self))

    @override
    def _on_stopped(self):
        """On stopped."""
        super()._on_stopped()
        self._module.on_test_epoch_end(weakref.proxy(self))

    @override
    def _step(self) -> MetricResults:
        """Step function."""
        return self._module.predict_step(self.batch, self._iter)


class Predict(EpochStage):
    def __init__(
        self,
        module: "reax.Module",
        dataloader,
        strategy: "reax.Strategy",
        max_batches: Union[int, float] = float("inf"),
        parent: Optional["reax.Stage"] = None,
    ):
        """Init function."""
        super().__init__(
            "Predicting", module, dataloader, strategy, max_batches=max_batches, parent=parent
        )
        self._all_outputs = []

    @override
    def _on_starting(self):
        """On starting."""
        self._module.setup(self)
        params = self._strategy.to_device(self._module.parameters())
        self._module.set_parameters(params)
        super()._on_starting()

    @override
    def _step(self) -> MetricResults:
        """Step function."""
        return self._module.predict_step(self.batch, self._iter)

    @override
    def _on_iteration_finishing(self, outputs: Any):
        """On iteration finishing."""
        self._all_outputs.append(outputs)


class FitEpoch(Train):
    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        module: "reax.Module",
        train_dataloaders: "reax.DataLoader",
        val_dataloaders: "Optional[reax.DataLoader]",
        optimizers: list["reax.Optimizer"],
        strategy: "reax.Strategy",
        min_updates: Optional[int] = None,
        max_updates: Union[int, float] = float("inf"),
        limit_train_batches: Optional[Union[int, float]] = 1.0,
        accumulate_grad_batches: int = 1,
        limit_val_batches: Optional[Union[int, float]] = 1.0,
        val_check_interval: Optional[Union[int, float]] = 1.0,
        check_val_every_n_epoch: int = 1,
        parent: Optional["reax.Stage"] = None,
        stopper: Optional[_Stopper] = None,
    ):
        """Init function."""
        super().__init__(
            module,
            train_dataloaders,
            strategy,
            optimizers,
            min_updates=min_updates,
            max_updates=max_updates,
            max_batches=limit_train_batches,
            accumulate_grad_batches=accumulate_grad_batches,
            parent=parent,
            stopper=stopper,
        )
        # Params
        self._val_check_interval = val_check_interval
        self._check_val_every_n_epoch = check_val_every_n_epoch
        self._val_check_batch = self._setup_val_check_batch_(
            val_check_interval, self.max_batches, check_val_every_n_epoch, train_dataloaders
        )

        # State
        if (
            val_dataloaders is None
            or limit_val_batches == 0.0
            or not overrides.is_overridden("validation_step", module, modules.Module)
        ):
            # No validation
            self._validate = None
        else:
            self._validate = Validate(
                module, val_dataloaders, strategy, max_batches=limit_val_batches, parent=parent
            )

    @property
    def val_check_interval(self) -> Optional[Union[int, float]]:
        """Val check interval."""
        return self._val_check_interval

    @property
    def check_val_every_n_epoch(self) -> Optional[int]:
        """Check val every n epoch."""
        return self._check_val_every_n_epoch

    @property
    def validate(self) -> Optional[Validate]:
        """Validate function."""
        return self._validate

    @override
    def _on_iteration_finished(self, outputs: Any) -> None:
        """On iteration finished."""
        super()._on_iteration_finished(outputs)

        # We've finished the train iteration, so check if we should do a validation
        if (
            isinstance(self._val_check_interval, int)
            and self.iteration % self._val_check_interval == 0
        ):
            self._run_child(self._validate)

    @override
    def _on_stopping(self) -> None:
        """On stopping."""
        if (
            self._validate is not None
            and self._check_val_every_n_epoch is not None
            and self.epoch % self._check_val_every_n_epoch == 0
        ):
            self._run_child(self._validate)
        super()._on_stopping()

    @override
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
        """Log function."""
        if self._child is not None:
            self._child.log(name, value, batch_size, prog_bar, logger, on_step, on_epoch)
        else:
            super().log(name, value, batch_size, prog_bar, logger, on_step, on_epoch)

    def _should_check_val_fx(self) -> bool:
        """Decide if we should run validation."""
        if not self._should_check_val_epoch():
            return False

        # val_check_batch is inf for iterable datasets with no length defined
        is_infinite_dataset = self._val_check_batch == float("inf")
        is_last_batch = self.batch_progress.is_last_batch
        if is_last_batch and is_infinite_dataset:
            return True

        if self._stopper.do_stop():
            # allow validation if requesting to stop early through `Trainer.should_stop`
            # (e.g. by early stopping) and when the loop allows to stop (min_epochs/steps met)
            return True

        # TODO: let training/eval loop handle logic around limit_*_batches and val_check_batch
        is_val_check_batch = is_last_batch
        if isinstance(self.max_batches, int) and is_infinite_dataset:
            is_val_check_batch = (self.batch_idx + 1) % self.max_batches == 0
        elif self._val_check_batch != float("inf"):
            # if `check_val_every_n_epoch is `None`, run a validation loop every n training batches
            # else condition it based on the batch_idx of the current epoch
            current_iteration = (
                self.total_batch_idx if self._check_val_every_n_epoch is None else self.batch_idx
            )
            is_val_check_batch = (current_iteration + 1) % self._val_check_batch == 0

        return is_val_check_batch

    def _should_check_val_epoch(self) -> bool:
        """Should check val epoch."""
        return self.validate and (
            self._check_val_every_n_epoch is None
            or (self.epoch + 1) % self._check_val_every_n_epoch == 0
        )

    @staticmethod
    def _setup_val_check_batch_(
        val_check_interval: Union[int, float],
        max_batches: Union[int, float],
        check_val_every_n_epoch: int,
        dataloader: "reax.DataLoader",
    ) -> Optional[Union[int, float]]:
        """Setup val check batch."""
        if max_batches == 0:
            return None

        if isinstance(val_check_interval, int):
            val_check_batch = val_check_interval
            if val_check_batch > max_batches and check_val_every_n_epoch is not None:
                raise ValueError(
                    f" `val_check_interval` ({val_check_interval}) must be less than or equal"
                    f" to the number of the training batches ({max_batches})."
                    " If you want to disable validation set `limit_val_batches` to 0.0 instead."
                    " If you want to validate based on the total training batches, set `check_val_every_n_epoch=None`."
                )
        else:
            dataloader_size = data.sized_len(dataloader)
            has_len_all_ranks_ = dataloader_size is not None
            if not has_len_all_ranks_:
                if val_check_interval == 1.0:
                    val_check_batch = float("inf")
                else:
                    raise exceptions.MisconfigurationException(
                        "When using an IterableDataset for `train_dataloader`,"
                        " `Trainer(val_check_interval)` must be `1.0` or an int. An int k specifies"
                        " checking validation every k training batches."
                    )
            else:
                val_check_batch = int(max_batches * val_check_interval)
                val_check_batch = max(1, val_check_batch)

        # if loggers and max_batches < log_every_n_steps and not fast_dev_run:
        #     rank_zero_warn(
        #         f"The number of training batches ({max_batches}) is smaller than the logging interval"
        #         f" Trainer(log_every_n_steps={log_every_n_steps}). Set a lower value for log_every_n_steps if"
        #         " you want to see logs for the training epoch.",
        #         category=PossibleUserWarning,
        #     )

        return val_check_batch


class Fit(Stage):
    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        module: "reax.Module",
        train_dataloaders: "reax.DataLoader",
        val_dataloaders: "Optional[reax.DataLoader]",
        optimizers: list["reax.Optimizer"],
        strategy: "reax.Strategy",
        max_epochs: Union[int, float] = float("inf"),
        min_epochs: int = -1,
        min_updates: int = -1,
        max_updates: Union[int, float] = float("inf"),
        limit_train_batches: Optional[Union[int, float]] = 1.0,
        accumulate_grad_batches: int = 1,
        limit_val_batches: Optional[Union[int, float]] = 1.0,
        val_check_interval: Optional[Union[int, float]] = 1.0,
        check_val_every_n_epoch: int = 1,
        parent: Optional["reax.Stage"] = None,
    ):
        """Init function."""
        super().__init__(
            "Fitting",
            module,
            strategy,
            max_iters=max_epochs,
            min_iters=min_epochs,
            parent=parent,
        )

        # State
        self._fit_epoch = FitEpoch(
            module,
            train_dataloaders,
            val_dataloaders,
            optimizers,
            strategy,
            min_updates=min_updates,
            max_updates=max_updates,
            limit_train_batches=limit_train_batches,
            accumulate_grad_batches=accumulate_grad_batches,
            limit_val_batches=limit_val_batches,
            val_check_interval=val_check_interval,
            check_val_every_n_epoch=check_val_every_n_epoch,
            parent=self,
            stopper=self._stopper,
        )

    @property
    def updates(self) -> int:
        """Updates function."""
        return self._fit_epoch.updates

    @property
    def validate(self) -> Optional[Validate]:
        """Validate function."""
        return self._fit_epoch.validate

    @property
    def val_check_interval(self):
        """Val check interval."""
        return self._fit_epoch.val_check_interval

    @property
    def check_val_every_n_epoch(self) -> Optional[int]:
        """Check val every n epoch."""
        return self._fit_epoch.check_val_every_n_epoch

    @override
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
        """Log function."""
        self._fit_epoch.log(name, value, batch_size, prog_bar, logger, on_step, on_epoch)

    @override
    def _step(self) -> Any:
        """Step function."""
        self._run_child(self._fit_epoch)


def _batches_limit(
    batch_limit: Union[int, float], dataloader: "reax.DataLoader"
) -> Union[int, float]:
    """Return a maximum number of batches given a dataloader and an optional batches limit.

    If the dataloader has fewer entries than the batch limit, then this will be used, otherwise
    the batches limit.

    .. note:: Will return `float("inf")` if there is no limit.
    :param batch_limit: The batches limit.
    :type batch_limit: Union[int, float]
    :param dataloader: The dataloader.
    :type dataloader: "reax.DataLoader"
    :return: The maximum number of batches.
    :rtype: Union[int, float]
    """
    dataloader_size = data.sized_len(dataloader)
    if isinstance(batch_limit, int):
        if dataloader_size is None:
            return batch_limit

        return min(batch_limit, dataloader_size)

    if isinstance(batch_limit, float):
        if batch_limit == float("inf") or batch_limit == 1.0:
            if dataloader_size is not None:
                return dataloader_size
            return float("inf")

        if dataloader_size is not None:
            # batch_limit is a finite float and we have a dataloader size
            batch_limit = cast(float, batch_limit)
            return int(round(batch_limit * dataloader_size))

        raise ValueError(
            f"Cannot determine number of batches from dataloader and batch_limit is "
            f"{batch_limit}"
        )

    # We can't say anything other than just 'go to the end'
    return float("inf")
