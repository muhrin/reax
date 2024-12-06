"""
Stages that perform the actions expected over the lifetime of a model e.g. training, testing,
predicting etc.
"""

import abc
import logging
from typing import TYPE_CHECKING, Any, Optional, TypedDict, Union, cast
import weakref

import beartype
import jax
import jaxtyping as jt
import optax
from typing_extensions import override

from . import data, keys
from . import optimizers as optimizers_
from . import results
from .utils import arrays, events

# Note: We do not import the trainer here, the relationship is deliberately one way i.e. `Trainer`
# knows about stages, but stages don't know about the trainer.  This helps to reduce coupling.

if TYPE_CHECKING:
    import reax

_LOGGER = logging.getLogger(__name__)


class MetricResults(TypedDict):
    callback: dict[str, jax.Array]
    log: dict[str, jax.Array]
    pbar: dict[str, float]


class StageListener:
    def on_stage_starting(self, stage: "reax.Stage", /):
        """The stage is about to start"""

    def on_stage_iter_starting(self, stage: "reax.Stage", step: int, /):
        """The stage is about to start an iteration"""

    def on_stage_iter_ending(self, stage: "reax.Stage", step: int, outputs: Any, /):
        """The stage just finished processing an iteration"""

    def on_stage_ending(self, stage: "reax.Stage", /):
        """The stage is about to finish"""


class Stage(metaclass=abc.ABCMeta):
    """Interface for loops"""

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        name: str,
        strategy: "reax.Strategy",
        max_iters: int = -1,
        min_iters: Optional[int] = None,
        parent: Optional["reax.Stage"] = None,
    ):
        # Params
        self._name = name
        self._strategy = strategy
        self._min_iters = min_iters
        self._max_iters = max_iters

        # State
        self._iter = -1
        self._should_stop: bool = False
        self._stop_reason: str = ""
        self._run_count = 0
        self._parent = parent
        self._events = events.EventGenerator[StageListener]() if parent is None else parent.events

    @property
    def name(self) -> str:
        return self._name

    @property
    def iteration(self) -> int:
        """Get the current stage iteration"""
        return self._iter

    @property
    def max_iters(self) -> int:
        return self._max_iters

    @property
    def run_count(self) -> int:
        return self._run_count

    @property
    def events(self) -> events.EventGenerator[StageListener]:
        return self._events

    @property
    def should_stop(self) -> bool:
        return self._should_stop

    @property
    def parent(self) -> Optional["reax.Stage"]:
        return self._parent

    def stop(self, reason: str):
        self._should_stop = True
        self._stop_reason = reason

    def cancel_stop(self):
        """Cancel a stop request"""
        self._should_stop = False
        self._stop_reason = ""

    @property
    def stop_reason(self) -> str:
        return self._stop_reason

    def run(self) -> None:
        """Run the loop until the end or `max_iters`"""
        while True:
            try:
                self.step()
            except StopIteration:
                break

    def step(self) -> None:
        """Advance the loop by one iteration"""
        if self._iter == -1:
            self._on_starting()

        try:
            if self.should_stop:
                raise StopIteration(self.should_stop)

            self._on_iteration_starting()
            result = self._step()
            self._on_iteration_finishing(result)
            self._on_iteration_finished(result)
        except StopIteration as exc:
            self._stop_reason = str(exc)
            self._on_stopping()
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
        """Log a result while the stage is running"""

    def _on_starting(self):
        """Stage is starting"""
        self._iter = 0
        self._should_stop = False
        self.events.fire_event(StageListener.on_stage_starting, weakref.proxy(self))

    def _on_iteration_starting(self):
        self.events.fire_event(
            StageListener.on_stage_iter_starting, weakref.proxy(self), self._iter
        )

    @abc.abstractmethod
    def _step(self) -> Any:
        """The advance logic that should be implemented by subclasses"""

    def _on_iteration_finishing(self, outputs: Any):
        """The iteration is about to finish"""

    def _on_iteration_finished(self, outputs: Any):
        """The iteration has finished.  Set ourselves up for the next iteration (if there is one)"""
        self.events.fire_event(
            StageListener.on_stage_iter_ending, weakref.proxy(self), self._iter, outputs
        )

        # Set ourselves up for the next iteration
        self._iter += 1
        if self._max_iters != -1 and self._iter >= self._max_iters:
            self.stop("Max iterations reached")

        if self.should_stop and self._min_iters is not None and self._iter < self._min_iters:
            message = "%s `min_iters=%i` has not been met. Stage will continue"
            _LOGGER.info(message, self.name, self._min_iters)
            self.cancel_stop()

    def _on_stopping(self):
        """The stage is stopping"""
        self.events.fire_event(StageListener.on_stage_ending, weakref.proxy(self))
        self._iter = -1
        self._run_count += 1


class EpochStage(Stage, abc.ABC):
    """Stage that represents a loop over batches i.e. a single epoch"""

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        name: str,
        dataloader: "reax.DataLoader",
        strategy: "reax.Strategy",
        min_batches: Optional[int] = None,
        max_batches: Union[int, float] = -1,
        parent: Optional["reax.Stage"] = None,
    ):
        max_batches = _batches_limit(max_batches, dataloader)
        super().__init__(
            name, strategy, min_iters=min_batches, max_iters=max_batches, parent=parent
        )
        if dataloader is None:
            raise ValueError(f"Stage {name} requires a data loader, got `None`")

        # Params
        self._dataloader = dataloader

        # State
        self._iterator = None
        self._batch: Optional[Any] = None
        self._metrics: Optional["reax.results.ResultCollection"] = None
        self._metrics_results: Optional[MetricResults] = None
        self._outputs = None

    @property
    def dataloader(self) -> "reax.DataLoader":
        return self._dataloader

    @property
    def batch(self) -> Optional[Any]:
        """Get the current batch"""
        return self._batch

    @property
    def epoch(self) -> int:
        """Get the current epoch"""
        return self._run_count

    @property
    def max_batches(self) -> Optional[int]:
        return self._max_iters

    @property
    def metrics(self) -> "reax.results.ResultCollection":
        return self._metrics

    @property
    def results(self) -> Optional[dict]:
        """The results for this epoch (if any)"""
        return self._metrics_results

    @property
    def callback_metrics(self) -> dict:
        """Get the metrics available to callbacks"""
        if not self._metrics_results:
            return dict()

        return self._metrics_results[keys.CALLBACK]

    @property
    def logged_metrics(self) -> dict:
        """Get the metrics available to loggers"""
        if not self._metrics_results:
            return dict()

        return self._metrics_results[keys.LOG]

    @property
    def progress_bar_metrics(self) -> dict:
        """Get the metrics available to progress indicators"""
        if not self._metrics_results:
            return dict()

        return self._metrics_results[keys.PBAR]

    @property
    def outputs(self) -> Any:
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
        """Log metrics during the current epoch"""
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
        self._metrics = results.ResultCollection()
        self._iterator = iter(self._dataloader)
        self._metrics_results = None
        super()._on_starting()

    @override
    def _on_iteration_starting(self):
        batch = next(self._iterator)
        self._strategy.to_device(batch)
        self._batch = batch
        super()._on_iteration_starting()

    @override
    def _on_iteration_finishing(self, outputs: Any) -> None:
        """
        Look through the logged metrics and extract those where a user logged a results during this
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
    def _on_stopping(self) -> None:
        """
        Look through the logged metrics and extract those where a user logged a results during this
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
    """One training epoch"""

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        module: "reax.Module",
        dataloader: "reax.DataLoader",
        strategy: "reax.Strategy",
        optimizers: "list[reax.Optimizer]",
        min_updates: Optional[int] = None,
        max_updates: int = -1,
        max_batches: Union[int, float] = -1,
        accumulate_grad_batches: int = 1,
        parent: Optional["reax.Stage"] = None,
    ):
        super().__init__(
            "Training epoch", dataloader, strategy, max_batches=max_batches, parent=parent
        )
        # Params
        self._min_updates = min_updates
        self._max_updates = max_updates
        self._accumulate_grad_batches = accumulate_grad_batches

        # State
        self._module = module
        self._optimizers = optimizers

    @property
    def updates(self) -> int:
        """Get the number of gradient updates that have been applied"""
        return sum(opt.update_count for opt in self._optimizers)

    @property
    def optimizers(self) -> Optional[list["reax.Optimizer"]]:
        return self._optimizers

    def run(self) -> list["reax.Optimizer"]:
        super().run()
        return self._optimizers

    def _on_starting(self):
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

        super()._on_starting()

    def _step(self) -> Any:
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
            self._max_updates != -1 and self.updates >= self._max_updates
        ):
            self.stop("Max updates reached")

        return res


class Validate(EpochStage):
    def __init__(
        self,
        module: "reax.Module",
        dataloader: "reax.DataLoader",
        strategy: "reax.Strategy",
        max_batches: Union[int, float] = -1,
        parent: Optional["reax.Stage"] = None,
    ):
        super().__init__("Validation", dataloader, strategy, max_batches=max_batches, parent=parent)
        self._module = module

    def _on_starting(self):
        self._module.setup(self)
        params = self._strategy.to_device(self._module.parameters())
        self._module.set_parameters(params)
        super()._on_starting()

    def _step(self) -> MetricResults:
        return self._module.validation_step(self.batch, self._iter)


class Test(EpochStage):
    def __init__(
        self,
        module: "reax.Module",
        dataloader,
        strategy: "reax.Strategy",
        max_batches: Union[int, float] = -1,
        parent: Optional["reax.Stage"] = None,
    ):
        super().__init__("test", dataloader, strategy, max_batches=max_batches, parent=parent)
        self._module = module

    @override
    def _on_starting(self):
        self._module.setup(self)
        params = self._strategy.to_device(self._module.parameters())
        self._module.set_parameters(params)
        super()._on_starting()

    @override
    def _step(self) -> MetricResults:
        return self._module.predict_step(self.batch, self._iter)


class Predict(EpochStage):
    def __init__(
        self,
        module: "reax.Module",
        dataloader,
        strategy: "reax.Strategy",
        max_batches: Union[int, float] = -1,
        parent: Optional["reax.Stage"] = None,
    ):
        super().__init__("Predicting", dataloader, strategy, max_batches=max_batches, parent=parent)
        self._module = module
        self._all_outputs = []

    @override
    def _on_starting(self):
        self._module.setup(self)
        params = self._strategy.to_device(self._module.parameters())
        self._module.set_parameters(params)
        super()._on_starting()

    @override
    def _step(self) -> MetricResults:
        return self._module.predict_step(self.batch, self._iter)

    @override
    def _on_iteration_finishing(self, outputs: Any):
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
        max_updates: int = -1,
        limit_train_batches: Optional[Union[int, float]] = 1.0,
        accumulate_grad_batches: int = 1,
        limit_val_batches: Optional[Union[int, float]] = 1.0,
        val_check_interval: Optional[Union[int, float]] = 1.0,
        check_val_every_n_epoch: int = 1,
        parent: Optional["reax.Stage"] = None,
    ):
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
        )
        # Params
        self._val_check_interval = val_check_interval
        self._check_val_every_n_epoch = check_val_every_n_epoch

        # State
        if val_dataloaders is None or limit_val_batches == 0.0:
            # No validation
            self._validate = None
        else:
            self._validate = Validate(
                module, val_dataloaders, strategy, max_batches=limit_val_batches, parent=parent
            )
        self._validating = False

    @property
    def validate(self) -> Optional[Validate]:
        return self._validate

    @override
    def _on_iteration_finished(self, outputs: Any) -> None:
        super()._on_iteration_finished(outputs)
        # We've finished the train iteration, so check if we should do a validation
        if (
            isinstance(self._val_check_interval, int)
            and self.iteration % self._val_check_interval == 0
        ):
            self._do_validate()

    @override
    def _on_stopping(self) -> None:
        super()._on_stopping()
        if (
            self._check_val_every_n_epoch is not None
            and self.epoch % self._check_val_every_n_epoch == 0
        ):
            self._do_validate()

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
        if self._validating:
            self._validate.log(name, value, batch_size, prog_bar, logger, on_step, on_epoch)
        else:
            super().log(name, value, batch_size, prog_bar, logger, on_step, on_epoch)

    def _do_validate(self):
        try:
            self._validating = True
            self._validate.run()
        finally:
            self._validating = False


class Fit(Stage):
    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        module: "reax.Module",
        train_dataloaders: "reax.DataLoader",
        val_dataloaders: "Optional[reax.DataLoader]",
        optimizers: list["reax.Optimizer"],
        strategy: "reax.Strategy",
        max_epochs: int = -1,
        min_epochs: Optional[int] = None,
        min_updates: Optional[int] = None,
        max_updates: int = -1,
        limit_train_batches: Optional[Union[int, float]] = 1.0,
        accumulate_grad_batches: int = 1,
        limit_val_batches: Optional[Union[int, float]] = 1.0,
        val_check_interval: Optional[Union[int, float]] = 1.0,
        check_val_every_n_epoch: int = 1,
        parent: Optional["reax.Stage"] = None,
    ):
        super().__init__(
            "Fitting",
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
        )

    @property
    def updates(self) -> int:
        return self._fit_epoch.updates

    @property
    def validate(self) -> Optional[Validate]:
        return self._fit_epoch.validate

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
        self._fit_epoch.log(name, value, batch_size, prog_bar, logger, on_step, on_epoch)

    @override
    def _step(self) -> Any:
        self._fit_epoch.run()


def _batches_limit(batch_limit: Union[int, float], dataloader: "reax.DataLoader") -> int:
    """
    Return a maximum number of batches given a dataloader and an optional batches limit.
    If the dataloader has fewer entries than the batch limit, then this will be used, otherwise
    the batches limit.

    .. note:: Will return `-1` if there is no limit.

    :param batch_limit: the batches limit
    :param dataloader: the dataloader
    :return: the maximum number of batches
    """
    dataloader_size = data.sized_len(dataloader)
    if isinstance(batch_limit, int):
        if dataloader_size is None:
            return batch_limit
        elif batch_limit == -1:
            return dataloader_size

        return min(batch_limit, dataloader_size)

    if dataloader_size is None:
        # We can't say anything other than just 'go to the end'
        return -1

    # batch_limit is a float
    batch_limit = cast(float, batch_limit)
    return int(round(batch_limit * dataloader_size))
