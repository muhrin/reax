import abc
import dataclasses
import itertools
import logging
from typing import TYPE_CHECKING, Any, Optional, TypedDict, Union, cast

import beartype
import jax
import jaxtyping as jt
import optax

from . import data, keys
from . import optimizers as optimizers_
from . import results
from .utils import arrays, events

if TYPE_CHECKING:
    import reax

_LOGGER = logging.getLogger(__name__)


class MetricResults(TypedDict):
    callback: dict[str, jax.Array]
    log: dict[str, jax.Array]
    pbar: dict[str, float]


class StageListener:
    def on_stage_starting(self, stage: "Stage"):
        """The stage is about to start"""

    def on_stage_iter_starting(self, stage: "Stage", step: int):
        """The stage is about to start processing a batch"""

    def on_stage_iter_ending(self, stage: "Stage", step: int, metrics: MetricResults):
        """The stage just finished processing a batch"""

    def on_stage_ending(self, stage: "Stage"):
        """Called when the stage is about to finish"""


class PassthroughStageListener(StageListener):
    """
    Pass all events that we get through to listeners subscribed to the events generator given
    to us
    """

    def __init__(self, events_generator: events.EventGenerator[StageListener]):
        self._events = events_generator

    def on_stage_starting(self, stage: "Stage"):
        self._events.fire_event(StageListener.on_stage_starting, stage)

    def on_stage_iter_starting(self, stage: "Stage", step: int):
        self._events.fire_event(StageListener.on_stage_iter_starting, stage, step)

    def on_stage_iter_ending(self, stage: "Stage", step: int, metrics: MetricResults):
        self._events.fire_event(StageListener.on_stage_iter_ending, stage, step, metrics)

    def on_stage_ending(self, stage: "Stage"):
        self._events.fire_event(StageListener.on_stage_ending, stage)


class Stage(metaclass=abc.ABCMeta):
    """Interface for loops"""

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        name: str,
        strategy: "reax.Strategy",
        min_iters: Optional[int] = None,
        max_iters: int = -1,
        parent: "Stage" = None,
    ):
        self._name = name
        self._strategy = strategy
        self._min_iters = min_iters
        self._max_iters = max_iters
        self._parent = parent

        self._iter = -1
        self._run_count = 0
        self.events = events.EventGenerator[StageListener]()
        self._should_stop = ""
        self._stop_reason = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def iteration(self) -> int:
        """Get the current stage iteration"""
        return self._iter

    @property
    def run_count(self) -> int:
        return self._run_count

    @property
    def parent(self) -> Optional["Stage"]:
        """Optional parent to the stage"""
        return self._parent

    @property
    def should_stop(self) -> str:
        return self._should_stop

    @should_stop.setter
    def should_stop(self, reason: str):
        self._should_stop = reason

    @property
    def stop_reason(self) -> str:
        return self._stop_reason

    def run(self) -> Any:
        """Run the loop until the end or `max_iters`"""
        iterator = itertools.count()
        self._on_starting()

        while True:
            self._iter = next(iterator)
            if self._max_iters != -1 and self._iter >= self._max_iters:
                self._stop_reason = "max iters"
                break

            try:
                self.next()
            except StopIteration as exc:
                self._stop_reason = str(exc)
                break

        self._on_stopping()
        self.events.fire_event(StageListener.on_stage_ending, self)

    def next(self) -> Any:
        """Advance the loop by one iteration"""
        assert self._max_iters == -1 or self._iter <= self._max_iters

        if self.should_stop:
            raise StopIteration(self.should_stop)
        if self._max_iters != -1 and self._iter >= self._max_iters:
            raise StopIteration("Max iterations reached")

        self._on_iteration_starting()
        result = self._next()
        self._on_iteration_finished(result)

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

    @abc.abstractmethod
    def _next(self) -> Any:
        """The advance logic that should be implemented by subclasses"""

    def _on_starting(self):
        """Stage is starting"""
        self._iter = -1
        self._should_stop = ""
        self.events.fire_event(StageListener.on_stage_starting, self)

    def _on_iteration_starting(self):
        self.events.fire_event(StageListener.on_stage_iter_starting, self, self._iter)

    def _on_iteration_finished(self, result: Any):
        self.events.fire_event(StageListener.on_stage_iter_ending, self, self._iter, result)
        if self.should_stop and self._min_iters is not None and self._iter < self._min_iters:
            message = "%s `min_iters=%i` has not been met. Stage will continue"
            _LOGGER.info(message, self.name, self._min_iters)
            self.should_stop = False

    def _on_stopping(self):
        """The stage is stopping"""
        self._run_count += 1


class EpochStage(Stage, abc.ABC):
    """Stage that runs one complete epoch from a dataloader in each iteration"""

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        name: str,
        dataloader: "reax.DataLoader",
        strategy: "reax.Strategy",
        min_iters: Optional[int] = None,
        max_iters: Union[int, float] = -1,
        parent: "Stage" = None,
    ):
        max_iters = _batches_limit(max_iters, dataloader)
        super().__init__(name, strategy, min_iters=min_iters, max_iters=max_iters, parent=parent)
        if dataloader is None:
            raise ValueError(f"Stage {name} requires a data loader, got `None`")

        self._dataloader = dataloader
        self._iterator = None
        self._batch = None
        self._metrics: Optional["reax.results.ResultCollection"] = None
        self._results: Optional[MetricResults] = None

    @property
    def dataloader(self) -> "reax.DataLoader":
        return self._dataloader

    @property
    def batch(self):
        """Get the current batch"""
        return self._batch

    @property
    def max_batches(self) -> Optional[int]:
        return data.sized_len(self._dataloader)

    # region Metrics and metric results

    @property
    def metrics(self) -> "reax.results.ResultCollection":
        return self._metrics

    @property
    def results(self) -> Optional[dict]:
        """The results for this epoch (if any)"""
        return self._results

    @property
    def callback_metrics(self) -> dict:
        """Get the metrics available to callbacks"""
        if self._results is None:
            return dict()

        return self._results[keys.CALLBACK]

    @property
    def logged_metrics(self) -> dict:
        """Get the metrics available to loggers"""
        if self._results is None:
            return dict()

        return self._results[keys.LOG]

    @property
    def progress_bar_metrics(self) -> dict:
        """Get the metrics availabe to progress indicators"""
        if self._results:
            return dict()

        return self._results[keys.PBAR]

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

    # endregion

    def _on_starting(self):
        self._metrics = results.ResultCollection()
        self._iterator = iter(self._dataloader)
        self._results = None
        super()._on_starting()

    def _on_iteration_starting(self):
        batch = next(self._iterator)
        self._strategy.to_device(batch)
        self._batch = batch
        super()._on_iteration_starting()

    def _on_stopping(self):
        """
        Look through the logged metrics and extract those where a user logged a results during this
        step and used the `on_step=True` option
        """
        metrics = {keys.PBAR: {}, keys.LOG: {}, keys.CALLBACK: {}}
        for name, entry in self.metrics.items():
            if entry.meta.on_epoch:
                # Ask the metric to compute itself using results accumulated during the epoch
                value = entry.metric.compute()
                metrics[keys.CALLBACK][name] = value
                if entry.meta.logger:
                    metrics[keys.LOG][entry.meta.name] = value
                if entry.meta.prog_bar:
                    metrics[keys.PBAR][entry.meta.name] = value

        # Convert tensors to python scalars
        self._results = jax.tree_map(arrays.to_base, metrics)
        super()._on_stopping()

    def _get_iteration_results(self) -> MetricResults:
        """
        Look through the logged metrics and extract those where a user logged a results during this
        step and used the `on_step=True` option
        """
        metrics = {keys.PBAR: {}, keys.LOG: {}, keys.CALLBACK: {}}
        for _name, entry in self.metrics.items():
            if entry.meta.on_step:
                value = entry.last_value
                if entry.meta.logger:
                    metrics[keys.LOG][entry.meta.name] = value
                if entry.meta.prog_bar:
                    metrics[keys.PBAR][entry.meta.name] = value

        # Convert tensors to python scalars
        return jax.tree_map(arrays.to_base, metrics)


class Train(EpochStage):
    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        module: "reax.Module",
        dataloader: "reax.DataLoader",
        strategy: "reax.Strategy",
        optimizers: "list[reax.Optimizer]",
        min_updates: Optional[int] = None,
        max_updates: int = -1,
        max_iters: Union[int, float] = -1,
        accumulate_grad_batches: int = 1,
        parent=None,
    ):
        super().__init__("training", dataloader, strategy, max_iters=max_iters, parent=parent)
        self._module = module
        self._optimizers = optimizers
        self._min_updates = min_updates
        self._max_updates = max_updates
        self._accumulate_grad_batches = accumulate_grad_batches

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

    def _next(self) -> MetricResults:
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
            self.should_stop = "Max updates reached"

        return self._get_iteration_results()


class Validate(EpochStage):
    def __init__(
        self,
        module: "reax.Module",
        dataloader: "reax.DataLoader",
        strategy: "reax.Strategy",
        max_iters: Union[int, float] = -1,
        parent: Stage = None,
    ):
        super().__init__("validation", dataloader, strategy, max_iters=max_iters, parent=parent)
        self._module = module

    def _on_starting(self):
        self._module.setup(self)
        params = self._strategy.to_device(self._module.parameters())
        self._module.set_parameters(params)
        super()._on_starting()

    def _next(self) -> MetricResults:
        self._module.validation_step(self.batch, self._iter)
        return self._get_iteration_results()


class Test(EpochStage):
    def __init__(
        self,
        module: "reax.Module",
        dataloader,
        strategy: "reax.Strategy",
        max_iters: Union[int, float] = -1,
        parent: Stage = None,
    ):
        super().__init__("test", dataloader, strategy, max_iters=max_iters, parent=parent)
        self._module = module

    def _on_starting(self):
        self._module.setup(self)
        params = self._strategy.to_device(self._module.parameters())
        self._module.set_parameters(params)
        super()._on_starting()

    def _next(self) -> MetricResults:
        return self._module.predict_step(self.batch, self._iter)


class Predict(EpochStage):
    def __init__(
        self,
        module: "reax.Module",
        dataloader,
        strategy: "reax.Strategy",
        max_iters: Union[int, float] = -1,
        parent: Stage = None,
    ):
        super().__init__("predict", dataloader, strategy, max_iters=max_iters, parent=parent)
        self._module = module

    def _on_starting(self):
        self._module.setup(self)
        params = self._strategy.to_device(self._module.parameters())
        self._module.set_parameters(params)
        super()._on_starting()

    def _next(self) -> MetricResults:
        self._module.predict_step(self.batch, self._iter)
        return self._get_iteration_results()


@dataclasses.dataclass
class StageInfo:
    stage: "EpochStage"
    run_every_n: int = 1


class MultiStage(Stage):
    """A stage that can have one or more children e.g. fit which has train + validate"""

    def __init__(
        self,
        name: str,
        children: list[EpochStage],
        strategy: "reax.Strategy",
        min_iters: Optional[int] = None,
        max_iters: int = -1,
        passthrough_listeners=True,
    ):
        if max_iters < -1:
            raise ValueError("`max_iters` must be a non-negative integer or -1")

        super().__init__(
            name,
            strategy,
            min_iters=min_iters,
            max_iters=max_iters,
        )

        children = [
            StageInfo(child) if not isinstance(child, StageInfo) else child for child in children
        ]

        self._children: list[StageInfo] = children
        if passthrough_listeners:
            for info in children:
                info.stage.events.add_listener(PassthroughStageListener(self.events))

        self._running: Optional[EpochStage] = None

    def _next(self) -> MetricResults:
        metrics = {keys.PBAR: {}, keys.LOG: {}, keys.CALLBACK: {}}
        for info in self._children:
            if self.iteration % info.run_every_n != 0:
                continue

            child = info.stage
            self._running = child
            try:
                child.run()
            finally:
                self._running = None
                if child.should_stop:
                    self.should_stop = child.should_stop

            metrics.update(child.results)

        return metrics

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
        """Pass logging onto the currently running stage"""
        self._running.log(
            name, value, batch_size, prog_bar, logger=logger, on_step=on_step, on_epoch=on_epoch
        )


class Fit(MultiStage):
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
        min_iters: Optional[int] = None,
        max_iters: int = -1,
        limit_train_batches: Optional[Union[int, float]] = 1.0,
        accumulate_grad_batches: int = 1,
        limit_val_batches: Optional[Union[int, float]] = 1.0,
        check_val_every_n_epoch: int = 1,
    ):
        children = [
            Train(
                module=module,
                dataloader=train_dataloaders,
                optimizers=optimizers,
                max_iters=limit_train_batches,
                min_updates=min_updates,
                max_updates=max_updates,
                accumulate_grad_batches=accumulate_grad_batches,
                strategy=strategy,
                parent=self,
            )
        ]
        if val_dataloaders is not None:
            # Add the validation stage to fitting
            children.append(
                StageInfo(
                    Validate(
                        module=module,
                        dataloader=val_dataloaders,
                        max_iters=limit_val_batches,
                        strategy=strategy,
                        parent=self,
                    ),
                    run_every_n=check_val_every_n_epoch,
                )
            )

        super().__init__(
            "fit",
            children,
            strategy,
            min_iters=min_iters,
            max_iters=max_iters,
        )

    @property
    def updates(self) -> int:
        """Get the number of gradient updates that have been applied"""
        return self.train.updates

    @property
    def train(self) -> Train:
        train_stage = self._children[0].stage
        assert isinstance(train_stage, Train)
        return train_stage

    @property
    def validate(self) -> Validate:
        validate_stage = self._children[1].stage
        assert isinstance(validate_stage, Validate)
        return validate_stage


def _batches_limit(limit_batches: Union[int, float], dataloader: "reax.DataLoader") -> int:
    if isinstance(limit_batches, int):
        return limit_batches

    limit_batches = cast(float, limit_batches)

    if limit_batches == 1.0:
        return -1

    n_train = len(dataloader)
    return int(round(limit_batches * n_train))
