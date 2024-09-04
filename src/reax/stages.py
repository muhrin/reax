import abc
import dataclasses
import itertools
from typing import TYPE_CHECKING, Any, Optional, Union

import jax

from . import data
from . import optimizers as optimizers_
from . import results
from .utils import events

if TYPE_CHECKING:
    import reax


@dataclasses.dataclass
class MetricResult:
    meta: "reax.results.Metadata"
    value: Any


MetricResults = dict[str, MetricResult]


class StageListener:
    def on_stage_starting(self, stage: "Stage"):
        """The stage is about to start"""

    def on_stage_step_start(self, stage: "Stage", step: int):
        """The stage is about to start processing a batch"""

    def on_stage_step_end(self, stage: "Stage", step: int, metrics: MetricResults):
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

    def on_stage_step_start(self, stage: "Stage", step: int):
        self._events.fire_event(StageListener.on_stage_step_start, stage, step)

    def on_stage_step_end(self, stage: "Stage", step: int, metrics: MetricResults):
        self._events.fire_event(StageListener.on_stage_step_end, stage, step, metrics)

    def on_stage_ending(self, stage: "Stage"):
        self._events.fire_event(StageListener.on_stage_ending, stage)


class Stage(metaclass=abc.ABCMeta):
    """Interface for loops"""

    def __init__(
        self,
        name: str,
        strategy: "reax.Strategy",
        min_steps: Optional[int] = None,
        max_steps: int = -1,
        parent: "Stage" = None,
    ):
        self._name = name
        self._strategy = strategy
        self._min_steps = min_steps
        self._max_steps = max_steps
        self._parent = parent

        self._step = -1
        self._run_count = 0
        self.events = events.EventGenerator[StageListener]()
        self._should_stop = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def step(self) -> int:
        return self._step

    @property
    def run_count(self) -> int:
        return self._run_count

    @property
    def parent(self) -> Optional["Stage"]:
        """Optional parent to the stage"""
        return self._parent

    @property
    def should_stop(self) -> bool:
        return self._should_stop

    @should_stop.setter
    def should_stop(self, stop: bool):
        self._should_stop = stop

    def run(self) -> Any:
        """Run the loop until the end or max_steps"""
        iterator = itertools.count() if self._max_steps == -1 else iter(range(self._max_steps))
        self._on_starting()

        while True:
            try:
                self._step = next(iterator)
                self.next()
            except StopIteration:
                break

        self._on_stopping()
        self.events.fire_event(StageListener.on_stage_ending, self)

    @abc.abstractmethod
    def log(
        self,
        name: str,
        value,
        batch_size: Optional[int] = None,
        prog_bar: bool = False,
        on_step=True,
        on_epoch=True,
    ) -> None:
        """Log a result while the stage is running"""

    @abc.abstractmethod
    def _next(self) -> Any:
        """The advance logic that should be implemented by subclasses"""

    def _on_starting(self):
        """Stage is starting"""
        self._step = -1
        self.events.fire_event(StageListener.on_stage_starting, self)

    def _on_iteration_starting(self):
        self.events.fire_event(StageListener.on_stage_step_start, self, self._step)

    def next(self) -> Any:
        """Advance the loop by one iteration"""
        assert self._max_steps == -1 or self._step <= self._max_steps

        if self._should_stop:
            self._should_stop = False
            raise StopIteration
        if self._max_steps != -1 and self._step >= self._max_steps:
            raise StopIteration

        self._on_iteration_starting()
        result = self._next()
        self._on_iteration_finished(result)

    def _on_iteration_finished(self, result: Any):
        self.events.fire_event(StageListener.on_stage_step_end, self, self._step, result)

    def _on_stopping(self):
        """The stage is stopping"""
        self._run_count += 1


class EpochStage(Stage):
    """Stage that runs one complete epoch from a dataloader at a time"""

    def __init__(
        self,
        name: str,
        dataloader: data.DataLoader,
        strategy: "reax.Strategy",
        min_steps: Optional[int] = None,
        max_steps: int = -1,
        parent: "Stage" = None,
    ):
        super().__init__(name, strategy, min_steps=min_steps, max_steps=max_steps, parent=parent)
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
    def metrics(self) -> "reax.results.ResultCollection":
        return self._metrics

    @property
    def batch(self):
        """Get the current batch"""
        return self._batch

    @property
    def max_batches(self) -> Optional[int]:
        return data.sized_len(self._dataloader)

    @property
    def results(self) -> Optional[MetricResults]:
        """The results for this epoch (if any)"""
        return self._results

    def log(
        self,
        name: str,
        value: Union[jax.typing.ArrayLike, "reax.Metric"],
        batch_size: Optional[int] = None,
        prog_bar: bool = False,
        on_step=True,
        on_epoch=True,
    ) -> None:
        assert self._batch is not None
        if batch_size is None:
            batch_size = data.extract_batch_size(self._batch)

        self._metrics.log(
            self._name,
            name,
            value,
            prog_bar=prog_bar,
            batch_size=batch_size,
            batch_idx=self._step,
            on_step=on_step,
            on_epoch=on_epoch,
        )

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

    @abc.abstractmethod
    def _next(self) -> Any:
        """The advance logic that should be implemented by subclasses"""

    def _on_stopping(self):
        """
        Look through the logged metrics and extract those where a user logged a results during this
        step and used the `on_step=True` option
        """
        res = {}
        for name, entry in self._metrics.items():
            if entry.meta.on_epoch:
                # Ask the metric to calculate the overall result and store it
                res[name] = MetricResult(entry.meta, entry.metric.compute())

        self._results = res
        super()._on_stopping()

    def _get_iteration_results(self) -> MetricResults:
        """
        Look through the logged metrics and extract those where a user logged a results during this
        step and used the `on_step=True` option
        """
        res = {}
        for name, entry in self._metrics.items():
            if entry.meta.on_step and entry.meta.batch_idx == self.step:
                # Get the last value (i.e. the one from this step)
                res[name] = MetricResult(entry.meta, entry.last_value)

        return res


class Train(EpochStage):
    def __init__(
        self,
        module: "reax.Module",
        dataloader: "reax.DataLoader",
        strategy: "reax.Strategy",
        optimizers: list["reax.Optimizer"],
        parent=None,
    ):
        super().__init__("training", dataloader, strategy, parent=parent)
        self._module = module
        self._optimizers = optimizers

    @property
    def optimizers(self) -> Optional[list["reax.Optimizer"]]:
        return self._optimizers

    def run(self) -> list["reax.Optimizer"]:
        super().run()
        return self._optimizers

    def _on_starting(self):
        self._module.setup(self.name)
        params = self._strategy.to_device(self._module.parameters())
        self._module.set_parameters(params)

        if not self._optimizers:
            opts = self._module.configure_optimizers()
            if not isinstance(opts, list):
                opts = [opts]

            # Move optimizer parameters to device
            opts = [(opt, self._strategy.to_device(state)) for opt, state in opts]

            # Create the `Optimizer` instances
            self._optimizers = list(map(lambda opt: optimizers_.Optimizer(*opt), opts))

        super()._on_starting()

    def _next(self) -> MetricResults:
        res = self._module.training_step(self.batch, self._step)
        if self._module.automatic_optimization:
            _loss, grads = res
            opt = self._optimizers[0]
            params, opt = opt.update(self._module.parameters(), grads)
            self._optimizers = [opt]
            self._module.set_parameters(params)

        return self._get_iteration_results()


class Validate(EpochStage):
    def __init__(
        self,
        module: "reax.Module",
        dataloader: "reax.DataLoader",
        strategy: "reax.Strategy",
        parent: Stage = None,
    ):
        super().__init__("validation", dataloader, strategy, parent=parent)
        self._module = module

    def _on_starting(self):
        self._module.setup(self.name)
        params = self._strategy.to_device(self._module.parameters())
        self._module.set_parameters(params)
        super()._on_starting()

    def _next(self) -> MetricResults:
        self._module.validation_step(self.batch, self._step)
        return self._get_iteration_results()


class Test(EpochStage):
    def __init__(
        self, module: "reax.Module", dataloader, strategy: "reax.Strategy", parent: Stage = None
    ):
        super().__init__("test", dataloader, strategy, parent=parent)
        self._module = module

    def _on_starting(self):
        self._module.setup(self.name)
        params = self._strategy.to_device(self._module.parameters())
        self._module.set_parameters(params)
        super()._on_starting()

    def _next(self) -> MetricResults:
        self._module.test_step(self.batch, self._step)
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
        min_steps: Optional[int] = None,
        max_steps: int = -1,
        passthrough_listeners=True,
    ):
        super().__init__(
            name,
            strategy,
            min_steps=min_steps,
            max_steps=max_steps,
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
        res = {}
        for info in self._children:
            if self.step % info.run_every_n != 0:
                continue

            child = info.stage
            self._running = child
            try:
                child.run()
            finally:
                self._running = None

            for name, entry in child.metrics.items():
                if entry.meta.on_epoch and entry.meta.batch_idx == self.step:
                    # Compute the final metric
                    res[name] = MetricResult(entry.meta, entry.metric.compute())

        return res

    def log(
        self,
        name: str,
        value,
        batch_size: Optional[int] = None,
        prog_bar: bool = False,
        on_step=True,
        on_epoch=True,
    ) -> None:
        """Pass logging onto the currently running stage"""
        self._running.log(name, value, batch_size, prog_bar, on_step=on_step, on_epoch=on_epoch)


class Fit(MultiStage):
    def __init__(
        self,
        module: "reax.Module",
        train_dataloaders: "reax.DataLoader",
        val_dataloaders: Optional["reax.DataLoader"],
        optimizers: list["reax.Optimizer"],
        strategy: "reax.Strategy",
        min_steps: Optional[int] = None,
        max_steps: int = -1,
        check_val_every_n_epoch: int = 1,
    ):
        children = [
            Train(
                module=module,
                dataloader=train_dataloaders,
                optimizers=optimizers,
                strategy=strategy,
                parent=self,
            )
        ]
        if val_dataloaders is not None:
            # Add the validation stage to fitting
            children.append(
                StageInfo(
                    Validate(
                        module=module, dataloader=val_dataloaders, strategy=strategy, parent=self
                    ),
                    run_every_n=check_val_every_n_epoch,
                )
            )

        super().__init__(
            "fit",
            children,
            strategy,
            min_steps=min_steps,
            max_steps=max_steps,
        )

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
