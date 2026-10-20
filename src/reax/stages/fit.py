import datetime
from typing import TYPE_CHECKING, Any, Final, TypeVar
import weakref

import beartype
import jaxtyping as jt
from lightning_utilities.core import overrides
from typing_extensions import override

from .. import exceptions, modules
from .. import optimizers as optimizers_
from ..lightning import rank_zero
from . import _loops, _timer, stages, train, validation

if TYPE_CHECKING:
    import reax

__all__ = "Fit", "FitLoop"


_T_co = TypeVar("_T_co", covariant=True)


class Fit(stages.ModuleStage):
    """Top-level stage that trains a module.

    Owns the high-level configuration and exposes its epoch counter and
    dataloaders. The actual work is delegated to ``FitLoop`` (one
    iteration per epoch), which in turn drives ``TrainEpoch`` (one
    iteration per batch) and the associated validation schedule.
    """

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        module: "reax.Module",
        datamanager: "reax.data.DataSourceManager",
        optimizers: "list[reax.Optimizer]",
        engine: "reax.Engine",
        *,
        fast_dev_run: bool | int = False,
        max_epochs: int | None = None,
        min_epochs: int = 0,
        min_updates: int = 0,
        max_updates: int | float | None = None,
        max_time: str | datetime.timedelta | dict[str, int] | None = None,
        limit_train_batches: int | float | None = 1.0,
        accumulate_grad_batches: int = 1,
        limit_val_batches: int | float | None = 1.0,
        val_check_interval: int | float = 1.0,
        check_val_every_n_epoch: int = 1,
        num_sanity_val_steps: int | None = 2,
        reload_dataloaders_every_n_epochs: int = 0,
    ):
        """Initialize the fit stage and its training loop.

        Accepts the training configuration (epoch/update limits, batch and
        gradient-accumulation settings, and the validation schedule) and
        forwards it to the ``FitLoop`` that actually runs the epochs.
        """
        if reload_dataloaders_every_n_epochs < 0:
            raise exceptions.MisconfigurationException(
                f"`reload_dataloaders_every_n_epochs` should be an int >= 0, got "
                f"{reload_dataloaders_every_n_epochs}."
            )

        if fast_dev_run:
            max_epochs = 1
            num_sanity_val_steps = 0

        super().__init__(
            "fit", module, datamanager, engine, enable_checkpointing=True, fast_dev_run=fast_dev_run
        )

        self.global_updates: int = 0
        self._train_loop = FitLoop(
            self,
            optimizers,
            fast_dev_run=fast_dev_run,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            min_updates=min_updates,
            max_updates=max_updates,
            limit_train_batches=limit_train_batches,
            accumulate_grad_batches=accumulate_grad_batches,
            limit_val_batches=limit_val_batches,
            val_check_interval=val_check_interval,
            check_val_every_n_epoch=check_val_every_n_epoch,
            num_sanity_val_steps=num_sanity_val_steps,
            reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs,
        )

        self._timer = self._init_timer(max_time)
        if self._timer is not None:
            self.events.add_listener(self._timer)
        self._stopper.add_condition(lambda: self.global_updates >= min_updates)

    @staticmethod
    def _init_timer(
        max_time: str | datetime.timedelta | dict[str, int] | None,
    ) -> _timer.StageTimer | None:
        """Create the stage timer, or ``None`` when no ``max_time`` was set."""
        if max_time is None:
            return None
        return _timer.StageTimer(max_time)

    @property
    def epoch(self) -> int:
        """Zero-based index of the epoch currently in progress."""
        return self._train_loop.epoch

    @property
    def min_epochs(self) -> int:
        return self._train_loop.min_iters

    @property
    def max_epochs(self) -> int | None:
        return self._train_loop.max_iters

    @property
    def check_val_every_n_epoch(self) -> int | None:
        return self._train_loop.check_val_every_n_epoch

    @property
    def val_check_interval(self) -> int | float:
        return self._train_loop.val_check_interval

    @property
    def validation_mode(self) -> "reax.stages.ValidationMode | None":
        return self._train_loop.validation_mode

    @property
    def sanity_checking(self) -> bool:
        return self._train_loop.sanity_checking

    @property
    def train_dataloader(self) -> "reax.DataLoader | None":
        return self._train_loop.train_dataloader

    @property
    def train_dataloaders(self) -> "reax.DataLoader | None":
        """Training dataloaders."""
        return self.train_dataloader

    @property
    def val_dataloader(self) -> "reax.DataLoader | None":
        return self._train_loop.val_dataloader

    @property
    def val_dataloaders(self) -> "reax.DataLoader | None":
        """Validation dataloaders."""
        return self.val_dataloader

    @property
    def fast_dev_run(self) -> bool | int:
        return self._train_loop.fast_dev_run

    @property
    def limit_train_batches(self) -> int | float | None:
        return self._train_loop.limit_train_batches

    @property
    def limit_val_batches(self) -> int | float | None:
        return self._train_loop.limit_val_batches

    @property
    def num_sanity_val_steps(self) -> int | None:
        return self._train_loop.num_sanity_val_steps

    @property
    def enable_validation(self) -> bool:
        return self._train_loop.enable_validation

    @property
    def num_training_batches(self) -> int | float | None:
        return self._train_loop.num_training_batches

    @property
    def num_val_batches(self) -> int | float | None:
        return self._train_loop.num_val_batches

    @property
    def timer(self) -> _timer.StageTimer | None:
        return self._timer

    @override
    def _on_started(self):
        super()._on_started()
        self._module.on_fit_start(weakref.proxy(self))

    def _run(self) -> None:
        self.run_loop(self._train_loop)

    @override
    def _on_stopping(self) -> None:
        super()._on_stopping()
        self._module.on_fit_end(weakref.proxy(self))


def _is_validation_possible(
    module: "reax.Module",
    datamanager: "reax.data.DataSourceManager",
    limit_val_batches: int | float | None,
) -> bool:
    if (
        not datamanager.has_dataloader("val")
        or limit_val_batches == 0.0
        or not overrides.is_overridden("validation_step", module, modules.Module)
    ):
        return False
    return True


class FitLoop(_loops.Loop[Fit]):
    """Loop over training epochs.

    Owns the epoch counter and the validation schedule. Constructs its child
    loops (`TrainEpoch` and its `ValidationEpoch`) lazily in ``_on_starting``,
    once the datamanager has been prepared and the dataloaders are available.
    """

    def __init__(
        self,
        stage: Fit,
        optimizers: "list[reax.Optimizer]",
        *,
        fast_dev_run: bool | int = False,
        max_epochs: int | None = None,
        min_epochs: int = 0,
        min_updates: int = 0,
        max_updates: int | float | None = None,
        limit_train_batches: int | float | None = 1.0,
        accumulate_grad_batches: int = 1,
        limit_val_batches: int | float | None = 1.0,
        val_check_interval: int | float = 1.0,
        check_val_every_n_epoch: int | None = 1,
        num_sanity_val_steps: int | None = 2,
        reload_dataloaders_every_n_epochs: int = 0,
    ):
        if stage.data_manager is None:
            raise exceptions.MisconfigurationException(
                "FitLoop requires a datamanager to be set on the stage."
            )
        if fast_dev_run:
            if isinstance(fast_dev_run, int):
                if fast_dev_run < 0:
                    raise exceptions.MisconfigurationException("`fast_dev_run` should be >= 0")
                if fast_dev_run == 1:
                    fast_dev_run = True

            num_batches = 1 if fast_dev_run is True else fast_dev_run
            max_epochs = 1
            num_sanity_val_steps = 0
            limit_train_batches = num_batches
            limit_val_batches = num_batches
            val_check_interval = 1.0

        super().__init__(
            "train",
            stage,
            min_iters=min_epochs,
            max_iters=max_epochs,
        )

        # Config
        self._fast_dev_run: Final[bool | int] = fast_dev_run
        self._min_updates: Final[int] = min_updates
        self._max_updates: Final[int | float | None] = max_updates
        self._limit_train_batches: Final[int | float | None] = limit_train_batches
        self._accumulate_grad_batches: Final[int] = accumulate_grad_batches
        self._limit_val_batches: Final[int | float | None] = limit_val_batches
        self._val_check_interval: Final[int | float] = val_check_interval
        self._check_val_every_n_epoch: Final[int | None] = check_val_every_n_epoch
        self._num_sanity_val_steps: Final[int | None] = num_sanity_val_steps
        self._reload_dataloaders_every_n_epochs: Final[int] = reload_dataloaders_every_n_epochs

        # State
        self._optimizers: list[optimizers_.Optimizer] = optimizers
        self._sanity_check: validation.ValidationEpoch | None = None
        self._train_epoch: train.TrainEpoch | None = None
        self._validation_epoch: validation.ValidationEpoch | None = None
        self._sanity_checking: bool = False
        # Only allow the fit loop to stop if we have reached the minimum number of epochs
        self._stopper.add_condition(lambda: self._num_completed >= min_epochs)

    @property
    def fast_dev_run(self) -> bool | int:
        return self._fast_dev_run

    @property
    def check_val_every_n_epoch(self) -> int | None:
        return self._check_val_every_n_epoch

    @property
    def val_check_interval(self) -> int | float:
        return self._val_check_interval

    @property
    def validation_mode(self) -> "reax.stages.ValidationMode | None":
        if self._train_epoch is None:
            return None
        return self._train_epoch.validation_mode

    @property
    def sanity_checking(self) -> bool:
        return self._sanity_checking

    @property
    def epoch(self) -> int:
        """Zero-based index of the epoch currently in progress."""
        return self._iter

    @property
    def max_epochs(self) -> int | None:
        """Maximum number of epochs this loop will run (``None`` for unbounded)."""
        return self.max_iters

    @property
    def train_dataloader(self) -> "reax.DataLoader | None":
        if self._train_epoch is None:
            return None
        return self._train_epoch.train_dataloader

    @property
    def val_dataloader(self) -> "reax.DataLoader | None":
        if self._train_epoch is None:
            return None
        return self._train_epoch.val_dataloader

    @property
    def limit_train_batches(self) -> int | float | None:
        return self._limit_train_batches

    @property
    def limit_val_batches(self) -> int | float | None:
        return self._limit_val_batches

    @property
    def num_sanity_val_steps(self) -> int | None:
        return self._num_sanity_val_steps

    @property
    def enable_validation(self) -> bool:
        return self._validation_epoch is not None

    @property
    def num_training_batches(self) -> int | float | None:
        if self._train_epoch is None:
            return None
        return self._train_epoch.num_training_batches

    @property
    def num_val_batches(self) -> int | float | None:
        if self._train_epoch is None:
            return None
        return self._train_epoch.num_val_batches

    @override
    def _on_starting(self):
        super()._on_starting()

        # Data is now prepared; build the data-dependent loops.
        self._init_loops()

    def _init_loops(self) -> None:
        """Construct the train and validation loops.

        Called from ``_on_starting``, after the datamanager has been prepared
        and the dataloaders are available. Whether validation is enabled is a
        policy decision made here; the concrete schedule (interval arithmetic)
        is derived by ``TrainEpoch`` from the config it is handed.
        """
        assert self._stage.data_manager is not None
        validation_possible = _is_validation_possible(
            self._stage.module, self._stage.data_manager, self._limit_val_batches
        )
        if validation_possible:
            # Sanity check uses the validation step
            self._sanity_check = self._init_sanity_check(
                num_sanity_val_steps=self._num_sanity_val_steps,
                limit_val_batches=self._limit_val_batches,
            )

            self._validation_epoch = validation.ValidationEpoch(
                self._stage,
                fast_dev_run=self._fast_dev_run,
                limit_batches=self._limit_val_batches,
            )
        else:
            self._validation_epoch = None

        self._train_epoch = train.TrainEpoch(
            self._stage,
            self._optimizers,
            fast_dev_run=self._fast_dev_run,
            min_updates=self._min_updates,
            max_updates=self._max_updates,
            limit_batches=self._limit_train_batches,
            accumulate_grad_batches=self._accumulate_grad_batches,
            validation_epoch=self._validation_epoch,
            val_check_interval=self._val_check_interval,
            check_val_every_n_epoch=self._check_val_every_n_epoch,
        )

    def _init_sanity_check(
        self,
        *,
        num_sanity_val_steps: int | None,
        limit_val_batches: int | float | None,
    ) -> validation.ValidationEpoch | None:
        if not num_sanity_val_steps:
            return None

        if num_sanity_val_steps is not None and limit_val_batches is not None:
            batches_limit = min(num_sanity_val_steps, limit_val_batches)
        elif num_sanity_val_steps is not None:
            batches_limit = num_sanity_val_steps
        else:
            batches_limit = limit_val_batches

        return validation.ValidationEpoch(
            self._stage,
            name="sanity_check",
            fast_dev_run=self._fast_dev_run,
            limit_batches=batches_limit,
        )

    @override
    def _on_started(self):
        super()._on_started()
        if self._sanity_check:
            self._sanity_checking = True
            try:
                self._stage.run_loop(self._sanity_check)
            finally:
                self._sanity_checking = False

        self._stage.module.on_train_start(weakref.proxy(self._stage), weakref.proxy(self))

    @override
    def _on_iteration_starting(self):
        super()._on_iteration_starting()

        # Reload the dataloaders at the start of every epoch.  The first epoch's
        # dataloaders are loaded freshly during `_on_starting`, so only reset for
        # subsequent epochs where the reload interval is due.
        data_manager = self._stage.data_manager
        if (
            data_manager is not None
            and self._reload_dataloaders_every_n_epochs
            and self.epoch % self._reload_dataloaders_every_n_epochs == 0
        ):
            data_manager.reset()

    @override
    def _step(self) -> Any:
        assert self._train_epoch is not None, (
            "TrainEpoch was not constructed. `_on_starting` should have built it."
        )
        return self._stage.run_loop(self._train_epoch)

    @override
    def _done(self) -> bool:
        if super()._done():
            return True

        if self._max_updates is not None and self._stage.global_updates >= self._max_updates:
            rank_zero.rank_zero_debug(
                f"`{type(self).__name__}` done: `max_updates={self._max_updates!r}` reached."
            )
            return True

        return False

    @override
    def _on_stopping(self):
        super()._on_stopping()
        self._stage.module.on_train_end(weakref.proxy(self._stage), weakref.proxy(self))
