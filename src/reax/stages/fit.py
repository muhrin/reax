from typing import TYPE_CHECKING, Any, Optional, Union
import weakref

import beartype
import jax
import jaxtyping as jt
from lightning_utilities.core import overrides
from typing_extensions import override

from . import common, stages, train, validation
from .. import data, exceptions, modules

if TYPE_CHECKING:
    import reax

__all__ = "FitEpoch", "Fit"


class FitEpoch(train.Train):
    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        module: "reax.Module",
        train_dataloaders: "reax.DataLoader",
        val_dataloaders: "Optional[reax.DataLoader]",
        optimizers: list["reax.Optimizer"],
        strategy: "reax.Strategy",
        *,
        min_updates: Optional[int] = None,
        max_updates: Union[int, float] = float("inf"),
        limit_train_batches: Optional[Union[int, float]] = 1.0,
        accumulate_grad_batches: int = 1,
        limit_val_batches: Optional[Union[int, float]] = 1.0,
        val_check_interval: Optional[Union[int, float]] = 1.0,
        check_val_every_n_epoch: int = 1,
        parent: Optional["reax.Stage"] = None,
        stopper: Optional[common.Stopper] = None,
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
            self._validate = validation.Validate(
                module,
                val_dataloaders,
                strategy,
                max_batches=limit_val_batches,
                parent=weakref.proxy(self),
            )

    @property
    def train_dataloader(self) -> "reax.DataLoader":
        return self.dataloader

    @property
    def val_check_interval(self) -> Optional[Union[int, float]]:
        """Val check interval."""
        return self._val_check_interval

    @property
    def check_val_every_n_epoch(self) -> Optional[int]:
        """Check val every n epoch."""
        return self._check_val_every_n_epoch

    @property
    def validate(self) -> Optional[validation.Validate]:
        """Validate function."""
        return self._validate

    @override
    def _on_starting(self):
        """On starting."""
        super()._on_starting()

        # Only the root stage does setup as this only needs to be done once per stage tree
        if self.is_root and self._module is not None:
            self._module.setup(self, next(iter(self.train_dataloader)))
            params = self._strategy.to_device(self._module.parameters())
            self._module.set_parameters(params)

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


class Fit(stages.Stage):
    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        module: "reax.Module",
        train_dataloaders: "reax.DataLoader",
        val_dataloaders: "Optional[reax.DataLoader]",
        optimizers: list["reax.Optimizer"],
        strategy: "reax.Strategy",
        *,
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
            "fit",
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
    def validate(self) -> Optional[validation.Validate]:
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

    @property
    def train_dataloader(self) -> "reax.DataLoader":
        return self._fit_epoch.train_dataloader

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
    def _on_starting(self):
        """On starting."""
        super()._on_starting()

        # Only the root stage does setup as this only needs to be done once per stage tree
        if self.is_root and self._module is not None:
            self._module.setup(self, next(iter(self.train_dataloader)))
            params = self._strategy.to_device(self._module.parameters())
            self._module.set_parameters(params)

    @override
    def _step(self) -> Any:
        """Step function."""
        self._run_child(self._fit_epoch)
