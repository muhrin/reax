from typing import TYPE_CHECKING, Any, Final, Optional, Union
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
        optimizers: list["reax.Optimizer"],
        strategy: "reax.Strategy",
        rng: "reax.Generator",
        *,
        train_dataloaders: "Optional[reax.DataLoader]" = None,
        val_dataloaders: "Optional[reax.DataLoader]" = None,
        datamodule: "Optional[reax.DataModule]" = None,
        fast_dev_run: Union[bool, int] = False,
        min_updates: Optional[int] = None,
        max_updates: Optional[Union[int, float]] = None,
        limit_train_batches: Optional[Union[int, float]] = 1.0,
        accumulate_grad_batches: int = 1,
        limit_val_batches: Optional[Union[int, float]] = 1.0,
        val_check_interval: Optional[Union[int, float]] = 1.0,
        check_val_every_n_epoch: int = 1,
        parent: Optional["reax.Stage"] = None,
        stopper: Optional[common.Stopper] = None,
    ):
        """Init function."""
        if train_dataloaders is None:
            datamanager = common.get_datasource(datamodule, module)
            train_dataloaders = datamanager.get_loader_proxy("train_dataloader")
            val_dataloaders = datamanager.get_loader_proxy("val_dataloader")
        else:
            datamanager = None

        if fast_dev_run:
            num_batches = 1 if fast_dev_run is True else fast_dev_run
            limit_train_batches = num_batches
            limit_val_batches = num_batches
            val_check_interval = 1.0
            check_val_every_n_epoch = 1

        super().__init__(
            module,
            strategy,
            optimizers,
            rng,
            dataloader=train_dataloaders,
            fast_dev_run=fast_dev_run,
            min_updates=min_updates,
            max_updates=max_updates,
            limit_batches=limit_train_batches,
            accumulate_grad_batches=accumulate_grad_batches,
            parent=parent,
            stopper=stopper,
            datamanager=datamanager,
        )
        # Params
        self._val_check_interval: Final[Union[int, float]] = val_check_interval
        self._check_val_every_n_epoch: Final[int] = check_val_every_n_epoch

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
                strategy,
                dataloader=val_dataloaders,
                fast_dev_run=fast_dev_run,
                limit_batches=limit_val_batches,
                parent=weakref.proxy(self),
            )

    @property
    def train_dataloader(self) -> "reax.DataLoader":
        """Train dataloader."""
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
    def _on_iteration_finished(self, outputs: Any, /) -> None:
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
                    f" to the number of the training batches ({max_batches}). "
                    "If you want to disable validation set `limit_val_batches` to 0.0 instead."
                    "If you want to validate based on the total training batches, set "
                    "`check_val_every_n_epoch=None`."
                )
        else:
            dataloader_size = data.sized_len(dataloader)
            has_len_all_ranks_ = dataloader_size is not None
            if not has_len_all_ranks_:
                if val_check_interval == 1.0:
                    val_check_batch = None
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
        #     rank_zero.rank_zero_warn(
        #         f"The number of training batches ({max_batches}) is smaller than the logging "
        #         f"interval. Trainer(log_every_n_steps={log_every_n_steps}). Set a lower value "
        #         f"for log_every_n_steps if you want to see logs for the training epoch.",
        #         category=PossibleUserWarning,
        #     )

        return val_check_batch


class Fit(stages.Stage):
    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        module: "reax.Module",
        optimizers: list["reax.Optimizer"],
        strategy: "reax.Strategy",
        rng: "reax.Generator",
        *,
        train_dataloaders: "Optional[reax.DataLoader]" = None,
        val_dataloaders: "Optional[reax.DataLoader]" = None,
        datamodule: "Optional[reax.DataModule]" = None,
        fast_dev_run: Union[bool, int] = False,
        max_epochs: Optional[int] = None,
        min_epochs: int = 0,
        min_updates: int = 0,
        max_updates: Optional[Union[int, float]] = None,
        limit_train_batches: Optional[Union[int, float]] = 1.0,
        accumulate_grad_batches: int = 1,
        limit_val_batches: Optional[Union[int, float]] = 1.0,
        val_check_interval: Optional[Union[int, float]] = 1.0,
        check_val_every_n_epoch: int = 1,
        reload_dataloaders_every_n_epochs: int = 0,
        parent: Optional["reax.Stage"] = None,
    ):
        """Init function."""
        if fast_dev_run:
            max_epochs = 1

        super().__init__(
            "fit",
            module,
            strategy,
            rng,
            max_iters=max_epochs,
            min_iters=min_epochs,
            parent=parent,
        )

        # Params
        self._reload_dataloaders_every_n_epochs = reload_dataloaders_every_n_epochs

        # State
        self._fit_epoch = FitEpoch(
            module,
            optimizers,
            strategy,
            rng,
            train_dataloaders=train_dataloaders,
            val_dataloaders=val_dataloaders,
            datamodule=datamodule,
            fast_dev_run=fast_dev_run,
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
        """Train dataloader."""
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
    def _step(self) -> Any:
        """Step function."""
        self._run_child(self._fit_epoch)
