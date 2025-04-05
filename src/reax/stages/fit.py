from typing import TYPE_CHECKING, Any, Final, Optional, Union

import beartype
import jax
import jaxtyping as jt
from lightning_utilities.core import overrides
from typing_extensions import override

from reax import data, exceptions, modules

from . import common, stages, train, validation

if TYPE_CHECKING:
    import reax

__all__ = "FitEpoch", "Fit"


class FitEpoch(train.Train):
    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        module: "reax.Module",
        datamanager: "reax.data.DataSourceManager",
        optimizers: list["reax.Optimizer"],
        strategy: "reax.Strategy",
        rng: "reax.Generator",
        *,
        fast_dev_run: Union[bool, int] = False,
        min_updates: Optional[int] = None,
        max_updates: Optional[Union[int, float]] = None,
        limit_train_batches: Optional[Union[int, float]] = 1.0,
        accumulate_grad_batches: int = 1,
        limit_val_batches: Optional[Union[int, float]] = 1.0,
        val_check_interval: Optional[Union[int, float]] = 1.0,
        check_val_every_n_epoch: int = 1,
        stopper: Optional[common.Stopper] = None,
    ):
        """Init function."""
        if fast_dev_run:
            if isinstance(fast_dev_run, int):
                if fast_dev_run < 0:
                    raise exceptions.MisconfigurationException("`fast_dev_run` should be >= 0")
                if fast_dev_run == 1:
                    fast_dev_run = True

            num_batches = 1 if fast_dev_run is True else fast_dev_run
            limit_train_batches = num_batches
            limit_val_batches = num_batches
            val_check_interval = 1.0
            check_val_every_n_epoch = 1

        super().__init__(
            module,
            datamanager,
            strategy,
            optimizers,
            rng,
            fast_dev_run=fast_dev_run,
            min_updates=min_updates,
            max_updates=max_updates,
            limit_batches=limit_train_batches,
            accumulate_grad_batches=accumulate_grad_batches,
            stopper=stopper,
        )
        # Params
        self._val_check_interval: Final[Union[int, float]] = val_check_interval
        self._check_val_every_n_epoch: Final[int] = check_val_every_n_epoch

        # State
        if (
            not self._datamanager.has_dataloader("val")
            or limit_val_batches == 0.0
            or not overrides.is_overridden("validation_step", module, modules.Module)
        ):
            # No validation
            self._validate = None
        else:
            self._validate = validation.Validate(
                module,
                datamanager,
                strategy,
                fast_dev_run=fast_dev_run,
                limit_batches=limit_val_batches,
            )
        self._val_check_batch = None

    @property
    def train_dataloader(self) -> "reax.DataLoader":
        """Train dataloader."""
        return self.dataloader

    @property
    def val_dataloader(self) -> "Optional[reax.DataLoader]":
        """Train dataloader."""
        if self._validate is None:
            return None

        return self._validate.dataloader

    @property
    def val_check_interval(self) -> Optional[Union[int, float]]:
        """Val check interval."""
        return self._val_check_interval

    @property
    def check_val_every_n_epoch(self) -> Optional[int]:
        """Check val every n epoch."""
        return self._check_val_every_n_epoch

    @property
    def limit_train_batches(self) -> Optional[Union[int, float]]:
        """The limit to the number of batches in a train epoch"""
        return self._limit_batches

    @property
    def limit_val_batches(self) -> Optional[Union[int, float]]:
        """The limit to the number of batches in a validation epoch"""
        if self._validate is None:
            return None

        return self._validate.limit_batches

    @property
    def validate(self) -> Optional[validation.Validate]:
        """Validate function."""
        return self._validate

    @override
    def _on_starting(self):
        """On starting."""
        super()._on_starting()
        # The superclass will have prepared the dataloader so we can now configure when to check
        # validation (if at all)
        self._val_check_batch = self._setup_val_check_batch_(
            self._val_check_interval,
            self.max_batches,
            self._check_val_every_n_epoch,
            self.train_dataloader,
        )

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
        # if (
        #     isinstance(self._val_check_interval, int)
        #     and self.iteration % self._val_check_interval == 0
        # ):
        #     self._run_child(self._validate)

        if self._should_check_val():
            self._run_child(self._validate)

    @override
    def _on_stopping(self) -> None:
        """On stopping."""
        # if (
        #     self._validate is not None
        #     and self._check_val_every_n_epoch is not None
        #     and (self.epoch + 1) % self._check_val_every_n_epoch == 0
        # ):
        #     self._run_child(self._validate)
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

    def _should_check_val(self):
        """Decide if we should run validation."""
        if not self._should_check_val_epoch():
            return False

        # val_check_batch is inf for iterable datasets with no length defined
        is_infinite_dataset = self._val_check_batch == float("inf")
        # is_last_batch = self.batch_progress.is_last_batch
        # if is_last_batch and (
        #     is_infinite_dataset or isinstance(data_fetcher, _DataLoaderIterDataFetcher)
        # ):
        #     return True

        if self.should_stop:
            # allow validation if requesting to stop early through `Trainer.should_stop`
            # (e.g. by early stopping) and when the loop allows to stop (min_epochs/steps met)
            return True

        # TODO: let training/eval loop handle logic around limit_*_batches and val_check_batch
        # is_val_check_batch = is_last_batch
        is_val_check_batch = False
        if isinstance(self.limit_train_batches, int) and is_infinite_dataset:
            is_val_check_batch = (self.batch_idx + 1) % self.limit_train_batches == 0
        elif self._val_check_batch != float("inf"):
            # if `check_val_every_n_epoch is `None`, run a validation loop every n training batches
            # else condition it based on the batch_idx of the current epoch
            current_iteration = (
                self.total_batch_idx if self.check_val_every_n_epoch is None else self.batch_idx
            )
            is_val_check_batch = (current_iteration) % self._val_check_batch == 0

        return is_val_check_batch

    @staticmethod
    def _setup_val_check_batch_(
        val_check_interval: Union[int, float],
        max_batches: Union[int, float],
        check_val_every_n_epoch: Optional[int],
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
        datamanager: "reax.data.DataSourceManager",
        optimizers: list["reax.Optimizer"],
        strategy: "reax.Strategy",
        rng: "reax.Generator",
        *,
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
        num_sanity_val_steps: Optional[int] = 2,
        reload_dataloaders_every_n_epochs: int = 0,
    ):
        """Init function."""
        if fast_dev_run:
            max_epochs = 1
            num_sanity_val_steps = 0

        super().__init__(
            "fit",
            module,
            strategy,
            rng,
            datamanager=datamanager,
            max_iters=max_epochs,
            min_iters=min_epochs,
        )

        # Params
        self._num_sanity_val_steps: Final[Optional[int]] = num_sanity_val_steps
        self._reload_dataloaders_every_n_epochs: Final[int] = reload_dataloaders_every_n_epochs

        # State
        self._fit_epoch = FitEpoch(
            module,
            datamanager,
            optimizers,
            strategy,
            rng,
            fast_dev_run=fast_dev_run,
            min_updates=min_updates,
            max_updates=max_updates,
            limit_train_batches=limit_train_batches,
            accumulate_grad_batches=accumulate_grad_batches,
            limit_val_batches=limit_val_batches,
            val_check_interval=val_check_interval,
            check_val_every_n_epoch=check_val_every_n_epoch,
            stopper=self._stopper,
        )
        self._sanity_check: Optional[validation.Validate] = None
        if self._fit_epoch.validate and num_sanity_val_steps:
            self._sanity_check = validation.Validate(
                module,
                datamanager,
                strategy,
                fast_dev_run=fast_dev_run,
                limit_batches=num_sanity_val_steps,
                name="sanity_check",
                enable_checkpointing=False,
            )

    @property
    def epoch(self) -> int:
        return self.iteration

    @property
    def fast_dev_run(self) -> Union[bool, int]:
        return self._fit_epoch.fast_dev_run

    @property
    def sanity_checking(self) -> bool:
        """`True` if currently sanity checking, `False` otherwise"""
        return self._sanity_check is not None and self._child is self._sanity_check

    @property
    def max_epochs(self) -> Optional[int]:
        return self.max_iters

    @property
    def updates(self) -> int:
        """Updates function."""
        return self._fit_epoch.updates

    @property
    def enable_validation(self) -> bool:
        """Returns `True` of validation is enabled, `False` otherwise"""
        return self._fit_epoch.validate is not None

    @property
    def validate(self) -> Optional[validation.Validate]:
        """Validate function."""
        return self._fit_epoch.validate

    @property
    def limit_train_batches(self) -> Optional[Union[int, float]]:
        """The limit on the number of training batches per epoch"""
        return self._fit_epoch.limit_train_batches

    @property
    def num_training_batches(self) -> Optional[Union[int, float]]:
        return self._fit_epoch.num_training_batches

    @property
    def limit_val_batches(self) -> Optional[Union[int, float]]:
        """The limit on the number of training batches per epoch"""
        return self._fit_epoch.limit_val_batches

    @property
    def num_val_batches(self) -> Optional[Union[int, float]]:
        if not self.enable_validation:
            return None

        return self._fit_epoch.validate.num_batches

    @property
    def num_sanity_val_steps(self) -> Optional[int]:
        return self._num_sanity_val_steps

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

    @property
    def train_dataloaders(self) -> "reax.DataLoader":
        return self.train_dataloader

    @property
    def val_dataloader(self) -> "reax.DataLoader":
        """Train dataloader."""
        return self._fit_epoch.val_dataloader

    @property
    def val_dataloaders(self) -> "reax.DataLoader":
        return self.val_dataloader

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
        if self._child is None:
            raise RuntimeError(
                "Fitting is not running a train, validation or sanity check epoch, "
                "so cannot currently log anything"
            )

        self._child.log(name, value, batch_size, prog_bar, logger, on_step, on_epoch)

    @override
    def _on_iteration_starting(self):
        super()._on_iteration_starting()
        if (
            self._reload_dataloaders_every_n_epochs
            and self.epoch % self._reload_dataloaders_every_n_epochs == 0
        ):
            self._datamanager.reset()

    @override
    def _step(self) -> Any:
        """Step function."""
        if self.iteration == 0 and self._sanity_check:
            self._run_child(self._sanity_check)

        self._run_child(self._fit_epoch)
