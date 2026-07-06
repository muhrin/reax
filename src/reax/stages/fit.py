import datetime
import enum
from typing import TYPE_CHECKING, Any, Final

import beartype
from flax import nnx
import jaxtyping as jt
from lightning_utilities.core import overrides
from typing_extensions import override

from . import _timer, common, stages, train, validation
from .. import data, exceptions, modules

if TYPE_CHECKING:
    import reax

__all__ = "FitEpoch", "Fit", "ValidationMode"


class ValidationMode(enum.Enum):
    OFF = enum.auto()  # No validation configured
    EVERY_N_STEPS = enum.auto()  # Global (check_val_every_n_epoch is None)
    EVERY_N_EPOCHS = enum.auto()  # Local (val_check_interval is 1.0)
    HYBRID_MID_EPOCH = enum.auto()  # Local (val_check_interval < 1.0 or int)


def _setup_val_check_batch(
    val_check_interval: int | float,
    max_batches: int | float,
    check_val_every_n_epoch: int | None,
    dataloader: "reax.DataLoader",
) -> int | float | None:
    """
    Calculates the concrete batch frequency for validation.

    Returns:
        int: The number of batches to wait between validation runs.
        float: float("inf") if validation should only happen at the end of an
               iterable stream with no known length.
        None: If validation is disabled.
    """
    # 1. Quick exit if no training batches are allowed
    if max_batches == 0:
        return None

    # Case 1: Fixed Step Interval (e.g., val_check_interval = 100)
    if isinstance(val_check_interval, int):
        val_check_batch = val_check_interval

        # Logic Guard: If we are in an epoch-based mode (EVERY_N_EPOCHS or HYBRID),
        # the interval shouldn't exceed the epoch length.
        if check_val_every_n_epoch is not None and val_check_batch > max_batches:
            raise ValueError(
                f"`val_check_interval` ({val_check_interval}) must be less than or equal to the "
                f"number of the training batches ({max_batches}) when "
                f"`check_val_every_n_epoch` is set. "
                "To validate across total training steps regardless of epochs, "
                "set `check_val_every_n_epoch=None`."
            )
        return val_check_batch

    # Case 2: Percentage-based Interval (e.g., val_check_interval = 0.5)
    # We need an "effective size" to calculate the percentage.
    # Priority: 1. User-defined max_batches (int) | 2. Dataloader __len__
    dataloader_size = data.sized_len(dataloader)

    effective_epoch_size = max_batches if isinstance(max_batches, int) else dataloader_size

    if effective_epoch_size is None:
        # If we have no way of knowing the epoch size (IterableDataset with no limit)
        if val_check_interval == 1.0:
            return float("inf")

        raise exceptions.MisconfigurationException(
            "When using an IterableDataset with no length limit, "
            "`val_check_interval` must be 1.0 (end of stream) or an integer."
        )

    # Calculate the batch count from percentage.
    # Use int() for truncation or round() for nearest-neighbor logic.
    # We use max(1, ...) to ensure we don't return 0 for very small epochs.
    val_check_batch = max(1, int(effective_epoch_size * val_check_interval))

    return val_check_batch


class FitEpoch(train.Train):
    """
    Orchestrates the execution of a single training epoch with integrated validation.

    This class manages the lifecycle of training batches, gradient accumulation,
    and determines when to intersperse validation runs based on `val_check_interval`
    and `check_val_every_n_epoch`.

    **Internal Logic Pseudocode:**

        # High-level execution flow within an epoch
        for batch_idx, batch in enumerate(train_dataloader):
            # 1. Perform training step (handled by superclass)
            outputs = training_step(batch)

            # 2. Check if validation should trigger
            if should_check_val(batch_idx, epoch, total_batch_idx):
                # Run the validation stage (Validate class)
                run_child_stage(validation_stage)

            # 3. Check for early stopping
            if stopper.should_stop:
                break

    Args:
        module: The Reax module to train.
        datamanager: Manager for training and validation data sources.
        optimizers: List of optimizers for the module parameters.
        engine: The compute engine (e.g., JAX/Device).
        fast_dev_run: Runs a small number of batches for debugging.
        min_updates/max_updates: Constraints on the number of optimization steps.
        limit_train_batches: Maximum number of training batches to process per epoch.
        accumulate_grad_batches: Number of batches to accumulate before an optimizer step.
        limit_val_batches: Maximum number of validation batches to process.
        val_check_interval: How often to run validation within a training epoch
            (float for percentage of epoch, int for fixed batch count).
        check_val_every_n_epoch: Frequency of validation runs across epochs.
        stopper: Optional early stopping utility.
    """

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        module: "reax.Module",
        datamanager: "reax.data.DataSourceManager",
        optimizers: list["reax.Optimizer"],
        engine: "reax.Engine",
        *,
        rngs: nnx.Rngs = None,
        fast_dev_run: bool | int = False,
        min_updates: int = 0,
        max_updates: int | float | None = None,
        limit_train_batches: int | float | None = 1.0,
        accumulate_grad_batches: int = 1,
        limit_val_batches: int | float | None = 1.0,
        val_check_interval: int | float | None = 1.0,
        check_val_every_n_epoch: int | None = 1,
        stopper: common.Stopper | None = None,
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

        super().__init__(
            module,
            datamanager,
            engine,
            optimizers,
            rngs=rngs,
            fast_dev_run=fast_dev_run,
            min_updates=min_updates,
            max_updates=max_updates,
            limit_batches=limit_train_batches,
            accumulate_grad_batches=accumulate_grad_batches,
            stopper=stopper,
        )
        # Params
        self._val_check_interval: Final[int | float | None] = val_check_interval
        self._check_val_every_n_epoch: Final[int | None] = check_val_every_n_epoch
        self._limit_val_batches: Final[int | float | None] = limit_val_batches

        # State
        self._validate: validation.Validate | None = None
        self._check_val_every_n_batch: int | float | None = None
        self._validate_mode = self._init_validate_mode(module, datamanager, limit_val_batches)

    @property
    def train_dataloader(self) -> "reax.DataLoader":
        """Train dataloader."""
        return self.dataloader

    @property
    def val_dataloader(self) -> "reax.DataLoader | None":
        """Train dataloader."""
        if self._validate is None:
            return None

        return self._validate.dataloader

    @property
    def val_check_interval(self) -> int | float | None:
        """Val check interval."""
        return self._val_check_interval

    @property
    def check_val_every_n_epoch(self) -> int | None:
        """Check val every n epoch."""
        return self._check_val_every_n_epoch

    @property
    def limit_train_batches(self) -> int | float | None:
        """The limit to the number of batches in a train epoch"""
        return self._limit_batches

    @property
    def limit_val_batches(self) -> int | float | None:
        """The limit to the number of batches in a validation epoch"""
        if self._validate is None:
            return None

        return self._validate.limit_batches

    @property
    def validate(self) -> validation.Validate | None:
        """Validate function."""
        return self._validate

    @property
    def validation_mode(self) -> ValidationMode | None:
        return self._validate_mode

    @override
    def _on_starting(self):
        """On starting."""
        super()._on_starting()

        # Now that the dataloader has prepared data, we can initialise validation
        self._init_validation()

        # Only the root stage does setup as this only needs to be done once per stage tree
        if self.is_root and self._module is not None:
            self._module.setup(self, next(iter(self.train_dataloader)))
            params = self._engine.to_device(self._module.parameters())
            self._module.set_parameters(params)

    @override
    def _on_iteration_finished(self, outputs: Any, /) -> None:
        """On iteration finished."""
        super()._on_iteration_finished(outputs)

        # We've finished the train iteration, so check if we should do a validation
        if self._should_check_val():
            assert self._validate is not None
            self._run_child(self._validate)

    @override
    def log(
        self,
        name: str,
        value: "jt.ArrayLike | reax.typing.MetricInstance",
        batch_size: int | None = None,
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

    def _should_check_val(self) -> bool:
        """Decide if we should run validation based on the current mode and progress."""
        # 1. Quick exit if validation is disabled or not configured
        if self._validate_mode == ValidationMode.OFF or self._validate is None:
            return False

        # 2. Handle Global Step-based validation (EVERY_N_STEPS)
        # This ignores epoch boundaries and uses the total count of batches processed.
        if self._validate_mode == ValidationMode.EVERY_N_STEPS:
            return (self.total_batch_idx + 1) % self._check_val_every_n_batch == 0

        # 3. Handle Epoch-aware modes (EVERY_N_EPOCHS and HYBRID)
        # First, check if we are in an epoch where validation is scheduled to run.
        # (e.g., if check_val_every_n_epoch = 2, we only check on epochs 1, 3, 5...)
        is_val_epoch = (self.epoch + 1) % self._check_val_every_n_epoch == 0
        if not is_val_epoch:
            return False

        # 4. Determine if the current batch within the epoch is a trigger point
        # For EVERY_N_EPOCHS: This usually triggers on the last batch of the epoch.
        # For HYBRID: This triggers at specific intervals (e.g., every 100 batches).
        if self._validate_mode == ValidationMode.EVERY_N_EPOCHS:
            # Trigger at the very end of the training dataloader
            return self._next_batch is None

        if self._validate_mode == ValidationMode.HYBRID_MID_EPOCH:
            # Trigger based on the local batch index within this specific epoch
            return self.batch_idx % self._check_val_every_n_batch == 0

        return False

    @staticmethod
    def _init_validate_mode(
        module: "reax.Module",
        datamanager: "reax.data.DataSourceManager",
        limit_val_batches: int | float | None,
    ) -> ValidationMode | None:
        if (
            not datamanager.has_dataloader("val")
            or limit_val_batches == 0.0
            or not overrides.is_overridden("validation_step", module, modules.Module)
        ):
            # No validation
            return ValidationMode.OFF

        return None

    def _init_validation(self):
        if self._validate_mode is not None or self._validate_mode == ValidationMode.OFF:
            return  # Already initialized

        module = self._module
        datamanager = self._datamanager
        assert module is not None
        assert datamanager is not None

        validate = validation.Validate(
            module,
            datamanager,
            self._engine,
            fast_dev_run=self._fast_dev_run,
            limit_batches=self._limit_val_batches,
        )

        # 1. Determine the validation mode
        if self._check_val_every_n_epoch is None:
            mode = ValidationMode.EVERY_N_STEPS
        elif self._val_check_interval == 1.0:
            mode = ValidationMode.EVERY_N_EPOCHS
        else:
            mode = ValidationMode.HYBRID_MID_EPOCH

        # 2. Determine the specific batch interval
        val_check_batch = _setup_val_check_batch(
            val_check_interval=self._val_check_interval,
            max_batches=self.max_batches,
            check_val_every_n_epoch=self._check_val_every_n_epoch,
            dataloader=self.train_dataloader,
        )

        self._validate = validate
        self._validate_mode = mode
        self._check_val_every_n_batch = val_check_batch


class Fit(stages.Stage):
    def __init__(
        self,
        module: "reax.Module",
        datamanager: "reax.data.DataSourceManager",
        optimizers: "list[reax.Optimizer]",
        engine: "reax.Engine",
        *,
        rngs: nnx.Rngs = None,
        fast_dev_run: bool | int = False,
        max_epochs: int | None = None,
        min_epochs: int = 0,
        min_updates: int = 0,
        max_updates: int | float | None = None,
        max_time: str | datetime.timedelta | dict[str, int] | None = None,
        limit_train_batches: int | float | None = 1.0,
        accumulate_grad_batches: int = 1,
        limit_val_batches: int | float | None = 1.0,
        val_check_interval: int | float | None = 1.0,
        check_val_every_n_epoch: int = 1,
        num_sanity_val_steps: int | None = 2,
        reload_dataloaders_every_n_epochs: int = 0,
    ):
        """Init function."""
        if (
            not isinstance(reload_dataloaders_every_n_epochs, int)
            or reload_dataloaders_every_n_epochs < 0
        ):
            raise exceptions.MisconfigurationException(
                f"`reload_dataloaders_every_n_epochs` should be an int >= 0, got "
                f"{reload_dataloaders_every_n_epochs}."
            )

        if fast_dev_run:
            max_epochs = 1
            num_sanity_val_steps = 0

        super().__init__(
            "fit",
            module,
            engine,
            rngs=rngs,
            datamanager=datamanager,
            max_iters=max_epochs,
            min_iters=min_epochs,
        )

        # Params
        self._num_sanity_val_steps: Final[int | None] = num_sanity_val_steps
        self._reload_dataloaders_every_n_epochs: Final[int] = reload_dataloaders_every_n_epochs

        # State
        self._timer = self._init_timer(max_time)
        if self._timer is not None:
            self.events.add_listener(self._timer)

        self._fit_epoch = FitEpoch(
            module,
            datamanager,
            optimizers,
            engine,
            rngs=rngs,
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

        self._sanity_check: validation.Validate | None = None
        if self._fit_epoch.validation_mode != ValidationMode.OFF and num_sanity_val_steps:
            if num_sanity_val_steps is not None and limit_val_batches is not None:
                batches_limit = min(num_sanity_val_steps, limit_val_batches)
            elif num_sanity_val_steps is not None:
                batches_limit = num_sanity_val_steps
            else:
                batches_limit = limit_val_batches

            self._sanity_check = validation.Validate(
                module,
                datamanager,
                engine,
                fast_dev_run=fast_dev_run,
                limit_batches=batches_limit,
                name="sanity_check",
                enable_checkpointing=False,
            )

    @staticmethod
    def _init_timer(
        max_time: str | datetime.timedelta | dict[str, int] | None,
    ) -> _timer.StageTimer | None:
        if max_time is None:
            return None

        timer = _timer.StageTimer(max_time)
        return timer

    @property
    def epoch(self) -> int:
        return self.iteration

    @property
    def fast_dev_run(self) -> bool | int:
        return self._fit_epoch.fast_dev_run

    @property
    def sanity_checking(self) -> bool:
        """`True` if currently sanity checking, `False` otherwise"""
        return self._sanity_check is not None and self._child is self._sanity_check

    @property
    def num_sanity_val_steps(self) -> int | None:
        return self._num_sanity_val_steps

    @property
    def max_epochs(self) -> int | None:
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
    def validate(self) -> validation.Validate | None:
        """Validate function."""
        return self._fit_epoch.validate

    @property
    def validation_mode(self) -> ValidationMode | None:
        return self._fit_epoch.validation_mode

    @property
    def limit_train_batches(self) -> int | float | None:
        """The limit on the number of training batches per epoch"""
        return self._fit_epoch.limit_train_batches

    @property
    def num_training_batches(self) -> int | float | None:
        return self._fit_epoch.num_training_batches

    @property
    def limit_val_batches(self) -> int | float | None:
        """The limit on the number of training batches per epoch"""
        return self._fit_epoch.limit_val_batches

    @property
    def num_val_batches(self) -> int | float | None:
        if not self.enable_validation:
            return None

        return self._fit_epoch.validate.num_batches

    @property
    def val_check_interval(self):
        """Val check interval."""
        return self._fit_epoch.val_check_interval

    @property
    def check_val_every_n_epoch(self) -> int | None:
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
    def val_dataloader(self) -> "reax.DataLoader | None":
        """Train dataloader."""
        return self._datamanager.get_dataloader("val")

    @property
    def val_dataloaders(self) -> "reax.DataLoader | None":
        return self.val_dataloader

    @property
    def timer(self) -> "reax.stages.StageTimer | None":
        return self._timer

    @override
    def log(
        self,
        name: str,
        value,
        batch_size: int | None = None,
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
