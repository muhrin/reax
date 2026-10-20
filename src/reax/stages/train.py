import enum
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable
import weakref

import beartype
import jaxtyping as jt
import optax
from typing_extensions import override

from .. import data, exceptions
from .. import optimizers as optimizers_
from ..lightning import rank_zero
from . import _loops, common, validation

if TYPE_CHECKING:
    import reax

__all__ = "TrainEpoch", "TrainStage", "ValidationMode"


@runtime_checkable
class TrainStage(_loops.ModuleBatchHost, Protocol):
    """Structural interface a stage must provide to run a :class:`TrainEpoch`.

    ``TrainEpoch`` is generic over this protocol rather than the concrete
    :class:`~reax.stages.Fit` stage, so custom fitting stages can drive training
    without inheriting from ``Fit`` as long as they expose the members below.
    """

    @property
    def epoch(self) -> int: ...

    @property
    def min_epochs(self) -> int: ...

    global_updates: int


class TrainEpoch(_loops.LoopOverBatches[TrainStage]):
    """Loop over training batches within a single epoch.

    Validation is enabled by supplying a :class:`~reax.stages.ValidationEpoch`
    (``None`` disables it). The schedule is derived from the validation config
    (``val_check_interval`` and ``check_val_every_n_epoch``), which this loop
    turns into a :class:`~reax.stages.ValidationTrigger` lazily once its
    dataloader and epoch size are known.
    """

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        stage: TrainStage,
        optimizers: "list[reax.Optimizer]",
        *,
        fast_dev_run: bool | int = False,
        min_updates: int = 0,
        max_updates: int | float | None = None,
        limit_batches: int | float | None = None,
        accumulate_grad_batches: int = 1,
        validation_epoch: validation.ValidationEpoch | None = None,
        val_check_interval: int | float = 1.0,
        check_val_every_n_epoch: int | None = 1,
    ):
        super().__init__("train", stage, fast_dev_run=fast_dev_run, limit_batches=limit_batches)

        # Params
        self._fast_dev_run: Final[bool | int] = fast_dev_run
        self._min_updates: Final[int] = min_updates
        self._max_updates: Final[int | float | None] = max_updates
        self._accumulate_grad_batches: Final[int] = accumulate_grad_batches
        self._val_check_interval: Final[int | float] = val_check_interval
        self._check_val_every_n_epoch: Final[int | None] = check_val_every_n_epoch
        self._validation_mode = _determine_validation_mode(
            self._check_val_every_n_epoch, self._val_check_interval
        )
        # State
        self._optimizers: list[optimizers_.Optimizer] = optimizers
        self._validation_epoch: validation.ValidationEpoch | None = validation_epoch
        self._validation_trigger: ValidationTrigger | None = None

        self._stopper = common.Stopper([lambda: self.updates >= self._min_updates])
        self._warning_cache = rank_zero.WarningCache()

    @property
    def _trigger(self) -> "ValidationTrigger | None":
        if self._validation_epoch is None:
            return None
        if self._validation_trigger is None:
            self._validation_trigger = self._build_trigger()
        return self._validation_trigger

    def _build_trigger(self) -> "ValidationTrigger":
        # By here the loop has fetched its dataloader and knows its epoch size.
        val_check_batch = _setup_val_check_batch(
            val_check_interval=self._val_check_interval,
            max_batches=self.max_batches,
            check_val_every_n_epoch=self._check_val_every_n_epoch,
            dataloader=self.dataloader,
        )
        return ValidationTrigger(
            self._validation_mode,
            every_n_epochs=self._check_val_every_n_epoch,
            every_n_batches=val_check_batch,
        )

    @property
    def validation_mode(self) -> "reax.stages.ValidationMode":
        return self._validation_mode

    @property
    def num_training_batches(self) -> int | float | None:
        """Number of training batches to run this epoch (``None`` if unbounded)."""
        return self.max_batches

    @property
    def num_val_batches(self) -> int | float | None:
        """Number of validation batches that will run this epoch (``None`` if unset)."""
        if self._validation_epoch is None:
            return None
        return self._validation_epoch.num_batches

    @property
    def updates(self) -> int:
        """Get the number of gradient updates that have been applied."""
        return int(sum(opt.update_count for opt in self._optimizers))

    @property
    def train_dataloader(self) -> "reax.DataLoader | None":
        """Get the training dataloader."""
        return self.dataloader

    @property
    def val_dataloader(self) -> "reax.DataLoader | None":
        """Get the validation dataloader."""
        if self._validation_epoch is None:
            return None
        return self._validation_epoch.dataloader

    @property
    def optimizers(self) -> "list[reax.Optimizer]":
        return self._optimizers

    @override
    def _on_starting(self):
        super()._on_starting()

        self._stage.configure_module(self._next_batch)
        self._configure_optimizers()

    @override
    def _on_started(self):
        super()._on_started()
        self._stage.module.on_train_epoch_start(weakref.proxy(self._stage), weakref.proxy(self))

    @override
    def _on_iteration_starting(self):
        super()._on_iteration_starting()
        self._stage.module.on_train_batch_start(
            weakref.proxy(self._stage), weakref.proxy(self), self.batch
        )

    @override
    def _step(self) -> Any:
        module = self._stage.module
        optimizer = self._optimizers[0]

        if module.parameters() is None:
            raise exceptions.MisconfigurationException(
                "Module does not have any parameters set, this should have been done in "
                ".configure_model()."
            )

        res = module.training_step(self.batch, self._iter)
        if module.automatic_optimization:
            if isinstance(res, dict):
                grad = res["grad"]
                loss = res.get("loss", None)
            elif isinstance(res, tuple):
                loss, grad = res
            else:
                raise RuntimeError(
                    f"Unexpected return type from `training_step`: {type(res)}. "
                    "Expected a dict or a tuple of (loss, grad)."
                )

            module.on_before_optimizer_step(optimizer, grad)
            updates_before = int(optimizer.update_count)
            opt = optimizer.update_module(module, grad, value=loss)
            self._optimizers = [opt]
            # Update the stage's global updates count only when the optimizer
            # actually performed a step (i.e. `accumulate_grad_batches` flushed).
            self._stage.global_updates += int(opt.update_count) - updates_before

        if (self._min_updates is None or self.updates >= self._min_updates) and (
            self._max_updates is not None and self.updates >= self._max_updates
        ):
            self.stop("Max updates reached")

        return res

    @override
    def _on_iteration_finishing(self, outputs: Any, /):
        super()._on_iteration_finishing(outputs)
        self._stage.module.on_train_batch_end(
            weakref.proxy(self._stage), weakref.proxy(self), outputs
        )

    @override
    def _on_iteration_finished(self, outputs: Any, /) -> None:
        super()._on_iteration_finished(outputs)

        if (
            self._validation_epoch is not None
            and self._trigger is not None
            and self._trigger.should_run(
                epoch=self._stage.epoch,
                batch_idx=self.batch_idx,
                total_batch_idx=self.total_batch_idx,
                is_last_batch=self._next_batch is None,
            )
        ):
            self._stage.run_loop(self._validation_epoch)

    @override
    def _on_epoch_end(self) -> None:
        self._stage.module.on_train_epoch_end(weakref.proxy(self._stage), weakref.proxy(self))

    @override
    def _done(self) -> bool:
        """Done function."""
        if super()._done():
            return True

        if self.max_batches is not None and self.batch_idx + 1 >= self.max_batches:
            rank_zero.rank_zero_debug(
                f"`{type(self).__name__}` done: max_batches.{self.max_batches!r}` reached."
            )
            return True

        if self._max_updates is not None and self.updates >= self._max_updates:
            rank_zero.rank_zero_debug(
                f"`{type(self).__name__}` done: `max_updates={self._max_updates!r}` reached."
            )
            return True

        return False

    @override
    def _stopping(self) -> bool:
        if self._done():
            return True

        if self._stage.stop_requested:
            if self._stage.can_stop:
                rank_zero.rank_zero_debug(
                    f"`{type(self).__name__}` stopped: `{type(self).__name__}.should_stop` was set."
                )
            else:
                lines = ["Trainer was signaled to stop but the required "]
                min_epochs = self._stage.min_epochs
                if min_epochs:
                    lines.append(f"`min_epochs={min_epochs!r}` or")
                lines.append(
                    f" `min_steps={self._min_updates!r}` has not been met. "
                    f"Training will continue..."
                )
                self._warning_cache.info("".join(lines))

            return self._stage.can_stop

        return False

    def _configure_optimizers(self):
        if not self._optimizers:
            module = self._stage.module
            engine = self._stage.engine

            opts = module.configure_optimizers()
            if opts is None:
                rank_zero.rank_zero_warn(
                    "`reax.Module.configure_optimizers` returned `None`, this fit will run "
                    "with no optimizer"
                )
                params = module.parameters()
                if params is None:
                    raise RuntimeError(
                        "Module does not have any parameters set, this should have been done in "
                        ".configure_model()."
                    )
                opt = optimizers_.mock_optimizer
                opts = opt, opt.init(params)

            if not isinstance(opts, list):
                opts = [opts]

            optimizers: list[optimizers_.Optimizer] = []
            for opt, state in opts:
                state = engine.to_device(state)
                if self._accumulate_grad_batches > 1:
                    stepper = optax.MultiSteps(opt, every_k_schedule=self._accumulate_grad_batches)
                    state = stepper.init(module.parameters())
                    opt = stepper.gradient_transformation()

                optimizers.append(optimizers_.Optimizer(opt, state))

            self._optimizers = optimizers


class ValidationMode(enum.Enum):
    """How validation is scheduled relative to training."""

    OFF = enum.auto()
    EVERY_N_STEPS = enum.auto()
    EVERY_N_EPOCHS = enum.auto()
    HYBRID_MID_EPOCH = enum.auto()


def _determine_validation_mode(
    check_val_every_n_epoch: int | None,
    val_check_interval: int | float,
) -> ValidationMode:
    """Return the validation mode implied by the given schedule configuration."""
    if check_val_every_n_epoch is None:
        return ValidationMode.EVERY_N_STEPS
    if val_check_interval == 1.0:
        return ValidationMode.EVERY_N_EPOCHS
    return ValidationMode.HYBRID_MID_EPOCH


class ValidationTrigger:
    """Decides whether validation should run at a given point in training."""

    def __init__(
        self,
        mode: ValidationMode,
        *,
        every_n_epochs: int | None = 1,
        every_n_batches: int | float | None = None,
    ) -> None:
        if mode is ValidationMode.OFF:
            every_n_epochs = None
            every_n_batches = None
        else:
            if every_n_epochs is not None and every_n_epochs < 1:
                raise exceptions.MisconfigurationException(
                    f"`every_n_epochs` must be >= 1 or None, got {every_n_epochs}"
                )
            if every_n_batches is not None and every_n_batches < 1:
                raise exceptions.MisconfigurationException(
                    f"`every_n_batches` must be >= 1 or None, got {every_n_batches}"
                )
            if mode in (ValidationMode.EVERY_N_STEPS, ValidationMode.HYBRID_MID_EPOCH):
                if every_n_batches is None:
                    raise exceptions.MisconfigurationException(
                        f"`every_n_batches` is required for mode {mode.name}"
                    )
            if mode in (ValidationMode.EVERY_N_EPOCHS, ValidationMode.HYBRID_MID_EPOCH):
                if every_n_epochs is None:
                    raise exceptions.MisconfigurationException(
                        f"`every_n_epochs` is required for mode {mode.name}"
                    )

        self._mode: Final[ValidationMode] = mode
        self._every_n_epochs: Final[int | None] = every_n_epochs
        self._every_n_batches: Final[int | float | None] = every_n_batches

    @property
    def mode(self) -> ValidationMode:
        return self._mode

    @property
    def every_n_epochs(self) -> int | None:
        return self._every_n_epochs

    @property
    def every_n_batches(self) -> int | float | None:
        return self._every_n_batches

    def should_run(
        self,
        *,
        epoch: int,
        batch_idx: int,
        total_batch_idx: int,
        is_last_batch: bool,
    ) -> bool:
        """Whether validation should run at the given point in training.

        The check depends on the configured mode: ``EVERY_N_STEPS`` fires on
        a fixed global-batch cadence, ``EVERY_N_EPOCHS`` only at the end of
        the epoch, and ``HYBRID_MID_EPOCH`` at mid-epoch steps and at the end.
        """
        mode = self._mode

        if mode is ValidationMode.OFF:
            return False

        if mode is ValidationMode.EVERY_N_STEPS:
            assert self._every_n_batches is not None
            interval = int(self._every_n_batches)
            return (total_batch_idx + 1) % interval == 0

        assert self._every_n_epochs is not None
        if (epoch + 1) % self._every_n_epochs != 0:
            return False

        if mode is ValidationMode.EVERY_N_EPOCHS:
            return is_last_batch

        if mode is ValidationMode.HYBRID_MID_EPOCH:
            assert self._every_n_batches is not None
            interval = int(self._every_n_batches)
            return (batch_idx + 1) % interval == 0

        raise AssertionError(f"Unhandled validation mode: {mode}")


def _setup_val_check_batch(
    val_check_interval: int | float,
    max_batches: int | float | None,
    check_val_every_n_epoch: int | None,
    dataloader: "reax.DataLoader",
) -> int | float | None:
    """Convert a user-facing `val_check_interval` into a concrete batch count."""
    if max_batches == 0:
        return None

    if isinstance(val_check_interval, int):
        val_check_batch = val_check_interval
        if (
            check_val_every_n_epoch is not None
            and max_batches is not None
            and val_check_batch > max_batches
        ):
            raise ValueError(
                f"`val_check_interval` ({val_check_interval}) must be less than or equal to the "
                f"number of the training batches ({max_batches}) when "
                f"`check_val_every_n_epoch` is set. "
                "To validate across total training steps regardless of epochs, "
                "set `check_val_every_n_epoch=None`."
            )
        return val_check_batch

    dataloader_size = data.sized_len(dataloader)
    effective_epoch_size = max_batches if isinstance(max_batches, int) else dataloader_size

    if effective_epoch_size is None:
        if val_check_interval == 1.0:
            return float("inf")
        raise exceptions.MisconfigurationException(
            "When using an IterableDataset with no length limit, "
            "`val_check_interval` must be 1.0 (end of stream) or an integer."
        )

    return max(1, int(effective_epoch_size * val_check_interval))
