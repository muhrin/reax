from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Generic, Optional, TypedDict, TypeVar, Union

import beartype
import jax
import jaxtyping as jt
from lightning_utilities.core import rank_zero
import optax

from . import _module_hooks

if TYPE_CHECKING:
    import reax

__all__ = ("Module",)

MetricType = Union["reax.Metric", jax.typing.ArrayLike]
OutputT_co = TypeVar("OutputT_co", covariant=True)
BatchT = TypeVar("BatchT")
OptimizerData = tuple[optax.GradientTransformation, Any]

LossAndGradDict = TypedDict("LossAndGradDict", {"loss": jax.Array, "grad": jt.PyTree}, total=False)
LossAndGrad = tuple[jax.Array, jax.Array]
TrainOutput = Union[LossAndGrad, LossAndGradDict]


class Module(Generic[BatchT, OutputT_co], _module_hooks.ModuleHooks):
    example_input_array: Optional[BatchT]

    def __init__(self, rng_key: jax.Array = None):
        """Init function."""
        self._trainer: Optional["reax.Trainer"] = None
        self._rng_key = rng_key or jax.random.key(0)
        self._parameters = None
        self._automatic_optimization = True

    @property
    def automatic_optimization(self) -> bool:
        """Automatic optimization."""
        return self._automatic_optimization

    @automatic_optimization.setter
    def automatic_optimization(self, automatic_optimization: bool) -> None:
        """Automatic optimization."""
        self._automatic_optimization = automatic_optimization

    @property
    def trainer(self) -> "reax.Trainer":
        """Trainer function."""
        return self._trainer

    @trainer.setter
    def trainer(self, trainer):
        """Trainer function."""
        if self._trainer is not None and trainer is not None:
            raise RuntimeError("Cannot set trainer, it is already set.")

        self._trainer = trainer

    @property
    def global_updates(self) -> int:
        """Get the global number of optimizer updates."""
        return self._trainer.global_updates

    @property
    def current_epoch(self) -> int:
        """Get the current fitting epoch."""
        return self._trainer.current_epoch

    def parameters(self) -> Optional[jt.PyTree]:
        """Parameters function."""
        return self._parameters

    def set_parameters(self, params: jt.PyTree):
        """Set parameters."""
        self._parameters = params

    def rng_key(self, num=1) -> jax.Array:
        """Rng key."""
        return self._trainer.rng_key(num=num)

    def optimizers(self) -> Union["reax.Optimizer", list["reax.Optimizer"]]:
        """Optimizers function."""
        optimizers = self.trainer.optimizers

        # Check for a single optimiser
        if (
            isinstance(optimizers, list)
            and len(optimizers) == 1
            and isinstance(optimizers[0], optax.GradientTransformation)
        ):
            return optimizers[0]

        # Multiple optimisers
        return optimizers

    def setup(self, stage: "reax.Stage", batch: Any, /) -> None:
        """Called at the beginning of each stage.

        A chance to perform some setup on the module.
        """

    def training_step(self, batch: BatchT, batch_idx: int, /) -> Optional[TrainOutput]:
        """Train step."""

    def validation_step(self, batch: BatchT, batch_idx: int, /):
        """Validate step."""

    def predict_step(self, batch: BatchT, batch_idx: int, /) -> OutputT_co:
        """Make a model prediction and return the result."""

    def test_step(self, batch: BatchT, batch_idx: int, /):
        """Test step."""

    @jt.jaxtyped(typechecker=beartype.beartype)
    def configure_optimizers(
        self,
    ) -> Optional[Union[OptimizerData, Sequence[OptimizerData]]]:
        """Create the optimizer(s) to use during training."""
        return None

    def log(
        self,
        name: str,
        value: MetricType,
        *,
        prog_bar: bool = False,
        batch_size: Optional[int] = None,
        logger: Optional[bool] = None,
        on_step=True,
        on_epoch=True,
    ) -> None:
        """Log a key, value pair.

        Example::

            self.log('train_loss', loss)
        """
        trainer = self._trainer
        if trainer is None:
            # not an error to support testing the `*_step` methods without a `Trainer` reference
            rank_zero.rank_zero_warn(
                "`self.log()` was called before `self.trainer` was set. "
                "Probably, the model was not passed to `Trainer`"
            )
            return

        if logger and trainer.logger is None:
            rank_zero.rank_zero_warn(
                f"You called `self.log({name!r}, ..., logger=True)` but have no logger "
                f"configured. You can enable one by using `Trainer(logger=ALogger(...))`"
            )
        if logger is None:
            # we could set false here if there's no configured logger, however, we still need to
            # compute the "logged" metrics anyway because that's what the evaluation loops use as
            # return value
            logger = True

        trainer.log(
            name,
            value,
            prog_bar=prog_bar,
            batch_size=batch_size,
            logger=logger,
            on_step=on_step,
            on_epoch=on_epoch,
        )


PyTree = Any
InputT_co = TypeVar("InputT_co", covariant=True)
ModelT = Callable[[PyTree, InputT_co], OutputT_co]
LabelT_co = TypeVar("LabelT_co", covariant=True)
LossFn = Callable[[OutputT_co, LabelT_co], jax.Array]
