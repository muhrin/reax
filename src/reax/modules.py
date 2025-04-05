from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Generic, Mapping, Optional, TypedDict, TypeVar, Union

import beartype
import jax
import jaxtyping as jt
from lightning_utilities.core import rank_zero
import optax
from typing_extensions import deprecated

from . import _module_hooks
from .data import _datasources

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


class Module(
    Generic[BatchT, OutputT_co], _module_hooks.ModuleHooks, _datasources.DataSource[BatchT]
):
    example_input_array: Optional[BatchT]

    def __init__(self):
        """Init function."""
        super().__init__()
        self._trainer: Optional["reax.Trainer"] = None
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

    def configure_model(self, stage: "reax.Stage", batch: Any, /) -> None:
        """Called at the beginning of each stage.

        A chance to configure the model.  This method should be idempotent, i.e. calling it a second
        should do nothing.
        """

    def training_step(self, batch: BatchT, batch_idx: int, /) -> Optional[TrainOutput]:
        """Train step."""

    def validation_step(self, batch: BatchT, batch_idx: int, /):
        """Validate step."""

    def predict_step(self, batch: BatchT, batch_idx: int, /) -> OutputT_co:
        """Make a model prediction and return the result."""

    def test_step(self, batch: BatchT, batch_idx: int, /):
        """Test step."""

    def configure_listeners(self) -> "Union[Sequence[reax.TrainerListener], reax.TrainerListener]":
        """Configure model-specific listeners. When the model gets attached, e.g., when ``.fit()``
        or ``.test()`` gets called, the list or a listener returned here will be merged with the
        list of listeners passed to the Trainer's ``listeners`` argument.
        If a listener returned here has the same type as one or several listeners already
        present in the Trainer's listeners list, it will take priority and replace them.
        In addition, REAX will make sure
        :class:`~reax.listeners.model_checkpoint.ModelCheckpoint` listeners run last.

        Return:
            A listener or a list of listeners which will extend the list of listeners in the
            Trainer.

        Example::

            def configure_listeners(self):
                early_stop = EarlyStopping(monitor="val_acc", mode="max")
                checkpoint = ModelCheckpoint(monitor="val_loss")
                return [early_stop, checkpoint]

        """
        return []

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

    def log_dict(
        self,
        dictionary: "Union[Mapping[str, MetricType], reax.metrics.MetricCollection]",
        prog_bar: bool = False,
        logger: Optional[bool] = None,
        on_step: Optional[bool] = None,
        on_epoch: Optional[bool] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        """Log a dictionary of values at once.
        :param dictionary: Key value pairs.
            Keys must be identical across all processes if using DDP or any other distributed
            strategy.
            The values can be a ``float``, ``Array``, ``Metric``, or ``MetricCollection``.
        :type dictionary: "Union[Mapping[str, MetricType], reax.metrics.MetricCollection]"
        :param prog_bar: If ``True`` logs to the progress base, defaults to False.
        :type prog_bar: bool, optional
        :param logger: If ``True`` logs to the logger, defaults to None.
        :type logger: Optional[bool], optional
        :param on_step: If ``True`` logs at this step.
            ``None`` auto-logs for training_step but not validation/test_step.
            The default value is determined by the hook.
            See :ref:`extensions/logging:Automatic Logging` for details, defaults to None.
        :type on_step: Optional[bool], optional
        :param on_epoch: If ``True`` logs epoch accumulated metrics.
            ``None`` auto-logs for val/test step but not ``training_step``.
            The default value is determined by the hook.
            See :ref:`extensions/logging:Automatic Logging` for details, defaults to None.
        :type on_epoch: Optional[bool], optional
        :param batch_size: Current batch size. This will be directly inferred from the loaded batch,
            but some data structures might need to explicitly provide it, defaults to None.
        :type batch_size: Optional[int], optional

        Example::

            values = {'loss': loss, 'acc': acc, ..., 'metric_n': metric_n}
            self.log_dict(values)
        """
        for key, val in dictionary.items():
            self.log(
                name=key,
                value=val,
                prog_bar=prog_bar,
                logger=logger,
                on_step=on_step,
                on_epoch=on_epoch,
                batch_size=batch_size,
            )

    def state_dict(self) -> dict[str, Any]:
        """Save any additional module state"""
        return {}

    def load_state(self, state_dict: dict[str, Any]) -> None:
        """Load module state from the passed state dictionary"""

    @property
    @deprecated("REAX uses the term 'update' instead of 'step', please use `.global_updates`")
    def global_step(self) -> int:
        """Get the global number of optimizer updates."""
        return self.global_updates


PyTree = Any
InputT_co = TypeVar("InputT_co", covariant=True)
ModelT = Callable[[PyTree, InputT_co], OutputT_co]
LabelT_co = TypeVar("LabelT_co", covariant=True)
LossFn = Callable[[OutputT_co, LabelT_co], jax.Array]
