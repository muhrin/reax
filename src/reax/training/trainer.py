import contextlib
import functools
import logging
import os
import pickle  # nosec
import signal
import sys
from typing import TYPE_CHECKING, Any, Callable, Final, Iterable, Literal, Optional, Union, cast
import weakref

import beartype
import fsspec
import jax
import jaxtyping as jt
from lightning_utilities.core import rank_zero
from typing_extensions import override

from . import _logger_connector
from .. import exceptions, hooks, keys
from .. import listeners as listeners_
from .. import loggers as loggers_
from .. import modules, stages, strategies, typing, utils
from ..utils import events

if TYPE_CHECKING:
    import reax

_LOGGER = logging.getLogger(__name__)

__all__ = ("Trainer",)


class Trainer(stages.StageListener):
    # pylint: disable=too-many-public-methods

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        *,
        accelerator: Literal["auto", "cpu", "gpu"] = "auto",
        devices: Union[list[int], str, int] = "auto",
        logger: Optional[Union["reax.Logger", Iterable["reax.Logger"], bool]] = None,
        fast_dev_run: Union[int, bool] = False,
        listeners: "Optional[list[reax.TrainerListener], reax.TrainerListener]" = None,
        log_every_n_steps: int = 50,
        enable_checkpointing: Optional[bool] = True,
        enable_progress_bar: bool = True,
        enable_model_summary: Optional[bool] = None,
        deterministic: bool = False,
        rng_key: jax.Array = None,
        default_root_dir: Optional[typing.Path] = None,
    ):
        """Init function."""
        if deterministic:
            _LOGGER.warning("`deterministic=True` is not supported yet, ignoring.")
        if devices != "auto":
            _LOGGER.warning("`devices` other than 'auto' is not supported yet, ignoring.")

        self._accelerator = (
            jax.devices()[0] if accelerator == "auto" else jax.devices(accelerator)[0]
        )
        # Params
        self._fast_dev_run: Final[Union[int, bool]] = fast_dev_run
        self._log_every_n_steps: Final[int] = log_every_n_steps

        self._strategy = strategies.SingleDevice(self._accelerator)
        self._default_root_dir = (
            os.fspath(default_root_dir) if default_root_dir is not None else os.getcwd()
        )

        self._automatic_optimization = True
        self._optimizers = []
        self._stage: Optional[stages.Stage] = None

        # State
        self._rng = utils.rngs.Generator(key=rng_key if rng_key is not None else jax.random.key(0))
        self._current_epoch: int = 0
        self._global_updates: int = 0

        # Init
        self._events = events.EventGenerator[hooks.TrainerListener](
            default_args=(weakref.proxy(self),)
        )

        self._loggers = _init_loggers(logger, self.default_root_dir)

        self._logging = _logger_connector.TrainerLogging()
        self._events.add_listener(self._logging)

        if isinstance(listeners, hooks.TrainerListener):
            listeners = [listeners]
        if listeners:
            for listener in listeners:
                self._events.add_listener(listener)

        if enable_progress_bar:
            self._events.add_listener(listeners_.TqdmProgressBar())

        if enable_model_summary:
            _LOGGER.warning("`enable_model_summary` is not supported yet, ignoring")

        if self.checkpoint_listeners:
            if not enable_checkpointing:
                raise exceptions.MisconfigurationException(
                    "Trainer was configured with `enable_checkpointing=False`"
                    " but found `ModelCheckpoint` in listeners list."
                )
        elif enable_checkpointing:
            # Create the default checkpointer
            self._events.add_listener(listeners_.ModelCheckpoint())

    def finalize(self):
        """Clean up the trainer.

        After this called this object should no longer be interacted with.
        """
        self._events = None
        self._loggers = None

    def __del__(self):
        """Del function."""
        self.finalize()

    @property
    def strategy(self) -> strategies.Strategy:
        """Strategy function."""
        return self._strategy

    @property
    def is_global_zero(self) -> bool:
        """Whether this process is the global zero in multi-node training.

        .. code-block:: python

            def training_step(self, batch, batch_idx):
                if self.trainer.is_global_zero:
                    print("in node 0, accelerator 0")
        """
        return self.strategy.is_global_zero

    @property
    def fast_dev_run(self) -> Union[int, bool]:
        """Fast dev run."""
        return self._fast_dev_run

    @property
    def current_epoch(self) -> int:
        """Get the current fitting epoch."""
        return self._current_epoch

    @property
    def global_updates(self) -> int:
        """Get the global number of optimizer updates."""
        if self._stage is not None and isinstance(self._stage, stages.Fit):
            # The stage is running, so count add the number of updates it has performed so far
            return self._global_updates + self._stage.updates

        return self._global_updates

    @property
    def optimizers(self) -> list["reax.Optimizer"]:
        """Optimizers function."""
        return self._optimizers

    @optimizers.setter
    def optimizers(self, opts: list["reax.Optimizer"]) -> None:
        """Optimizers function."""
        self._optimizers = opts

    @property
    def should_stop(self) -> bool:
        """Should stop."""
        if self._stage is None:
            return False
        return self._stage.should_stop

    @should_stop.setter
    def should_stop(self, stop: bool):
        """Should stop."""
        if stop:
            self._stage.stop("None")

    @property
    def default_root_dir(self) -> str:
        """Get the fallback directory used for loggers and other components when not explicitly
        specified."""
        if _is_local_file_protocol(self._default_root_dir):
            return os.path.normpath(os.path.expanduser(self._default_root_dir))

        return self._default_root_dir

    @property
    def early_stopping_listener(self) -> Optional[listeners_.EarlyStopping]:
        """The first :class:`~reax.listeners.early_stopping.EarlyStopping` listener in the
        Trainer.listeners list, or ``None`` if it doesn't exist."""
        listeners = self.early_stopping_listeners
        return listeners[0] if len(listeners) > 0 else None

    @property
    def early_stopping_listeners(self) -> list[listeners_.EarlyStopping]:
        """A list of all instances of :class:`~reax.listeners.early_stopping.EarlyStopping` found in
        the Trainer.listeners list."""
        return self._events.find(type=listeners_.EarlyStopping)

    @property
    def early_stopping_callback(self) -> Optional[listeners_.EarlyStopping]:
        """The first :class:`~reax.listeners.early_stopping.EarlyStopping` listener in the
        Trainer.callbacks list, or ``None`` if it doesn't exist."""
        return self.early_stopping_listener

    @property
    def early_stopping_callbacks(self) -> list[listeners_.EarlyStopping]:
        """A list of all instances of :class:`~reax.listeners.early_stopping.EarlyStopping` found in
        the Trainer.callbacks list."""
        return self.early_stopping_listeners

    @property
    def checkpoint_listeners(self) -> list["reax.listeners.Checkpointer"]:
        """The first :class:`~reax.listeners.model_checkpoint.ModelCheckpoint` listener in the
        Trainer.listeners list, or ``None`` if it doesn't exist."""
        return self._events.find(type=listeners_.Checkpointer)

    @property
    def checkpoint_listener(self) -> Optional["reax.listeners.Checkpointer"]:
        """The first :class:`~reax.listeners.model_checkpoint.ModelCheckpoint` listener in the
        Trainer.listeners list, or ``None`` if it doesn't exist."""
        listeners = self.checkpoint_listeners
        return listeners[0] if len(listeners) > 0 else None

    @property
    def checkpoint_callbacks(self) -> list["reax.listeners.Checkpointer"]:
        """A list of all instances of :class:`~reax.listeners.model_checkpoint.ModelCheckpoint`
        found in the Trainer.listeners list."""
        # Kept for compatibility with Lightning
        return self.checkpoint_listeners

    @property
    def checkpoint_callback(self) -> Optional["reax.listeners.Checkpointer"]:
        """The first :class:`~reax.listeners.model_checkpoint.ModelCheckpoint` callback in the
        Trainer.listeners list, or ``None`` if it doesn't exist."""
        # Kept for compatibility with Lightning
        return self.checkpoint_listener

    @property
    def logger(self) -> Optional["reax.Logger"]:
        r"""Get the first (and main) logger."""
        if not self._loggers:
            return None

        return self._loggers[0]

    @property
    def loggers(self) -> list["reax.Logger"]:
        """Get all the loggers."""
        return self._loggers

    @loggers.setter
    def loggers(self, loggers: Optional[list["reax.Logger"]]) -> None:
        """Loggers function."""
        self._loggers = loggers if loggers else []

    @property
    def log_dir(self) -> Optional[str]:
        """The directory for the current experiment. Use this to save images to, etc...

        .. note:: You must call this on all processes. Failing to do so will cause your program to
            stall forever.

        .. code-block:: python

            def training_step(self, batch, batch_idx):
                img = ...
                save_img(img, self.trainer.log_dir)
        """
        if len(self.loggers) > 0:
            if not isinstance(self.loggers[0], loggers_.TensorBoardLogger):
                dirpath = self.loggers[0].save_dir
            else:
                dirpath = self.loggers[0].log_dir
        else:
            dirpath = self.default_root_dir

        dirpath = self._strategy.broadcast(dirpath)
        return dirpath

    @property
    def stage(self) -> "Optional[reax.Stage]":
        """Get the current stage if there is one running, otherwise None."""
        return self._stage

    @property
    def progress_bar_metrics(self) -> dict:
        """Progress bar metrics."""
        return self._logging.progress_bar_metrics

    @property
    def listener_metrics(self) -> dict:
        """Callback metrics."""
        return self._logging.listener_metrics

    @property
    def callback_metrics(self) -> dict:
        """Callback metrics."""
        # Kept for compatibility with Lightning
        return self.listener_metrics

    @property
    def logger_metrics(self) -> dict:
        """Logger metrics."""
        return self._logging.logger_metrics

    @property
    def global_rank(self) -> int:
        """Global rank."""
        return getattr(self._strategy, "global_rank", 0)

    @property
    def local_rank(self) -> int:
        """Local rank."""
        return getattr(self._strategy, "local_rank", 0)

    @property
    def node_rank(self) -> int:
        """Node rank."""
        return getattr(self._strategy, "node_rank", 0)

    @property
    def world_size(self) -> int:
        """World size."""
        return getattr(self._strategy, "world_size", 1)

    @property
    def num_nodes(self) -> int:
        """Num nodes."""
        return getattr(self._strategy, "num_nodes", 1)

    @property
    def train_dataloader(self) -> Optional:
        """Train dataloader."""
        if self._stage is not None:
            return getattr(self._stage, "train_dataloader", None)
        return None

    @property
    def _module(self) -> "reax.Module":
        if self._stage is None:
            raise AttributeError("There is no stage running, so module is not available")
        if self._stage.module is None:
            raise AttributeError(
                f"The running stage '{self._stage.__class__.__name__}', does not have a module"
            )
        return self._stage.module

    def rng_key(self, num: Union[int, tuple[int, ...]] = 1) -> jax.Array:
        """Get a new RNG key.

        This will update the state in the `Trainer`.
        """
        return self._rng.make_key(num=num)

    def log(
        # pylint: disable=unused-argument
        self,
        name: str,
        value,
        *,
        prog_bar: bool = False,
        batch_size: Optional[int] = None,
        logger: bool = None,
        on_step=True,
        on_epoch=True,
    ) -> None:
        """Log function."""
        if self._stage is None:
            raise RuntimeError(
                "Logging is only supported during one of the train/validate/test stages. "
                "There is currently no stage running."
            )

        # Let the stage take care of it
        self._stage.log(
            name,
            value,
            prog_bar=prog_bar,
            logger=logger,
            batch_size=batch_size,
            on_step=on_step,
            on_epoch=on_epoch,
        )

    # region Stages

    @jt.jaxtyped(typechecker=beartype.beartype)
    def fit(
        self,
        module: modules.Module,
        train_dataloaders: "Optional[reax.DataLoader]" = None,
        val_dataloaders: "Optional[reax.DataLoader]" = None,
        *,
        datamodule: "Optional[reax.DataModule]" = None,
        max_epochs: Optional[int] = 1_000,
        min_epochs: int = 0,
        min_updates: int = 0,
        max_updates: Union[int, float] = None,
        limit_train_batches: Optional[Union[int, float]] = 1.0,
        accumulate_grad_batches: int = 1,
        limit_val_batches: Optional[Union[int, float]] = 1.0,
        val_check_interval: Optional[Union[int, float]] = 1.0,
        check_val_every_n_epoch: int = 1,
        num_sanity_val_steps: Optional[int] = None,
    ):
        """Fit function."""
        if max_updates is not None and max_updates < 0:
            raise ValueError("`max_updates` must be a non-negative integer")
        if max_epochs is not None and max_epochs < 0:
            raise ValueError(f"`max_epochs` must be a non-negative integer or {keys.NO_LIMIT}")
        if num_sanity_val_steps:
            _LOGGER.warning("`num_sanity_val_steps` is not supported yet, ignoring.")

        if self._fast_dev_run:
            num_batches = 1
            max_epochs = 1
            limit_train_batches = num_batches
            limit_val_batches = num_batches
            val_check_interval = 1.0
            check_val_every_n_epoch = 1

            rank_zero.rank_zero_info(
                f"Running in `fast_dev_run` mode: will run the requested loop using {num_batches} "
                f"batch(es). Logging and checkpointing is suppressed."
            )

        fit = stages.Fit(
            module,
            self.optimizers,
            self._strategy,
            train_dataloaders=train_dataloaders,
            val_dataloaders=val_dataloaders,
            datamodule=datamodule,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            min_updates=min_updates,
            max_updates=max_updates,
            limit_train_batches=limit_train_batches,
            accumulate_grad_batches=accumulate_grad_batches,
            limit_val_batches=limit_val_batches,
            val_check_interval=val_check_interval,
            check_val_every_n_epoch=check_val_every_n_epoch,
        )
        self._run_stage(fit)
        # Update state variables
        self._global_updates += fit.updates

    @jt.jaxtyped(typechecker=beartype.beartype)
    def validate(
        self,
        module: modules.Module,
        dataloaders=None,
        datamodule=None,
        max_batches: Union[int, type(keys.NO_LIMIT)] = keys.NO_LIMIT,
    ):
        """Test function."""
        self._run_stage(
            stages.Validate(
                module,
                self._strategy,
                dataloader=dataloaders,
                datamodule=datamodule,
                max_batches=max_batches,
            )
        )

    @jt.jaxtyped(typechecker=beartype.beartype)
    def test(
        self,
        module: modules.Module,
        dataloaders=None,
        datamodule=None,
    ):
        """Test function."""
        self._run_stage(
            stages.Test(module, self._strategy, dataloader=dataloaders, datamodule=datamodule)
        )

    @jt.jaxtyped(typechecker=beartype.beartype)
    def predict(
        self,
        module: modules.Module,
        dataloaders: "Optional[reax.DataLoader]" = None,
        datamodule: "Optional[reax.DataModule]" = None,
        return_predictions: Optional[bool] = None,
        limit_batches: Union[int, float] = keys.NO_LIMIT,
    ) -> Optional[Union[list[Any], list[list[Any]]]]:
        r"""Run inference on the data.

        Logging is disabled in the predict hooks.
        """
        if self._fast_dev_run:
            limit_batches = 1

        if return_predictions is None:
            return_predictions = True

        predict = stages.Predict(
            module,
            self._strategy,
            dataloader=dataloaders,
            datamodule=datamodule,
            max_batches=limit_batches,
            keep_predictions=return_predictions,
        )
        self._run_stage(predict)
        if return_predictions:
            return predict.all_outputs

        return None

    def _run_stage(self, stage: stages.Stage) -> stages.Stage:
        """Run stage."""
        with jax.default_device(jax.devices("cpu")[0]):
            try:
                with self._attach(stage):
                    self._logging.reset_metrics()
                    with stage.events.listen_context(self):
                        stage.run()
            except KeyboardInterrupt:
                # Disable further Ctrl+C presses while we respond to this one
                signal.signal(signal.SIGINT, signal.SIG_IGN)
                sys.exit(1)
            else:
                for logger in self.loggers:
                    logger.finalize("success")

        return stage

    # endregion

    @contextlib.contextmanager
    def _attach(self, stage: stages.Stage):
        """Attach function."""
        self._stage = stage
        if stage.module is not None:
            stage.module.trainer = self
        try:
            yield
        finally:
            self._stage = None
            if stage.module is not None:
                stage.module.trainer = None

    @override
    def on_stage_starting(self, stage: "stages.Stage", /) -> None:
        """The stage is about to start."""
        if not self._optimizers and isinstance(stage, stages.Train):
            self._optimizers = stage.optimizers

        if stage.is_root:
            # Only setup for the root loop
            self._events.fire_event(hooks.TrainerListener.setup, stage)

        self._events.fire_event(hooks.TrainerListener.on_stage_starting, stage)

        event = hook_map(stage).get(hooks.TrainerListener.on_stage_starting)
        if event is not None:
            self._events.fire_event(event, stage)

    @override
    def on_stage_started(self, stage: "reax.Stage", /):
        """On stage started."""
        self._events.fire_event(hooks.TrainerListener.on_stage_started, stage)

        event = hook_map(stage).get(hooks.TrainerListener.on_stage_started)
        if event is not None:
            self._events.fire_event(event, stage)

    @override
    def on_stage_iter_starting(self, stage: "stages.Stage", step: int, /):
        """On stage iter starting."""
        self._events.fire_event(hooks.TrainerListener.on_stage_iter_starting, stage, step)

        event = hook_map(stage).get(hooks.TrainerListener.on_stage_iter_starting)
        if event is not None:
            stage = cast(stages.EpochStage, stage)
            self._events.fire_event(event, stage, stage.batch, step)

    @override
    def on_stage_iter_ending(self, stage: "stages.Stage", step: int, outputs: Any, /):
        """On stage iter ending."""
        if isinstance(stage, stages.EpochStage):
            logging_metrics = {"epoch": self.current_epoch, "stage": stage.name}
            logging_metrics.update(stage.logged_metrics)
            for logger in self.loggers:
                logger.log_metrics(metrics=logging_metrics, step=self.global_updates - 1)
                logger.save()

            self._events.fire_event(
                hooks.TrainerListener.on_batch_ending,
                stage,
                step,
                outputs,
            )

        event = hook_map(stage).get(hooks.TrainerListener.on_stage_iter_ending)
        if event is not None:
            stage = cast(stages.EpochStage, stage)
            self._events.fire_event(event, stage, outputs, stage.batch, step)

        if isinstance(stage, stages.Fit):
            # Keep track of the number of completed training epochs
            self._current_epoch += 1

    @override
    def on_stage_ending(self, stage: "stages.Stage", /) -> None:
        """On stage ending."""
        if isinstance(stage, stages.EpochStage):
            self._events.fire_event(
                hooks.TrainerListener.on_epoch_ending, stage, stage.listener_metrics
            )

            metrics = stage.results
            if metrics[keys.LOG]:
                logging_metrics = {"epoch": self.current_epoch, "stage": stage.name}
                logging_metrics.update(metrics[keys.LOG])
                for logger in self.loggers:
                    logger.log_metrics(metrics=logging_metrics, step=self.global_updates - 1)
                    logger.save()

        self._events.fire_event(hooks.TrainerListener.on_stage_ending, stage)

        event = hook_map(stage).get(hooks.TrainerListener.on_stage_ending)
        if event is not None:
            stage = cast(stages.EpochStage, stage)
            self._events.fire_event(event, stage)

    @override
    def on_stage_ended(self, stage: "reax.Stage", /):
        """On stage ended."""
        self._events.fire_event(hooks.TrainerListener.on_stage_ended, stage)

        event = hook_map(stage).get(hooks.TrainerListener.on_stage_ended)
        if event is not None:
            stage = cast(stages.EpochStage, stage)
            self._events.fire_event(event, stage)

    # region Checkpointing

    def save_checkpoint(self, filepath: typing.Path, weights_only: bool = True):
        """For now, we just save the model weights.

        The user has to store the model definition themselves.
        """
        if not weights_only:
            _LOGGER.warning("`weights_only=False` is not supported yet, ignoring")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as file:
            pickle.dump(self._module.parameters(), file)

    def _get_checkpoint_path(self) -> Optional[str]:
        """Get checkpoint path."""
        checkpointers = self.checkpoint_listeners
        if not checkpointers:
            return None

        if len(checkpointers) > 1:
            fn = self._get_checkpoint_path
            rank_zero.rank_zero_warn(
                f'`.{fn}(ckpt_path="best")` found Trainer with multiple `ModelCheckpoint`'
                " listeners. The best checkpoint path from first checkpoint listener will be used."
            )

        checkpointer = checkpointers[0]
        return getattr(checkpointer, "best_model_path", None)

    # endregion


@jt.jaxtyped(typechecker=beartype.beartype)
def _init_loggers(
    logger: Optional[Union["reax.Logger", Iterable["reax.Logger"], bool]],
    default_root_dir: str,
) -> list["reax.Logger"]:
    """Init loggers."""
    if isinstance(logger, loggers_.Logger):
        return [logger]

    if isinstance(logger, (bool, type(None))):
        if logger:
            return [loggers_.TensorBoardLogger(default_root_dir)]

        return []

    return list(logger)


def _is_local_file_protocol(path: typing.Path) -> bool:
    """Is local file protocol."""
    return fsspec.utils.get_protocol(str(path)) == "file"


@functools.singledispatch
def hook_map(_stage) -> dict[Callable, Callable]:
    """Hook map."""
    return dict()


@hook_map.register
def _(_stage: stages.Train) -> dict[Callable, Callable]:
    """Function."""
    return {
        hooks.TrainerListener.on_stage_starting: hooks.TrainerListener.on_train_start,
        hooks.TrainerListener.on_stage_started: hooks.TrainerListener.on_train_epoch_start,
        hooks.TrainerListener.on_stage_iter_starting: hooks.TrainerListener.on_train_batch_start,
        hooks.TrainerListener.on_stage_iter_ending: hooks.TrainerListener.on_train_batch_end,
        hooks.TrainerListener.on_stage_ending: hooks.TrainerListener.on_train_epoch_end,
        hooks.TrainerListener.on_stage_ended: hooks.TrainerListener.on_train_end,
    }


@hook_map.register
def _(_stage: stages.Validate) -> dict[Callable, Callable]:
    """Function."""
    return {
        hooks.TrainerListener.on_stage_starting: hooks.TrainerListener.on_validation_start,
        hooks.TrainerListener.on_stage_started: hooks.TrainerListener.on_validation_epoch_start,
        # pylint: disable=line-too-long
        hooks.TrainerListener.on_stage_iter_starting: hooks.TrainerListener.on_validation_batch_start,
        hooks.TrainerListener.on_stage_iter_ending: hooks.TrainerListener.on_validation_batch_end,
        hooks.TrainerListener.on_stage_ending: hooks.TrainerListener.on_validation_epoch_end,
        hooks.TrainerListener.on_stage_ended: hooks.TrainerListener.on_validation_end,
    }


@hook_map.register
def _(_stage: stages.Test) -> dict[Callable, Callable]:
    """Function."""
    return {
        hooks.TrainerListener.on_stage_starting: hooks.TrainerListener.on_test_start,
        hooks.TrainerListener.on_stage_started: hooks.TrainerListener.on_test_epoch_start,
        hooks.TrainerListener.on_stage_iter_starting: hooks.TrainerListener.on_test_batch_start,
        hooks.TrainerListener.on_stage_iter_ending: hooks.TrainerListener.on_test_batch_end,
        hooks.TrainerListener.on_stage_ending: hooks.TrainerListener.on_test_epoch_end,
        hooks.TrainerListener.on_stage_ended: hooks.TrainerListener.on_test_end,
    }


@hook_map.register
def _(_stage: stages.Predict) -> dict[Callable, Callable]:
    """Function."""
    return {
        hooks.TrainerListener.on_stage_starting: hooks.TrainerListener.on_predict_start,
        hooks.TrainerListener.on_stage_started: hooks.TrainerListener.on_predict_epoch_start,
        hooks.TrainerListener.on_stage_iter_starting: hooks.TrainerListener.on_predict_batch_start,
        hooks.TrainerListener.on_stage_iter_ending: hooks.TrainerListener.on_predict_batch_end,
        hooks.TrainerListener.on_stage_ending: hooks.TrainerListener.on_predict_epoch_end,
        hooks.TrainerListener.on_stage_ended: hooks.TrainerListener.on_predict_end,
    }


@hook_map.register
def _(_stage: stages.Fit) -> dict[Callable, Callable]:
    """Function."""
    return {
        hooks.TrainerListener.on_stage_starting: hooks.TrainerListener.on_fit_start,
        hooks.TrainerListener.on_stage_ending: hooks.TrainerListener.on_fit_end,
    }
