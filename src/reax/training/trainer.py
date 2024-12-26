import contextlib
import functools
import logging
import os
import pickle
import signal
import sys
from typing import TYPE_CHECKING, Any, Callable, Iterable, Literal, Optional, Union, cast
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
from .. import modules, stages, strategies, typing
from ..utils import containers, events

if TYPE_CHECKING:
    import reax

_LOGGER = logging.getLogger(__name__)


__all__ = ("Trainer",)


class Trainer(stages.StageListener):
    def __init__(
        self,
        module: modules.Module,
        accelerator: Literal["auto", "cpu", "gpu"] = "auto",
        logger: Optional[Union["reax.Logger", Iterable["reax.Logger"], bool]] = None,
        fast_dev_run: Union[int, bool] = False,
        listeners: "Optional[list[reax.TrainerListener], reax.TrainerListener]" = None,
        log_every_n_steps: int = 50,
        enable_checkpointing: Optional[bool] = True,
        check_val_every_n_epoch: int = 1,
        enable_progress_bar: bool = True,
        enable_model_summary: Optional[bool] = None,
        rng_key: jax.Array = None,
        default_root_dir: Optional[typing.Path] = None,
    ):
        self._accelerator = (
            jax.devices()[0] if accelerator == "auto" else jax.devices(accelerator)[0]
        )

        self._strategy = strategies.SingleDevice(self._accelerator)
        self._fast_dev_run = fast_dev_run
        self._log_every_n_steps = log_every_n_steps
        self.check_val_every_n_epoch = check_val_every_n_epoch
        self._rng_key = rng_key if rng_key is not None else jax.random.key(0)
        self._default_root_dir = (
            os.fspath(default_root_dir) if default_root_dir is not None else os.getcwd()
        )

        self._automatic_optimization = True
        self._optimizers = []
        self._stage: Optional[stages.Stage] = None

        # State indexes
        self._current_epoch: int = 0
        self._global_updates: int = 0

        self.events = events.EventGenerator[hooks.TrainerListener]()

        # Attach the trainer to the module
        self._module: modules.Module = module
        module.trainer = self

        self._loggers = _init_loggers(logger, self.default_root_dir)

        self._logging = _logger_connector.TrainerLogging()
        self.events.add_listener(self._logging)

        if isinstance(listeners, hooks.TrainerListener):
            listeners = [listeners]
        if listeners:
            for listener in listeners:
                self.events.add_listener(listener)

        if enable_progress_bar:
            self.events.add_listener(listeners_.TqdmProgressBar())

        if enable_model_summary:
            _LOGGER.warning("`enable_model_summary` is not supported yet, ignoring")

        if self.checkpoint_callbacks:
            if not enable_checkpointing:
                raise exceptions.MisconfigurationException(
                    "Trainer was configured with `enable_checkpointing=False`"
                    " but found `ModelCheckpoint` in callbacks list."
                )
        elif enable_checkpointing:
            # Create the default checkpointer
            self.events.add_listener(listeners_.ModelCheckpoint())

    def finalize(self):
        """
        Clean up the trainer.  After this called this object should no longer be interacted with
        """
        if self._module is None:
            return

        if self._module.trainer is self:
            self._module.trainer = None
        self.events = None
        self._loggers = None

    def __del__(self):
        self.finalize()

    @property
    def fast_dev_run(self) -> Union[int, bool]:
        return self._fast_dev_run

    @property
    def current_epoch(self) -> int:
        """Get the current fitting epoch"""
        return self._current_epoch

    @property
    def global_updates(self) -> int:
        """Get the global number of optimizer updates"""
        if self._stage is not None and isinstance(self._stage, stages.Fit):
            # The stage is running, so count add the number of updates it has performed so far
            return self._global_updates + self._stage.updates

        return self._global_updates

    @property
    def optimizers(self) -> list["reax.Optimizer"]:
        return self._optimizers

    @optimizers.setter
    def optimizers(self, opts: list["reax.Optimizer"]) -> None:
        self._optimizers = opts

    @property
    def should_stop(self) -> bool:
        if self._stage is None:
            return False
        return self._stage.should_stop

    @should_stop.setter
    def should_stop(self, stop: bool):
        if stop:
            self._stage.stop("None")

    @property
    def default_root_dir(self) -> str:
        """
        Get the fallback directory used for loggers and other components when not explicitly
        specified
        """
        if _is_local_file_protocol(self._default_root_dir):
            return os.path.normpath(os.path.expanduser(self._default_root_dir))

        return self._default_root_dir

    @property
    def early_stopping_callback(self) -> Optional[listeners_.EarlyStopping]:
        """The first :class:`~reax.listeners.early_stopping.EarlyStopping` callback in the
        Trainer.callbacks list, or ``None`` if it doesn't exist."""
        callbacks = self.early_stopping_callbacks
        return callbacks[0] if len(callbacks) > 0 else None

    @property
    def early_stopping_callbacks(self) -> list[listeners_.EarlyStopping]:
        """A list of all instances of :class:`~reax.listeners.early_stopping.EarlyStopping` found in the
        Trainer.callbacks list."""
        return self.events.find(listeners_.EarlyStopping)

    @property
    def logger(self) -> "reax.Logger":
        """Get the first (and main) logger"""
        if not self._loggers:
            return None

        return self._loggers[0]

    @property
    def loggers(self) -> list["reax.Logger"]:
        """Get all the loggers"""
        return self._loggers

    @loggers.setter
    def loggers(self, loggers: Optional[list["reax.Logger"]]) -> None:
        self._loggers = loggers if loggers else []

    @property
    def log_dir(self) -> Optional[str]:
        """The directory for the current experiment. Use this to save images to, etc..."""
        if len(self.loggers) > 0:
            dirpath = self.loggers[0].log_dir
        else:
            dirpath = self.default_root_dir

        # dirpath = self.strategy.broadcast(dirpath)
        return dirpath

    @property
    def stage(self) -> "Optional[reax.Stage]":
        """Get the current stage if there is one running, otherwise None"""
        return self._stage

    @property
    def checkpoint_callbacks(self) -> list[listeners_.Checkpointer]:
        """A list of all instances of :class:`~reax.listeners.Checkpointer` found in the Trainer.events list."""
        return self.events.find(type=listeners_.Checkpointer)

    @property
    def progress_bar_metrics(self) -> dict:
        return self._logging.progress_bar_metrics

    @property
    def callback_metrics(self) -> dict:
        return self._logging.callback_metrics

    @property
    def logger_metrics(self) -> dict:
        return self._logging.logger_metrics

    @property
    def global_rank(self) -> int:
        return getattr(self._strategy, "global_rank", 0)

    @property
    def local_rank(self) -> int:
        return getattr(self._strategy, "local_rank", 0)

    @property
    def node_rank(self) -> int:
        return getattr(self._strategy, "node_rank", 0)

    @property
    def world_size(self) -> int:
        return getattr(self._strategy, "world_size", 1)

    @property
    def num_nodes(self) -> int:
        return getattr(self._strategy, "num_nodes", 1)

    def rng_key(self, num=1) -> jax.Array:
        """Get a new RNG key.  This will update the state in the `Trainer`"""
        self._rng_key, subkey = jax.random.split(self._rng_key, num=num + 1)
        return subkey

    def log(
        # pylint: disable=unused-argument
        self,
        name: str,
        value,
        prog_bar: bool = False,
        batch_size: Optional[int] = None,
        logger: bool = None,
        on_step=True,
        on_epoch=True,
    ) -> None:
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
        train_dataloaders: "Optional[reax.DataLoader]" = None,
        val_dataloaders: "Optional[reax.DataLoader]" = None,
        datamodule: "Optional[reax.DataModule]" = None,
        max_epochs: Union[int, float] = 1_000,
        min_epochs: int = -1,
        min_updates: int = -1,
        max_updates: Union[int, float] = float("inf"),
        limit_train_batches: Optional[Union[int, float]] = 1.0,
        accumulate_grad_batches: int = 1,
        limit_val_batches: Optional[Union[int, float]] = 1.0,
        num_sanity_val_steps: Optional[int] = None,
    ):
        if max_updates < -1:
            raise ValueError("`max_updates` must be a non-negative integer or -1")
        if max_epochs is None:
            max_epochs = -1
        elif max_epochs < -1:
            raise ValueError("`max_epochs` must be a non-negative integer or -1")
        if num_sanity_val_steps:
            _LOGGER.warning("`num_sanity_val_steps` is not supported yet, ignoring.")

        # Must choose either a datamodule or train/val dataloaders
        if datamodule is not None:
            if train_dataloaders is not None or val_dataloaders is not None:
                raise ValueError(
                    "You cannot pass `train_dataloader` or `val_dataloaders` and `datamodule` to "
                    "`Trainer.fit()`"
                )
            train_dataloaders = datamodule.train_dataloader()
            val_dataloaders = datamodule.val_dataloader()

        # Fallbacks
        if train_dataloaders is None:
            train_dataloaders = self._module.train_dataloader()

        if val_dataloaders is None:
            val_dataloaders = self._module.val_dataloader()

        if self._fast_dev_run:
            num_batches = 1
            max_epochs = 1
            limit_train_batches = num_batches
            limit_val_batches = num_batches
            val_check_interval = 1.0
            check_val_every_n_epochs = 1

            rank_zero.rank_zero_info(
                f"Running in `fast_dev_run` mode: will run the requested loop using {num_batches} "
                f"batch(es). Logging and checkpointing is suppressed."
            )

        fit = stages.Fit(
            self._module,
            train_dataloaders,
            val_dataloaders,
            optimizers=self.optimizers,
            strategy=self._strategy,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            min_updates=min_updates,
            max_updates=max_updates,
            limit_train_batches=limit_train_batches,
            accumulate_grad_batches=accumulate_grad_batches,
            limit_val_batches=limit_val_batches,
            check_val_every_n_epoch=self.check_val_every_n_epoch,
        )
        self._run_stage(fit)
        # Update state variables
        self._global_updates += fit.updates

    def test(
        self,
        dataloaders=None,
        datamodule=None,
    ):
        if datamodule is not None:
            if dataloaders is not None:
                raise ValueError("Cannot supply dataloaders and datamodule to Trainer.test()")

            dataloaders = datamodule.test_dataloader()

        self._run_stage(stages.Test(self._module, dataloaders, self._strategy))

    def predict(
        self,
        dataloaders: "Optional[reax.DataLoader]" = None,
        datamodule: "Optional[reax.DataModule]" = None,
        return_predictions: Optional[bool] = None,
        limit_batches=float("inf"),
    ) -> Optional[Union[list[Any], list[list[Any]]]]:
        r"""
        Run inference on the data.  Logging is disabled in the predict hooks.
        """
        if datamodule is not None:
            if dataloaders is not None:
                raise ValueError("Cannot supply dataloaders and datamodule to Trainer.test()")

            dataloaders = datamodule.predict_dataloader()

        if self._fast_dev_run:
            limit_batches = 1

        predict = stages.Predict(
            self._module, dataloaders, self._strategy, max_batches=limit_batches
        )
        self._run_stage(predict)
        return predict._all_outputs

    def _run_stage(self, stage: stages.Stage) -> stages.Stage:
        try:
            with self._attach(stage):
                self._logging.reset_metrics()
                with stage.events.listen_context(self):
                    stage.run()
        except KeyboardInterrupt:
            # Disable further Ctrl+C presses while we respond to this one
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            sys.exit(1)

        return stage

    # endregion

    @contextlib.contextmanager
    def _attach(self, stage: stages.Stage):
        self._stage = stage
        try:
            yield
        finally:
            self._stage = None

    @override
    def on_stage_starting(self, stage: "stages.Stage") -> None:
        """The stage is about to start"""
        if not self._optimizers and isinstance(stage, stages.Train):
            self._optimizers = stage.optimizers

        self.events.fire_event(hooks.TrainerListener.on_stage_starting, weakref.proxy(self), stage)

        event = hook_map(stage).get(hooks.TrainerListener.on_stage_starting)
        if event is not None:
            self.events.fire_event(event, self, stage)

    @override
    def on_stage_started(self, stage: "reax.Stage", /):
        self.events.fire_event(hooks.TrainerListener.on_stage_started, weakref.proxy(self), stage)
        event = hook_map(stage).get(hooks.TrainerListener.on_stage_started)
        if event is not None:
            self.events.fire_event(event, self, stage)

    @override
    def on_stage_iter_starting(self, stage: "stages.Stage", step: int):
        self.events.fire_event(
            hooks.TrainerListener.on_stage_iter_starting, weakref.proxy(self), stage, step
        )

        event = hook_map(stage).get(hooks.TrainerListener.on_stage_iter_starting)
        if event is not None:
            stage = cast(stages.EpochStage, stage)
            self.events.fire_event(event, self, stage, stage.batch, step)

    @override
    def on_stage_iter_ending(self, stage: "stages.Stage", step: int, outputs: Any, /):
        if isinstance(stage, stages.EpochStage):
            logging_metrics = {"epoch": self.current_epoch, "stage": stage.name}
            logging_metrics.update(stage.logged_metrics)
            for logger in self.loggers:
                logger.log_metrics(metrics=logging_metrics, step=self.global_updates - 1)
                logger.save()

            self.events.fire_event(
                hooks.TrainerListener.on_batch_ending,
                self,
                stage,
                batch_idx=step,
                metrics=outputs,
            )

        event = hook_map(stage).get(hooks.TrainerListener.on_stage_iter_ending)
        if event is not None:
            stage = cast(stages.EpochStage, stage)
            self.events.fire_event(event, self, stage, outputs, stage.batch, step)

        if isinstance(stage, stages.Fit):
            # Keep track of the number of completed training epochs
            self._current_epoch += 1

    @override
    def on_stage_ending(self, stage: "stages.Stage") -> None:
        if isinstance(stage, stages.EpochStage):
            self.events.fire_event(
                hooks.TrainerListener.on_epoch_ending, self, stage, stage.callback_metrics
            )

            metrics = stage.results
            if metrics[keys.LOG]:
                logging_metrics = {"epoch": self.current_epoch, "stage": stage.name}
                logging_metrics.update(metrics[keys.LOG])
                for logger in self.loggers:
                    logger.log_metrics(metrics=logging_metrics, step=self.global_updates - 1)
                    logger.save()

        self.events.fire_event(hooks.TrainerListener.on_stage_ending, weakref.proxy(self), stage)

        event = hook_map(stage).get(hooks.TrainerListener.on_stage_ending)
        if event is not None:
            stage = cast(stages.EpochStage, stage)
            self.events.fire_event(event, self, stage)

    @override
    def on_stage_ended(self, stage: "reax.Stage", /):
        self.events.fire_event(hooks.TrainerListener.on_stage_ended, weakref.proxy(self), stage)

        event = hook_map(stage).get(hooks.TrainerListener.on_stage_ended)
        if event is not None:
            stage = cast(stages.EpochStage, stage)
            self.events.fire_event(event, weakref.proxy(self), stage)

    # region Checkpointing

    def save_checkpoint(self, filepath: typing.Path):
        """
        For now, we just save the model weights.  The user has to store the model definition
        themselves
        """
        with open(filepath, "wb") as file:
            pickle.dump(self._module.parameters(), file)

    def _get_checkpoint_path(self) -> Optional[str]:
        checkpointers = self.checkpoint_callbacks
        if not checkpointers:
            return None

        if len(checkpointers) > 1:
            fn = self._get_checkpoint_path
            rank_zero.rank_zero_warn(
                f'`.{fn}(ckpt_path="best")` found Trainer with multiple `ModelCheckpoint`'
                " callbacks. The best checkpoint path from first checkpoint callback will be used."
            )

        checkpointer = checkpointers[0]
        return getattr(checkpointer, "best_model_path", None)

    # endregion


@jt.jaxtyped(typechecker=beartype.beartype)
def _init_loggers(
    logger: Optional[Union["reax.Logger", Iterable["reax.Logger"], bool]],
    default_root_dir: str,
) -> list["reax.Logger"]:
    if isinstance(logger, loggers_.Logger):
        return [logger]

    if isinstance(logger, (bool, type(None))):
        if logger:
            return [loggers_.TensorBoardLogger(default_root_dir)]

        return []

    return list(logger)


def _is_local_file_protocol(path: typing.Path) -> bool:
    return fsspec.utils.get_protocol(str(path)) == "file"


_BATCH = Any
_OUTPUTS = Any
_BATCH_IDX = int


@functools.singledispatch
def hook_map(stage) -> dict[Callable, Callable]:
    return dict()


@hook_map.register
def _(stage: stages.Train) -> dict[Callable, Callable]:
    return {
        hooks.TrainerListener.on_stage_starting: hooks.TrainerListener.on_train_epoch_start,
        hooks.TrainerListener.on_stage_iter_starting: hooks.TrainerListener.on_train_batch_start,
        hooks.TrainerListener.on_stage_iter_ending: hooks.TrainerListener.on_train_batch_end,
        hooks.TrainerListener.on_stage_ending: hooks.TrainerListener.on_train_epoch_end,
    }


@hook_map.register
def _(stage: stages.Validate) -> dict[Callable, Callable]:
    return {
        hooks.TrainerListener.on_stage_starting: hooks.TrainerListener.on_validation_start,
        hooks.TrainerListener.on_stage_started: hooks.TrainerListener.on_validation_epoch_start,
        hooks.TrainerListener.on_stage_iter_starting: hooks.TrainerListener.on_validation_batch_start,
        hooks.TrainerListener.on_stage_iter_ending: hooks.TrainerListener.on_validation_batch_end,
        hooks.TrainerListener.on_stage_ending: hooks.TrainerListener.on_validation_epoch_end,
        hooks.TrainerListener.on_stage_ended: hooks.TrainerListener.on_validation_end,
    }


@hook_map.register
def _(stage: stages.Test) -> dict[Callable, Callable]:
    return {
        hooks.TrainerListener.on_stage_starting: hooks.TrainerListener.on_test_epoch_start,
        hooks.TrainerListener.on_stage_iter_starting: hooks.TrainerListener.on_test_batch_start,
        hooks.TrainerListener.on_stage_iter_ending: hooks.TrainerListener.on_test_batch_end,
        hooks.TrainerListener.on_stage_ending: hooks.TrainerListener.on_test_epoch_end,
    }


@hook_map.register
def _(stage: stages.Predict) -> dict[Callable, Callable]:
    return {
        hooks.TrainerListener.on_stage_starting: hooks.TrainerListener.on_predict_epoch_start,
        hooks.TrainerListener.on_stage_iter_starting: hooks.TrainerListener.on_predict_batch_start,
        hooks.TrainerListener.on_stage_iter_ending: hooks.TrainerListener.on_predict_batch_end,
        hooks.TrainerListener.on_stage_ending: hooks.TrainerListener.on_predict_epoch_end,
    }


@hook_map.register
def _(stage: stages.Fit) -> dict[Callable, Callable]:
    return {
        hooks.TrainerListener.on_stage_starting: hooks.TrainerListener.on_fit_start,
        hooks.TrainerListener.on_stage_ending: hooks.TrainerListener.on_fit_end,
    }
