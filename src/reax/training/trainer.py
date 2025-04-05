from collections.abc import Sequence
import contextlib
import functools
import logging
import os
import signal
import sys
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Final,
    Generator,
    Iterable,
    Literal,
    Optional,
    Union,
)
import weakref

import beartype
import fsspec
import jax
import jaxtyping as jt
from lightning_utilities.core import rank_zero
from typing_extensions import override

from . import _checkpointing, _deprecated, _logger_connector
from .. import data, exceptions, hooks, keys
from .. import listeners as listeners_
from .. import loggers as loggers_
from .. import modules, random, stages, strategies, typing
from ..utils import events

if TYPE_CHECKING:
    import reax

_LOGGER = logging.getLogger(__name__)

__all__ = ("Trainer",)


class Trainer(stages.StageListener, _deprecated.TrainerDeprecatedMixin):
    # pylint: disable=too-many-public-methods

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        *,
        accelerator: Literal["auto", "cpu", "gpu"] = "auto",
        devices: Union[list[int], str, int] = "auto",
        precision=None,
        logger: Optional[Union["reax.Logger", Iterable["reax.Logger"], bool]] = True,
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
        if precision is not None:
            _LOGGER.warning(
                "`precision` other than None is not supported yet, ignoring.  "
                "There is still ongoing discussion on how to support this in JAX, "
                "see e.g.: https://github.com/jax-ml/jax/issues/22688"
            )

        self._accelerator = (
            jax.devices()[0] if accelerator == "auto" else jax.devices(accelerator)[0]
        )
        # Params
        self._log_every_n_steps: Final[int] = log_every_n_steps

        self._strategy = strategies.SingleDevice(self._accelerator)
        self._default_root_dir = (
            os.fspath(default_root_dir) if default_root_dir is not None else os.getcwd()
        )

        self._automatic_optimization = True
        self._optimizers = []
        self._stage: Optional[stages.Stage] = None

        # State
        self._rng = random.Generator(key=rng_key if rng_key is not None else jax.random.key(0))
        self._current_epoch: int = 0
        self._global_updates: int = 0

        # Init
        self._events = events.EventGenerator[hooks.TrainerListener](
            default_args=(weakref.proxy(self),)
        )

        self._loggers: list[loggers_.Logger] = _init_loggers(logger, self.default_root_dir)

        self._logging = _logger_connector.TrainerLogging()

        if isinstance(listeners, hooks.TrainerListener):
            listeners = [listeners]
        if listeners:
            for listener in listeners:
                self._events.add_listener(listener)

        pbar = _init_progress_bar(self._events.find(), enable_progress_bar)
        if pbar is not None:
            self._events.add_listener(pbar)

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
    def rng(self) -> "reax.random.Generator":
        return self._rng

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
    def sanity_checking(self) -> bool:
        """`True` if currently sanity checking, `False` otherwise"""
        return isinstance(self._stage, stages.Fit) and self._stage.sanity_checking

    @property
    def default_root_dir(self) -> str:
        """Get the fallback directory used for loggers and other components when not explicitly
        specified."""
        if _is_local_file_protocol(self._default_root_dir):
            return os.path.normpath(os.path.expanduser(self._default_root_dir))

        return self._default_root_dir

    # region Listeners

    @property
    def listeners(self) -> "list[reax.TrainerListener]":
        return self._events.find()

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
    def progress_bar_listeners(self) -> list["reax.listeners.ProgressBar"]:
        """The first :class:`~reax.listeners.ProgressBar` listener in the
        Trainer.listeners list, or ``None`` if it doesn't exist."""
        return self._events.find(type=listeners_.ProgressBar)

    @property
    def progress_bar_listener(self) -> Optional["reax.listeners.ProgressBar"]:
        """The first :class:`~reax.listeners.ProgressBar` listener in the
        Trainer.listeners list, or ``None`` if it doesn't exist."""
        listeners = self.progress_bar_listeners
        return listeners[0] if len(listeners) > 0 else None

    # end region

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
    def logged_metrics(self) -> dict:
        """Logger metrics."""
        return self._logging.logged_metrics

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
    def train_dataloader(self) -> "Optional[reax.data.DataLoader]":
        """Train dataloader."""
        if self._stage is not None:
            return getattr(self._stage, "train_dataloader", None)
        return None

    @property
    def val_dataloaders(self) -> "Optional[reax.data.DataLoader]":
        """Train dataloader."""
        if self._stage is not None:
            return getattr(self._stage, "val_dataloader", None)
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
        ckpt_path: Optional[typing.Path] = None,
        fast_dev_run: Union[bool, int] = False,
        max_epochs: Optional[int] = 1_000,
        min_epochs: int = 0,
        min_updates: int = 0,
        max_updates: Union[int, float] = None,
        limit_train_batches: Optional[Union[int, float]] = 1.0,
        accumulate_grad_batches: int = 1,
        limit_val_batches: Optional[Union[int, float]] = 1.0,
        val_check_interval: Optional[Union[int, float]] = 1.0,
        check_val_every_n_epoch: int = 1,
        num_sanity_val_steps: Optional[int] = 2,
        reload_dataloaders_every_n_epochs: int = 0,
    ) -> "reax.stages.Fit":
        """Fit function.


        :param ckpt_path: Path/URL of the checkpoint from which training is resumed. Could also be
            one of two special keywords ``"last"`` and ``"hpc"``. If there is no checkpoint file at
            the path, an exception is raised.
        """
        if max_updates is not None and max_updates < 0:
            raise ValueError("`max_updates` must be a non-negative integer")
        if max_epochs is not None and max_epochs < 0:
            raise ValueError(f"`max_epochs` must be a non-negative integer or {keys.NO_LIMIT}")

        if fast_dev_run:
            num_batches = 1 if fast_dev_run is True else fast_dev_run
            rank_zero.rank_zero_info(
                f"Running in `fast_dev_run` mode: will run the requested loop using {num_batches} "
                f"batch(es). Logging and checkpointing is suppressed."
            )

        if ckpt_path:
            _checkpointing.load_checkpoint(module, ckpt_path, weights_only=False)

        datamanager = data.create_manager(
            module=module, datamodule=datamodule, train=train_dataloaders, val=val_dataloaders
        )
        fit = stages.Fit(
            module,
            datamanager,
            self.optimizers,
            self._strategy,
            self._rng,
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
        self._run_stage(fit)
        # Update state variables
        self._global_updates += fit.updates

        return fit

    @jt.jaxtyped(typechecker=beartype.beartype)
    def validate(
        self,
        module: modules.Module,
        dataloaders=None,
        datamodule=None,
        ckpt_path: Optional[typing.Path] = None,
        limit_batches: Union[int, type(keys.NO_LIMIT)] = keys.NO_LIMIT,
    ) -> "reax.stages.Validate":
        """Validate function."""
        if ckpt_path:
            _checkpointing.load_checkpoint(module, ckpt_path, weights_only=False)

        datamanager = data.create_manager(module, datamodule, val=dataloaders)
        validate = stages.Validate(module, datamanager, self._strategy, limit_batches=limit_batches)
        self._run_stage(validate)

        return validate

    @jt.jaxtyped(typechecker=beartype.beartype)
    def test(
        self,
        module: modules.Module,
        dataloaders=None,
        datamodule=None,
        ckpt_path: Optional[typing.Path] = None,
        fast_dev_run: Union[bool, int] = False,
        limit_batches: Optional[Union[int, float]] = 1.0,
    ) -> "reax.stages.Test":
        """Test function."""
        if ckpt_path:
            _checkpointing.load_checkpoint(module, ckpt_path, weights_only=False)

        datamanager = data.create_manager(module, datamodule, test=dataloaders)
        test = stages.Test(
            module,
            datamanager,
            self._strategy,
            self._rng,
            fast_dev_run=fast_dev_run,
            limit_batches=limit_batches,
        )
        self._run_stage(test)

        return test

    @jt.jaxtyped(typechecker=beartype.beartype)
    def predict(
        self,
        module: modules.Module,
        dataloaders: "Optional[reax.DataLoader]" = None,
        datamodule: "Optional[reax.DataModule]" = None,
        ckpt_path: Optional[typing.Path] = None,
        return_predictions: Optional[bool] = None,
        fast_dev_run: Union[bool, int] = False,
        limit_batches: Union[int, float] = keys.NO_LIMIT,
    ) -> "reax.stages.Predict":
        r"""Run inference on the data.

        Logging is disabled in the predict hooks.
        """
        if ckpt_path:
            _checkpointing.load_checkpoint(module, ckpt_path, weights_only=True)

        if fast_dev_run:
            limit_batches = 1

        if return_predictions is None:
            return_predictions = True

        datamanager = data.create_manager(module, datamodule, predict=dataloaders)
        predict = stages.Predict(
            module,
            datamanager,
            self._strategy,
            limit_batches=limit_batches,
            keep_predictions=return_predictions,
        )
        self._run_stage(predict)

        return predict

    @jt.jaxtyped(typechecker=beartype.beartype)
    def eval_stats(
        self,
        stats: Union["reax.Metric", Sequence["reax.Metric"], dict[str, "reax.Metric"]],
        dataloaders: "Optional[reax.DataLoader]" = None,
        datamodule: "Optional[reax.DataModule]" = None,
        dataset_name: str = "train",
        fast_dev_run: Union[bool, int] = False,
        limit_batches: Union[int, float] = keys.NO_LIMIT,
    ) -> "reax.stages.EvaluateStats":
        r"""Evaluate metrics on a dataset to get statistics about it.

        The calculated statistics will be stored on `stage.logged_metrics` where `stage` is the
        return value of this call.
        """
        if fast_dev_run:
            limit_batches = 1

        datamanager = data.create_manager(datamodule=datamodule, **{dataset_name: dataloaders})
        eval_stats = stages.EvaluateStats(
            stats,
            datamanager,
            self._strategy,
            self._rng,
            dataset_name=dataset_name,
            limit_batches=limit_batches,
        )
        self._run_stage(eval_stats)

        return eval_stats

    def run(self, stage: "reax.stages.Stage") -> "reax.stages.Stage":
        return self._run_stage(stage)

    def _run_stage(self, stage: stages.Stage) -> stages.Stage:
        """Run stage."""
        with jax.default_device(jax.devices("cpu")[0]):
            try:
                with self._attach(stage):
                    with stage.events.listen_context(self):
                        stage.run()
            except KeyboardInterrupt as exc:
                rank_zero.rank_zero_info(
                    "\nDetected KeyboardInterrupt, attempting graceful shutdown ..."
                )
                # Disable further Ctrl+C presses while we respond to this one
                signal.signal(signal.SIGINT, signal.SIG_IGN)
                self._interrupt(stage, exc)
                sys.exit(1)
            except BaseException as exc:
                self._interrupt(stage, exc)
                raise

            else:
                for logger in self.loggers:
                    logger.finalize("success")

        return stage

    def _interrupt(self, stage: "reax.Stage", exception: BaseException):
        # Inform our listeners of the exception
        self._events.fire_event(hooks.TrainerListener.on_exception, stage, exception)

        # TODO: self._strategy.on_exception(exc)
        for logger in self.loggers:
            logger.finalize("failed")

    # endregion

    @contextlib.contextmanager
    def _attach(self, stage: stages.Stage):
        """Attach function."""
        self._stage = stage
        if stage.module is not None:
            stage.module.trainer = self
            listeners = stage.module.configure_listeners()
        else:
            listeners = []

        try:
            with self._attach_model_listeners(listeners):
                yield
        finally:
            self._stage = None
            if stage.module is not None:
                stage.module.trainer = None

    @contextlib.contextmanager
    def _attach_model_listeners(
        self, listeners: Union["reax.TrainerListener", Sequence["reax.TrainerListener"]]
    ) -> Generator[None, Any, None]:
        """Attaches the listeners defined in the model.

        If a listener returned by the model's configure_listener method has the same type as one or
        several listeners already present in the trainer listeners list, it will replace them.
        In addition, all :class:`~lightning.pytorch.listeners.model_checkpoint.ModelCheckpoint`
        listeners will be pushed to the end of the list, ensuring they run last.
        """
        if listeners:
            listeners = [listeners] if not isinstance(listeners, Sequence) else listeners
            model_listener_types = {type(c) for c in listeners}
            trainer_listener_types = {type(c) for c in self.listeners}
            # edge case: if an unmodified listener was added, the logic below would filter it
            trainer_listener_types.discard(hooks.TrainerListener)
            # exclude trainer listeners of the same class or subclass
            override_types = set()
            for model_cb in model_listener_types:
                for trainer_cb in trainer_listener_types:
                    if issubclass(model_cb, trainer_cb):
                        override_types.add(trainer_cb)
                        break
            if override_types:
                rank_zero.rank_zero_info(
                    "The following listeners returned in `Module.configure_listeners` will override"
                    " existing listeners passed to Trainer:"
                    f" {', '.join(sorted(t.__name__ for t in override_types))}"
                )
            # remove all listeners with a type that occurs in model listeners
            all_listeners = [
                listener for listener in self.listeners if type(listener) not in override_types
            ]
            all_listeners.extend(listeners)
            all_listeners = _reorder_listeners(all_listeners)
        else:
            all_listeners = None

        original_events = self._events
        try:
            if listeners:
                self._events = events.EventGenerator[hooks.TrainerListener](
                    default_args=(weakref.proxy(self),)
                )
                for listener in all_listeners:
                    self._events.add_listener(listener)

            yield
        finally:
            if listeners:
                self._events = original_events

    @override
    def on_stage_start(self, stage: "stages.Stage", /) -> None:
        """The stage is starting."""
        if not self._optimizers and isinstance(stage, stages.Train):
            self._optimizers = stage.optimizers

        if stage.is_root:
            self._logging.reset_metrics()
            self._events.fire_event(hooks.TrainerListener.setup, stage)

        self._events.fire_event(hooks.TrainerListener.on_stage_start, stage)
        self._fire_stage_event(stage, stages.StageListener.on_stage_start)

    @override
    def on_stage_started(self, stage: "reax.Stage", /):
        if isinstance(stage, stages.EpochStage):
            self._events.fire_event(hooks.TrainerListener.on_epoch_start, stage)
        self._fire_stage_event(stage, stages.StageListener.on_stage_started)

    @override
    def on_stage_iter_starting(self, stage: "stages.Stage", step: int, /):
        """On stage iter starting."""
        self._events.fire_event(hooks.TrainerListener.on_stage_iter_start, stage, step)
        if isinstance(stage, stages.EpochStage):
            self._fire_stage_event(
                stage, stages.StageListener.on_stage_iter_starting, stage.batch, step
            )

    @override
    def on_stage_iter_ended(self, stage: "stages.Stage", step: int, outputs: Any, /):
        """On stage iter ending."""
        if isinstance(stage, stages.EpochStage):
            self._logging.update(stage)
            logging_metrics = {"epoch": self.current_epoch, "stage": stage.name}
            logging_metrics.update(stage.logged_metrics)
            for logger in self.loggers:
                logger.log_metrics(metrics=logging_metrics, step=self.global_updates - 1)
                logger.save()

            self._events.fire_event(hooks.TrainerListener.on_batch_end, stage, step, outputs)

            self._fire_stage_event(
                stage, stages.StageListener.on_stage_iter_ending, outputs, stage.batch, step
            )

        self._events.fire_event(hooks.TrainerListener.on_stage_iter_end, stage, step, outputs)

        if isinstance(stage, stages.Fit):
            # Keep track of the number of completed training epochs
            self._current_epoch += 1

    @override
    def on_stage_end(self, stage: "reax.Stage", /):
        if isinstance(stage, stages.EpochStage):
            self._logging.update(stage)
            self._events.fire_event(
                hooks.TrainerListener.on_epoch_end, stage, stage.listener_metrics
            )

            metrics = stage.results
            if metrics[keys.LOG]:
                logging_metrics = {"epoch": self.current_epoch, "stage": stage.name}
                logging_metrics.update(self.logged_metrics)
                for logger in self.loggers:
                    logger.log_metrics(metrics=logging_metrics, step=self.global_updates - 1)
                    logger.save()

        self._fire_stage_event(stage, stages.StageListener.on_stage_end)

    @override
    def on_stage_ended(self, stage: "reax.Stage", /):
        self._events.fire_event(hooks.TrainerListener.on_stage_end, stage)
        self._fire_stage_event(stage, stages.StageListener.on_stage_ended)
        # Finally, call teardown
        if stage.is_root:
            self._events.fire_event(hooks.TrainerListener.teardown, stage)

    # region Checkpointing

    def save_checkpoint(self, filepath: typing.Path, weights_only: bool = True):
        """For now, we just save the model weights.

        The user has to store the model definition themselves.
        """
        _checkpointing.save_checkpoint(self._module, filepath, weights_only=weights_only)

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

    def _fire_stage_event(self, stage, hook: Callable, *args):
        event = hook_map(stage).get(hook)
        if event is not None:
            self._events.fire_event(event, stage, *args)


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
            return [loggers_.CsvLogger(default_root_dir)]

        return []

    return list(logger)


def _init_progress_bar(
    listeners: list[hooks.TrainerListener], enable_progress_bar: bool = True
) -> Optional[listeners_.ProgressBar]:
    progress_bars = [
        listener for listener in listeners if isinstance(listener, listeners_.progress.ProgressBar)
    ]
    if len(progress_bars) > 1:
        raise exceptions.MisconfigurationException(
            "You added multiple progress bar listeners to the Trainer, but currently only one"
            " progress bar is supported."
        )
    if len(progress_bars) == 1:
        # the user specified the progress bar in the listeners list
        # so the trainer doesn't need to provide a default one
        if enable_progress_bar:
            return None

        # otherwise the user specified a progress bar listener but also
        # elected to disable the progress bar with the trainer flag
        progress_bar_listener = progress_bars[0]
        raise exceptions.MisconfigurationException(
            "Trainer was configured with `enable_progress_bar=False`"
            f" but found `{progress_bar_listener.__class__.__name__}` in listeners list."
        )

    if enable_progress_bar:
        return listeners_.progress.TqdmProgressBar()

    return None


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
        stages.StageListener.on_stage_start: hooks.TrainerListener.on_train_start,
        stages.StageListener.on_stage_started: hooks.TrainerListener.on_train_epoch_start,
        stages.StageListener.on_stage_iter_starting: hooks.TrainerListener.on_train_batch_start,
        stages.StageListener.on_stage_iter_ending: hooks.TrainerListener.on_train_batch_end,
        stages.StageListener.on_stage_end: hooks.TrainerListener.on_train_epoch_end,
        stages.StageListener.on_stage_ended: hooks.TrainerListener.on_train_end,
    }


@hook_map.register
def _(_stage: stages.Validate) -> dict[Callable, Callable]:
    """Function."""
    return {
        stages.StageListener.on_stage_start: hooks.TrainerListener.on_validation_start,
        stages.StageListener.on_stage_started: hooks.TrainerListener.on_validation_epoch_start,
        # pylint: disable=line-too-long
        stages.StageListener.on_stage_iter_starting: hooks.TrainerListener.on_validation_batch_start,
        stages.StageListener.on_stage_iter_ending: hooks.TrainerListener.on_validation_batch_end,
        stages.StageListener.on_stage_end: hooks.TrainerListener.on_validation_epoch_end,
        stages.StageListener.on_stage_ended: hooks.TrainerListener.on_validation_end,
    }


@hook_map.register
def _(_stage: stages.Test) -> dict[Callable, Callable]:
    """Function."""
    return {
        stages.StageListener.on_stage_start: hooks.TrainerListener.on_test_start,
        stages.StageListener.on_stage_started: hooks.TrainerListener.on_test_epoch_start,
        stages.StageListener.on_stage_iter_starting: hooks.TrainerListener.on_test_batch_start,
        stages.StageListener.on_stage_iter_ending: hooks.TrainerListener.on_test_batch_end,
        stages.StageListener.on_stage_end: hooks.TrainerListener.on_test_epoch_end,
        stages.StageListener.on_stage_ended: hooks.TrainerListener.on_test_end,
    }


@hook_map.register
def _(_stage: stages.Predict) -> dict[Callable, Callable]:
    """Function."""
    return {
        stages.StageListener.on_stage_start: hooks.TrainerListener.on_predict_start,
        stages.StageListener.on_stage_started: hooks.TrainerListener.on_predict_epoch_start,
        stages.StageListener.on_stage_iter_starting: hooks.TrainerListener.on_predict_batch_start,
        stages.StageListener.on_stage_iter_ending: hooks.TrainerListener.on_predict_batch_end,
        stages.StageListener.on_stage_end: hooks.TrainerListener.on_predict_epoch_end,
        stages.StageListener.on_stage_ended: hooks.TrainerListener.on_predict_end,
    }


@hook_map.register
def _(_stage: stages.Fit) -> dict[Callable, Callable]:
    """Function."""
    return {
        stages.StageListener.on_stage_start: hooks.TrainerListener.on_fit_start,
        stages.StageListener.on_stage_end: hooks.TrainerListener.on_fit_end,
    }


def _reorder_listeners(listeners: list["reax.TrainerListener"]) -> list["reax.TrainerListener"]:
    """Moves all the tuner specific listeners at the beginning of the list and all the
    `ModelCheckpoint` listeners to the end of the list. The sequential order within the group of
    checkpoint listeners is preserved, as well as the order of all other listeners.

    Args:
        listeners: A list of listeners.

    Return:
        A new list in which the first elements are tuner specific listeners and last elements are
        ModelCheckpoints if there were any present in the input.

    """
    tuner_listeners: list["reax.TrainerListener"] = []
    other_listeners: list["reax.TrainerListener"] = []
    checkpoint_listeners: list["reax.TrainerListener"] = []

    for cb in listeners:
        # if isinstance(cb, (BatchSizeFinder, LearningRateFinder)):
        #     tuner_listeners.append(cb)
        if isinstance(cb, listeners_.Checkpointer):
            checkpoint_listeners.append(cb)
        else:
            other_listeners.append(cb)

    return tuner_listeners + other_listeners + checkpoint_listeners
