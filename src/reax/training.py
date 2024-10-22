import contextlib
import os
import signal
import sys
from typing import TYPE_CHECKING, Iterable, Literal, Optional, Union

import beartype
import fsspec
import jax
import jaxtyping as jt

from . import hooks, keys
from . import listeners as listeners_
from . import loggers as loggers_
from . import modules, stages, strategies, typing
from .utils import events

if TYPE_CHECKING:
    import reax

__all__ = ("Trainer",)


class Trainer(stages.StageListener):
    def __init__(
        self,
        module: modules.Module,
        accelerator: Literal["auto", "cpu", "gpu"] = "auto",
        logger: Optional[Union["reax.Logger", Iterable["reax.Logger"], bool]] = None,
        listeners: "Optional[list[reax.TrainerListener]]" = None,
        log_every_n_steps: int = 50,
        check_val_every_n_epoch: int = 1,
        enable_progress_bar: bool = True,
        rng_key: jax.Array = None,
        default_root_dir: Optional[typing.Path] = None,
    ):
        self._accelerator = (
            jax.devices()[0] if accelerator == "auto" else jax.devices(accelerator)[0]
        )

        self._strategy = strategies.SingleDevice(self._accelerator)
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
        self._current_epoch = 0
        self._global_updates = 0

        self.events = events.EventGenerator[hooks.TrainerListener]()

        # Attach the trainer to the module
        self._module = module
        module.trainer = self

        self._loggers = _init_loggers(logger, self.default_root_dir)

        if enable_progress_bar:
            self.events.add_listener(listeners_.TqdmProgressBar())
        if listeners:
            for listener in listeners:
                self.events.add_listener(listener)

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
    def train_dataloader(self) -> Optional["reax.DataLoader"]:
        if isinstance(self._stage, stages.Fit):
            return self._stage.train.dataloader

        return None

    @property
    def current_epoch(self) -> int:
        """Get the current fitting epoch"""
        if self._stage is not None and isinstance(self._stage, stages.Fit):
            # The stage is running, so count add the number of updates it has performed so far
            return self._current_epoch + self._stage.iteration

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
        return self._stage.should_stop

    @should_stop.setter
    def should_stop(self, stop: bool):
        self._stage.should_stop = stop

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
    def logger(self) -> "reax.Logger":
        """Get the first (and main) logger"""
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
        """The directory for the current experiment. Use this to save images to, etc...

        .. note:: You must call this on all processes. Failing to do so will cause your program to
            stall forever.

         .. code-block:: python

             def training_step(self, batch, batch_idx):
                 img = ...
                 save_img(img, self.trainer.log_dir)

        """
        if len(self.loggers) > 0:
            dirpath = self.loggers[0].log_dir
        else:
            dirpath = self.default_root_dir

        # dirpath = self.strategy.broadcast(dirpath)
        return dirpath

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

        self._stage.log(
            name,
            value,
            prog_bar=prog_bar,
            logger=logger,
            batch_size=batch_size,
            on_step=on_step,
            on_epoch=on_epoch,
        )

    @jt.jaxtyped(typechecker=beartype.beartype)
    def fit(
        # pylint: disable=unused-argument
        self,
        train_dataloaders: "Optional[reax.DataLoader]" = None,
        val_dataloaders: "Optional[reax.DataLoader]" = None,
        datamodule: "Optional[reax.DataModule]" = None,
        ckpt_path=None,
        max_epochs: Optional[int] = 1_000,
        min_epochs: Optional[int] = None,
        min_updates: Optional[int] = None,
        max_updates: int = -1,
        limit_train_batches: Optional[Union[int, float]] = 1.0,
        accumulate_grad_batches: int = 1,
        limit_val_batches: Optional[Union[int, float]] = 1.0,
    ):
        if max_updates < -1:
            raise ValueError("`max_updates` must be a non-negative integer or -1")
        if max_epochs is None:
            max_epochs = -1
        elif max_epochs < -1:
            raise ValueError("`max_epochs` must be a non-negative integer or -1")

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

        fit = stages.Fit(
            self._module,
            train_dataloaders,
            val_dataloaders,
            optimizers=self.optimizers,
            strategy=self._strategy,
            min_updates=min_updates,
            max_updates=max_updates,
            min_iters=min_epochs,
            max_iters=max_epochs,
            limit_train_batches=limit_train_batches,
            accumulate_grad_batches=accumulate_grad_batches,
            limit_val_batches=limit_val_batches,
            check_val_every_n_epoch=self.check_val_every_n_epoch,
        )
        self._run_stage(fit)
        # Update state variables
        self._global_updates += fit.updates
        self._current_epoch += fit.iteration

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

    def _run_stage(self, stage: stages.Stage) -> stages.Stage:
        try:
            with self._attach(stage):
                with stage.events.listen_context(self):
                    stage.run()
        except KeyboardInterrupt:
            # Disable further Ctrl+C presses while we respond to this one
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            sys.exit(1)

        return stage

    @contextlib.contextmanager
    def _attach(self, stage: stages.Stage):
        self._stage = stage
        try:
            yield
        finally:
            self._stage = None

    def on_stage_starting(self, stage: "stages.Stage") -> None:
        """The stage is about to start"""
        if not self._optimizers and isinstance(stage, stages.Train):
            self._optimizers = stage.optimizers

        self.events.fire_event(hooks.TrainerListener.on_stage_starting, self, stage)

        if isinstance(stage, stages.EpochStage):
            self.events.fire_event(hooks.TrainerListener.on_epoch_starting, self, stage)

    def on_stage_iter_starting(self, stage: "stages.Stage", step: int):
        if isinstance(stage, stages.EpochStage):
            self.events.fire_event(hooks.TrainerListener.on_batch_starting, self, stage, step)

    def on_stage_iter_ending(self, stage: "stages.Stage", step: int, metrics: dict):
        if isinstance(stage, stages.EpochStage):
            if metrics[keys.LOG]:
                logging_metrics = {"epoch": self.current_epoch, "stage": stage.name}
                logging_metrics.update(metrics[keys.LOG])
                for logger in self.loggers:
                    logger.log_metrics(metrics=logging_metrics, step=self.global_updates - 1)
                    logger.save()

            self.events.fire_event(
                hooks.TrainerListener.on_batch_ending,
                self,
                stage,
                batch_idx=step,
                metrics=metrics,
            )

    def on_stage_ending(self, stage: "stages.Stage") -> None:
        """Called when the stage has finished a full epoch"""
        if isinstance(stage, stages.EpochStage):
            self.events.fire_event(
                hooks.TrainerListener.on_epoch_ending, self, stage, stage.results[keys.CALLBACK]
            )

            metrics = stage.results
            if metrics[keys.LOG]:
                logging_metrics = {"epoch": self.current_epoch, "stage": stage.name}
                logging_metrics.update(metrics[keys.LOG])
                for logger in self.loggers:
                    logger.log_metrics(metrics=logging_metrics, step=self.global_updates - 1)
                    logger.save()

        self.events.fire_event(hooks.TrainerListener.on_stage_ending, self, stage)


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
