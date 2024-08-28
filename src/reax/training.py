import contextlib
import signal
import sys
from typing import Literal, Optional

import jax

from . import hooks, listeners, modules, stages
from .optimizers import Optimizer
from .utils import events

__all__ = ("Trainer",)


class Trainer(stages.StageListener):
    def __init__(
        self,
        module: modules.Module,
        accelerator: Literal["auto", "cpu", "gpu"] = "auto",
        log_every_n_steps: int = 50,
        check_val_every_n_epoch: int = 1,
        enable_progress_bar: bool = True,
    ):
        self._accelerator = (
            jax.devices()[0] if accelerator == "auto" else jax.devices(accelerator)[0]
        )
        self._log_every_n_steps = log_every_n_steps
        self.check_val_every_n_epoch = check_val_every_n_epoch

        self._automatic_optimization = True
        self._optimizers = []
        self._stage: Optional[stages.Stage] = None
        self._current_epoch: Optional[int] = None

        self.events = events.EventGenerator[hooks.TrainerListener]()

        self._num_batches = None

        # Attach the trainer to the module
        self._module = module
        module.trainer = self

        if enable_progress_bar:
            self.events.add_listener(listeners.TqdmProgressBar())

    @property
    def current_epoch(self) -> Optional[int]:
        return self._current_epoch

    @property
    def optimizers(self) -> list[Optimizer]:
        return self._optimizers

    @optimizers.setter
    def optimizers(self, opts: list[Optimizer]) -> None:
        self._optimizers = opts

    @property
    def should_stop(self):
        return self._stage.should_stop

    @should_stop.setter
    def should_stop(self, stop: bool):
        self._stage.should_stop = stop

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
                "There is currently not stage running."
            )

        self._stage.log(
            name,
            value,
            prog_bar=prog_bar,
            batch_size=batch_size,
            on_step=on_step,
            on_epoch=on_epoch,
        )

    def fit(
        # pylint: disable=unused-argument
        self,
        train_dataloaders=None,
        val_dataloaders=None,
        datamodule=None,
        ckpt_path=None,
        max_epochs: int = 1_000,
        min_epochs: Optional[int] = None,
    ):

        # Must choose either a datamodule or train/val dataloaders
        if datamodule is not None:
            if train_dataloaders is not None or val_dataloaders is not None:
                raise ValueError(
                    "You cannot pass `train_dataloader` or `val_dataloaders` and `datamodule` to "
                    "`Trainer.fit()`"
                )
            train_dataloaders = datamodule.train_dataloader()
            val_dataloaders = datamodule.val_dataloader()

        if self._module.parameters() is None:
            batch = next(iter(train_dataloaders))
            self._module.configure_model(batch)

        self._configure_optimizers(self._module)

        fit = stages.Fit(
            self._module,
            train_dataloaders,
            val_dataloaders,
            optimizers=self.optimizers,
            min_steps=min_epochs,
            max_steps=max_epochs,
            check_val_every_n_epoch=self.check_val_every_n_epoch,
        )
        self._run_stage(fit)
        self._current_epoch = None

    def test(
        self,
        dataloaders=None,
        datamodule=None,
    ):
        if datamodule is not None:
            if dataloaders is not None:
                raise ValueError("Cannot supply dataloaders and datamodule to Trainer.test()")

            dataloaders = datamodule.test_dataloader()

        self._run_stage(stages.Test(self._module, dataloaders))

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

    def _configure_optimizers(self, module):
        opts = module.configure_optimizers()
        if not isinstance(opts, list):
            opts = [opts]
        self.optimizers = list(map(lambda opt: Optimizer(*opt), opts))

    def on_stage_starting(self, stage: "stages.Stage") -> None:
        """The stage is about to start"""
        self.events.fire_event(hooks.TrainerListener.on_stage_starting, self, stage)

        if isinstance(stage, stages.EpochStage):
            self.events.fire_event(hooks.TrainerListener.on_epoch_starting, self, stage)

    def on_stage_step_start(self, stage: "stages.Stage", step: int):
        if isinstance(stage, stages.EpochStage):
            self.events.fire_event(hooks.TrainerListener.on_batch_starting, self, stage, step)

    def on_stage_step_end(self, stage: "stages.Stage", step: int, metrics: dict):
        if isinstance(stage, stages.EpochStage):
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
                hooks.TrainerListener.on_epoch_ending, self, stage, stage.results
            )

        self.events.fire_event(hooks.TrainerListener.on_stage_ending, self, stage)
