from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import reax


class TrainerListener:
    def on_stage_starting(self, trainer: "reax.Trainer", stage: "reax.stages.Stage") -> None:
        """A trainer stage is about to begin"""

    def on_epoch_starting(self, trainer: "reax.Trainer", stage: "reax.stages.EpochStage") -> None:
        """An epoch is just about to begin"""

    def on_batch_starting(
        self, trainer: "reax.Trainer", stage: "reax.stages.EpochStage", batch_idx: int
    ) -> None:
        """A batch is just about to be processed"""

    def on_batch_ending(
        self,
        trainer: "reax.Trainer",
        stage: "reax.stages.EpochStage",
        batch_idx: int,
        metrics: dict,
    ) -> None:
        """A batch has just been processed"""

    def on_epoch_ending(
        self, trainer: "reax.Trainer", stage: "reax.stages.EpochStage", metrics: dict
    ) -> None:
        """An epoch is ending"""

    def on_stage_ending(self, trainer: "reax.Trainer", stage: "reax.stages.Stage") -> None:
        """A trainer stage is ending"""


class ModelHooks:
    def on_train_start(self) -> None:
        """Training is about to begin"""

    def on_train_end(self) -> None:
        """Training is ending"""

    def configure_model(self, batch):
        """
        Configure the model before a fit/va/test/predict stage.  This will be called at the
        beginning of each of these stages, so it's important that the implementation is a no-op
        after the first time it is called.
        """
