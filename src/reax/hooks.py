from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import reax


__all__ = ("TrainerListener",)


class TrainerListener:
    # pylint: disable=too-many-public-methods

    def on_fit_start(self, trainer: "reax.Trainer", stage: "reax.stages.Fit", /) -> None:
        """A fitting stage is about to begin."""

    def on_fit_end(self, trainer: "reax.Trainer", stage: "reax.stages.Fit", /) -> None:
        """A fitting stage is about to end."""

    def on_train_start(self, trainer: "reax.Trainer", stage: "reax.stages.Train", /):
        """Training is starting"""

    def on_train_epoch_start(self, trainer: "reax.Trainer", stage: "reax.stages.Train", /) -> None:
        """A training epoch is about to begin."""

    def on_train_batch_start(
        self, trainer: "reax.Trainer", stage: "reax.stages.Train", batch: Any, batch_idx: int, /
    ) -> None:
        """The training stage if about to process a batch."""

    def on_train_batch_end(
        self,
        trainer: "reax.Trainer",
        stage: "reax.stages.Train",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        /,
    ) -> None:
        """The training stage has just finished processing a batch."""

    def on_train_epoch_end(self, trainer: "reax.Trainer", stage: "reax.stages.Train", /) -> None:
        """A training epoch is about to end."""

    def on_train_end(self, trainer: "reax.Trainer", stage: "reax.stages.Train", /):
        """Training is ending"""

    def on_validation_start(
        self, trainer: "reax.Trainer", stage: "reax.stages.Validate", /
    ) -> None:
        """Validation is starting."""

    def on_validation_epoch_start(
        self, trainer: "reax.Trainer", stage: "reax.stages.Validate", /
    ) -> None:
        """A validation epoch is about to begin."""

    def on_validation_batch_start(
        self, trainer: "reax.Trainer", stage: "reax.stages.Validate", batch: Any, batch_idx: int, /
    ) -> None:
        """The validation stage if about to process a batch."""

    def on_validation_batch_end(
        self,
        trainer: "reax.Trainer",
        stage: "reax.stages.Train",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        /,
    ) -> None:
        """The validation stage has just finished processing a batch."""

    def on_validation_epoch_end(
        self, trainer: "reax.Trainer", stage: "reax.stages.Validate", /
    ) -> None:
        """A validation epoch is about to end."""

    def on_validation_end(self, trainer: "reax.Trainer", stage: "reax.stages.Validate", /) -> None:
        """Validation has ended."""

    def on_test_start(self, trainer: "reax.Trainer", stage: "reax.stages.Test", /) -> None:
        """Test is starting."""

    def on_test_epoch_start(self, trainer: "reax.Trainer", stage: "reax.stages.Test", /) -> None:
        """A test epoch is about to begin."""

    def on_test_batch_start(
        self, trainer: "reax.Trainer", stage: "reax.stages.Test", batch: Any, batch_idx: int, /
    ) -> None:
        """The test stage if about to process a batch."""

    def on_test_batch_end(
        self,
        trainer: "reax.Trainer",
        stage: "reax.stages.Test",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        /,
    ) -> None:
        """The test stage has just finished processing a batch."""

    def on_test_epoch_end(self, trainer: "reax.Trainer", stage: "reax.stages.Test", /) -> None:
        """A test epoch is about to end."""

    def on_test_end(self, trainer: "reax.Trainer", stage: "reax.stages.Test", /) -> None:
        """Test is ending."""

    def on_predict_start(self, trainer: "reax.Trainer", stage: "reax.stages.Predict", /) -> None:
        """Predict is starting."""

    def on_predict_epoch_start(
        self, trainer: "reax.Trainer", stage: "reax.stages.Predict", /
    ) -> None:
        """A predict epoch is about to begin."""

    def on_predict_batch_start(
        self, trainer: "reax.Trainer", stage: "reax.stages.Predict", batch: Any, batch_idx: int, /
    ) -> None:
        """The test stage if about to process a batch."""

    def on_predict_batch_end(
        self,
        trainer: "reax.Trainer",
        stage: "reax.stages.Train",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        /,
    ) -> None:
        """The predict stage has just finished processing a batch."""

    def on_predict_epoch_end(self, trainer: "reax.Trainer", stage: "reax.stages.Train", /) -> None:
        """A predict epoch is about to end."""

    def on_predict_end(self, trainer: "reax.Trainer", stage: "reax.stages.Predict", /) -> None:
        """Predict is ending."""

    def on_epoch_start(self, trainer: "reax.Trainer", stage: "reax.stages.EpochStage", /) -> None:
        """An epoch is just about to begin."""

    def on_batch_end(
        self,
        trainer: "reax.Trainer",
        stage: "reax.stages.EpochStage",
        batch_idx: int,
        metrics: dict,
        /,
    ) -> None:
        """A batch has just been processed."""

    def on_epoch_end(
        self, trainer: "reax.Trainer", stage: "reax.stages.EpochStage", metrics: dict, /
    ) -> None:
        """An epoch is ending."""

    def on_exception(
        self,
        trainer: "reax.Trainer",
        stage: "reax.Stage",
        exception: BaseException,
        /,
    ) -> None:
        """Called when any trainer execution is interrupted by an exception."""

    # region Generic messages

    def setup(self, trainer: "reax.Trainer", stage: "reax.Stage", /) -> None:
        """Called when a stage (e.g. fit, validate, test, predict) begins."""

    def on_stage_start(self, trainer: "reax.Trainer", stage: "reax.stages.Stage", /) -> None:
        """A trainer stage is starting."""

    def on_stage_iter_start(
        self, trainer: "reax.Trainer", stage: "reax.Stage", step: int, /
    ) -> None:
        """A stage is about to start an interation."""

    def on_stage_iter_end(
        self, trainer: "reax.Trainer", stage: "reax.Stage", step: int, outputs: Any, /
    ):
        """The stage just finished processing an iteration."""

    def on_stage_end(self, trainer: "reax.Trainer", stage: "reax.Stage", /) -> None:
        """The stage is ending."""

    def teardown(self, trainer: "reax.Trainer", state: "reax.Stage", /) -> None:
        """Called when a stage (e.g. fit, validate, test, predict, or tune ends)."""

    # endregion
