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

    def on_train_start(
        self, trainer: "reax.Trainer", stage: "reax.Stage", loop: "reax.stages.FitLoop", /
    ) -> None:
        """Training is starting."""

    def on_train_epoch_start(
        self, trainer: "reax.Trainer", stage: "reax.Stage", loop: "reax.stages.TrainEpoch", /
    ) -> None:
        """A training epoch is about to begin."""

    def on_train_batch_start(
        self,
        trainer: "reax.Trainer",
        stage: "reax.Stage",
        loop: "reax.stages.TrainEpoch",
        batch: Any,
        /,
    ) -> None:
        """The training stage is about to process a batch."""

    def on_train_batch_end(
        self,
        trainer: "reax.Trainer",
        stage: "reax.Stage",
        loop: "reax.stages.TrainEpoch",
        outputs: Any,
        /,
    ) -> None:
        """The training stage has just finished processing a batch."""

    def on_train_epoch_end(
        self, trainer: "reax.Trainer", stage: "reax.Stage", loop: "reax.stages.TrainEpoch", /
    ) -> None:
        """A training epoch is about to end."""

    def on_train_end(
        self, trainer: "reax.Trainer", stage: "reax.Stage", loop: "reax.stages.FitLoop", /
    ) -> None:
        """Training is ending."""

    def on_validation_start(
        self, trainer: "reax.Trainer", stage: "reax.stages.Validate", /
    ) -> None:
        """Validation is starting."""

    def on_validation_epoch_start(
        self, trainer: "reax.Trainer", stage: "reax.Stage", loop: "reax.stages.ValidationEpoch", /
    ) -> None:
        """A validation epoch is about to begin."""

    def on_validation_batch_start(
        self,
        trainer: "reax.Trainer",
        stage: "reax.Stage",
        loop: "reax.stages.ValidationEpoch",
        batch: Any,
        /,
    ) -> None:
        """The validation stage is about to process a batch."""

    def on_validation_batch_end(
        self,
        trainer: "reax.Trainer",
        stage: "reax.Stage",
        loop: "reax.stages.ValidationEpoch",
        outputs: Any,
        /,
    ) -> None:
        """The validation stage has just finished processing a batch."""

    def on_validation_epoch_end(
        self, trainer: "reax.Trainer", stage: "reax.Stage", loop: "reax.stages.ValidationEpoch", /
    ) -> None:
        """A validation epoch is about to end."""

    def on_validation_end(self, trainer: "reax.Trainer", stage: "reax.stages.Validate", /) -> None:
        """Validation has ended."""

    def on_test_start(self, trainer: "reax.Trainer", stage: "reax.stages.Test", /) -> None:
        """Test is starting."""

    def on_test_epoch_start(
        self, trainer: "reax.Trainer", stage: "reax.Stage", loop: "reax.stages.TestEpoch", /
    ) -> None:
        """A test epoch is about to begin."""

    def on_test_batch_start(
        self,
        trainer: "reax.Trainer",
        stage: "reax.Stage",
        loop: "reax.stages.TestEpoch",
        batch: Any,
        /,
    ) -> None:
        """The test stage is about to process a batch."""

    def on_test_batch_end(
        self,
        trainer: "reax.Trainer",
        stage: "reax.Stage",
        loop: "reax.stages.TestEpoch",
        outputs: Any,
        /,
    ) -> None:
        """The test stage has just finished processing a batch."""

    def on_test_epoch_end(
        self, trainer: "reax.Trainer", stage: "reax.Stage", loop: "reax.stages.TestEpoch", /
    ) -> None:
        """A test epoch is about to end."""

    def on_test_end(self, trainer: "reax.Trainer", stage: "reax.stages.Test", /) -> None:
        """Test is ending."""

    def on_predict_start(self, trainer: "reax.Trainer", stage: "reax.stages.Predict", /) -> None:
        """Predict is starting."""

    def on_predict_epoch_start(
        self, trainer: "reax.Trainer", stage: "reax.Stage", loop: "reax.stages.PredictionEpoch", /
    ) -> None:
        """A predict epoch is about to begin."""

    def on_predict_batch_start(
        self,
        trainer: "reax.Trainer",
        stage: "reax.Stage",
        loop: "reax.stages.PredictionEpoch",
        batch: Any,
        /,
    ) -> None:
        """The predict stage is about to process a batch."""

    def on_predict_batch_end(
        self,
        trainer: "reax.Trainer",
        stage: "reax.Stage",
        loop: "reax.stages.PredictionEpoch",
        outputs: Any,
        /,
    ) -> None:
        """The predict stage has just finished processing a batch."""

    def on_predict_epoch_end(
        self, trainer: "reax.Trainer", stage: "reax.Stage", loop: "reax.stages.PredictionEpoch", /
    ) -> None:
        """A predict epoch is about to end."""

    def on_predict_end(self, trainer: "reax.Trainer", stage: "reax.stages.Predict", /) -> None:
        """Predict is ending."""

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

    def teardown(self, trainer: "reax.Trainer", stage: "reax.Stage", /) -> None:
        """Called when a stage (e.g. fit, validate, test, predict, or tune ends)."""

    def on_stage_start(self, trainer: "reax.Trainer", stage: "reax.Stage", /) -> None:
        """A trainer stage is starting."""

    def on_stage_end(self, trainer: "reax.Trainer", stage: "reax.Stage", /) -> None:
        """The stage is ending."""

    def on_loop_start(
        self, trainer: "reax.Trainer", stage: "reax.Stage", loop: "reax.stages.Loop", /
    ) -> None:
        """A stage is about to start a loop."""

    def on_loop_iter_start(
        self, trainer: "reax.Trainer", stage: "reax.Stage", loop: "reax.stages.Loop", /
    ) -> None:
        """A stage is about to start an iteration."""

    def on_loop_iter_end(
        self,
        trainer: "reax.Trainer",
        stage: "reax.Stage",
        loop: "reax.stages.Loop",
        outputs: Any,
        /,
    ) -> None:
        """The stage just finished processing an iteration."""

    def on_loop_end(
        self,
        trainer: "reax.Trainer",
        stage: "reax.Stage",
        loop: "reax.stages.Loop",
        outputs: Any,
        /,
    ) -> None:
        """A stage is about to end a loop."""

    # endregion
