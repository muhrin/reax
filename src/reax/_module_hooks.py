from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import reax


class ModuleHooks:
    def on_fit_start(self, stage: "reax.stages.Fit", /) -> None:
        """A fitting stage is about to begin."""

    def on_fit_end(self, stage: "reax.stages.Fit", /) -> None:
        """A fitting stage is about to end."""

    # region Train

    def on_train_start(self, stage: "reax.stages.Train", /) -> None:
        """Training is about to start."""

    def on_train_epoch_start(self, stage: "reax.stages.Train", /) -> None:
        """A training epoch is about to begin."""

    def on_train_batch_start(
        self, stage: "reax.stages.Train", batch: Any, batch_idx: int, /
    ) -> None:
        """The training stage if about to process a batch."""

    def on_train_batch_end(
        self, stage: "reax.stages.Train", outputs: Any, batch: Any, batch_idx: int, /
    ) -> None:
        """The training stage has just finished processing a batch."""

    def on_train_epoch_end(self, stage: "reax.stages.Train", /) -> None:
        """A training epoch is about to end."""

    def on_train_end(self, stage: "reax.stages.Train", /) -> None:
        """Training is about to end."""

    # endregion

    # region Validation

    def on_validation_start(self, stage: "reax.stages.Validate", /) -> None:
        """Validation is starting."""

    def on_validation_epoch_start(self, stage: "reax.stages.Validate", /) -> None:
        """A validation epoch is about to begin."""

    def on_validation_batch_start(
        self, stage: "reax.stages.Validate", batch: Any, batch_idx: int
    ) -> None:
        """The validation stage if about to process a batch."""

    def on_validation_batch_end(
        self, stage: "reax.stages.Train", outputs: Any, batch: Any, batch_idx: int, /
    ) -> None:
        """The validation stage has just finished processing a batch."""

    def on_validation_epoch_end(self, stage: "reax.stages.Validate", /) -> None:
        """A validation epoch is about to end."""

    def on_validation_end(self, stage: "reax.stages.Validate", /) -> None:
        """Validation has ended."""

    # endregion

    # region Test

    def on_test_epoch_start(self, stage: "reax.stages.Test", /) -> None:
        """A test epoch is about to begin."""

    def on_test_batch_start(self, stage: "reax.stages.Test", batch: Any, batch_idx: int, /) -> None:
        """The test stage if about to process a batch."""

    def on_test_batch_end(
        self, stage: "reax.stages.Test", outputs: Any, batch: Any, batch_idx: int, /
    ) -> None:
        """The test stage has just finished processing a batch."""

    def on_test_epoch_end(self, stage: "reax.stages.Test", /) -> None:
        """A test epoch is about to end."""

    # endregion

    # region Predict

    def on_predict_epoch_start(self, stage: "reax.stages.Predict", /) -> None:
        """A predict epoch is about to begin."""

    def on_predict_batch_start(
        self, stage: "reax.stages.Predict", batch: Any, batch_idx: int, /
    ) -> None:
        """The test stage if about to process a batch."""

    def on_predict_batch_end(
        self, stage: "reax.stages.Train", outputs: Any, batch: Any, batch_idx: int, /
    ) -> None:
        """The predict stage has just finished processing a batch."""

    def on_predict_epoch_end(self, stage: "reax.stages.Train", /) -> None:
        """A predict epoch is about to end."""

    # endregion

    # region Generic stage messages

    def on_epoch_starting(self, stage: "reax.stages.EpochStage", /) -> None:
        """An epoch is just about to begin."""

    def on_batch_ending(
        self, stage: "reax.stages.EpochStage", batch_idx: int, metrics: dict, /
    ) -> None:
        """A batch has just been processed."""

    def on_epoch_ending(self, stage: "reax.stages.EpochStage", metrics: dict, /) -> None:
        """An epoch is ending."""

    def on_stage_starting(self, stage: "reax.stages.Stage", /) -> None:
        """A trainer stage is about to begin."""

    def on_stage_started(self, stage: "reax.stages.Stage", /) -> None:
        """A trainer has started."""

    def on_stage_iter_starting(self, stage: "reax.Stage", step: int, /) -> None:
        """A stage is about to start an interation."""

    def on_stage_iter_ending(self, stage: "reax.Stage", step: int, outputs: Any, /):
        """The stage just finished processing an iteration."""

    def on_stage_ending(self, stage: "reax.Stage", /) -> None:
        """The stage is about to finish."""

    def on_stage_ended(self, stage: "reax.Stage", /) -> None:
        """The stage has ended."""

    # endregion
