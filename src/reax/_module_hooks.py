from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import reax


class ModuleHooks:
    # pylint: disable=too-many-public-methods

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
        self, stage: "reax.stages.Validate", outputs: Any, batch: Any, batch_idx: int, /
    ) -> None:
        """The validation stage has just finished processing a batch."""

    def on_validation_epoch_end(self, stage: "reax.stages.Validate", /) -> None:
        """A validation epoch is about to end."""

    def on_validation_end(self, stage: "reax.stages.Validate", /) -> None:
        """Validation has ended."""

    # endregion

    # region Test

    def on_test_start(self, stage: "reax.stages.Test", /) -> None:
        """The test stage is starting"""

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

    def on_test_end(self, stage: "reax.stages.Test", /) -> None:
        """The test stage has finished"""

    # endregion

    # region Predict

    def on_predict_start(self, stage: "reax.stages.Predict", /) -> None:
        """Prediction is starting."""

    def on_predict_epoch_start(self, stage: "reax.stages.Predict", /) -> None:
        """A predict epoch is about to begin."""

    def on_predict_batch_start(
        self, stage: "reax.stages.Predict", batch: Any, batch_idx: int, /
    ) -> None:
        """The test stage if about to process a batch."""

    def on_predict_batch_end(
        self, stage: "reax.stages.Predict", outputs: Any, batch: Any, batch_idx: int, /
    ) -> None:
        """The predict stage has just finished processing a batch."""

    def on_predict_epoch_end(self, stage: "reax.stages.Train", /) -> None:
        """A predict epoch is about to end."""

    def on_predict_end(self, stage: "reax.stages.Predict", /) -> None:
        """Prediction is starting."""

    # endregion

    def on_before_optimizer_step(
        self, optimizer: "reax.Optimizer", grad: dict[str, Any], /
    ) -> None:
        """Called before ``optimizer.step()``.

        If using gradient accumulation, the hook is called once the gradients have been accumulated.
        See: :paramref:`~reax.Trainer.accumulate_grad_batches`.

        If clipping gradients, the gradients will not have been clipped yet.

        :param optimizer: Current optimizer being used.
        :param grad: The gradients dictionary from JAX
        """

    # region Generic stage messages

    def on_stage_start(self, stage: "reax.stages.Stage", /) -> None:
        """A stage is about to start."""

    def on_stage_iter_starting(self, stage: "reax.Stage", step: int, /) -> None:
        """A stage is about to start an interation."""

    def on_stage_iter_ending(self, stage: "reax.Stage", step: int, outputs: Any, /):
        """The stage just finished processing an iteration."""

    def on_stage_end(self, stage: "reax.Stage", /) -> None:
        """The stage is ending."""

    # endregion
