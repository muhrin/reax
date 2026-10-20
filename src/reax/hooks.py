from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import reax


__all__ = ("TrainerListener",)


class TrainerListener:
    """Hooks that a listener can override to be notified of trainer events.

    The stage and loop specific hooks (e.g. :meth:`on_train_epoch_start`) are fired after the
    corresponding generic hook (e.g. :meth:`on_loop_start`), and after the equivalent
    :class:`~reax.Module` hook (if any).
    """

    # pylint: disable=too-many-public-methods

    def on_fit_start(self, trainer: "reax.Trainer", stage: "reax.stages.Fit", /) -> None:
        """A fitting stage has started."""

    def on_fit_end(self, trainer: "reax.Trainer", stage: "reax.stages.Fit", /) -> None:
        """A fitting stage has ended."""

    def on_train_start(
        self, trainer: "reax.Trainer", stage: "reax.Stage", loop: "reax.stages.FitLoop", /
    ) -> None:
        """Training (the loop over epochs) has started.

        This is called after the validation sanity check (if any), and before the first epoch.
        """

    def on_train_epoch_start(
        self, trainer: "reax.Trainer", stage: "reax.Stage", loop: "reax.stages.TrainEpoch", /
    ) -> None:
        """A training epoch has started (before the first batch)."""

    def on_train_batch_start(
        self,
        trainer: "reax.Trainer",
        stage: "reax.Stage",
        loop: "reax.stages.TrainEpoch",
        batch: Any,
        /,
    ) -> None:
        """The training loop is about to process a batch."""

    def on_train_batch_end(
        self,
        trainer: "reax.Trainer",
        stage: "reax.Stage",
        loop: "reax.stages.TrainEpoch",
        outputs: Any,
        /,
    ) -> None:
        """The training loop has just processed a batch (and applied any optimizer update).

        Args:
            outputs: The value returned by :meth:`~reax.Module.training_step`.
        """

    def on_train_epoch_end(
        self, trainer: "reax.Trainer", stage: "reax.Stage", loop: "reax.stages.TrainEpoch", /
    ) -> None:
        """A training epoch has ended (the epoch-level metrics are available)."""

    def on_train_end(
        self, trainer: "reax.Trainer", stage: "reax.Stage", loop: "reax.stages.FitLoop", /
    ) -> None:
        """Training (the loop over epochs) has ended."""

    def on_validation_start(
        self, trainer: "reax.Trainer", stage: "reax.stages.Validate", /
    ) -> None:
        """A validation stage has started.

        Only called for a standalone validation stage (i.e. :meth:`reax.Trainer.validate`), not
        for validation during fitting.
        """

    def on_validation_epoch_start(
        self, trainer: "reax.Trainer", stage: "reax.Stage", loop: "reax.stages.ValidationEpoch", /
    ) -> None:
        """A validation epoch has started (before the first batch).

        The stage is either a :class:`~reax.stages.Validate` stage, or the
        :class:`~reax.stages.Fit` stage when validating during fitting.
        """

    def on_validation_batch_start(
        self,
        trainer: "reax.Trainer",
        stage: "reax.Stage",
        loop: "reax.stages.ValidationEpoch",
        batch: Any,
        /,
    ) -> None:
        """The validation loop is about to process a batch."""

    def on_validation_batch_end(
        self,
        trainer: "reax.Trainer",
        stage: "reax.Stage",
        loop: "reax.stages.ValidationEpoch",
        outputs: Any,
        /,
    ) -> None:
        """The validation loop has just processed a batch.

        Args:
            outputs: The value returned by :meth:`~reax.Module.validation_step`.
        """

    def on_validation_epoch_end(
        self, trainer: "reax.Trainer", stage: "reax.Stage", loop: "reax.stages.ValidationEpoch", /
    ) -> None:
        """A validation epoch has ended (the epoch-level metrics are available)."""

    def on_validation_end(self, trainer: "reax.Trainer", stage: "reax.stages.Validate", /) -> None:
        """A validation stage has ended.

        Only called for a standalone validation stage (i.e. :meth:`reax.Trainer.validate`), not
        for validation during fitting.
        """

    def on_test_start(self, trainer: "reax.Trainer", stage: "reax.stages.Test", /) -> None:
        """A test stage has started."""

    def on_test_epoch_start(
        self, trainer: "reax.Trainer", stage: "reax.Stage", loop: "reax.stages.TestEpoch", /
    ) -> None:
        """A test epoch has started (before the first batch)."""

    def on_test_batch_start(
        self,
        trainer: "reax.Trainer",
        stage: "reax.Stage",
        loop: "reax.stages.TestEpoch",
        batch: Any,
        /,
    ) -> None:
        """The test loop is about to process a batch."""

    def on_test_batch_end(
        self,
        trainer: "reax.Trainer",
        stage: "reax.Stage",
        loop: "reax.stages.TestEpoch",
        outputs: Any,
        /,
    ) -> None:
        """The test loop has just processed a batch.

        Args:
            outputs: The value returned by :meth:`~reax.Module.test_step`.
        """

    def on_test_epoch_end(
        self, trainer: "reax.Trainer", stage: "reax.Stage", loop: "reax.stages.TestEpoch", /
    ) -> None:
        """A test epoch has ended (the epoch-level metrics are available)."""

    def on_test_end(self, trainer: "reax.Trainer", stage: "reax.stages.Test", /) -> None:
        """A test stage has ended."""

    def on_predict_start(self, trainer: "reax.Trainer", stage: "reax.stages.Predict", /) -> None:
        """A predict stage has started."""

    def on_predict_epoch_start(
        self, trainer: "reax.Trainer", stage: "reax.Stage", loop: "reax.stages.PredictionEpoch", /
    ) -> None:
        """A predict epoch has started (before the first batch)."""

    def on_predict_batch_start(
        self,
        trainer: "reax.Trainer",
        stage: "reax.Stage",
        loop: "reax.stages.PredictionEpoch",
        batch: Any,
        /,
    ) -> None:
        """The predict loop is about to process a batch."""

    def on_predict_batch_end(
        self,
        trainer: "reax.Trainer",
        stage: "reax.Stage",
        loop: "reax.stages.PredictionEpoch",
        outputs: Any,
        /,
    ) -> None:
        """The predict loop has just processed a batch.

        Args:
            outputs: The value returned by :meth:`~reax.Module.predict_step`.
        """

    def on_predict_epoch_end(
        self, trainer: "reax.Trainer", stage: "reax.Stage", loop: "reax.stages.PredictionEpoch", /
    ) -> None:
        """A predict epoch has ended."""

    def on_predict_end(self, trainer: "reax.Trainer", stage: "reax.stages.Predict", /) -> None:
        """A predict stage has ended."""

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
        """A stage (e.g. fit, validate, test, predict) is starting.

        Called before :meth:`on_stage_start`.
        """

    def teardown(self, trainer: "reax.Trainer", stage: "reax.Stage", /) -> None:
        """A stage (e.g. fit, validate, test, predict) has ended.

        Called last, after :meth:`on_stage_end` and the stage specific end hook (e.g.
        :meth:`on_fit_end`).
        """

    def on_stage_start(self, trainer: "reax.Trainer", stage: "reax.Stage", /) -> None:
        """A stage has started."""

    def on_stage_end(self, trainer: "reax.Trainer", stage: "reax.Stage", /) -> None:
        """A stage has ended."""

    def on_loop_start(
        self, trainer: "reax.Trainer", stage: "reax.Stage", loop: "reax.stages.Loop", /
    ) -> None:
        """A loop has started (before its first iteration)."""

    def on_loop_iter_start(
        self, trainer: "reax.Trainer", stage: "reax.Stage", loop: "reax.stages.Loop", /
    ) -> None:
        """A loop is about to run an iteration.

        This is called for every loop, including those whose iterations run other loops (e.g. the
        :class:`~reax.stages.FitLoop`, once per epoch).
        """

    def on_loop_iter_end(
        self,
        trainer: "reax.Trainer",
        stage: "reax.Stage",
        loop: "reax.stages.Loop",
        outputs: Any,
        /,
    ) -> None:
        """A loop has just finished an iteration.

        Args:
            outputs: The result of the iteration (e.g. the value returned by the step method for
                a loop over batches).
        """

    def on_loop_end(
        self,
        trainer: "reax.Trainer",
        stage: "reax.Stage",
        loop: "reax.stages.Loop",
        outputs: Any,
        /,
    ) -> None:
        """A loop has ended.

        Args:
            outputs: Currently always ``None``.
        """

    # endregion
