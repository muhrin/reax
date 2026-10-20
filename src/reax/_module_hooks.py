from typing import TYPE_CHECKING, Any

import optax

if TYPE_CHECKING:
    import reax


class ModuleHooks:
    """Hooks that a :class:`~reax.Module` can override.

    These mirror the :class:`~reax.TrainerListener` hooks, without the ``trainer`` argument (the
    module can get it from :attr:`~reax.Module.trainer`). For each event, the module hook is called
    before the corresponding listener hook.

    The batch hooks don't receive the batch index, it is available as
    :attr:`loop.batch_idx <reax.stages.LoopOverBatches.batch_idx>`.
    """

    # pylint: disable=too-many-public-methods

    # region Stages

    def on_fit_start(self, stage: "reax.stages.Fit", /) -> None:
        """A fit stage has started.

        Args:
            stage: The fit stage.
        """

    def on_fit_end(self, stage: "reax.stages.Fit", /) -> None:
        """A fit stage is ending.

        Args:
            stage: The fit stage.
        """

    def on_validation_start(self, stage: "reax.stages.Validate", /) -> None:
        """A validation stage has started.

        Only called for a standalone validation stage (i.e. :meth:`reax.Trainer.validate`), not
        for validation during fitting.

        Args:
            stage: The validation stage.
        """

    def on_validation_end(self, stage: "reax.stages.Validate", /) -> None:
        """A validation stage is ending.

        Only called for a standalone validation stage (i.e. :meth:`reax.Trainer.validate`), not
        for validation during fitting.

        Args:
            stage: The validation stage.
        """

    def on_test_start(self, stage: "reax.stages.Test", /) -> None:
        """A test stage has started.

        Args:
            stage: The test stage.
        """

    def on_test_end(self, stage: "reax.stages.Test", /) -> None:
        """A test stage is ending.

        Args:
            stage: The test stage.
        """

    def on_predict_start(self, stage: "reax.stages.Predict", /) -> None:
        """A predict stage has started.

        Args:
            stage: The predict stage.
        """

    def on_predict_end(self, stage: "reax.stages.Predict", /) -> None:
        """A predict stage is ending.

        Args:
            stage: The predict stage.
        """

    # endregion

    # region Train

    def on_train_start(self, stage: "reax.Stage", loop: "reax.stages.FitLoop", /) -> None:
        """Training has started (after the validation sanity check, before the first epoch).

        Args:
            stage: The stage running the loop (the fit stage).
            loop: The loop over training epochs.
        """

    def on_train_epoch_start(self, stage: "reax.Stage", loop: "reax.stages.TrainEpoch", /) -> None:
        """A training epoch has started (before the first batch).

        Args:
            stage: The stage running the loop (the fit stage).
            loop: The loop over training batches.
        """

    def on_train_batch_start(
        self, stage: "reax.Stage", loop: "reax.stages.TrainEpoch", batch: Any, /
    ) -> None:
        """A training batch is about to be processed.

        Args:
            stage: The stage running the loop (the fit stage).
            loop: The loop over training batches.
            batch: The batch.
        """

    def on_train_batch_end(
        self, stage: "reax.Stage", loop: "reax.stages.TrainEpoch", outputs: Any, /
    ) -> None:
        """A training batch has been processed (and any optimizer update applied).

        Args:
            stage: The stage running the loop (the fit stage).
            loop: The loop over training batches.
            outputs: The value returned by :meth:`~reax.Module.training_step`.
        """

    def on_train_epoch_end(self, stage: "reax.Stage", loop: "reax.stages.TrainEpoch", /) -> None:
        """A training epoch is ending (after the last batch).

        This is called before the epoch-level metrics are gathered, so metrics logged here are
        included in them.

        Args:
            stage: The stage running the loop (the fit stage).
            loop: The loop over training batches.
        """

    def on_train_end(self, stage: "reax.Stage", loop: "reax.stages.FitLoop", /) -> None:
        """Training is ending (after the last epoch).

        Args:
            stage: The stage running the loop (the fit stage).
            loop: The loop over training epochs.
        """

    # endregion

    # region Validation

    def on_validation_epoch_start(
        self, stage: "reax.Stage", loop: "reax.stages.ValidationEpoch", /
    ) -> None:
        """A validation epoch has started (before the first batch).

        Args:
            stage: The stage running the loop (the validation stage, or the fit stage when
                validating during fitting).
            loop: The loop over validation batches.
        """

    def on_validation_batch_start(
        self, stage: "reax.Stage", loop: "reax.stages.ValidationEpoch", batch: Any, /
    ) -> None:
        """A validation batch is about to be processed.

        Args:
            stage: The stage running the loop (the validation stage, or the fit stage when
                validating during fitting).
            loop: The loop over validation batches.
            batch: The batch.
        """

    def on_validation_batch_end(
        self, stage: "reax.Stage", loop: "reax.stages.ValidationEpoch", outputs: Any, /
    ) -> None:
        """A validation batch has been processed.

        Args:
            stage: The stage running the loop (the validation stage, or the fit stage when
                validating during fitting).
            loop: The loop over validation batches.
            outputs: The value returned by :meth:`~reax.Module.validation_step`.
        """

    def on_validation_epoch_end(
        self, stage: "reax.Stage", loop: "reax.stages.ValidationEpoch", /
    ) -> None:
        """A validation epoch is ending (after the last batch).

        This is called before the epoch-level metrics are gathered, so metrics logged here are
        included in them.

        Args:
            stage: The stage running the loop (the validation stage, or the fit stage when
                validating during fitting).
            loop: The loop over validation batches.
        """

    # endregion

    # region Test

    def on_test_epoch_start(self, stage: "reax.Stage", loop: "reax.stages.TestEpoch", /) -> None:
        """A test epoch has started (before the first batch).

        Args:
            stage: The stage running the loop (the test stage).
            loop: The loop over test batches.
        """

    def on_test_batch_start(
        self, stage: "reax.Stage", loop: "reax.stages.TestEpoch", batch: Any, /
    ) -> None:
        """A test batch is about to be processed.

        Args:
            stage: The stage running the loop (the test stage).
            loop: The loop over test batches.
            batch: The batch.
        """

    def on_test_batch_end(
        self, stage: "reax.Stage", loop: "reax.stages.TestEpoch", outputs: Any, /
    ) -> None:
        """A test batch has been processed.

        Args:
            stage: The stage running the loop (the test stage).
            loop: The loop over test batches.
            outputs: The value returned by :meth:`~reax.Module.test_step`.
        """

    def on_test_epoch_end(self, stage: "reax.Stage", loop: "reax.stages.TestEpoch", /) -> None:
        """A test epoch is ending (after the last batch).

        This is called before the epoch-level metrics are gathered, so metrics logged here are
        included in them.

        Args:
            stage: The stage running the loop (the test stage).
            loop: The loop over test batches.
        """

    # endregion

    # region Predict

    def on_predict_epoch_start(
        self, stage: "reax.Stage", loop: "reax.stages.PredictionEpoch", /
    ) -> None:
        """A predict epoch has started (before the first batch).

        Args:
            stage: The stage running the loop (the predict stage).
            loop: The loop over predict batches.
        """

    def on_predict_batch_start(
        self, stage: "reax.Stage", loop: "reax.stages.PredictionEpoch", batch: Any, /
    ) -> None:
        """A predict batch is about to be processed.

        Args:
            stage: The stage running the loop (the predict stage).
            loop: The loop over predict batches.
            batch: The batch.
        """

    def on_predict_batch_end(
        self, stage: "reax.Stage", loop: "reax.stages.PredictionEpoch", outputs: Any, /
    ) -> None:
        """A predict batch has been processed.

        Args:
            stage: The stage running the loop (the predict stage).
            loop: The loop over predict batches.
            outputs: The value returned by :meth:`~reax.Module.predict_step`.
        """

    def on_predict_epoch_end(
        self, stage: "reax.Stage", loop: "reax.stages.PredictionEpoch", /
    ) -> None:
        """A predict epoch is ending (after the last batch).

        This is called before the epoch-level metrics are gathered, so metrics logged here are
        included in them.

        Args:
            stage: The stage running the loop (the predict stage).
            loop: The loop over predict batches.
        """

    # endregion

    # region Optimization

    def on_before_optimizer_step(self, optimizer: "reax.Optimizer", grad: optax.Params, /) -> None:
        """The optimizer is about to be updated with the gradients of a training batch.

        Only called when using automatic optimization (see
        :attr:`~reax.Module.automatic_optimization`). This is called for every training batch,
        including when gradients are being accumulated (see
        :paramref:`~reax.Trainer.fit.accumulate_grad_batches`), as accumulation is performed by
        the optimizer. Any transformations that are part of the optimizer (e.g. gradient
        clipping) will not have been applied yet.

        Args:
            optimizer: The optimizer that is about to be updated.
            grad: The gradients of the batch, as returned by :meth:`~reax.Module.training_step`.
        """

    # endregion
