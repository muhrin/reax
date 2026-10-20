from collections.abc import Callable
import functools

from .. import hooks, stages


@functools.singledispatch
def hook_map(_sender) -> dict[Callable, Callable]:
    """Hook map.

    Maps internal lifecycle events fired by Stages and Loops to external
    TrainerListener hooks.

    The rule is absolute:

    - A Stage fires `on_stage_*` events, once per `Stage.run()`.
    - A Loop fires `on_loop_*` events, once per `Loop.run()`.
    - A leaf Loop additionally fires `on_iter_*` events, once per iteration.

    Every external hook maps to exactly one internal event, from exactly one
    object, at exactly one level. There are no exceptions and no
    cross-level coordination. A hook named after a stage fires when that
    stage runs; a hook named after a loop fires when that loop runs.

    External hooks attach to the `_started` and `_finished` variants of each
    pair. The `_starting` / `_finishing` variants are available for internal
    listeners (profilers, tracers) but are not translated.
    """
    return dict()


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


@hook_map.register
def _(_stage: stages.Fit) -> dict[Callable, Callable]:
    """Fit stage (top-level training operation)."""
    return {
        stages.StageListener.on_stage_started: hooks.TrainerListener.on_fit_start,
        stages.StageListener.on_stage_finished: hooks.TrainerListener.on_fit_end,
    }


@hook_map.register
def _(_stage: stages.Validate) -> dict[Callable, Callable]:
    """Validate stage (top-level validation operation)."""
    return {
        stages.StageListener.on_stage_started: hooks.TrainerListener.on_validation_start,
        stages.StageListener.on_stage_finished: hooks.TrainerListener.on_validation_end,
    }


@hook_map.register
def _(_stage: stages.Test) -> dict[Callable, Callable]:
    """Test stage (top-level test operation)."""
    return {
        stages.StageListener.on_stage_started: hooks.TrainerListener.on_test_start,
        stages.StageListener.on_stage_finished: hooks.TrainerListener.on_test_end,
    }


@hook_map.register
def _(_stage: stages.Predict) -> dict[Callable, Callable]:
    """Predict stage (top-level predict operation)."""
    return {
        stages.StageListener.on_stage_started: hooks.TrainerListener.on_predict_start,
        stages.StageListener.on_stage_finished: hooks.TrainerListener.on_predict_end,
    }


# ---------------------------------------------------------------------------
# Loops
# ---------------------------------------------------------------------------


@hook_map.register
def _(_loop: stages.FitLoop) -> dict[Callable, Callable]:
    """Loop over training epochs. Hierarchical (iterations are TrainEpoch loops),
    so no iter events."""
    return {
        stages.StageListener.on_loop_started: hooks.TrainerListener.on_train_start,
        stages.StageListener.on_loop_finished: hooks.TrainerListener.on_train_end,
    }


@hook_map.register
def _(_loop: stages.TrainEpoch) -> dict[Callable, Callable]:
    """Loop over training batches. Leaf loop."""
    return {
        stages.StageListener.on_loop_started: hooks.TrainerListener.on_train_epoch_start,
        stages.StageListener.on_loop_finished: hooks.TrainerListener.on_train_epoch_end,
        stages.StageListener.on_loop_iter_started: hooks.TrainerListener.on_train_batch_start,
        stages.StageListener.on_loop_iter_finished: hooks.TrainerListener.on_train_batch_end,
    }


@hook_map.register
def _(_loop: stages.ValidationEpoch) -> dict[Callable, Callable]:
    """Loop over validation batches. Leaf loop."""
    return {
        stages.StageListener.on_loop_started: hooks.TrainerListener.on_validation_epoch_start,
        stages.StageListener.on_loop_finished: hooks.TrainerListener.on_validation_epoch_end,
        stages.StageListener.on_loop_iter_started: hooks.TrainerListener.on_validation_batch_start,
        stages.StageListener.on_loop_iter_finished: hooks.TrainerListener.on_validation_batch_end,
    }


@hook_map.register
def _(_loop: stages.TestEpoch) -> dict[Callable, Callable]:
    """Loop over test batches. Leaf loop."""
    return {
        stages.StageListener.on_loop_started: hooks.TrainerListener.on_test_epoch_start,
        stages.StageListener.on_loop_finished: hooks.TrainerListener.on_test_epoch_end,
        stages.StageListener.on_loop_iter_started: hooks.TrainerListener.on_test_batch_start,
        stages.StageListener.on_loop_iter_finished: hooks.TrainerListener.on_test_batch_end,
    }


@hook_map.register
def _(_loop: stages.PredictionEpoch) -> dict[Callable, Callable]:
    """Loop over predict batches. Leaf loop."""
    return {
        stages.StageListener.on_loop_started: hooks.TrainerListener.on_predict_epoch_start,
        stages.StageListener.on_loop_finished: hooks.TrainerListener.on_predict_epoch_end,
        stages.StageListener.on_loop_iter_started: hooks.TrainerListener.on_predict_batch_start,
        stages.StageListener.on_loop_iter_finished: hooks.TrainerListener.on_predict_batch_end,
    }
