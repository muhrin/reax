from typing import TYPE_CHECKING, Any

from typing_extensions import override

from .. import hooks, stages

if TYPE_CHECKING:
    import reax


class TrainerLogging(hooks.TrainerListener):
    def __init__(self):
        self._progress_bar_metrics = {}
        self._callback_metrics = {}
        self._logger_metrics = {}

    @property
    def progress_bar_metrics(self) -> dict:
        return self._progress_bar_metrics

    @property
    def callback_metrics(self) -> dict:
        return self._callback_metrics

    @property
    def logger_metrics(self) -> dict:
        return self._logger_metrics

    @override
    def on_stage_iter_ending(
        self, trainer: "reax.Trainer", stage: "reax.Stage", step: int, outputs: Any
    ):
        """The stage just finished processing an iteration"""
        if isinstance(stage, stages.EpochStage):
            self._progress_bar_metrics.update(stage.progress_bar_metrics)
            self._callback_metrics.update(stage.callback_metrics)
            self._logger_metrics.update(stage.logged_metrics)

    @override
    def on_stage_ending(self, trainer: "reax.Trainer", stage: "reax.Stage") -> None:
        """The stage is about to finish"""
        if isinstance(stage, stages.EpochStage):
            self._progress_bar_metrics.update(stage.progress_bar_metrics)
            self._callback_metrics.update(stage.callback_metrics)
            self._logger_metrics.update(stage.logged_metrics)

    def reset_metrics(self):
        self._progress_bar_metrics = {}
        self._callback_metrics = {}
        self._logger_metrics = {}
