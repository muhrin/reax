import sys
from typing import TYPE_CHECKING, Optional

import tqdm

from reax import hooks

if TYPE_CHECKING:
    import reax

__all__ = ("TqdmProgressBar",)


class TqdmProgressBar(hooks.TrainerListener):
    BAR_FORMAT = (
        "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}{postfix}]"
    )

    def __init__(self):
        self._bar: Optional[tqdm.tqdm] = None

    def on_epoch_starting(  # pylint: disable=unused-argument
        self, trainer: "reax.Trainer", stage: "reax.stages.EpochStage"
    ) -> None:
        self._bar = tqdm.tqdm(
            total=stage.max_batches,
            desc=f"{stage.name} (epoch {stage.run_count})",
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
            bar_format=self.BAR_FORMAT,
        )

    def on_batch_ending(
        # pylint: disable=unused-argument
        self,
        trainer: "reax.Trainer",
        stage: "reax.stages.EpochStage",
        batch_idx: int,
        metrics: "reax.stages.MetricResults",
    ) -> None:
        self._bar.n = batch_idx + 1
        postfix = {}
        if metrics:
            for _name, result in metrics.items():
                if result.meta.prog_bar:
                    postfix[result.meta.name] = result.value
        if postfix:
            self._bar.set_postfix(postfix)

        self._bar.refresh()

    def on_epoch_ending(  # pylint: disable=unused-argument
        self, trainer: "reax.Trainer", stage: "reax.stages.EpochStage", metrics: dict
    ) -> None:
        pass

    def on_stage_ending(  # pylint: disable=unused-argument
        self, trainer: "reax.Trainer", stage: "reax.stages.Stage"
    ) -> None:
        self._bar.close()
