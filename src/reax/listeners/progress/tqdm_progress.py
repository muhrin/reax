import math
import os
import sys
from typing import TYPE_CHECKING, Any, Optional, Union

from lightning_utilities.core import rank_zero
import tqdm
from typing_extensions import override

from . import progress_bar

if TYPE_CHECKING:
    import reax

__all__ = ("TqdmProgressBar",)


class TqdmProgressBar(progress_bar.ProgressBar):
    BAR_FORMAT = (
        "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}{postfix}]"
    )

    def __init__(self, refresh_rate: int = 1, process_position: int = 0):
        super().__init__()
        self._refresh_rate = self._resolve_refresh_rate(refresh_rate)
        self._process_position = process_position
        self._enabled = True
        self._train_progress_bar: Optional[tqdm.tqdm] = None
        self._val_progress_bar: Optional[tqdm.tqdm] = None
        self._test_progress_bar: Optional[tqdm.tqdm] = None
        self._predict_progress_bar: Optional[tqdm.tqdm] = None

    @property
    def train_progress_bar(self) -> tqdm.tqdm:
        if self._train_progress_bar is None:
            raise TypeError(
                f"The `{self.__class__.__name__}._train_progress_bar` reference has not been set yet."
            )
        return self._train_progress_bar

    @train_progress_bar.setter
    def train_progress_bar(self, bar: tqdm.tqdm) -> None:
        self._train_progress_bar = bar

    @property
    def val_progress_bar(self) -> tqdm.tqdm:
        if self._val_progress_bar is None:
            raise TypeError(
                f"The `{self.__class__.__name__}._val_progress_bar` reference has not been set yet."
            )
        return self._val_progress_bar

    @val_progress_bar.setter
    def val_progress_bar(self, bar: tqdm.tqdm) -> None:
        self._val_progress_bar = bar

    @property
    def test_progress_bar(self) -> tqdm.tqdm:
        if self._test_progress_bar is None:
            raise TypeError(
                f"The `{self.__class__.__name__}._test_progress_bar` reference has not been set yet."
            )
        return self._test_progress_bar

    @test_progress_bar.setter
    def test_progress_bar(self, bar: tqdm.tqdm) -> None:
        self._test_progress_bar = bar

    @property
    def predict_progress_bar(self) -> tqdm.tqdm:
        if self._predict_progress_bar is None:
            raise TypeError(
                f"The `{self.__class__.__name__}._predict_progress_bar` reference has not been set yet."
            )
        return self._predict_progress_bar

    @predict_progress_bar.setter
    def predict_progress_bar(self, bar: tqdm.tqdm) -> None:
        self._predict_progress_bar = bar

    @property
    def refresh_rate(self) -> int:
        return self._refresh_rate

    @property
    def process_position(self) -> int:
        return self._process_position

    @property
    def is_enabled(self) -> bool:
        return self._enabled and self.refresh_rate > 0

    @property
    def is_disabled(self) -> bool:
        return not self.is_enabled

    @override
    def disable(self) -> None:
        self._enabled = False

    @override
    def enable(self) -> None:
        self._enabled = True

    def init_train_tqdm(self, stage: "reax.stages.Train") -> tqdm.tqdm:
        """Override this to customize the tqdm bar for training."""
        return tqdm.tqdm(
            desc=stage.name,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
            bar_format=self.BAR_FORMAT,
        )

    def init_predict_tqdm(self, stage: "reax.stages.Predict") -> tqdm.tqdm:
        """Override this to customize the tqdm bar for predicting."""
        return tqdm.tqdm(
            desc=stage.name,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
            bar_format=self.BAR_FORMAT,
        )

    def init_validation_tqdm(self, stage: "reax.stages.Validate") -> tqdm.tqdm:
        """Override this to customize the tqdm bar for validation."""
        # Check if there is a 'parent' in which case it will have a progress bar
        has_main_bar = stage.parent is not None
        return tqdm.tqdm(
            desc=stage.name,
            position=(2 * self.process_position + has_main_bar),
            disable=self.is_disabled,
            leave=not has_main_bar,
            dynamic_ncols=True,
            file=sys.stdout,
            bar_format=self.BAR_FORMAT,
        )

    def init_test_tqdm(self) -> tqdm.tqdm:
        """Override this to customize the tqdm bar for testing."""
        return tqdm.tqdm(
            desc="Testing",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            bar_format=self.BAR_FORMAT,
        )

    @override
    def on_fit_start(self, _trainer: "reax.Trainer", stage: "reax.stages.Fit") -> None:
        self.train_progress_bar = self.init_train_tqdm(stage)

    @override
    def on_train_epoch_start(self, trainer: "reax.Trainer", stage: "reax.stages.Train") -> None:
        self.train_progress_bar.reset(total=convert_inf(stage.max_iters))
        self.train_progress_bar.initial = 0
        self.train_progress_bar.set_description(f"Epoch {trainer.current_epoch}")

    @override
    def on_train_batch_end(
        self,
        trainer: "reax.Trainer",
        stage: "reax.stages.Train",
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        n = batch_idx + 1
        if self._should_update(n, self.train_progress_bar.total):
            _update_n(self.train_progress_bar, n)
            self.train_progress_bar.set_postfix(trainer.progress_bar_metrics)

    @override
    def on_train_epoch_end(self, trainer: "reax.Trainer", _stage: "reax.stages.Train") -> None:
        if not self.train_progress_bar.disable:
            self.train_progress_bar.set_postfix(trainer.progress_bar_metrics)

    @override
    def on_fit_end(self, *_: Any) -> None:
        self.train_progress_bar.close()

    @override
    def on_validation_epoch_start(
        self,
        _trainer: "reax.Trainer",
        stage: "reax.stages.Validate",
    ) -> None:
        self.val_progress_bar = self.init_validation_tqdm(stage)

    @override
    def on_validation_batch_start(
        self,
        trainer: "reax.Trainer",
        stage: "reax.stages.Validate",
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.val_progress_bar.reset(total=convert_inf(stage.max_iters))
        self.val_progress_bar.initial = 0
        self.val_progress_bar.set_description(stage.name)

    @override
    def on_validation_batch_end(
        self,
        trainer: "reax.Trainer",
        stage: "reax.Stage",
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        n = batch_idx + 1
        if self._should_update(n, self.val_progress_bar.total):
            _update_n(self.val_progress_bar, n)

    @override
    def on_validation_epoch_end(
        self, trainer: "reax.Trainer", stage: "reax.stages.Validate"
    ) -> None:
        self.val_progress_bar.close()
        if self._train_progress_bar is not None and trainer.stage != stage:
            self.train_progress_bar.set_postfix(trainer.progress_bar_metrics)

    @override
    def on_test_epoch_start(self, trainer: "reax.Trainer", stage: "reax.Stage") -> None:
        self.test_progress_bar = self.init_test_tqdm()

    @override
    def on_test_batch_start(
        self,
        trainer: "reax.Trainer",
        stage: "reax.stages.Test",
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.test_progress_bar.reset(total=convert_inf(stage.max_iters))
        self.test_progress_bar.initial = 0
        self.test_progress_bar.set_description(f"{stage.name}")

    @override
    def on_test_batch_end(
        self,
        trainer: "reax.Trainer",
        stage: "reax.Stage",
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        n = batch_idx + 1
        if self._should_update(n, self.test_progress_bar.total):
            _update_n(self.test_progress_bar, n)

    @override
    def on_test_epoch_end(self, _trainer: "reax.Trainer", _stage: "reax.Stage") -> None:
        self.test_progress_bar.close()

    @override
    def on_predict_epoch_start(
        self, _trainer: "reax.Trainer", stage: "reax.stages.Predict"
    ) -> None:
        self.predict_progress_bar = self.init_predict_tqdm(stage)
        self.predict_progress_bar.reset(total=convert_inf(stage.max_iters))
        self.predict_progress_bar.initial = 0
        self.predict_progress_bar.set_description(stage.name)

    @override
    def on_predict_batch_end(
        self,
        trainer: "reax.Trainer",
        stage: "reax.Stage",
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        n = batch_idx + 1
        if self._should_update(n, self.predict_progress_bar.total):
            _update_n(self.predict_progress_bar, n)

    @override
    def on_predict_epoch_end(self, trainer: "reax.Trainer", stage: "reax.Stage") -> None:
        self.predict_progress_bar.close()

    @override
    def print(self, *args: Any, sep: str = " ", **kwargs: Any) -> None:
        active_progress_bar = None

        if self._train_progress_bar is not None and not self.train_progress_bar.disable:
            active_progress_bar = self.train_progress_bar
        elif self._val_progress_bar is not None and not self.val_progress_bar.disable:
            active_progress_bar = self.val_progress_bar
        elif self._test_progress_bar is not None and not self.test_progress_bar.disable:
            active_progress_bar = self.test_progress_bar
        elif self._predict_progress_bar is not None and not self.predict_progress_bar.disable:
            active_progress_bar = self.predict_progress_bar

        if active_progress_bar is not None:
            s = sep.join(map(str, args))
            active_progress_bar.write(s, **kwargs)

    def _should_update(self, current: int, total: int) -> bool:
        return self.is_enabled and (current % self.refresh_rate == 0 or current == total)

    @staticmethod
    def _resolve_refresh_rate(refresh_rate: int) -> int:
        if os.getenv("COLAB_GPU") and refresh_rate == 1:
            # smaller refresh rate on colab causes crashes, choose a higher value
            rank_zero.rank_zero_debug("Using a higher refresh rate on Colab. Setting it to `20`")
            return 20
        # Support TQDM_MINITERS environment variable, which sets the minimum refresh rate
        if "TQDM_MINITERS" in os.environ:
            return max(int(os.environ["TQDM_MINITERS"]), refresh_rate)
        return refresh_rate


def convert_inf(x: Optional[Union[int, float]]) -> Optional[Union[int, float]]:
    """The tqdm doesn't support inf/nan values.

    We have to convert it to None.

    """
    if x is None or math.isinf(x) or math.isnan(x):
        return None
    return x


def _update_n(bar: tqdm.tqdm, value: int) -> None:
    if not bar.disable:
        bar.n = value
        bar.refresh()
