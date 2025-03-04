import os
import sys
from typing import TYPE_CHECKING, Any, Optional

from lightning_utilities.core import rank_zero
import tqdm.auto as tqdm
from typing_extensions import override

from . import progress_bar
from .. import utils

if TYPE_CHECKING:
    import reax

__all__ = ("TqdmProgressBar",)


class TqdmProgressBar(progress_bar.ProgressBar):
    # pylint: disable=too-many-public-methods
    BAR_FORMAT = (
        "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}{postfix}]"
    )

    def __init__(self, refresh_rate: int = 1, process_position: int = 0):
        """Init function."""
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
        """Train progress bar."""
        if self._train_progress_bar is None:
            raise TypeError(
                f"The `{self.__class__.__name__}._train_progress_bar` reference has not been set "
                f"yet."
            )
        return self._train_progress_bar

    @train_progress_bar.setter
    def train_progress_bar(
        # pylint: disable=disallowed-name
        self,
        bar: tqdm.tqdm,
    ) -> None:
        """Train progress bar."""
        self._train_progress_bar = bar

    @property
    def val_progress_bar(self) -> tqdm.tqdm:
        """Val progress bar."""
        if self._val_progress_bar is None:
            raise TypeError(
                f"The `{self.__class__.__name__}._val_progress_bar` reference has not been set yet."
            )
        return self._val_progress_bar

    @val_progress_bar.setter
    def val_progress_bar(
        # pylint: disable=disallowed-name
        self,
        bar: tqdm.tqdm,
    ) -> None:
        """Val progress bar."""
        self._val_progress_bar = bar

    @property
    def test_progress_bar(self) -> tqdm.tqdm:
        """Test progress bar."""
        if self._test_progress_bar is None:
            raise TypeError(
                f"The `{self.__class__.__name__}._test_progress_bar` reference has not been set "
                f"yet."
            )
        return self._test_progress_bar

    @test_progress_bar.setter
    def test_progress_bar(
        # pylint: disable=disallowed-name
        self,
        bar: tqdm.tqdm,
    ) -> None:
        """Test progress bar."""
        self._test_progress_bar = bar

    @property
    def predict_progress_bar(self) -> tqdm.tqdm:
        """Predict progress bar."""
        if self._predict_progress_bar is None:
            raise TypeError(
                f"The `{self.__class__.__name__}._predict_progress_bar` reference has not been set "
                f"yet."
            )
        return self._predict_progress_bar

    @predict_progress_bar.setter
    def predict_progress_bar(
        # pylint: disable=disallowed-name
        self,
        bar: tqdm.tqdm,
    ) -> None:
        """Predict progress bar."""
        self._predict_progress_bar = bar

    @property
    def refresh_rate(self) -> int:
        """Refresh rate."""
        return self._refresh_rate

    @property
    def process_position(self) -> int:
        """Process position."""
        return self._process_position

    @property
    def is_enabled(self) -> bool:
        """Is enabled."""
        return self._enabled and self.refresh_rate > 0

    @property
    def is_disabled(self) -> bool:
        """Is disabled."""
        return not self.is_enabled

    @override
    def disable(self) -> None:
        """Disable function."""
        self._enabled = False

    @override
    def enable(self) -> None:
        """Enable function."""
        self._enabled = True

    def init_train_tqdm(self, stage: "reax.stages.Fit") -> tqdm.tqdm:
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
    def on_fit_start(self, _trainer: "reax.Trainer", stage: "reax.stages.Fit", /) -> None:
        """On fit start."""
        self.train_progress_bar = self.init_train_tqdm(stage)

    @override
    def on_train_epoch_start(self, trainer: "reax.Trainer", stage: "reax.stages.Train", /) -> None:
        """On train epoch start."""
        self.train_progress_bar.reset(total=utils.convert_inf(stage.max_batches))
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
        /,
    ) -> None:
        """On train batch end."""
        n = batch_idx + 1
        if self._should_update(n, self.train_progress_bar.total):
            _update_n(self.train_progress_bar, n)
            self.train_progress_bar.set_postfix(trainer.progress_bar_metrics)

    @override
    def on_train_epoch_end(self, trainer: "reax.Trainer", _stage: "reax.stages.Train", /) -> None:
        """On train epoch end."""
        if not self.train_progress_bar.disable:
            self.train_progress_bar.set_postfix(trainer.progress_bar_metrics)

    @override
    def on_fit_end(self, _trainer: "reax.Trainer", _stage: "reax.stages.Fit", /) -> None:
        """On fit end."""
        self.train_progress_bar.close()

    @override
    def on_validation_epoch_start(
        self, _trainer: "reax.Trainer", stage: "reax.stages.Validate", /
    ) -> None:
        """On validation epoch start."""
        self.val_progress_bar = self.init_validation_tqdm(stage)
        self.val_progress_bar.reset(total=utils.convert_inf(stage.max_batches))
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
        /,
    ) -> None:
        """On validation batch end."""
        n = batch_idx + 1
        if self._should_update(n, self.val_progress_bar.total):
            _update_n(self.val_progress_bar, n)

    @override
    def on_validation_epoch_end(
        self, trainer: "reax.Trainer", stage: "reax.stages.Validate", /
    ) -> None:
        """On validation epoch end."""
        self.val_progress_bar.close()
        if self._train_progress_bar is not None and trainer.stage != stage:
            self.train_progress_bar.set_postfix(trainer.progress_bar_metrics)

    @override
    def on_test_epoch_start(self, trainer: "reax.Trainer", stage: "reax.Stage", /) -> None:
        """On test epoch start."""
        self.test_progress_bar = self.init_test_tqdm()

    @override
    def on_test_batch_start(
        self, trainer: "reax.Trainer", stage: "reax.stages.Test", batch: Any, batch_idx: int, /
    ) -> None:
        """On test batch start."""
        self.test_progress_bar.reset(total=utils.convert_inf(stage.max_batches))
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
        /,
    ) -> None:
        """On test batch end."""
        n = batch_idx + 1
        if self._should_update(n, self.test_progress_bar.total):
            _update_n(self.test_progress_bar, n)

    @override
    def on_test_epoch_end(self, _trainer: "reax.Trainer", _stage: "reax.Stage", /) -> None:
        """On test epoch end."""
        self.test_progress_bar.close()

    @override
    def on_predict_epoch_start(
        self, _trainer: "reax.Trainer", stage: "reax.stages.Predict", /
    ) -> None:
        """On predict epoch start."""
        self.predict_progress_bar = self.init_predict_tqdm(stage)
        self.predict_progress_bar.reset(total=utils.convert_inf(stage.max_batches))
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
        /,
    ) -> None:
        """On predict batch end."""
        n = batch_idx + 1
        if self._should_update(n, self.predict_progress_bar.total):
            _update_n(self.predict_progress_bar, n)

    @override
    def on_predict_epoch_end(self, trainer: "reax.Trainer", stage: "reax.Stage", /) -> None:
        """On predict epoch end."""
        self.predict_progress_bar.close()

    @override
    def print(self, *args: Any, sep: str = " ", **kwargs: Any) -> None:
        """Print function."""
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
        """Should update."""
        return self.is_enabled and (current % self.refresh_rate == 0 or current == total)

    @staticmethod
    def _resolve_refresh_rate(refresh_rate: int) -> int:
        """Resolve refresh rate."""
        if os.getenv("COLAB_GPU") and refresh_rate == 1:
            # smaller refresh rate on colab causes crashes, choose a higher value
            rank_zero.rank_zero_debug("Using a higher refresh rate on Colab. Setting it to `20`")
            return 20
        # Support TQDM_MINITERS environment variable, which sets the minimum refresh rate
        if "TQDM_MINITERS" in os.environ:
            return max(int(os.environ["TQDM_MINITERS"]), refresh_rate)
        return refresh_rate


def _update_n(
    # pylint: disable=disallowed-name
    bar: tqdm.tqdm,
    value: int,
) -> None:
    """Update n."""
    if not bar.disable:
        bar.n = value
        bar.refresh()
