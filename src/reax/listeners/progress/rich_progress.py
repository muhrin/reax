# Copyright (C) 2024  Martin Uhrin
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Most of this file is covered by the following license.  To find what has been modified you
# can perform a diff with the file at:
# https://github.com/Lightning-AI/pytorch-lightning/blob/0324a20f00235c7a10a235a44326811ba42b6ae4/src/lightning/pytorch/callbacks/progress/rich_progress.py
#
# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections.abc import Generator
from dataclasses import dataclass
from datetime import timedelta
import math
from typing import TYPE_CHECKING, Any, Final, Optional, Union, cast

from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override

from reax import stages

from . import progress_bar

if TYPE_CHECKING:
    import reax

__all__ = ("RichProgressBar",)

_RICH_AVAILABLE = RequirementCache("rich>=10.2.2")

if _RICH_AVAILABLE:  # noqa: C901
    import rich
    from rich import style as style_
    from rich import text as text_
    import rich.progress
    import rich.progress_bar

    class CustomBarColumn(rich.progress.BarColumn):
        """Overrides ``BarColumn`` to provide support for dataloaders that do not define a size
        (infinite size) such as ``IterableDataset``."""

        def render(self, task: rich.progress.Task) -> rich.progress_bar.ProgressBar:
            """Gets a progress bar widget for a task."""
            assert task.total is not None
            assert task.remaining is not None
            return rich.progress_bar.ProgressBar(
                total=max(0, task.total),
                completed=max(0, task.completed),
                width=None if self.bar_width is None else max(1, self.bar_width),
                pulse=not task.started or not math.isfinite(task.remaining),
                animation_time=task.get_time(),
                style=self.style,
                complete_style=self.complete_style,
                finished_style=self.finished_style,
                pulse_style=self.pulse_style,
            )

    @dataclass
    class CustomInfiniteTask(rich.progress.Task):
        """Overrides ``Task`` to define an infinite task.

        This is useful for datasets that do not define a size (infinite size) such as
        ``IterableDataset``.
        """

        @property
        def time_remaining(self) -> Optional[float]:
            return None

    class CustomProgress(rich.progress.Progress):
        """Overrides ``Progress`` to support adding tasks that have an infinite total size."""

        def add_task(
            self,
            description: str,
            start: bool = True,
            total: Optional[float] = 100.0,
            completed: int = 0,
            visible: bool = True,
            **fields: Any,
        ) -> rich.progress.TaskID:
            assert total is not None
            if not math.isfinite(total):
                task = CustomInfiniteTask(
                    self._task_index,
                    description,
                    total,
                    completed,
                    visible=visible,
                    fields=fields,
                    _get_time=self.get_time,
                    _lock=self._lock,
                )
                return self.add_custom_task(task)
            return super().add_task(description, start, total, completed, visible, **fields)

        def add_custom_task(
            self, task: CustomInfiniteTask, start: bool = True
        ) -> rich.progress.TaskID:
            with self._lock:
                self._tasks[self._task_index] = task
                if start:
                    self.start_task(self._task_index)
                new_task_index = self._task_index
                self._task_index = rich.progress.TaskID(int(self._task_index) + 1)
            self.refresh()
            return new_task_index

    class CustomTimeColumn(rich.progress.ProgressColumn):
        # Only refresh twice a second to prevent jitter
        max_refresh = 0.5

        def __init__(self, style: Union[str, style_.Style]) -> None:
            self.style = style
            super().__init__()

        def render(self, task: rich.progress.Task) -> text_.Text:
            elapsed = task.finished_time if task.finished else task.elapsed
            remaining = task.time_remaining
            elapsed_delta = "-:--:--" if elapsed is None else str(timedelta(seconds=int(elapsed)))
            remaining_delta = (
                "-:--:--" if remaining is None else str(timedelta(seconds=int(remaining)))
            )
            return text_.Text(f"{elapsed_delta} • {remaining_delta}", style=self.style)

    class BatchesProcessedColumn(rich.progress.ProgressColumn):
        def __init__(self, style: Union[str, style_.Style]):
            self.style = style
            super().__init__()

        def render(self, task: rich.progress.Task) -> rich.console.RenderableType:
            total = task.total if task.total != float("inf") else "--"
            return text_.Text(f"{int(task.completed)}/{total}", style=self.style)

    class ProcessingSpeedColumn(rich.progress.ProgressColumn):
        def __init__(self, style: Union[str, style_.Style]):
            self.style = style
            super().__init__()

        def render(self, task: rich.progress.Task) -> rich.console.RenderableType:
            task_speed = f"{task.speed:>.2f}" if task.speed is not None else "0.00"
            return text_.Text(f"{task_speed}it/s", style=self.style)

    class MetricsTextColumn(rich.progress.ProgressColumn):
        """A column containing text."""

        def __init__(
            self,
            trainer: "reax.Trainer",
            style: Union[str, "style_.Style"],
            text_delimiter: str,
            metrics_format: str,
        ):
            self._trainer = trainer
            self._tasks: dict[Union[int, rich.progress.TaskID], Any] = {}
            self._current_task_id = 0
            self._metrics: dict[Union[str, style_.Style], Any] = {}
            self._style = style
            self._text_delimiter = text_delimiter
            self._metrics_format = metrics_format
            super().__init__()

        def update(self, metrics: dict[Any, Any]) -> None:
            # Called when metrics are ready to be rendered.
            # This is to prevent render from causing deadlock issues by requesting metrics
            # in separate threads.
            self._metrics = metrics

        def render(self, task: rich.progress.Task) -> text_.Text:
            assert isinstance(self._trainer.progress_bar_listener, RichProgressBar)
            if (
                not isinstance(self._trainer.stage, stages.Fit)
                # TODO: or self._trainer.sanity_checking
                or self._trainer.progress_bar_listener.train_progress_bar_id != task.id
            ):
                return text_.Text()

            if isinstance(self._trainer.stage, stages.Fit) and task.id not in self._tasks:
                self._tasks[task.id] = "None"
                if self._renderable_cache:
                    self._current_task_id = cast(rich.progress.TaskID, self._current_task_id)
                    self._tasks[self._current_task_id] = self._renderable_cache[
                        self._current_task_id
                    ][1]
                self._current_task_id = task.id

            if isinstance(self._trainer.stage, stages.Fit) and task.id != self._current_task_id:
                return self._tasks[task.id]

            metrics_texts = self._generate_metrics_texts()
            text = self._text_delimiter.join(metrics_texts)
            return text_.Text(text, justify="left", style=self._style)

        def _generate_metrics_texts(self) -> Generator[str, None, None]:
            for name, value in self._metrics.items():
                if not isinstance(value, str):
                    value = f"{value:{self._metrics_format}}"
                yield f"{name}: {value}"


@dataclass
class RichProgressBarTheme:
    """Styles to associate to different base components.

    Args:
        description: Style for the progress bar description. For eg., Epoch x, Testing, etc.
        progress_bar: Style for the bar in progress.
        progress_bar_finished: Style for the finished progress bar.
        progress_bar_pulse: Style for the progress bar when `IterableDataset` is being processed.
        batch_progress: Style for the progress tracker (i.e 10/50 batches completed).
        time: Style for the processed time and estimate time remaining.
        processing_speed: Style for the speed of the batches being processed.
        metrics: Style for the metrics

    https://rich.readthedocs.io/en/stable/style.html

    """

    description: Union[str, "style_.Style"] = ""
    progress_bar: Union[str, "style_.Style"] = "#6206E0"
    progress_bar_finished: Union[str, "style_.Style"] = "#6206E0"
    progress_bar_pulse: Union[str, "style_.Style"] = "#6206E0"
    batch_progress: Union[str, "style_.Style"] = ""
    time: Union[str, "style_.Style"] = "dim"
    processing_speed: Union[str, "style_.Style"] = "dim underline"
    metrics: Union[str, "style_.Style"] = "italic"
    metrics_text_delimiter: str = " "
    metrics_format: str = ".3f"


class RichProgressBar(progress_bar.ProgressBar):
    """Create a progress bar with `rich text formatting <https://github.com/Textualize/rich>`_.

    Install it with pip:

    .. code-block:: bash

        pip install rich

    .. code-block:: python

        import reax

        trainer = reax.Trainer(callbacks=reax.listeners.progress.RichProgressBar())

    Args:
        refresh_rate: Determines at which rate (in number of batches) the progress bars get updated.
            Set it to ``0`` to disable the display.
        leave: Leaves the finished progress bar in the terminal at the end of the epoch.
            Default: False
        theme: Contains styles used to stylize the progress bar.
        console_kwargs: Args for constructing a `Console`

    Raises:
        ModuleNotFoundError:
            If required `rich` package is not installed on the device.

    Note:
        PyCharm users will need to enable “emulate terminal” in output console option in
        run/debug configuration to see styled output.
        Reference: https://rich.readthedocs.io/en/latest/introduction.html#requirements

    """

    # pylint: disable=too-many-public-methods

    def __init__(
        self,
        refresh_rate: int = 1,
        leave: bool = False,
        theme: RichProgressBarTheme = RichProgressBarTheme(),
        console_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        if not _RICH_AVAILABLE:
            raise ModuleNotFoundError(
                "`RichProgressBar` requires `rich` >= 10.2.2. Install it by running "
                "`pip install -U rich`."
            )

        super().__init__()
        # Params
        self._refresh_rate: Final[int] = refresh_rate
        self._leave: Final[bool] = leave

        # State
        self._console: Optional[rich.Console] = None
        self._console_kwargs = console_kwargs or {}
        self._enabled: bool = True
        self.progress: Optional[CustomProgress] = None
        self.train_progress_bar_id: Optional[rich.progress.TaskID]
        self.val_sanity_progress_bar_id: Optional[rich.progress.TaskID] = None
        self.val_progress_bar_id: Optional[rich.progress.TaskID]
        self.test_progress_bar_id: Optional[rich.progress.TaskID]
        self.predict_progress_bar_id: Optional[rich.progress.TaskID]
        self._reset_progress_bar_ids()
        self._metric_component: Optional[MetricsTextColumn] = None
        self._progress_stopped: bool = False
        self.theme = theme

    @property
    def refresh_rate(self) -> float:
        return self._refresh_rate

    @property
    def is_enabled(self) -> bool:
        return self._enabled and self.refresh_rate > 0

    @property
    def is_disabled(self) -> bool:
        return not self.is_enabled

    @property
    def train_progress_bar(self) -> rich.progress.Task:
        assert self.progress is not None
        assert self.train_progress_bar_id is not None
        return self.progress.tasks[self.train_progress_bar_id]

    @property
    def val_sanity_check_bar(self) -> rich.progress.Task:
        assert self.progress is not None
        assert self.val_sanity_progress_bar_id is not None
        return self.progress.tasks[  # pylint: disable=invalid-sequence-index
            self.val_sanity_progress_bar_id
        ]

    @property
    def val_progress_bar(self) -> rich.progress.Task:
        assert self.progress is not None
        assert self.val_progress_bar_id is not None
        return self.progress.tasks[self.val_progress_bar_id]

    @property
    def test_progress_bar(self) -> rich.progress.Task:
        assert self.progress is not None
        assert self.test_progress_bar_id is not None
        return self.progress.tasks[self.test_progress_bar_id]

    @override
    def disable(self) -> None:
        self._enabled = False

    @override
    def enable(self) -> None:
        self._enabled = True

    def _init_progress(self, trainer: "reax.Trainer") -> None:
        if self.is_enabled and (self.progress is None or self._progress_stopped):
            self._reset_progress_bar_ids()
            rich.reconfigure(**self._console_kwargs)
            self._console = rich.get_console()
            self._console.clear_live()
            self._metric_component = MetricsTextColumn(
                trainer,
                self.theme.metrics,
                self.theme.metrics_text_delimiter,
                self.theme.metrics_format,
            )
            self.progress = CustomProgress(
                *self._configure_columns(trainer),
                self._metric_component,
                auto_refresh=False,
                disable=self.is_disabled,
                console=self._console,
            )
            self.progress.start()
            # progress has started
            self._progress_stopped = False

    def refresh(self) -> None:
        if self.progress:
            self.progress.refresh()

    @override
    def on_train_start(self, trainer: "reax.Trainer", /, *_) -> None:
        self._init_progress(trainer)

    @override
    def on_predict_start(self, trainer: "reax.Trainer", /, *_) -> None:
        self._init_progress(trainer)

    @override
    def on_test_start(self, trainer: "reax.Trainer", /, *_) -> None:
        self._init_progress(trainer)

    @override
    def on_validation_start(self, trainer: "reax.Trainer", /, *_) -> None:
        self._init_progress(trainer)

    @override
    def on_sanity_check_start(self, trainer: "reax.Trainer", *_) -> None:
        self._init_progress(trainer)

    @override
    def on_train_epoch_start(self, trainer: "reax.Trainer", stage: "reax.stages.Train", /) -> None:
        if self.is_disabled:
            return

        total_batches = stage.max_batches
        train_description = self._get_train_description(trainer.current_epoch, stage)

        if self.train_progress_bar_id is not None and self._leave:
            self._stop_progress()
            self._init_progress(trainer)
        if self.progress is not None:
            if self.train_progress_bar_id is None:
                self.train_progress_bar_id = self._add_task(total_batches, train_description)
            else:
                self.progress.reset(
                    self.train_progress_bar_id,
                    total=total_batches,
                    description=train_description,
                    visible=True,
                )

        self.refresh()

    @override
    def on_train_batch_end(
        self,
        trainer: "reax.Trainer",
        stage: "reax.stages.Train",
        _outputs: Any,
        _batch: Any,
        batch_idx: int,
        /,
    ) -> None:
        self._update(self.train_progress_bar_id, batch_idx + 1)
        self._update_metrics(trainer, stage)
        self.refresh()

    @override
    def on_train_epoch_end(self, trainer: "reax.Trainer", stage: "reax.stages.Train", /) -> None:
        self._update_metrics(trainer, stage)

    @override
    def on_sanity_check_end(self, *_) -> None:
        if self.progress is not None:
            assert self.val_sanity_progress_bar_id is not None
            self.progress.update(self.val_sanity_progress_bar_id, advance=0, visible=False)
        self.refresh()

    @override
    def on_validation_epoch_start(
        self, _trainer: "reax.Trainer", stage: "reax.stages.Validate", /
    ) -> None:
        if self.is_disabled:  # TODO or not self.has_dataloader_changed(dataloader_idx):
            return

        assert self.progress is not None

        # TODO:
        # if trainer.sanity_checking:
        #     if self.val_sanity_progress_bar_id is not None:
        #         self.progress.update(self.val_sanity_progress_bar_id, advance=0, visible=False)
        #
        #     self.val_sanity_progress_bar_id = self._add_task(
        #         stage.max_batches, self.sanity_check_description, visible=False
        #     )
        # else:
        if self.val_progress_bar_id is not None:
            self.progress.update(self.val_progress_bar_id, advance=0, visible=False)

        # TODO: remove old tasks when new ones are created
        self.val_progress_bar_id = self._add_task(
            stage.max_batches, self.validation_description, visible=False
        )

        self.refresh()

    def _add_task(
        self, total_batches: Union[int, float], description: str, visible: bool = True
    ) -> rich.progress.TaskID:
        assert self.progress is not None
        return self.progress.add_task(
            f"[{self.theme.description}]{description}" if self.theme.description else description,
            total=total_batches,
            visible=visible,
        )

    def _update(
        self, progress_bar_id: Optional[rich.progress.TaskID], current: int, visible: bool = True
    ) -> None:
        if self.progress is not None and self.is_enabled:
            assert progress_bar_id is not None
            total = self.progress.tasks[progress_bar_id].total
            assert total is not None
            if not self._should_update(current, total):
                return

            leftover = current % self.refresh_rate
            advance = leftover if (current == total and leftover != 0) else self.refresh_rate
            self.progress.update(progress_bar_id, advance=advance, visible=visible)
            self.refresh()

    def _should_update(self, current: int, total: Union[int, float]) -> bool:
        return current % self.refresh_rate == 0 or current == total

    @override
    def on_validation_epoch_end(
        self, trainer: "reax.Trainer", stage: "reax.stages.Validate", /
    ) -> None:
        if (
            self.is_enabled
            and self.val_progress_bar_id is not None
            and isinstance(trainer.stage, stages.Fit)
        ):
            assert self.progress is not None
            self.progress.update(self.val_progress_bar_id, advance=0, visible=False)
            self.refresh()

    # @override
    # def on_validation_end(self, trainer: "reax.Trainer",
    #       stage: "reax.stages.Validate", /) -> None:
    #     if isinstance(trainer.stage, stages.Fit):
    #         self._update_metrics(trainer, stage)
    #     self.reset_dataloader_idx_tracker()
    #
    # @override
    # def on_test_end(self, trainer: "reax.Trainer", stage: "reax.stages.Test", /) -> None:
    #     self.reset_dataloader_idx_tracker()
    #
    # @override
    # def on_predict_end(self, trainer: "reax.Trainer", stage: "reax.stages.Predict", /) -> None:
    #     self.reset_dataloader_idx_tracker()

    @override
    def on_test_epoch_start(self, _trainer: "reax.Trainer", stage: "reax.stages.Test", /) -> None:
        if self.is_disabled:  # or not self.has_dataloader_changed(dataloader_idx):
            return

        if self.test_progress_bar_id is not None:
            assert self.progress is not None
            self.progress.update(self.test_progress_bar_id, advance=0, visible=False)
        self.test_progress_bar_id = self._add_task(stage.max_batches, self.test_description)
        self.refresh()

    @override
    def on_predict_epoch_start(
        self, _trainer: "reax.Trainer", stage: "reax.stages.Predict", /
    ) -> None:
        if self.is_disabled:  # or not self.has_dataloader_changed(dataloader_idx):
            return

        if self.predict_progress_bar_id is not None:
            assert self.progress is not None
            self.progress.update(self.predict_progress_bar_id, advance=0, visible=False)
        self.predict_progress_bar_id = self._add_task(stage.max_batches, self.predict_description)
        self.refresh()

    @override
    def on_validation_batch_end(
        self,
        trainer: "reax.Trainer",
        _stage: "reax.stages.Validate",
        _outputs: Any,
        _batch: Any,
        batch_idx: int,
        /,
    ) -> None:
        if self.is_disabled:
            return

        # TODO: if trainer.sanity_checking:
        #     self._update(self.val_sanity_progress_bar_id, batch_idx + 1)
        if self.val_progress_bar_id is not None:
            self._update(self.val_progress_bar_id, batch_idx + 1)

        self.refresh()

    @override
    def on_test_batch_end(
        self,
        _trainer: "reax.Trainer",
        _stage: "reax.stages.Test",
        _outputs: Any,
        _batch: Any,
        batch_idx: int,
        /,
        # dataloader_idx: int = 0,
    ) -> None:
        if self.is_disabled:
            return
        assert self.test_progress_bar_id is not None
        self._update(self.test_progress_bar_id, batch_idx + 1)
        self.refresh()

    @override
    def on_predict_batch_end(
        self,
        _trainer: "reax.Trainer",
        _stage: "reax.stages.Predict",
        _outputs: Any,
        _batch: Any,
        batch_idx: int,
        /,
    ) -> None:
        if self.is_disabled:
            return
        assert self.predict_progress_bar_id is not None
        self._update(self.predict_progress_bar_id, batch_idx + 1)
        self.refresh()

    def _get_train_description(self, current_epoch: int, stage: "reax.stages.Train") -> str:
        train_description = f"Epoch {current_epoch}"
        if stage.parent is not None and stage.parent.max_iters is not None:
            max_epochs = stage.parent.max_iters
            train_description += f"/{max_epochs - 1}"
        if len(self.validation_description) > len(train_description):
            # Padding is required to avoid flickering due of uneven lengths of "Epoch X"
            # and "Validation" Bar description
            train_description = f"{train_description:{len(self.validation_description)}}"
        return train_description

    def _stop_progress(self) -> None:
        if self.progress is not None:
            self.progress.stop()
            # # signals for progress to be re-initialized for next stages
            self._progress_stopped = True

    def _reset_progress_bar_ids(self) -> None:
        self.train_progress_bar_id = None
        self.val_sanity_progress_bar_id = None
        self.val_progress_bar_id = None
        self.test_progress_bar_id = None
        self.predict_progress_bar_id = None

    def _update_metrics(self, trainer: "reax.Trainer", stage: "reax.stages.Stage") -> None:
        metrics = self.get_metrics(trainer, stage)
        if self._metric_component:
            self._metric_component.update(metrics)

    @override
    def teardown(self, /, *_) -> None:
        self._stop_progress()

    @override
    def on_exception(self, /, *_) -> None:
        self._stop_progress()

    def _configure_columns(self, _trainer: "reax.Trainer", /) -> list:
        return [
            rich.progress.TextColumn("[progress.description]{task.description}"),
            CustomBarColumn(
                complete_style=self.theme.progress_bar,
                finished_style=self.theme.progress_bar_finished,
                pulse_style=self.theme.progress_bar_pulse,
            ),
            BatchesProcessedColumn(style=self.theme.batch_progress),
            CustomTimeColumn(style=self.theme.time),
            ProcessingSpeedColumn(style=self.theme.processing_speed),
        ]

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        # both the console and progress object can hold thread lock objects that are not pickleable
        state["progress"] = None
        state["_console"] = None
        return state
