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
# https://github.com/Lightning-AI/pytorch-lightning/blob/f9babd1def4c703e639dfc34fd1877ac4e7b9435/tests/tests_pytorch/listeners/progress/test_rich_progress_bar.py
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
from collections import defaultdict
import pickle
from unittest import mock
from unittest.mock import DEFAULT, Mock

import pytest

import reax
from reax import demos, loggers
from reax.listeners import progress
from reax.listeners.progress import rich_progress

rich = pytest.importorskip("rich")


def test_rich_progress_bar_listener():
    trainer = reax.Trainer(listeners=progress.RichProgressBar())

    progress_bars = [c for c in trainer.listeners if isinstance(c, progress.ProgressBar)]

    assert len(progress_bars) == 1
    assert isinstance(trainer.progress_bar_listener, progress.RichProgressBar)


def test_rich_progress_bar_refresh_rate_enabled():
    progress_bar = progress.RichProgressBar(refresh_rate=1)
    assert progress_bar.is_enabled
    assert not progress_bar.is_disabled
    progress_bar = progress.RichProgressBar(refresh_rate=0)
    assert not progress_bar.is_enabled
    assert progress_bar.is_disabled


@pytest.mark.parametrize(
    "dataset", [demos.RandomDataset(32, 64), demos.RandomIterableDataset(32, 64)]
)
def test_rich_progress_bar(tmp_path, dataset):
    class TestModel(demos.BoringModel):
        def train_dataloader(self):
            return reax.DataLoader(dataset=dataset)

        def val_dataloader(self):
            return reax.DataLoader(dataset=dataset)

        def test_dataloader(self):
            return reax.DataLoader(dataset=dataset)

        def predict_dataloader(self):
            return reax.DataLoader(dataset=dataset)

    trainer = reax.Trainer(
        default_root_dir=tmp_path,
        # TODO: num_sanity_val_steps=0,
        limit_train_batches=1,
        limit_val_batches=1,
        limit_test_batches=1,
        limit_predict_batches=1,
        max_epochs=1,
        listeners=progress.RichProgressBar(),
    )
    model = TestModel()

    with mock.patch("reax.listeners.progress.rich_progress.Progress.update") as mocked:
        trainer.fit(model)
    # 2 for train progress bar and 1 for val progress bar
    assert mocked.call_count == 3

    with mock.patch("reax.listeners.progress.rich_progress.Progress.update") as mocked:
        trainer.validate(model)
    assert mocked.call_count == 1

    with mock.patch("reax.listeners.progress.rich_progress.Progress.update") as mocked:
        trainer.test(model)
    assert mocked.call_count == 1

    with mock.patch("reax.listeners.progress.rich_progress.Progress.update") as mocked:
        trainer.predict(model)
    assert mocked.call_count == 1


def test_rich_progress_bar_import_error(monkeypatch):
    import reax.listeners.progress.rich_progress as imports

    monkeypatch.setattr(imports, "_RICH_AVAILABLE", False)
    with pytest.raises(ModuleNotFoundError, match="`RichProgressBar` requires `rich` >= 10.2.2."):
        progress.RichProgressBar()


def test_rich_progress_bar_custom_theme():
    """Test to ensure that custom theme styles are used."""
    with mock.patch.multiple(
        "reax.listeners.progress.rich_progress",
        CustomBarColumn=DEFAULT,
        BatchesProcessedColumn=DEFAULT,
        CustomTimeColumn=DEFAULT,
        ProcessingSpeedColumn=DEFAULT,
    ) as mocks:
        theme = rich_progress.RichProgressBarTheme()

        progress_bar = progress.RichProgressBar(theme=theme)
        progress_bar.on_train_start(reax.Trainer(), demos.BoringModel())

        assert progress_bar.theme == theme
        _, kwargs = mocks["CustomBarColumn"].call_args
        assert kwargs["complete_style"] == theme.progress_bar
        assert kwargs["finished_style"] == theme.progress_bar_finished

        _, kwargs = mocks["BatchesProcessedColumn"].call_args
        assert kwargs["style"] == theme.batch_progress

        _, kwargs = mocks["CustomTimeColumn"].call_args
        assert kwargs["style"] == theme.time

        _, kwargs = mocks["ProcessingSpeedColumn"].call_args
        assert kwargs["style"] == theme.processing_speed


def test_rich_progress_bar_keyboard_interrupt(tmp_path):
    """Test to ensure that when the user keyboard interrupts, we close the progress bar."""

    class TestModel(demos.BoringModel):
        def on_train_epoch_start(self, *_) -> None:
            raise KeyboardInterrupt

    model = TestModel()

    with (
        mock.patch("rich.progress.Progress.stop", autospec=True) as mock_progress_stop,
        pytest.raises(SystemExit),
    ):
        progress_bar = progress.RichProgressBar()
        trainer = reax.Trainer(
            default_root_dir=tmp_path,
            listeners=progress_bar,
        )

        trainer.fit(model, fast_dev_run=True)
    mock_progress_stop.assert_called_once()


def test_rich_progress_bar_configure_columns():
    from rich.progress import TextColumn

    custom_column = TextColumn("[progress.description]Testing Rich!")

    class CustomRichProgressBar(progress.RichProgressBar):
        def _configure_columns(self, trainer):
            return [custom_column]

    progress_bar = CustomRichProgressBar()

    progress_bar._init_progress(Mock())

    assert progress_bar.progress.columns[0] == custom_column
    assert len(progress_bar.progress.columns) == 2


@pytest.mark.parametrize(("leave", "reset_call_count"), ([(True, 0), (False, 3)]))
def test_rich_progress_bar_leave(tmp_path, leave, reset_call_count):
    # Calling `reset` means continuing on the same progress bar.
    model = demos.BoringModel()

    with mock.patch("rich.progress.Progress.reset", autospec=True) as mock_progress_reset:
        progress_bar = progress.RichProgressBar(leave=leave)
        trainer = reax.Trainer(
            default_root_dir=tmp_path,
            listeners=progress_bar,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
        )
        trainer.fit(
            model,
            # TODO: num_sanity_val_steps=0,
            limit_train_batches=1,
            limit_val_batches=0,
            max_epochs=4,
        )
    assert mock_progress_reset.call_count == reset_call_count


@mock.patch("rich.progress.Progress.update")
def test_rich_progress_bar_refresh_rate_disabled(progress_update, tmp_path):
    trainer = reax.Trainer(
        default_root_dir=tmp_path, listeners=progress.RichProgressBar(refresh_rate=0)
    )
    trainer.fit(demos.BoringModel(), fast_dev_run=4)
    assert progress_update.call_count == 0


@pytest.mark.parametrize(
    ("refresh_rate", "train_batches", "val_batches", "expected_call_count"),
    [
        # note: there is always one extra update at the very end (+1)
        (3, 6, 6, 2 + 2 + 1),
        (4, 6, 6, 2 + 2 + 1),
        (7, 6, 6, 1 + 1 + 1),
        (1, 2, 3, 2 + 3 + 1),
        (1, 0, 0, 0 + 0),
        (3, 1, 0, 1 + 0),
        (3, 1, 1, 1 + 1 + 1),
        (3, 5, 0, 2 + 0),
        (3, 5, 2, 2 + 1 + 1),
        (6, 5, 2, 1 + 1 + 1),
    ],
)
def test_rich_progress_bar_with_refresh_rate(
    tmp_path, refresh_rate, train_batches, val_batches, expected_call_count
):
    model = demos.BoringModel()
    trainer = reax.Trainer(
        default_root_dir=tmp_path,
        # TODO: num_sanity_val_steps=0,
        limit_train_batches=train_batches,
        limit_val_batches=val_batches,
        max_epochs=1,
        listeners=progress.RichProgressBar(refresh_rate=refresh_rate),
    )

    trainer.progress_bar_listener.on_train_start(trainer, model)
    with mock.patch.object(
        trainer.progress_bar_listener.progress,
        "update",
        wraps=trainer.progress_bar_listener.progress.update,
    ) as progress_update:
        trainer.fit(model)
        assert progress_update.call_count == expected_call_count

    if train_batches > 0:
        fit_main_bar = trainer.progress_bar_listener.progress.tasks[0]
        assert fit_main_bar.completed == train_batches
        assert fit_main_bar.total == train_batches
        assert fit_main_bar.visible
    if val_batches > 0:
        fit_val_bar = trainer.progress_bar_listener.progress.tasks[1]
        assert fit_val_bar.completed == val_batches
        assert fit_val_bar.total == val_batches
        assert not fit_val_bar.visible


@pytest.mark.skip(reason="Trainer doesn't support sanity check yet")
@pytest.mark.parametrize("limit_val_batches", [1, 5])
def test_rich_progress_bar_num_sanity_val_steps(tmp_path, limit_val_batches):
    model = demos.BoringModel()

    progress_bar = progress.RichProgressBar()
    num_sanity_val_steps = 3

    trainer = reax.Trainer(
        default_root_dir=tmp_path,
        num_sanity_val_steps=num_sanity_val_steps,
        limit_train_batches=1,
        limit_val_batches=limit_val_batches,
        max_epochs=1,
        listeners=progress_bar,
    )

    trainer.fit(model)
    assert progress_bar.progress.tasks[0].completed == min(num_sanity_val_steps, limit_val_batches)
    assert progress_bar.progress.tasks[0].total == min(num_sanity_val_steps, limit_val_batches)


def test_rich_progress_bar_counter_with_val_check_interval(tmp_path):
    """Test the completed and total counter for rich progress bar when using val_check_interval."""
    progress_bar = progress.RichProgressBar()
    model = demos.BoringModel()
    trainer = reax.Trainer(default_root_dir=tmp_path, listeners=[progress_bar])
    trainer.fit(
        model,
        val_check_interval=2,
        max_epochs=1,
        limit_train_batches=7,
        limit_val_batches=4,
    )

    fit_train_progress_bar = progress_bar.progress.tasks[1]
    assert fit_train_progress_bar.completed == 7
    assert fit_train_progress_bar.total == 7

    fit_val_bar = progress_bar.progress.tasks[2]
    assert fit_val_bar.completed == 4
    assert fit_val_bar.total == 4

    trainer.validate(model)
    val_bar = progress_bar.progress.tasks[0]
    assert val_bar.completed == 4
    assert val_bar.total == 4


def test_rich_progress_bar_metric_display_task_id(tmp_path):
    class CustomModel(demos.BoringModel):
        def training_step(self, *args, **kwargs):
            res = super().training_step(*args, **kwargs)
            self.log("train_loss", res["loss"], prog_bar=True)
            return res

    progress_bar = progress.RichProgressBar()
    model = CustomModel()
    trainer = reax.Trainer(
        default_root_dir=tmp_path,
        listeners=progress_bar,
        enable_checkpointing=False,
        enable_model_summary=False,
    )

    trainer.fit(model, limit_train_batches=1, limit_val_batches=1, max_epochs=1)
    train_progress_bar_id = progress_bar.train_progress_bar_id
    val_progress_bar_id = progress_bar.val_progress_bar_id
    rendered = progress_bar.progress.columns[-1]._renderable_cache

    for key in ("loss", "v_num", "train_loss"):
        assert key in rendered[train_progress_bar_id][1]
        assert key not in rendered[val_progress_bar_id][1]


def test_rich_progress_bar_metrics_fast_dev_run(tmp_path):
    """Test that `v_num` does not appear in the progress bar when a dummy logger is used (fast-dev-run)."""
    progress_bar = progress.RichProgressBar()
    trainer = reax.Trainer(default_root_dir=tmp_path, listeners=progress_bar)
    model = demos.BoringModel()
    trainer.fit(model, fast_dev_run=True)
    assert isinstance(trainer.logger, DummyLogger)
    train_progress_bar_id = progress_bar.train_progress_bar_id
    val_progress_bar_id = progress_bar.val_progress_bar_id
    rendered = progress_bar.progress.columns[-1]._renderable_cache
    assert "v_num" not in rendered[train_progress_bar_id][1]
    assert "v_num" not in rendered[val_progress_bar_id][1]


def test_rich_progress_bar_correct_value_epoch_end(tmp_path):
    """Rich counterpart to test_tqdm_progress_bar::test_tqdm_progress_bar_correct_value_epoch_end."""

    class MockedProgressBar(progress.RichProgressBar):
        calls = defaultdict(list)

        def get_metrics(self, trainer, pl_module):
            items = super().get_metrics(trainer, model)
            del items["v_num"]
            # this is equivalent to mocking `set_postfix` as this method gets called every time
            self.calls[trainer.state.fn].append(
                (
                    trainer.state.stage,
                    trainer.current_epoch,
                    trainer.global_updates,
                    items,
                )
            )
            return items

    class MyModel(demos.BoringModel):
        def training_step(self, batch, batch_idx):
            self.log(
                "a", self.global_updates, prog_bar=True, on_step=False, on_epoch=True, reduce_fx=max
            )
            return super().training_step(batch, batch_idx)

        def validation_step(self, batch, batch_idx):
            self.log(
                "b", self.global_updates, prog_bar=True, on_step=False, on_epoch=True, reduce_fx=max
            )
            return super().validation_step(batch, batch_idx)

        def test_step(self, batch, batch_idx):
            self.log(
                "c", self.global_updates, prog_bar=True, on_step=False, on_epoch=True, reduce_fx=max
            )
            return super().test_step(batch, batch_idx)

    model = MyModel()
    pbar = MockedProgressBar()
    trainer = reax.Trainer(
        default_root_dir=tmp_path,
        enable_model_summary=False,
        enable_checkpointing=False,
        log_every_n_steps=1,
        listeners=pbar,
        logger=loggers.CsvLogger(tmp_path),
    )

    trainer.fit(model, limit_train_batches=2, limit_val_batches=2, max_epochs=2)
    assert pbar.calls["fit"] == [
        ("sanity_check", 0, 0, {"b": 0}),
        ("train", 0, 1, {}),
        ("train", 0, 2, {}),
        ("validate", 0, 2, {"b": 2}),  # validation end
        # epoch end over, `on_epoch=True` metrics are computed
        ("train", 0, 2, {"a": 1, "b": 2}),  # training epoch end
        ("train", 1, 3, {"a": 1, "b": 2}),
        ("train", 1, 4, {"a": 1, "b": 2}),
        ("validate", 1, 4, {"a": 1, "b": 4}),  # validation end
        ("train", 1, 4, {"a": 3, "b": 4}),  # training epoch end
    ]

    trainer.validate(model, verbose=False, max_epochs=2)
    assert pbar.calls["validate"] == []

    trainer.test(model, verbose=False, limit_batches=2, max_epochs=2)
    assert pbar.calls["test"] == []


def test_rich_progress_bar_padding():
    progress_bar = progress.RichProgressBar()
    trainer = Mock()
    trainer.max_epochs = 1
    progress_bar._trainer = trainer

    train_description = progress_bar._get_train_description(current_epoch=0)
    assert "Epoch 0/0" in train_description
    assert len(progress_bar.validation_description) == len(train_description)


def test_rich_progress_bar_can_be_pickled(tmp_path):
    bar = progress.RichProgressBar()
    trainer = reax.Trainer(
        default_root_dir=tmp_path,
        listeners=[bar],
        logger=False,
        enable_model_summary=False,
    )
    model = demos.BoringModel()
    pickle.dumps(bar)
    trainer.fit(model, max_epochs=1, limit_train_batches=1, limit_val_batches=1)
    pickle.dumps(bar)
    trainer.validate(model, limit_val_batches=1)
    pickle.dumps(bar)
    trainer.test(model, limit_batches=1)
    pickle.dumps(bar)
    trainer.predict(model, limit_predict_batches=1)
    pickle.dumps(bar)


def test_rich_progress_bar_reset_bars():
    """Test that the progress bar resets all internal bars when a new trainer stage begins."""
    bar = progress.RichProgressBar()
    assert bar.is_enabled
    assert bar.progress is None
    assert bar._progress_stopped is False

    def _set_fake_bar_ids():
        bar.train_progress_bar_id = 0
        # TODO: bar.val_sanity_progress_bar_id = 1
        bar.val_progress_bar_id = 2
        bar.test_progress_bar_id = 3
        bar.predict_progress_bar_id = 4

    for stage in ("train", "validation", "test", "predict"):  # TODO: "sanity_check",
        hook_name = f"on_{stage}_start"
        hook = getattr(bar, hook_name)

        _set_fake_bar_ids()  # pretend that bars are initialized from a previous run
        hook(Mock(), Mock())
        bar.teardown(Mock(), Mock(), Mock())

        # assert all bars are reset
        assert bar.train_progress_bar_id is None
        #  TODO: assert bar.val_sanity_progress_bar_id is None
        assert bar.val_progress_bar_id is None
        assert bar.test_progress_bar_id is None
        assert bar.predict_progress_bar_id is None

        # the progress object remains in case we need it for the next stage
        assert bar.progress is not None


def test_rich_progress_bar_disabled(tmp_path):
    """Test that in a disabled bar there are no updates and no internal progress objects."""
    bar = progress.RichProgressBar()
    bar.disable()
    assert bar.is_disabled

    model = demos.BoringModel()
    trainer = reax.Trainer(
        default_root_dir=tmp_path,
        enable_model_summary=False,
        enable_checkpointing=False,
        listeners=[bar],
    )

    with mock.patch("reax.listeners.progress.rich_progress.CustomProgress") as mocked:
        trainer.fit(model, max_epochs=1, limit_train_batches=2, limit_val_batches=2)
        trainer.validate(model)
        trainer.test(model, limit_batches=2)
        trainer.predict(model, limit_batches=2)

    mocked.assert_not_called()
    assert bar.train_progress_bar_id is None
    # TODO: assert bar.val_sanity_progress_bar_id is None
    assert bar.val_progress_bar_id is None
    assert bar.test_progress_bar_id is None
    assert bar.predict_progress_bar_id is None


@pytest.mark.parametrize("metrics_format", [".3f", ".3e"])
def test_rich_progress_bar_metrics_format(tmp_path, metrics_format):
    metric_name = "train_loss"

    class CustomModel(demos.BoringModel):
        def training_step(self, *args, **kwargs):
            res = super().training_step(*args, **kwargs)
            self.log(metric_name, res["loss"], prog_bar=True)
            return res

    progress_bar = progress.RichProgressBar(
        theme=rich_progress.RichProgressBarTheme(metrics_format=metrics_format)
    )
    trainer = reax.Trainer(
        default_root_dir=tmp_path,
        listeners=progress_bar,
    )
    model = CustomModel()
    trainer.fit(model, fast_dev_run=True)

    def extract_rendered_value():
        rendered = progress_bar.progress.columns[-1]._renderable_cache
        train_progress_bar_id = progress_bar.train_progress_bar_id
        rendered_text = str(rendered[train_progress_bar_id][1])
        return rendered_text.split(f"{metric_name}: ")[1]

    rendered_value = extract_rendered_value()
    value = trainer.logged_metrics[metric_name]
    formatted_value = f"{value:{metrics_format}}"
    assert rendered_value == formatted_value


def test_rich_progress_bar_metrics_theme_update(*_):
    theme = progress.RichProgressBar().theme
    assert theme.metrics_format == ".3f"
    assert theme.metrics_text_delimiter == " "

    theme = progress.RichProgressBar(
        theme=rich_progress.RichProgressBarTheme(metrics_format=".3e", metrics_text_delimiter="\n")
    ).theme
    assert theme.metrics_format == ".3e"
    assert theme.metrics_text_delimiter == "\n"
