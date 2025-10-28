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
# https://github.com/Lightning-AI/pytorch-lightning/blob/f9babd1def4c703e639dfc34fd1877ac4e7b9435/tests/tests_pytorch/callbacks/test_early_stopping.py
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
import logging
import math
import os
import pickle
from test import helpers
from typing import Optional
from unittest import mock
from unittest.mock import Mock

import cloudpickle
import jax.numpy as jnp
import pytest
from typing_extensions import override

import reax
from reax import exceptions, listeners
from reax.demos import boring_classes

_logger = logging.getLogger(__name__)

# todo:
# @RunIf(sklearn=True, skip_windows=True)  # Flaky test on Windows for unknown reasons
# @mock.patch.dict(os.environ, os.environ.copy(), clear=True)
# def test_resume_early_stopping_from_checkpoint(tmp_path):
#     """Prevent regressions to bugs:
#
#     https://github.com/Lightning-AI/lightning/issues/1464
#     https://github.com/Lightning-AI/lightning/issues/1463
#
#     """
#     pytest.importorskip("sklearn")
#
#     reax.seed_everything(42)
#     model = helpers.ClassificationModel()
#     dm = helpers.ClassifDataModule()
#     checkpoint_callback = listeners.ModelCheckpoint(
#         dirpath=tmp_path, monitor="train_loss", save_top_k=1
#     )
#     early_stop_callback = EarlyStoppingTestRestore(None, monitor="train_loss")
#     trainer = reax.Trainer(
#         model,
#         default_root_dir=tmp_path,
#         listeners=[early_stop_callback, checkpoint_callback],
#     )
#     trainer.fit(datamodule=dm, max_epochs=4, num_sanity_val_steps=0)
#
#     assert len(early_stop_callback.saved_states) == 4
#
#     checkpoint_filepath = checkpoint_callback.kth_best_model_path
#     # ensure state is persisted properly
#     checkpoint = torch.load(checkpoint_filepath, weights_only=True)
#     # the checkpoint saves "epoch + 1"
#     early_stop_callback_state = early_stop_callback.saved_states[checkpoint["epoch"]]
#     assert len(early_stop_callback.saved_states) == 4
#     es_name = "EarlyStoppingTestRestore{'monitor': 'train_loss', 'mode': 'min'}"
#     assert checkpoint["callbacks"][es_name] == early_stop_callback_state
#
#     # ensure state is reloaded properly (assertion in the callback)
#     early_stop_callback = EarlyStoppingTestRestore(early_stop_callback_state, monitor="train_loss")
#     new_trainer = reax.Trainer(
#         default_root_dir=tmp_path,
#         listeners=[early_stop_callback],
#     )
#
#     with pytest.raises(
#         exceptions.MisconfigurationException, match=r"You restored a checkpoint with current_epoch"
#     ):
#         new_trainer.fit(model, datamodule=dm, ckpt_path=checkpoint_filepath, max_epochs=1)


def test_early_stopping_no_extraneous_invocations(tmp_path):
    """Test to ensure that callback methods aren't being invoked outside of the callback handler."""
    pytest.importorskip("sklearn")

    model = helpers.ClassificationModel()
    dm = helpers.ClassifDataModule()
    early_stop_listener = listeners.EarlyStopping(monitor="train_loss")
    early_stop_listener._run_early_stopping_check = Mock()
    expected_count = 4
    trainer = reax.Trainer(
        default_root_dir=tmp_path,
        listeners=[early_stop_listener],
        enable_checkpointing=False,
    )
    trainer.fit(
        model,
        datamodule=dm,
        limit_train_batches=4,
        limit_val_batches=4,
        max_epochs=expected_count,
    )

    assert trainer.early_stopping_listener == early_stop_listener
    assert trainer.early_stopping_listeners == [early_stop_listener]
    assert early_stop_listener._run_early_stopping_check.call_count == expected_count


@pytest.mark.parametrize(
    ("loss_values", "patience", "expected_stop_epoch"),
    [([6, 5, 5, 5, 5, 5], 3, 4), ([6, 5, 4, 4, 3, 3], 1, 3), ([6, 5, 6, 5, 5, 5], 3, 4)],
)
def test_early_stopping_patience(
    tmp_path, loss_values: list, patience: int, expected_stop_epoch: int
):
    """Test to ensure that early stopping is not triggered before patience is exhausted."""

    class ModelOverrideValidationReturn(boring_classes.BoringModel):
        validation_return_values = jnp.array(loss_values)

        @override
        def on_validation_epoch_end(self, stage: "reax.stages.Train", *_) -> None:
            loss = self.validation_return_values[self._trainer.current_epoch]
            self.log("test_val_loss", loss)

    model = ModelOverrideValidationReturn()
    early_stop_listener = listeners.EarlyStopping(
        monitor="test_val_loss", patience=patience, verbose=True
    )
    trainer = reax.Trainer(
        default_root_dir=tmp_path, listeners=[early_stop_listener], enable_progress_bar=False
    )
    trainer.fit(model, num_sanity_val_steps=0, max_epochs=10)
    assert trainer.current_epoch - 1 == expected_stop_epoch


@pytest.mark.parametrize("validation_step_none", [True, False])
@pytest.mark.parametrize(
    ("loss_values", "patience", "expected_stop_epoch"),
    [([6, 5, 5, 5, 5, 5], 3, 4), ([6, 5, 4, 4, 3, 3], 1, 3), ([6, 5, 6, 5, 5, 5], 3, 4)],
)
def test_early_stopping_patience_train(
    tmp_path, validation_step_none: bool, loss_values: list, patience: int, expected_stop_epoch: int
):
    """Test to ensure that early stopping is not triggered before patience is exhausted."""

    class ModelOverrideTrainReturn(boring_classes.BoringModel):
        train_return_values = jnp.array(loss_values)

        def on_train_epoch_end(self, *_):
            loss = self.train_return_values[self.current_epoch]
            self.log("train_loss", loss)

    model = ModelOverrideTrainReturn()

    if validation_step_none:
        model.validation_step = None

    early_stop_listener = listeners.EarlyStopping(
        monitor="train_loss", patience=patience, verbose=True, check_on_train_epoch_end=True
    )
    trainer = reax.Trainer(
        default_root_dir=tmp_path,
        listeners=[early_stop_listener],
        enable_progress_bar=False,
    )
    trainer.fit(
        model,
        num_sanity_val_steps=0,
        max_epochs=10,
    )
    assert trainer.current_epoch - 1 == expected_stop_epoch


def test_pickling():
    early_stopping = listeners.EarlyStopping(monitor="foo")

    early_stopping_pickled = pickle.dumps(early_stopping)
    early_stopping_loaded = pickle.loads(early_stopping_pickled)
    assert vars(early_stopping) == vars(early_stopping_loaded)

    early_stopping_pickled = cloudpickle.dumps(early_stopping)
    early_stopping_loaded = cloudpickle.loads(early_stopping_pickled)
    assert vars(early_stopping) == vars(early_stopping_loaded)


def test_early_stopping_no_val_step(tmp_path):
    """Test that early stopping listener falls back to training metrics when no validation defined."""
    pytest.importorskip("sklearn")

    max_epochs = 4
    model = helpers.ClassificationModel()
    dm = helpers.ClassifDataModule()
    model.validation_step = None
    model.val_dataloader = None

    stopping = listeners.EarlyStopping(
        monitor="train_loss", min_delta=0.1, patience=0, check_on_train_epoch_end=True
    )
    trainer = reax.Trainer(default_root_dir=tmp_path, listeners=[stopping])
    trainer.fit(
        model,
        datamodule=dm,
        # overfit_batches=0.20, # todo: add back
        max_epochs=max_epochs,
    )

    assert trainer.current_epoch < max_epochs - 1
    # todo: add back assert trainer.state.finished, f"Training failed with {trainer.state}"


@pytest.mark.parametrize(
    ("stopping_threshold", "divergence_threshold", "losses", "expected_epoch"),
    [
        (None, None, [8, 4, 2, 3, 4, 5, 8, 10], 5),
        (2.9, None, [9, 8, 7, 6, 5, 6, 4, 3, 2, 1], 8),
        (None, 15.9, [9, 4, 2, 16, 32, 64], 3),
    ],
)
def test_early_stopping_thresholds(
    tmp_path, stopping_threshold, divergence_threshold, losses, expected_epoch
):
    class CurrentModel(boring_classes.BoringModel):
        def on_validation_epoch_end(self, *_):
            val_loss = losses[self.current_epoch]
            self.log("abc", val_loss)

    model = CurrentModel()
    early_stopping = listeners.EarlyStopping(
        monitor="abc",
        stopping_threshold=stopping_threshold,
        divergence_threshold=divergence_threshold,
    )
    trainer = reax.Trainer(
        default_root_dir=tmp_path,
        listeners=[early_stopping],
    )
    trainer.fit(
        model,
        limit_train_batches=0.2,
        limit_val_batches=0.2,
        max_epochs=20,
    )
    assert trainer.current_epoch - 1 == expected_epoch, "early_stopping failed"


@pytest.mark.parametrize("stop_value", [jnp.array(jnp.inf), jnp.array(jnp.nan)])
def test_early_stopping_on_non_finite_monitor(tmp_path, stop_value):
    losses = [4, 3, stop_value, 2, 1]
    expected_stop_epoch = 2

    class CurrentModel(boring_classes.BoringModel):
        def on_validation_epoch_end(self, *_):
            val_loss = losses[self.current_epoch]
            self.log("val_loss", val_loss)

    model = CurrentModel()
    early_stopping = listeners.EarlyStopping(monitor="val_loss", check_finite=True)
    trainer = reax.Trainer(
        default_root_dir=tmp_path,
        listeners=[early_stopping],
    )
    trainer.fit(
        model,
        limit_train_batches=0.2,
        limit_val_batches=0.2,
        max_epochs=10,
    )
    assert trainer.current_epoch - 1 == expected_stop_epoch
    assert early_stopping.stopped_epoch == expected_stop_epoch


@pytest.mark.parametrize(
    ("limit_train_batches", "min_epochs", "min_steps", "stop_step"),
    [
        # IF `min_steps` was set to a higher value than the `trainer.global_step` when `early_stopping` is being
        # triggered, THEN the trainer should continue until reaching `trainer.global_step == min_steps` and stop
        (3, 0, 10, 10),
        (5, 0, 10, 10),
        # IF `min_epochs` resulted in a higher number of steps than the `trainer.global_step` when `early_stopping` is
        # being triggered, THEN the trainer should continue until reaching
        # `trainer.global_step` == `min_epochs * len(train_dataloader)`
        (3, 2, 0, 6),
        (5, 2, 0, 10),
        # IF both `min_epochs` and `min_steps` are provided and higher than the `trainer.global_step` when
        # `early_stopping` is being triggered, THEN the highest between `min_epochs * len(train_dataloader)` and
        # `min_steps` would be reached
        (3, 1, 10, 10),
        (5, 1, 10, 10),
        (3, 3, 10, 10),
        (5, 3, 10, 15),
    ],
)
def test_min_epochs_min_steps_global_step(
    tmp_path, limit_train_batches, min_epochs, min_steps, stop_step
):
    if min_steps:
        assert limit_train_batches < min_steps

    class TestModel(boring_classes.BoringModel):
        def training_step(self, batch, batch_idx):
            self.log("foo", batch_idx)
            return super().training_step(batch, batch_idx)

    es_listener = listeners.EarlyStopping("foo")
    model = TestModel()
    trainer = reax.Trainer(
        default_root_dir=tmp_path,
        listeners=es_listener,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    expected_epochs = max(math.ceil(min_steps / limit_train_batches), min_epochs)
    # trigger early stopping directly after the first epoch
    side_effect = [(True, "")] * expected_epochs
    with mock.patch.object(es_listener, "_evaluate_stopping_criteria", side_effect=side_effect):
        trainer.fit(
            model,
            limit_val_batches=0,
            limit_train_batches=limit_train_batches,
            min_epochs=min_epochs,
            min_updates=min_steps,
        )

    # epochs continue until min steps are reached
    assert trainer.current_epoch == expected_epochs
    # steps continue until min steps are reached AND the epoch is exhausted
    assert trainer.global_updates == stop_step


def test_early_stopping_mode_options():
    with pytest.raises(TypeError):
        listeners.EarlyStopping(monitor="foo", mode="unknown_option")


class EarlyStoppingModel(boring_classes.BoringModel):
    def __init__(
        self,
        expected_end_epoch: int,
        early_stop_on_train: bool,
        dist_diverge_epoch: int | None = None,
    ):
        super().__init__()
        self.expected_end_epoch = expected_end_epoch
        self.early_stop_on_train = early_stop_on_train
        self.dist_diverge_epoch = dist_diverge_epoch

    def _dist_diverge(self):
        should_diverge = (
            self.dist_diverge_epoch
            and self.current_epoch >= self.dist_diverge_epoch
            and self.trainer.process_index == 0
        )
        return 10 if should_diverge else None

    def _epoch_end(self) -> None:
        losses = [8, 4, 2, 3, 4, 5, 8, 10]
        loss = self._dist_diverge() or losses[self.current_epoch]
        self.log("abc", jnp.array(loss))
        self.log("cba", jnp.array(0))

    def on_train_epoch_end(self):
        if not self.early_stop_on_train:
            return
        self._epoch_end()

    def on_validation_epoch_end(self, *_):
        if self.early_stop_on_train:
            return
        self._epoch_end()

    @override
    def on_fit_end(self, trainer: "reax.Trainer", _stage: "reax.stages.Fit") -> None:
        assert trainer.current_epoch - 1 == self.expected_end_epoch, "Early Stopping Failed"


_ES_CHECK = {"check_on_train_epoch_end": True}
_ES_CHECK_P3 = {"patience": 3, "check_on_train_epoch_end": True}
# todo:
# _SPAWN_MARK = {"marks": RunIf(skip_windows=True)}
#
#
# @pytest.mark.parametrize(
#     (
#         "listeners",
#         "expected_stop_epoch",
#         "check_on_train_epoch_end",
#         "strategy",
#         "devices",
#         "dist_diverge_epoch",
#     ),
#     [
#         (
#             [listeners.EarlyStopping("abc"), listeners.EarlyStopping("cba", patience=3)],
#             3,
#             False,
#             "auto",
#             1,
#             None,
#         ),
#         (
#             [listeners.EarlyStopping("cba", patience=3), listeners.EarlyStopping("abc")],
#             3,
#             False,
#             "auto",
#             1,
#             None,
#         ),
#         pytest.param(
#             [listeners.EarlyStopping("abc", patience=1), listeners.EarlyStopping("cba")],
#             2,
#             False,
#             "ddp_spawn",
#             2,
#             2,
#             **_SPAWN_MARK,
#         ),
#         pytest.param(
#             [listeners.EarlyStopping("abc"), listeners.EarlyStopping("cba", patience=3)],
#             3,
#             False,
#             "ddp_spawn",
#             2,
#             None,
#             **_SPAWN_MARK,
#         ),
#         pytest.param(
#             [listeners.EarlyStopping("cba", patience=3), listeners.EarlyStopping("abc")],
#             3,
#             False,
#             "ddp_spawn",
#             2,
#             None,
#             **_SPAWN_MARK,
#         ),
#         (
#             [
#                 listeners.EarlyStopping("abc", **_ES_CHECK),
#                 listeners.EarlyStopping("cba", **_ES_CHECK_P3),
#             ],
#             3,
#             True,
#             "auto",
#             1,
#             None,
#         ),
#         (
#             [
#                 listeners.EarlyStopping("cba", **_ES_CHECK_P3),
#                 listeners.EarlyStopping("abc", **_ES_CHECK),
#             ],
#             3,
#             True,
#             "auto",
#             1,
#             None,
#         ),
#         pytest.param(
#             [
#                 listeners.EarlyStopping("abc", **_ES_CHECK),
#                 listeners.EarlyStopping("cba", **_ES_CHECK_P3),
#             ],
#             3,
#             True,
#             "ddp_spawn",
#             2,
#             None,
#             **_SPAWN_MARK,
#         ),
#         pytest.param(
#             [
#                 listeners.EarlyStopping("cba", **_ES_CHECK_P3),
#                 listeners.EarlyStopping("abc", **_ES_CHECK),
#             ],
#             3,
#             True,
#             "ddp_spawn",
#             2,
#             None,
#             **_SPAWN_MARK,
#         ),
#     ],
# )
# def test_multiple_early_stopping_listeners(
#     tmp_path,
#     listeners: list[listeners.EarlyStopping],
#     expected_stop_epoch: int,
#     check_on_train_epoch_end: bool,
#     strategy: str,
#     devices: int,
#     dist_diverge_epoch: Optional[int],
# ):
#     """Ensure when using multiple early stopping listeners we stop if any signals we should stop."""
#
#     model = EarlyStoppingModel(
#         expected_stop_epoch, check_on_train_epoch_end, dist_diverge_epoch=dist_diverge_epoch
#     )
#
#     trainer = reax.Trainer(
#         model,
#         default_root_dir=tmp_path,
#         listeners=listeners,
#         strategy=strategy,
#         accelerator="cpu",
#         devices=devices,
#     )
#     trainer.fit(
#         limit_train_batches=0.1,
#         limit_val_batches=0.1,
#         max_epochs=20,
#     )
#
#
# @pytest.mark.parametrize(
#     "case",
#     {
#         "val_check_interval": {
#             "val_check_interval": 0.3,
#             "limit_train_batches": 10,
#             "max_epochs": 10,
#         },
#         "check_val_every_n_epoch": {"check_val_every_n_epoch": 2, "max_epochs": 5},
#     }.items(),
# )
# def test_check_on_train_epoch_end_smart_handling(tmp_path, case):
#     class TestModel(boring_classes.BoringModel):
#         def validation_step(self, batch, batch_idx):
#             self.log("foo", 1)
#             return super().validation_step(batch, batch_idx)
#
#     case, kwargs = case
#     model = TestModel()
#     trainer = reax.Trainer(
#         default_root_dir=tmp_path,
#         listeners=listeners.EarlyStopping(monitor="foo"),
#         enable_progress_bar=False,
#     )
#
#     side_effect = [(False, "A"), (True, "B")]
#     with mock.patch(
#         "reax.listeners.EarlyStopping._evaluate_stopping_criteria",
#         side_effect=side_effect,
#     ) as es_mock:
#         fit = trainer.fit(
#             model,
#             limit_val_batches=1,
#             **kwargs,
#         )
#
#     assert es_mock.call_count == len(side_effect)
#     if case == "val_check_interval":
#         assert trainer.global_updates == len(side_effect) * int(
#             fit.limit_train_batches * fit.val_check_interval
#         )
#     else:
#         assert trainer.current_epoch == len(side_effect) * fit.check_val_every_n_epoch


def test_early_stopping_squeezes():
    early_stopping = listeners.EarlyStopping(monitor="foo")
    trainer = reax.Trainer()
    trainer.listener_metrics["foo"] = jnp.array([[[0]]])

    with mock.patch(
        "reax.listeners.EarlyStopping._evaluate_stopping_criteria",
        return_value=(False, ""),
    ) as es_mock:
        stage = mock.Mock(spec=reax.stages.EpochStage, fast_dev_run=False)
        early_stopping._run_early_stopping_check(trainer, stage)

    es_mock.assert_called_once_with(jnp.array(0))


@pytest.mark.parametrize(
    ("log_rank_zero_only", "process_count", "process_index", "expected_log"),
    [
        (False, 1, 0, "bar"),
        (False, 2, 0, "[rank: 0] bar"),
        (False, 2, 1, "[rank: 1] bar"),
        (True, 1, 0, "bar"),
        (True, 2, 0, "[rank: 0] bar"),
        (True, 2, 1, None),
    ],
)
def test_early_stopping_log_info(log_rank_zero_only, process_count, process_index, expected_log):
    """Checks if log.info() gets called with expected message when used within EarlyStopping."""
    # set the process_index and process_count if trainer is not None
    # or else always expect the simple logging message
    trainer = Mock(process_index=process_index, process_count=process_count)

    with mock.patch("reax.listeners.early_stopping._LOGGER.info") as log_mock:
        listeners.EarlyStopping._log_info(trainer, "bar", log_rank_zero_only)

    # check log.info() was called or not with expected arg
    if expected_log:
        log_mock.assert_called_once_with(expected_log)
    else:
        log_mock.assert_not_called()
