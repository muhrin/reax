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
# https://github.com/Lightning-AI/pytorch-lightning/blob/0324a20f00235c7a10a235a44326811ba42b6ae4/tests/tests_pytorch/loggers/test_tensorboard.py
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
import argparse
import os
from unittest import mock

import jax.numpy as jnp
import numpy as np
import pytest
import tensorboardX

import reax
from reax import demos, loggers
import reax.loggers.tensorboard


def test_tensorboard_automatic_versioning(tmp_path):
    """Verify that automatic versioning works."""
    root_dir = tmp_path / "tb_versioning"
    root_dir.mkdir()
    (root_dir / "version_0").mkdir()
    (root_dir / "version_1").mkdir()
    (root_dir / "version_nonumber").mkdir()
    (root_dir / "other").mkdir()

    logger = loggers.tensorboard.TensorBoardLogger(log_dir=tmp_path, name="tb_versioning")
    assert logger.version == 2


def test_tensorboard_manual_versioning(tmp_path):
    """Verify that manual versioning works."""
    root_dir = tmp_path / "tb_versioning"
    root_dir.mkdir()
    (root_dir / "version_0").mkdir()
    (root_dir / "version_1").mkdir()
    (root_dir / "version_2").mkdir()

    logger = loggers.tensorboard.TensorBoardLogger(
        log_dir=tmp_path, name="tb_versioning", version=1
    )

    assert logger.version == 1


def test_tensorboard_named_version(tmp_path):
    """Verify that manual versioning works for string versions, e.g. '2020-02-05-162402'."""
    name = "tb_versioning"
    (tmp_path / name).mkdir()
    expected_version = "2020-02-05-162402"

    logger = loggers.tensorboard.TensorBoardLogger(
        log_dir=tmp_path, name=name, version=expected_version
    )
    logger.log_hyperparams({"a": 1, "b": 2, 123: 3, 3.5: 4, 5j: 5})  # Force data to be written

    assert logger.version == expected_version
    assert os.listdir(tmp_path / name) == [expected_version]
    assert os.listdir(tmp_path / name / expected_version)


@pytest.mark.parametrize("name", ["", None])
def test_tensorboard_no_name(tmp_path, name):
    """Verify that None or empty name works."""
    logger = loggers.tensorboard.TensorBoardLogger(log_dir=tmp_path, name=name)
    logger.log_hyperparams({"a": 1, "b": 2, 123: 3, 3.5: 4, 5j: 5})  # Force data to be written
    assert os.path.normpath(logger.root_dir) == str(
        tmp_path
    )  # use os.path.normpath to handle trailing /
    assert os.listdir(tmp_path / "version_0")


def test_tensorboard_log_sub_dir(tmp_path):
    class TestLogger(loggers.tensorboard.TensorBoardLogger):
        # for reproducibility
        @property
        def version(self):
            return "version"

        @property
        def name(self):
            return "name"

    trainer_args = {"default_root_dir": tmp_path}

    # no sub_dir specified
    save_dir = tmp_path / "logs"
    logger = TestLogger(save_dir)
    trainer = reax.Trainer(**trainer_args, logger=logger)
    assert trainer.logger.log_dir == os.path.join(save_dir, "name", "version")

    # sub_dir specified
    logger = TestLogger(save_dir, sub_dir="sub_dir")
    trainer = reax.Trainer(**trainer_args, logger=logger)
    assert trainer.logger.log_dir == os.path.join(save_dir, "name", "version", "sub_dir")

    # test home dir (`~`) handling
    save_dir = "~/tmp"
    explicit_save_dir = os.path.expanduser(save_dir)
    logger = TestLogger(save_dir, sub_dir="sub_dir")
    trainer = reax.Trainer(**trainer_args, logger=logger)
    assert trainer.logger.log_dir == os.path.join(explicit_save_dir, "name", "version", "sub_dir")

    with mock.patch.dict(os.environ, {}):
        # test env var (`$`) handling
        test_env_dir = "some_directory"
        os.environ["TEST_ENV_DIR"] = test_env_dir
        save_dir = "$TEST_ENV_DIR/tmp"
        explicit_save_dir = f"{test_env_dir}/tmp"
        logger = TestLogger(save_dir, sub_dir="sub_dir")
        trainer = reax.Trainer(**trainer_args, logger=logger)
        assert trainer.logger.log_dir == os.path.join(
            explicit_save_dir, "name", "version", "sub_dir"
        )


@pytest.mark.parametrize("step_idx", [10, None])
def test_tensorboard_log_metrics(tmp_path, step_idx):
    logger = loggers.tensorboard.TensorBoardLogger(tmp_path)
    metrics = {
        "float": 0.3,
        "int": 1,
        "FloatTensor": jnp.array(0.1),
        "IntTensor": jnp.array(1),
    }
    logger.log_metrics(metrics, step_idx)


def test_tensorboard_log_hyperparams(tmp_path):
    logger = loggers.tensorboard.TensorBoardLogger(tmp_path)
    hparams = {
        "float": 0.3,
        "int": 1,
        "string": "abc",
        "bool": True,
        "dict": {"a": {"b": "c"}},
        "list": [1, 2, 3],
        "namespace": argparse.Namespace(foo=argparse.Namespace(bar="buzz")),
        # "layer": torch.nn.BatchNorm1d,
        "tensor": jnp.empty((2, 2, 2)),
        "array": np.empty([2, 2, 2]),
    }
    logger.log_hyperparams(hparams)


def test_tensorboard_log_hparams_and_metrics(tmp_path):
    logger = loggers.tensorboard.TensorBoardLogger(tmp_path, default_hp_metric=False)
    hparams = {
        "float": 0.3,
        "int": 1,
        "string": "abc",
        "bool": True,
        "dict": {"a": {"b": "c"}},
        "list": [1, 2, 3],
        "namespace": argparse.Namespace(foo=argparse.Namespace(bar="buzz")),
        # "layer": torch.nn.BatchNorm1d,
        "array": jnp.empty((2, 2, 2)),
    }
    metrics = {"abc": jnp.array([0.54])}
    logger.log_hyperparams(hparams, metrics)


def test_tensorboard_log_omegaconf_hparams_and_metrics(tmp_path):
    omegaconf = pytest.importorskip("omegaconf")

    logger = loggers.tensorboard.TensorBoardLogger(tmp_path, default_hp_metric=False)
    hparams = {
        "float": 0.3,
        "int": 1,
        "string": "abc",
        "bool": True,
        "dict": {"a": {"b": "c"}},
        "list": [1, 2, 3],
    }
    hparams = omegaconf.OmegaConf.create(hparams)

    metrics = {"abc": jnp.array([0.54])}
    logger.log_hyperparams(hparams, metrics)


@pytest.mark.parametrize("example_input_array", [None, np.random.rand(2, 32)])
def test_tensorboard_log_graph(tmp_path, example_input_array):
    """
    Test that log graph works with both model.example_input_array and if array is passed externally.
    """
    model = demos.BoringModel()
    if example_input_array is not None:
        model.example_input_array = None

    logger = loggers.tensorboard.TensorBoardLogger(tmp_path, log_graph=True)
    logger.log_graph(model, example_input_array)


@pytest.mark.skip(reason="Graph logging is not supported yet")
def test_tensorboard_log_graph_warning_no_example_input_array(tmp_path):
    """Test that log graph throws warning if model.example_input_array is None."""
    model = demos.BoringModel()
    model.example_input_array = None
    logger = loggers.tensorboard.TensorBoardLogger(tmp_path, log_graph=True)
    with pytest.warns(
        UserWarning,
        match="Could not log computational graph to TensorBoard: The `model.example_input_array` .* was not given",
    ):
        logger.log_graph(model)

    model.example_input_array = {"x": 1, "y": 2}
    with pytest.warns(
        UserWarning,
        match="Could not log computational graph to TensorBoard: .* can't be traced by TensorBoard",
    ):
        logger.log_graph(model)


@pytest.mark.skip(reason="Gradient accumulation is currently not supported")
@mock.patch("reax.loggers.tensorboard.TensorBoardLogger.log_metrics")
def test_tensorboard_with_accummulated_gradients(mock_log_metrics, tmp_path):
    """Tests to ensure that tensorboard log properly when accumulated_gradients > 1."""

    class TestModel(demos.BoringModel):
        def __init__(self):
            super().__init__()
            self.indexes = []

        def training_step(self, *args):
            self.log("foo", 1, on_step=True, on_epoch=True)
            # if (
            #     not self.trainer.fit_loop._should_accumulate()
            #     and self.trainer._logger_connector.should_update_logs
            # ):
            #     self.indexes.append(self.trainer.global_step)
            return super().training_step(*args)

    model = TestModel()
    logger_0 = loggers.tensorboard.TensorBoardLogger(tmp_path, default_hp_metric=False)
    trainer = reax.Trainer(
        default_root_dir=tmp_path,
        logger=[logger_0],
        log_every_n_steps=3,
    )
    trainer.fit(
        model,
        max_epochs=3,
        limit_train_batches=12,
        limit_val_batches=0,
        accumulate_grad_batches=2,
    )

    calls = [m[2] for m in mock_log_metrics.mock_calls]
    count_epochs = [c["step"] for c in calls if "foo_epoch" in c["metrics"]]
    assert count_epochs == [5, 11, 17]

    count_steps = [c["step"] for c in calls if "foo_step" in c["metrics"]]
    assert count_steps == model.indexes


def test_tensorboard_finalize(monkeypatch, tmp_path):
    """Test that the SummaryWriter closes in finalize."""
    monkeypatch.setattr(tensorboardX, "SummaryWriter", mock.Mock())
    logger = loggers.tensorboard.TensorBoardLogger(log_dir=tmp_path)
    assert logger._exp is None
    logger.finalize("any")

    # no log calls, no experiment created -> nothing to flush
    logger.experiment.assert_not_called()

    logger = loggers.tensorboard.TensorBoardLogger(log_dir=tmp_path)
    logger.log_metrics({"flush_me": 11.1})  # trigger creation of an experiment
    logger.finalize("any")

    # finalize flushes to experiment directory
    logger.experiment.flush.assert_called()
    logger.experiment.close.assert_called()


@pytest.mark.skip(reason="Saving hparams is currently not supported")
def test_tensorboard_save_hparams_to_yaml_once(tmp_path):
    model = demos.BoringModel()
    logger = loggers.tensorboard.TensorBoardLogger(log_dir=tmp_path, default_hp_metric=False)
    trainer = reax.Trainer(default_root_dir=tmp_path, logger=logger)
    assert trainer.log_dir == trainer.logger.log_dir
    trainer.fit(model, max_updates=1)

    hparams_file = "hparams.yaml"
    assert os.path.isfile(os.path.join(trainer.log_dir, hparams_file))
    assert not os.path.isfile(os.path.join(tmp_path, hparams_file))


def test_tensorboard_with_symlink(tmp_path, monkeypatch):
    """
    Tests a specific failure case when tensorboard logger is used with empty name, symbolic link
    ``log_dir``, and relative paths.
    """
    monkeypatch.chdir(tmp_path)  # need to use relative paths
    source = os.path.join(".", "lightning_logs")
    dest = os.path.join(".", "sym_lightning_logs")

    os.symlink(source, dest)

    logger = loggers.tensorboard.TensorBoardLogger(log_dir=dest, name="")
    _ = logger.version
