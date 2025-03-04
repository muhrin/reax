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
# https://github.com/Lightning-AI/pytorch-lightning/blob/9177ec09caadcf88859e1f1e3e10a18e8832069a/tests/tests_pytorch/loggers/test_csv.py
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
import os
from unittest import mock
from unittest.mock import MagicMock

import fsspec
import pytest
import torch

import reax
from reax import loggers, saving
from reax.loggers import csv_logs

from .. import helpers


def test_automatic_versioning(tmp_path):
    """Verify that automatic versioning works."""
    (tmp_path / "exp" / "version_0").mkdir(parents=True)
    (tmp_path / "exp" / "version_1").mkdir()
    (tmp_path / "exp" / "version_nonumber").mkdir()
    (tmp_path / "exp" / "other").mkdir()

    logger = loggers.CsvLogger(save_dir=tmp_path, name="exp")

    assert logger.version == 2


def test_manual_versioning(tmp_path):
    """Verify that manual versioning works."""
    root_dir = tmp_path / "exp"
    (root_dir / "version_0").mkdir(parents=True)
    (root_dir / "version_1").mkdir()
    (root_dir / "version_2").mkdir()

    logger = loggers.CsvLogger(save_dir=tmp_path, name="exp", version=1)

    assert logger.version == 1


def test_manual_versioning_file_exists(tmp_path):
    """Test that a warning is emitted and existing files get overwritten."""

    # Simulate an existing 'version_0' vrom a previous run
    (tmp_path / "exp" / "version_0").mkdir(parents=True)
    previous_metrics_file = tmp_path / "exp" / "version_0" / "metrics.csv"
    previous_metrics_file.touch()

    logger = loggers.CsvLogger(save_dir=tmp_path, name="exp", version=0)
    assert previous_metrics_file.exists()
    with pytest.warns(UserWarning, match="Experiment logs directory .* exists and is not empty"):
        _ = logger.experiment
    assert not previous_metrics_file.exists()


def test_named_version(tmp_path):
    """Verify that manual versioning works for string versions, e.g. '2020-02-05-162402'."""
    exp_name = "exp"
    (tmp_path / exp_name).mkdir()
    expected_version = "2020-02-05-162402"

    logger = loggers.CsvLogger(save_dir=tmp_path, name=exp_name, version=expected_version)
    logger.log_hyperparams({"a": 1, "b": 2})
    logger.save()
    assert logger.version == expected_version
    assert os.listdir(tmp_path / exp_name) == [expected_version]
    assert os.listdir(tmp_path / exp_name / expected_version)


@pytest.mark.parametrize("name", ["", None])
def test_no_name(tmp_path, name):
    """Verify that None or empty name works."""
    logger = loggers.CsvLogger(save_dir=tmp_path, name=name)
    logger.save()
    assert os.path.normpath(logger.root_dir) == str(
        tmp_path
    )  # use os.path.normpath to handle trailing /
    assert os.listdir(tmp_path / "version_0")


@pytest.mark.parametrize("step_idx", [10, None])
def test_log_metrics(tmp_path, step_idx):
    logger = loggers.CsvLogger(tmp_path)
    metrics = {
        "float": 0.3,
        "int": 1,
        "FloatTensor": torch.tensor(0.1),
        "IntTensor": torch.tensor(1),
    }
    logger.log_metrics(metrics, step_idx)
    logger.save()

    path_csv = os.path.join(logger.log_dir, csv_logs.ExperimentWriter.NAME_METRICS_FILE)
    with open(path_csv) as fp:
        lines = fp.readlines()
    assert len(lines) == 2
    assert all(n in lines[0] for n in metrics)


def test_log_hyperparams(tmp_path):
    logger = loggers.CsvLogger(tmp_path)
    hparams = {
        "float": 0.3,
        "int": 1,
        "string": "abc",
        "bool": True,
        "dict": {"a": {"b": "c"}},
        "list": [1, 2, 3],
        "layer": torch.nn.BatchNorm1d,
    }
    logger.log_hyperparams(hparams)
    logger.save()

    path_yaml = os.path.join(logger.log_dir, csv_logs.ExperimentWriter.NAME_HPARAMS_FILE)
    params = saving.load_hparams_from_yaml(path_yaml)
    assert all(n in params for n in hparams)


def test_fit_csv_logger(tmp_path):
    pytest.importorskip("sklearn")

    dm = helpers.ClassifDataModule()
    model = helpers.ClassificationModel()
    logger = loggers.CsvLogger(save_dir=tmp_path)
    trainer = reax.Trainer(default_root_dir=tmp_path, logger=logger, log_every_n_steps=1)
    trainer.fit(model, datamodule=dm, max_updates=10)
    metrics_file = os.path.join(logger.log_dir, csv_logs.ExperimentWriter.NAME_METRICS_FILE)
    assert os.path.isfile(metrics_file)


def test_csv_logger_remotefs():
    logger = loggers.CsvLogger(save_dir="memory://test_fit_csv_logger_remotefs")
    fs, _ = fsspec.core.url_to_fs("memory://test_fit_csv_logger_remotefs")
    exp = logger.experiment
    exp.log_metrics({"loss": 0.1})
    exp.save()
    metrics_file = os.path.join(logger.log_dir, csv_logs.ExperimentWriter.NAME_METRICS_FILE)
    assert fs.isfile(metrics_file)


def test_flush_n_steps(tmp_path):
    logger = loggers.CsvLogger(tmp_path, flush_logs_every_n_steps=2)
    metrics = {
        "float": 0.3,
        "int": 1,
        "FloatTensor": torch.tensor(0.1),
        "IntTensor": torch.tensor(1),
    }
    logger.save = MagicMock()
    logger.log_metrics(metrics, step=0)

    logger.save.assert_not_called()
    logger.log_metrics(metrics, step=1)
    logger.save.assert_called_once()


def test_metrics_reset_after_save(tmp_path):
    logger = loggers.CsvLogger(tmp_path, flush_logs_every_n_steps=2)
    metrics = {"test": 1}
    logger.log_metrics(metrics, step=0)
    assert logger.experiment.metrics
    logger.log_metrics(metrics, step=1)  # flush triggered
    assert not logger.experiment.metrics


@mock.patch(
    # Mock the existence check, so we can simulate appending to the metrics file
    "reax.loggers.csv_logs.ExperimentWriter._check_log_dir_exists"
)
def test_append_metrics_file(_, tmp_path):
    """Test that the logger appends to the file instead of rewriting it on every save."""
    logger = loggers.CsvLogger(tmp_path, name="test", version=0, flush_logs_every_n_steps=1)

    # initial metrics
    logger.log_metrics({"a": 1, "b": 2})
    logger.log_metrics({"a": 3, "b": 4})

    # create a new logger to show we append to the existing file
    logger = loggers.CsvLogger(tmp_path, name="test", version=0, flush_logs_every_n_steps=1)
    logger.log_metrics({"a": 100, "b": 200})

    with open(logger.experiment.metrics_file_path) as file:
        lines = file.readlines()
    assert len(lines) == 4  # 1 header + 3 lines of metrics


def test_append_columns(tmp_path):
    """Test that the CSV file gets rewritten with new headers if the columns change."""
    logger = loggers.CsvLogger(tmp_path, flush_logs_every_n_steps=1)

    # initial metrics
    logger.log_metrics({"a": 1, "b": 2})

    # new key appears
    logger.log_metrics({"a": 1, "b": 2, "c": 3})
    with open(logger.experiment.metrics_file_path) as file:
        header = file.readline().strip()
        assert set(header.split(",")) == {"step", "a", "b", "c"}

    # key disappears
    logger.log_metrics({"a": 1, "c": 3})
    with open(logger.experiment.metrics_file_path) as file:
        header = file.readline().strip()
        assert set(header.split(",")) == {"step", "a", "b", "c"}
