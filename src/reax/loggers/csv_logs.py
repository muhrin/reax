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
# https://github.com/Lightning-AI/pytorch-lightning/blob/9177ec09caadcf88859e1f1e3e10a18e8832069a/src/lightning/fabric/loggers/csv_logs.py
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
"""
CSV logger
----------

CSV logger for basic experiment logging that does not require opening ports

"""

from argparse import Namespace
import csv
import logging
import os
from typing import Any, Mapping, Optional, Union

import fsspec
import jax
from typing_extensions import override

from reax import typing
from reax.lightning import rank_zero

from . import _utils, logger
from .. import saving

_LOGGER = logging.getLogger(__name__)

__all__ = "CsvLogger", "CSVLogger"


class CsvLogger(logger.Logger):
    r"""Log to local file system in yaml and CSV format.

    Logs are saved to ``os.path.join(save_dir, name, version)``.

    Example:
        >>> import reax
        >>> logger = reax.loggers.CsvLogger("logs", name="my_exp_name")
        >>> trainer = reax.Trainer(logger=logger)

    Args:
        save_dir: Save directory
        name: Experiment name, optional. Defaults to ``'reax_logs'``. If name is ``None``, logs
            (versions) will be stored to the save dir directly.
        version: Experiment version. If version is not specified the logger inspects the save
            directory for existing versions, then automatically assigns the next available version.
        prefix: A string to put at the beginning of metric keys.
        flush_logs_every_n_steps: How often to flush logs to disk (defaults to every 100 steps).

    """

    LOGGER_JOIN_CHAR = "-"

    def __init__(
        self,
        save_dir: typing.Path,
        name: Optional[str] = "reax_logs",
        version: Optional[Union[int, str]] = None,
        prefix: str = "",
        flush_logs_every_n_steps: int = 100,
    ):
        super().__init__()
        save_dir = os.fspath(save_dir)
        self._root_dir = save_dir
        self._name = name or ""
        self._version = version
        self._prefix = prefix
        self._fs: fsspec.AbstractFileSystem = fsspec.url_to_fs(save_dir)[0]
        self._experiment: Optional[ExperimentWriter] = None
        self._flush_logs_every_n_steps = flush_logs_every_n_steps

    @property
    @override
    def name(self) -> str:
        """Gets the name of the experiment.

        Returns:
            The name of the experiment.

        """
        return self._name

    @property
    @override
    def version(self) -> Union[int, str]:
        """Gets the version of the experiment.

        Returns:
            The version of the experiment if it is specified, else the next version.

        """
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    @property
    @override
    def root_dir(self) -> str:
        """Gets the save directory where the versioned CSV experiments are saved."""
        return self._root_dir

    @property
    @override
    def log_dir(self) -> str:
        """The log directory for this run.

        By default, it is named ``'version_${self.version}'`` but it can be overridden by passing a
        string value for the constructor's version parameter instead of ``None`` or an int.

        """
        # create a pseudo standard path
        version = self.version if isinstance(self.version, str) else f"version_{self.version}"
        return os.path.join(self._root_dir, self.name, version)

    @property
    @override
    def save_dir(self) -> str:
        """The current directory where logs are saved.

        Returns:
            The path to current directory where logs are saved.

        """
        return self.log_dir

    @property
    @logger.rank_zero_experiment
    def experiment(self) -> "ExperimentWriter":
        """Actual ExperimentWriter object. To use ExperimentWriter features anywhere in your code,
        do the following.

        Example::

            self.logger.experiment.some_experiment_writer_function()

        """
        if self._experiment is not None:
            return self._experiment

        self._fs.makedirs(self.root_dir, exist_ok=True)
        self._experiment = ExperimentWriter(log_dir=self.log_dir)
        return self._experiment

    @override
    @rank_zero.rank_zero_only
    def log_hyperparams(self, params: Union[dict[str, Any], Namespace], *_, **__) -> None:
        params = _utils.convert_params(params)
        self.experiment.log_hparams(params)

    @override
    @rank_zero.rank_zero_only
    def log_metrics(
        self, metrics: Mapping[str, jax.typing.ArrayLike], step: Optional[int] = None
    ) -> None:
        metrics = _utils.add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)
        if step is None:
            step = len(self.experiment.metrics)
        self.experiment.log_metrics(metrics, step)
        if (step + 1) % self._flush_logs_every_n_steps == 0:
            self.save()

    @override
    @rank_zero.rank_zero_only
    def save(self) -> None:
        super().save()
        self.experiment.save()

    @override
    @rank_zero.rank_zero_only
    def finalize(self, status: str) -> None:
        if self._experiment is None:
            # When using multiprocessing, finalize() should be a no-op on the main process, as no
            # experiment has been initialized there
            return
        self.save()

    def _get_next_version(self) -> int:
        versions_root = os.path.join(self._root_dir, self.name)

        if not self._fs.isdir(versions_root):
            return 0

        existing_versions = []
        for d in self._fs.listdir(versions_root):
            full_path = d["name"]
            name = os.path.basename(full_path)
            if self._fs.isdir(full_path) and name.startswith("version_"):
                dir_ver = name.split("_")[1]
                if dir_ver.isdigit():
                    existing_versions.append(int(dir_ver))

        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1


CSVLogger = CsvLogger  # Alias for compatibility with lightning


class ExperimentWriter:
    r"""Experiment writer for CSVLogger.

    Args:
        log_dir: Directory for the experiment logs

    """

    NAME_METRICS_FILE = "metrics.csv"
    NAME_HPARAMS_FILE = "hparams.yaml"

    def __init__(self, log_dir: str) -> None:
        # Params
        self.log_dir = log_dir

        self._fs = fsspec.url_to_fs(log_dir)[0]
        self.metrics_file_path = os.path.join(self.log_dir, self.NAME_METRICS_FILE)

        self._check_log_dir_exists()
        self._fs.makedirs(self.log_dir, exist_ok=True)

        # State
        self.metrics: list[dict[str, float]] = []
        self.metrics_keys: list[str] = []
        self.hparams: dict[str, Any] = {}

    def log_metrics(self, metrics_dict: dict[str, float], step: Optional[int] = None) -> None:
        """Record metrics."""

        def _handle_value(value: Union[jax.Array, Any]) -> Any:
            if isinstance(value, jax.Array):
                return value.item()
            return value

        if step is None:
            step = len(self.metrics)

        metrics = {k: _handle_value(v) for k, v in metrics_dict.items()}
        metrics["step"] = step
        self.metrics.append(metrics)

    def log_hparams(self, params: dict[str, Any]) -> None:
        """Record hparams."""
        self.hparams.update(params)

    def save(self) -> None:
        """Save recorded hparams and metrics into files."""
        hparams_file = os.path.join(self.log_dir, self.NAME_HPARAMS_FILE)
        saving.save_hparams_to_yaml(hparams_file, self.hparams)

        if not self.metrics:
            return

        new_keys = self._record_new_keys()
        file_exists = self._fs.isfile(self.metrics_file_path)

        if new_keys and file_exists:
            # we need to re-write the file if the keys (header) change
            self._rewrite_with_new_header(self.metrics_keys)

        with self._fs.open(
            self.metrics_file_path, mode=("a" if file_exists else "w"), newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=self.metrics_keys)
            if not file_exists:
                # only write the header if we're writing a fresh file
                writer.writeheader()
            writer.writerows(self.metrics)

        self.metrics = []  # reset

    def _record_new_keys(self) -> set[str]:
        """Records new keys that have not been logged before."""
        current_keys = set().union(*self.metrics)
        new_keys = current_keys - set(self.metrics_keys)
        self.metrics_keys.extend(new_keys)
        self.metrics_keys.sort()
        return new_keys

    def _rewrite_with_new_header(self, fieldnames: list[str]) -> None:
        with self._fs.open(self.metrics_file_path, "r", newline="") as file:
            metrics = list(csv.DictReader(file))

        with self._fs.open(self.metrics_file_path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics)

    def _check_log_dir_exists(self) -> None:
        if self._fs.exists(self.log_dir) and self._fs.listdir(self.log_dir):
            rank_zero.rank_zero_warn(
                f"Experiment logs directory {self.log_dir} exists and is not empty."
                " Previous log files in this directory will be deleted when the new ones are saved!"
            )
            if self._fs.isfile(self.metrics_file_path):
                self._fs.rm_file(self.metrics_file_path)
