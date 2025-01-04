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
# https://github.com/Lightning-AI/pytorch-lightning/blob/0324a20f00235c7a10a235a44326811ba42b6ae4/src/lightning/pytorch/loggers/tensorboard.py
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
TensorBoard Logger
------------------
"""

import argparse
import os
from typing import Any, Callable, Final, Mapping, Optional, Union

import fsspec
import jax.typing
import numpy as np
import tensorboardX
from typing_extensions import override

try:
    import omegaconf
except ImportError:
    omegaconf = None

from reax import typing
from reax.lightning import rank_zero

from . import _utils, logger

__all__ = ("TensorBoardLogger",)


class TensorBoardLogger(logger.WithDdp["tensorboardX.SummaryWriter"], logger.Logger):
    r"""Log to local or remote file system in
    `TensorBoard <https://www.tensorflow.org/tensorboard>`_ format.

    Implemented using :class:`~tensorboardX.SummaryWriter`. Logs are saved to
    ``os.path.join(save_dir, name, version)``. This is the default logger in Lightning, it comes
    preinstalled.

    This logger supports logging to remote filesystems via ``fsspec``. Make sure you have it
    installed and you don't have tensorflow (otherwise it will use tf.io.gfile instead of fsspec).

    Example:

    .. testcode::
        :skipif: not _TENSORBOARD_AVAILABLE or not _TENSORBOARDX_AVAILABLE

        from reax import Trainer
        from reax.loggers import TensorBoardLogger

        logger = TensorBoardLogger("tb_logs", name="my_model")
        trainer = Trainer(logger=logger)
    """

    LOGGER_JOIN_CHAR: Final[str] = "-"
    NAME_HPARAMS_FILE: Final[str] = "hparams.yaml"

    def __init__(
        self,
        log_dir: typing.Path,
        name: Optional[str] = "reax_logs",
        *,
        version: Optional[Union[int, str]] = None,
        log_graph: bool = False,
        default_hp_metric: bool = True,
        prefix: str = "",
        sub_dir: Optional[typing.Path] = None,
        **kwargs: Any,
    ):
        """Init function.

        :param **kwargs:
        :type **kwargs: Any
        :param log_dir:
        :type log_dir: typing.Path
        :param save_dir: Save directory.
        :param name: Experiment name.  If it is the empty string then no per-experiment
            subdirectory is used, defaults to "reax_logs".
        :type name: Optional[str], optional
        :param version: Experiment version. If version is not specified the logger inspects the save
            directory for existing versions, then automatically assigns the next available version.
            If it is a string then it is used as the run-specific subdirectory name,
            otherwise ``'version_${version}'`` is used, defaults to None.
        :type version: Optional[Union[int, str]], optional
        :param log_graph: Adds the computational graph to tensorboard. This requires that
            the user has defined the `self.example_input_array` attribute in their
            model, defaults to False.
        :type log_graph: bool, optional
        :param default_hp_metric: Enables a placeholder metric with key `hp_metric` when
            `log_hyperparams` is called without a metric (otherwise calls to log_hyperparams
            without a metric are ignored), defaults to True.
        :type default_hp_metric: bool, optional
        :param prefix: A string to put at the beginning of metric keys, defaults to "".
        :type prefix: str, optional
        :param sub_dir: Sub-directory to group TensorBoard logs. If a sub_dir argument is passed
            then logs are saved in ``/save_dir/name/version/sub_dir/``.
            logs are saved in ``/save_dir/name/version/``, defaults to None.
        :type sub_dir: Optional[typing.Path], optional
        :param kwargs: Additional arguments used by :class:`tensorboardX.SummaryWriter` can be
            passed as keyword arguments in this logger. To automatically flush to disk, `max_queue`
            sets the sizeof the queue for pending logs before flushing. `flush_secs` determines how
            many seconds elapses before flushing.
        """
        super().__init__()
        self._root_dir = os.fspath(log_dir)
        self._name = name or ""
        self._version = version
        self._sub_dir = None if sub_dir is None else os.fspath(sub_dir)

        self._default_hp_metric = default_hp_metric
        self._prefix = prefix
        self._fs: fsspec.AbstractFileSystem = fsspec.url_to_fs(log_dir)[0]

        self._exp: Optional["tensorboardX.SummaryWriter"] = None
        self._kwargs = kwargs
        self._should_log_graph = log_graph
        self.hparams: Union[dict[str, Any], argparse.Namespace] = {}

    @property
    @override
    def name(self) -> str:
        """Name function."""
        return self._name

    @property
    @override
    def version(self) -> Union[int, str]:
        """Version function."""
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    @property
    @override
    def root_dir(self) -> str:
        """Parent directory for all tensorboard checkpoint subdirectories.

        If the experiment name parameter is an empty string, no experiment subdirectory is used and
        the checkpoint will be saved in "save_dir/version"
        """
        return os.path.join(self._root_dir, self.name)

    @property
    @override
    def log_dir(self) -> str:
        """The directory for this run's tensorboard checkpoint.

        By default, it is named ``'version_${self.version}'`` but it can be overridden by passing a
        string value for the constructor's version parameter instead of ``None`` or an int.
        """
        # create a pseudo standard path ala test-tube
        version = self.version if isinstance(self.version, str) else f"version_{self.version}"
        log_dir = os.path.join(self.root_dir, version)
        if isinstance(self.sub_dir, str):
            log_dir = os.path.join(log_dir, self.sub_dir)

        log_dir = os.path.expandvars(log_dir)
        log_dir = os.path.expanduser(log_dir)
        return log_dir

    @property
    def sub_dir(self) -> Optional[str]:
        """Sub dir."""
        return self._sub_dir

    @property
    @override
    def save_dir(self) -> str:
        """Gets the save directory where the TensorBoard experiments are saved.

        :rtype: The local path to the save directory where the TensorBoard experiments are saved.
        """
        return self._root_dir

    @property
    def _experiment(self) -> Optional["tensorboardX.SummaryWriter"]:
        """Get the tensorboard object."""
        if self._exp is not None:
            return self._exp

        assert rank_zero.rank_zero_only.rank == 0, "tried to init log dirs in non global_rank=0"

        if self.root_dir:
            self._fs.makedirs(self.root_dir, exist_ok=True)

        self._exp = tensorboardX.SummaryWriter(log_dir=self.log_dir, **self._kwargs)
        return self._exp

    @override
    def _log_metrics(
        self, metrics: Mapping[str, jax.typing.ArrayLike], step: Optional[int] = None
    ) -> None:
        """Log metrics."""
        metrics = _utils.add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)

        for key, val in metrics.items():
            if isinstance(val, (jax.Array, np.ndarray)):
                val = val.item()

            if isinstance(val, dict):
                self.experiment.add_scalars(key, val, step)
            else:
                try:
                    self.experiment.add_scalar(key, val, step)
                except Exception as ex:
                    raise ValueError(
                        f"Logging of type `{type(val).__name__}` is not supported. "
                        f"Use `dict`, scalar or array"
                    ) from ex

    @override
    def _log_hyperparams(
        # pylint: disable=arguments-differ
        self,
        params: Union[dict[str, Any], argparse.Namespace],
        metrics: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record hyperparameters.

        TensorBoard logs with and without saved hyperparameters are incompatible, the
        hyperparameters are then not displayed in the TensorBoard. Please delete or move the
        previously saved logs to display the new ones with hyperparameters.

        :param params: A dictionary-like container with the hyperparameters.
        :type params: Union[dict[str, Any], argparse.Namespace]
        :param metrics: Dictionary with metric names as keys and measured quantities as values,
            defaults to None.
        :type metrics: Optional[dict[str, Any]], optional
        :param step: Optional global step number for the logged metrics.
        """
        params = _utils.convert_params(params)

        if omegaconf is not None and isinstance(params, omegaconf.Container):
            # store params to output
            self.hparams = omegaconf.OmegaConf.merge(self.hparams, params)
        else:
            self.hparams.update(params)

        # format params into the suitable for tensorboard
        params = _utils.flatten_dict(params)
        params = self._sanitize_params(params)

        if metrics is None:
            if self._default_hp_metric:
                metrics = {"hp_metric": -1}
        elif not isinstance(metrics, dict):
            metrics = {"hp_metric": metrics}

        if metrics:
            self.log_metrics(metrics, 0)

            exp, ssi, sei = tensorboardX.summary.hparams(params, metrics)
            writer = self.experiment._get_file_writer()  # pylint: disable=protected-access
            writer.add_summary(exp)
            writer.add_summary(ssi)
            writer.add_summary(sei)

    @override
    def _log_graph(
        # pylint: disable=arguments-differ
        self,
        model: Callable,
        *inputs,
    ) -> None:
        """Log graph."""
        if not self._should_log_graph:
            return

        input_array = model.example_input_array if not inputs else inputs

        if input_array is None:
            rank_zero.rank_zero_warn(
                "Could not log computational graph to TensorBoard: The `model.example_input_array` "
                "attribute is not set or `input_array` was not given."
            )
        else:
            # TODO: Complete this
            # comp = jax.jit(model).lower(inputs).compiler_ir("hlo")
            # self.experiment.file_writer.add_graph
            pass

    @override
    def _save(self) -> None:
        """Save function."""
        self.experiment.flush()

    @override
    def _finalize(self, status: str) -> None:
        """Finalize function."""
        if status == "success":
            # saving hparams happens independent of experiment manager
            self.save()

        if self._exp is not None:
            self.experiment.flush()
            self.experiment.close()

    def _get_next_version(self) -> int:
        """Get next version."""
        root_dir = self.root_dir

        try:
            listdir_info = self._fs.listdir(root_dir)
        except OSError:
            return 0

        existing_versions = []
        for info in listdir_info:
            name = info["name"]
            basename = os.path.basename(name)
            if os.path.isdir(name) and basename.startswith("version_"):
                dir_ver = basename.split("_")[1].replace("/", "")
                if dir_ver.isdigit():
                    existing_versions.append(int(dir_ver))
        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1

    @staticmethod
    def _sanitize_params(params: dict[str, Any]) -> dict[str, Any]:
        """Sanitize params."""
        params = _utils.sanitize_params(params)
        # logging of arrays with dimension > 1 is not supported, sanitize as string
        return {
            key: str(val) if hasattr(val, "ndim") and val.ndim > 1 else val
            for key, val in params.items()
        }
