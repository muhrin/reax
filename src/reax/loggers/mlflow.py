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
# https://github.com/Lightning-AI/pytorch-lightning/blob/0324a20f00235c7a10a235a44326811ba42b6ae4/src/lightning/pytorch/loggers/mlflow.py
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
MLflow Logger
-------------
"""

from argparse import Namespace
from collections.abc import Mapping
import logging
import os
from pathlib import Path
import re
import tempfile
from time import time
from typing import TYPE_CHECKING, Any, Callable, Final, Literal, Optional, Union

import jax
from lightning_utilities.core import imports
from typing_extensions import override
import yaml

from reax.lightning import rank_zero

from . import _utils, logger

if TYPE_CHECKING:
    import mlflow
    import mlflow.tracking.context.registry

    import reax


__all__ = ("MlflowLogger", "MLFlowLogger")

_LOGGER = logging.getLogger(__name__)
LOCAL_FILE_URI_PREFIX = "file:"
_MLFLOW_AVAILABLE = imports.RequirementCache("mlflow>=1.0.0", "mlflow")
_MLFLOW_SYNCHRONOUS_AVAILABLE = imports.RequirementCache("mlflow>=2.8.0", "mlflow")
TagsDict = dict[str, str]


class MlflowLogger(logger.Logger):
    """Log using `MLflow <https://mlflow.org>`_.

    Install it with pip:

    .. code-block:: bash

        pip install mlflow  # or mlflow-skinny

    .. code-block:: python

        from reax import Trainer
        from reax.loggers import MLFlowLogger

        mlf_logger = MLFlowLogger(experiment_name="reax_logs", tracking_uri="file:./ml-runs")
        trainer = Trainer(logger=mlf_logger)

    Use the logger anywhere in your :class:`~reax.Module` as follows:

    .. code-block:: python

        from reax import Module

        class LitModel(Module):
            def training_step(self, batch, batch_idx):
                # example
                self.logger.experiment.whatever_ml_flow_supports(...)

            def any_reax_module_function_or_hook(self):
                self.logger.experiment.whatever_ml_flow_supports(...)
    """

    LOGGER_JOIN_CHAR = "-"

    def __init__(
        self,
        experiment_name: str = "reax_logs",
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = os.getenv("MLFLOW_TRACKING_URI"),
        *,
        tags: Optional[dict[str, Any]] = None,
        save_dir: Optional[str] = "./mlruns",
        log_model: Literal[True, False, "all"] = False,
        prefix: str = "",
        artifact_location: Optional[str] = None,
        run_id: Optional[str] = None,
        synchronous: Optional[bool] = None,
    ):
        """Init function.

        :param experiment_name: The name of the experiment, defaults to "reax_logs".
        :type experiment_name: str, optional
        :param run_name: Name of the new run. The `run_name` is internally stored as a
            ``mlflow.runName`` tag. If the ``mlflow.runName`` tag has already been set in `tags`,
            the value is overridden by the `run_name`, defaults to ``None``.
        :type run_name: Optional[str], optional
        :param tracking_uri: Address of local or remote tracking server.
            If not provided
            back to `file:<save_dir>`, defaults to os.getenv("MLFLOW_TRACKING_URI").
        :type tracking_uri: Optional[str], optional
        :param tags: A dictionary tags for the experiment, defaults to None.
        :type tags: Optional[dict[str, Any]], optional
        :param save_dir: A path to a local directory where the MLflow runs get saved.

            Has no effect if `tracking_uri` is provided, defaults to "./mlruns".
        :type save_dir: Optional[str], optional
        :param log_model: Log checkpoints created by
            :class:`~reax.listeners.model_checkpoint.ModelCheckpoint` as MLFlow artifacts.

            * if ``log_model == 'all'``, checkpoints are logged during training.
            * if ``log_model == True``, checkpoints are logged at the end of training, except when
                :paramref:`~reax.listeners.Checkpointer.save_top_k` ``== -1``
                which also logs every checkpoint during training.
            * if ``log_model == False`` (default), no checkpoint is logged, defaults to False.
        :type log_model: Literal[True, False, "all"], optional
        :param prefix: A string to put at the beginning of metric keys, defaults to "".
        :type prefix: str, optional
        :param artifact_location: The location to store run artifacts. If not provided, the server
            picks an appropriate default, defaults to None.
        :type artifact_location: Optional[str], optional
        :param run_id: The run identifier of the experiment. If not provided, a new run is started,
            defaults to None.
        :type run_id: Optional[str], optional
        :param synchronous: Hints mlflow whether to block the execution for every logging call until
            complete where applicable. Requires mlflow >= 2.8.0, defaults to None.
        :type synchronous: Optional[bool], optional
        :raises ModuleNotFoundError: If required MLFlow package is not installed on the device.
        """
        if not _MLFLOW_AVAILABLE:
            raise ModuleNotFoundError(str(_MLFLOW_AVAILABLE))
        if synchronous is not None and not _MLFLOW_SYNCHRONOUS_AVAILABLE:
            raise ModuleNotFoundError("`synchronous` requires mlflow>=2.8.0")
        super().__init__()
        if not tracking_uri:
            tracking_uri = f"{LOCAL_FILE_URI_PREFIX}{save_dir}"

        # Params
        self._experiment_name: Final[str] = experiment_name
        self._prefix: Final[str] = prefix
        self._log_model: Final[Literal[True, False, "all"]] = log_model
        self._artifact_location: Final[Optional[str]] = artifact_location

        # State
        self._experiment_id: Optional[str] = None
        self._tracking_uri = tracking_uri
        self._run_name = run_name
        self._run_id = run_id
        self.tags = tags
        self._logged_model_time: dict[str, float] = {}
        self._checkpoint_callback: Optional[reax.listeners.ModelCheckpoint] = None
        self._log_batch_kwargs = {} if synchronous is None else {"synchronous": synchronous}
        self._initialized = False
        self._warning_cache = rank_zero.WarningCache()

        from mlflow import tracking

        self._mlflow_client = tracking.MlflowClient(tracking_uri)

    @property
    @logger.rank_zero_experiment
    def experiment(self) -> "mlflow.tracking.MlflowClient":
        r"""Actual MLflow object.

        To use MLflow features in your :class:`~reax.Module` do the following.

                Example::

                    self.logger.experiment.some_mlflow_function()
        """
        import mlflow

        if self._initialized:
            return self._mlflow_client

        mlflow.set_tracking_uri(self._tracking_uri)

        if self._run_id is not None:
            run = self._mlflow_client.get_run(self._run_id)
            self._experiment_id = run.info.experiment_id
            self._initialized = True
            return self._mlflow_client

        if self._experiment_id is None:
            expt = self._mlflow_client.get_experiment_by_name(self._experiment_name)
            if expt is not None:
                self._experiment_id = expt.experiment_id
            else:
                _LOGGER.warning(
                    "Experiment with name %s not found. Creating it.", self._experiment_name
                )
                self._experiment_id = self._mlflow_client.create_experiment(
                    name=self._experiment_name, artifact_location=self._artifact_location
                )

        if self._run_id is None:
            if self._run_name is not None:
                self.tags = self.tags or {}

                from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME

                if MLFLOW_RUN_NAME in self.tags:
                    _LOGGER.warning(
                        "The tag %s is found in tags. The value will be overridden by %s.",
                        MLFLOW_RUN_NAME,
                        self._run_name,
                    )
                self.tags[MLFLOW_RUN_NAME] = self._run_name

            resolve_tags = _get_resolve_tags()
            run = self._mlflow_client.create_run(
                experiment_id=self._experiment_id, tags=resolve_tags(self.tags)
            )
            self._run_id = run.info.run_id
        self._initialized = True
        return self._mlflow_client

    @property
    def run_id(self) -> Optional[str]:
        """Create the experiment if it does not exist to get the run id.

        :returns: The run id.
        :rtype: Optional[str]
        """
        _ = self.experiment
        return self._run_id

    @property
    def experiment_id(self) -> Optional[str]:
        """Create the experiment if it does not exist to get the experiment id.

        :return s: The experiment id.
        :rtype s: Optional[str]
        """
        _ = self.experiment
        return self._experiment_id

    @override
    @rank_zero.rank_zero_only
    def log_hyperparams(self, params: Union[dict[str, Any], Namespace], *_, **__) -> None:
        """Log hyperparams."""
        params = _utils.convert_params(params)
        params = _utils.flatten_dict(params)

        from mlflow.entities import Param

        # Truncate parameter values to 250 characters.
        # TODO: MLflow 1.28 allows up to 500 characters:
        #  https://github.com/mlflow/mlflow/releases/tag/v1.28.0
        params_list = [Param(key=k, value=str(v)[:250]) for k, v in params.items()]

        # Log in chunks of 100 parameters (the maximum allowed by MLflow).
        for idx in range(0, len(params_list), 100):
            self.experiment.log_batch(
                run_id=self.run_id, params=params_list[idx : idx + 100], **self._log_batch_kwargs
            )

    @override
    @rank_zero.rank_zero_only
    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        """Log metrics."""
        assert rank_zero.rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"

        from mlflow.entities import Metric

        metrics = _utils.add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)
        metrics_list: list[Metric] = []

        timestamp_ms = int(time() * 1000)
        for name, value in metrics.items():
            if isinstance(value, str):
                self._warning_cache.warn(f"Discarding metric with string value {name}={value}.")
                continue
            if isinstance(value, list):
                self._warning_cache.warn(f"Discarding metric with list value {name}={value}.")
                continue

            new_k = re.sub("[^a-zA-Z0-9_/. -]+", "", name)
            if name != new_k:
                self._warning_cache.warn(
                    "MLFlow only allows '_', '/', '.' and ' ' special characters in metric name."
                    f" Replacing {name} with {new_k}.",
                    category=RuntimeWarning,
                )
                name = new_k
            metrics_list.append(
                Metric(key=name, value=value, timestamp=timestamp_ms, step=step or 0)
            )

        self.experiment.log_batch(
            run_id=self.run_id, metrics=metrics_list, **self._log_batch_kwargs
        )

    @override
    @rank_zero.rank_zero_only
    def finalize(self, status: str = "success") -> None:
        """Finalize function."""
        if not self._initialized:
            return
        if status == "success":
            status = "FINISHED"
        elif status == "failed":
            status = "FAILED"
        elif status == "finished":
            status = "FINISHED"

        # log checkpoints as artifacts
        if self._checkpoint_callback:
            self._scan_and_log_checkpoints(self._checkpoint_callback)

        if self.experiment.get_run(self.run_id):
            self.experiment.set_terminated(self.run_id, status)

    @property
    @override
    def save_dir(self) -> Optional[str]:
        """The root file directory in which MLflow experiments are saved.

                Local path to the root
        experiment directory if the tracking uri is local. Otherwise, returns ``None``.
        """
        if self._tracking_uri.startswith(LOCAL_FILE_URI_PREFIX):
            return self._tracking_uri[len(LOCAL_FILE_URI_PREFIX) :]
        return None

    @property
    @override
    def name(self) -> Optional[str]:
        """Get the experiment id.

        :return s: The experiment id.
        :rtype s: Optional[str]
        """
        return self.experiment_id

    @property
    @override
    def version(self) -> Optional[str]:
        """Get the run id.

        :return s: The run id.
        :rtype s: Optional[str]
        """
        return self.run_id

    @override
    def after_save_checkpoint(self, checkpoint_listener: "reax.listeners.ModelCheckpoint") -> None:
        """After save checkpoint."""
        # log checkpoints as artifacts
        if (
            self._log_model == "all"
            or self._log_model is True
            and checkpoint_listener.save_top_k == -1
        ):
            self._scan_and_log_checkpoints(checkpoint_listener)
        elif self._log_model is True:
            self._checkpoint_callback = checkpoint_listener

    def _scan_and_log_checkpoints(
        self, checkpoint_callback: "reax.listeners.ModelCheckpoint"
    ) -> None:
        """Scan and log checkpoints."""
        # get checkpoints to be saved with associated score
        checkpoints = _utils.scan_checkpoints(checkpoint_callback, self._logged_model_time)

        # log iteratively all new checkpoints
        for timestamp, path, score, _tag in checkpoints:
            metadata = {
                # Ensure .item() is called to store array contents
                "score": score.item() if isinstance(score, jax.Array) else score,
                "original_filename": Path(path).name,
                "Checkpoint": {
                    key: getattr(checkpoint_callback, key)
                    for key in [
                        "monitor",
                        "mode",
                        "save_last",
                        "save_top_k",
                        "save_weights_only",
                        "_every_n_train_steps",
                        "_every_n_val_epochs",
                    ]
                    # ensure it does not break if `Checkpoint` args change
                    if hasattr(checkpoint_callback, key)
                },
            }
            aliases = (
                ["latest", "best"] if path == checkpoint_callback.best_model_path else ["latest"]
            )

            # Artifact path on mlflow
            artifact_path = Path(path).stem

            # Log the checkpoint
            self.experiment.log_artifact(self._run_id, path, artifact_path)

            # Create a temporary directory to log on mlflow
            with tempfile.TemporaryDirectory(
                prefix="test", suffix="test", dir=os.getcwd()
            ) as tmp_dir:
                # Log the metadata
                with open(f"{tmp_dir}/metadata.yaml", "w", encoding="utf-8") as tmp_file_metadata:
                    yaml.dump(metadata, tmp_file_metadata, default_flow_style=False)

                # Log the aliases
                with open(f"{tmp_dir}/aliases.txt", "w", encoding="utf-8") as tmp_file_aliases:
                    tmp_file_aliases.write(str(aliases))

                # Log the metadata and aliases
                self.experiment.log_artifacts(self._run_id, tmp_dir, artifact_path)

            # remember logged models - timestamp needed in case filename didn't change (lastkckpt
            # or custom name)
            self._logged_model_time[path] = timestamp


def _get_resolve_tags() -> Callable[
    [Optional[TagsDict], Optional[list["mlflow.tracking.context.registry.RunContextProvider"]]],
    TagsDict,
]:
    """Get resolve tags."""
    from mlflow.tracking import context

    # before v1.1.0
    if hasattr(context, "resolve_tags"):
        from mlflow.tracking.context import resolve_tags
    # since v1.1.0
    elif hasattr(context, "registry"):
        from mlflow.tracking.context.registry import resolve_tags
    else:
        resolve_tags = _identity

    return resolve_tags


def _identity(x):
    """Identity function."""
    return x


# Alias for compatibility with pytorch lightning
MLFlowLogger = MlflowLogger
