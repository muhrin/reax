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
# https://github.com/Lightning-AI/pytorch-lightning/blob/9177ec09caadcf88859e1f1e3e10a18e8832069a/src/lightning/fabric/loggers/logger.py
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
"""Abstract base class used to build new loggers."""

import abc
import argparse
import functools
from typing import TYPE_CHECKING, Any, Callable, Generic, Optional, TypeVar, Union

from typing_extensions import override

from reax.lightning import rank_zero

if TYPE_CHECKING:
    import reax

__all__ = ("Logger",)

Exp = TypeVar("Exp")  # Experiment type


class Logger(abc.ABC):
    """Base class for REAX loggers."""

    @property
    @abc.abstractmethod
    def name(self) -> Optional[str]:
        """Return the name of the experiment."""

    @property
    @abc.abstractmethod
    def version(self) -> Optional[Union[int, str]]:
        """Return the experiment version."""

    @property
    def root_dir(self) -> Optional[str]:
        """Returns the root directory where all experimental versions are saved.

        Returns `None` if the logger does not save locally.
        """
        return None

    @property
    def save_dir(self) -> Optional[str]:
        """Get the directory where logs are saved.

        Returns `None` if logger does not save locally.
        """
        return None

    @property
    def log_dir(self) -> Optional[str]:
        """Returns the directory used to store logs for the current experiment or `None` if logger
        does not save locally."""
        return None

    @property
    def group_separator(self) -> str:
        """The separator used to store to group data into subfolders."""
        return "/"

    @abc.abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        """Log the passed metrics.

        :param metrics: Dictionary of metrics to log.
        :type metrics: dict[str, float]
        :param step: The current step, defaults to None.
        :type step: Optional[int], optional
        """

    @abc.abstractmethod
    def log_hyperparams(
        self, params: Union[dict[str, Any], argparse.Namespace], *args: Any, **kwargs: Any
    ) -> None:
        """Log the passed hyperparameters.

        :param params: A `dict` or :class:`~argparse.Namespace` containing hyperparameters.
        :type params: Union[dict[str, Any], argparse.Namespace]
        :param *args: Additional optional args.
        :param **kwargs: Additional optional kwargs.
        """

    def log_graph(self, model: Callable, *args, **kwargs) -> None:
        """Log the model graph.

        :param model: The model function.
        :type model: Callable :param *args: The args to pass to the model. :param **kwargs: The
            kwargs to pass to the model.
        """

    def save(self) -> None:
        """Save the log data."""

    def finalize(self, status: str) -> None:  # pylint: disable=unused-argument
        """Do postprocessing to be done at the end of an experiment.

        :param status: A status string indicating the outcome of the experiment.
        :type status: str
        """
        self.save()

    def after_save_checkpoint(self, checkpoint_listener: "reax.listeners.ModelCheckpoint") -> None:
        """Called after model checkpoint listener saves a new checkpoint.

        :param checkpoint_listener: The model checkpoint listener instance.
        :type checkpoint_listener: "reax.listeners.ModelCheckpoint"
        """


class DummyLogger(Logger):
    """Dummy logger for internal use.

    It is useful if we want to disable user's logger for a feature, but still ensure that user code
    can run
    """

    def __init__(self) -> None:
        super().__init__()
        self._experiment = _DummyExperiment()

    @property
    def experiment(self) -> "_DummyExperiment":
        """Return the experiment object associated with this logger."""
        return self._experiment

    @override
    def log_metrics(self, *args: Any, **kwargs: Any) -> None:
        pass

    @override
    def log_hyperparams(self, *args: Any, **kwargs: Any) -> None:
        pass

    @property
    @override
    def name(self) -> str:
        """Return the experiment name."""
        return ""

    @property
    @override
    def version(self) -> str:
        """Return the experiment version."""
        return ""

    def __getitem__(self, idx: int) -> "_DummyLogger":
        # enables self.logger[0].experiment.add_image(...)
        return self

    def __getattr__(self, name: str) -> Callable:
        """Allows the DummyLogger to be called with arbitrary methods, to avoid AttributeErrors."""

        def method(*_, **__) -> None:
            return None

        return method


class WithDdp(Generic[Exp], abc.ABC):
    """Mixins that allows a logger to be compatible with DPP strategy."""

    @property
    def experiment(self) -> Union[Exp, "_DummyExperiment"]:
        """Returns the actual experiment if on rank 0 and otherwise the _DummyExperiment."""
        if rank_zero.rank_zero_only.rank > 0:
            return _DummyExperiment()

        return self._experiment

    @rank_zero.rank_zero_only
    def log_hyperparams(
        self, params: Union[dict[str, Any], argparse.Namespace], *args: Any, **kwargs: Any
    ) -> None:
        """Log hyperparams."""
        return self._log_hyperparams(params, *args, **kwargs)

    @rank_zero.rank_zero_only
    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics."""
        return self._log_metrics(metrics, step)

    @rank_zero.rank_zero_only
    def log_graph(self, model: Callable, *args, **kwargs) -> None:
        """Log graph."""
        return self._log_graph(model, *args, **kwargs)

    @rank_zero.rank_zero_only
    def save(self) -> None:
        """Save function."""
        return self._save()

    @rank_zero.rank_zero_only
    def finalize(self, status: str) -> None:
        """Finalize function."""
        return self._finalize(status)

    @property
    def _experiment(self) -> Optional[Exp]:
        """Experiment function."""
        return None

    @abc.abstractmethod
    def _log_hyperparams(
        self, params: Union[dict[str, Any], argparse.Namespace], *args: Any, **kwargs: Any
    ) -> None:
        """Log hyperparams implementation."""

    @abc.abstractmethod
    def _log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics implementation."""

    def _log_graph(self, model: Callable, *args, **kwargs) -> None:
        """Log graph implementation."""

    def _save(self) -> None:
        """Save implementation."""

    def _finalize(self, status: str) -> None:  # pylint: disable=unused-argument
        """Finalize implementation."""
        self._save()


def rank_zero_experiment(fn: Callable) -> Callable:
    """Returns the real experiment on rank 0 and otherwise the _DummyExperiment."""

    @functools.wraps(fn)
    def experiment(self: Logger) -> Union[Any, _DummyExperiment]:
        """Get the experiment.

        ..note::
            ``self`` is a custom logger instance. The loggers typically wrap an ``experiment``
            method with a ``@rank_zero_experiment`` decorator.

        ``Union[Any, _DummyExperiment]`` is used because the wrapped hooks have several return
        types that are specific to the custom logger. The return type here can be considered as
        ``Union[return type of logger.experiment, _DummyExperiment]``.
        """
        if rank_zero.rank_zero_only.rank > 0:
            return _DummyExperiment()
        return fn(self)

    return experiment


class _DummyExperiment:
    """Dummy experiment."""

    def nop(self, *args: Any, **kw: Any) -> None:
        """Nop function."""

    def __getattr__(self, _: Any) -> Callable:
        """Getattr function."""
        return self.nop

    def __getitem__(self, idx: int) -> "_DummyExperiment":
        """Getitem function."""
        # enables self.logger.experiment[0].add_image(...)
        return self

    def __setitem__(self, *args: Any, **kwargs: Any) -> None:
        """Setitem function."""
