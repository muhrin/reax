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
from collections.abc import Callable, Iterator
import contextlib
import functools
import io
import posixpath
from typing import IO, TYPE_CHECKING, Any, Generic, TypeVar, Union

import fsspec
from typing_extensions import override

from reax.lightning import rank_zero

from . import _figures

if TYPE_CHECKING:
    from reax.listeners import ModelCheckpoint  # noqa: F401

__all__ = ("Logger",)

Exp = TypeVar("Exp")  # Experiment type


class Logger(abc.ABC):
    """Base class for REAX loggers."""

    @property
    @abc.abstractmethod
    def name(self) -> str | None:
        """Return the name of the experiment."""

    @property
    @abc.abstractmethod
    def version(self) -> int | str | None:
        """Return the experiment version."""

    @property
    def root_dir(self) -> str | None:
        """Returns the root directory where all experimental versions are saved.

        Returns `None` if the logger does not save locally.
        """
        return None

    @property
    def save_dir(self) -> str | None:
        """Get the directory where logs are saved.

        Returns `None` if logger does not save locally.
        """
        return None

    @property
    def log_dir(self) -> str | None:
        """Returns the directory used to store logs for the current experiment or `None` if logger
        does not save locally.
        """
        return None

    @property
    def group_separator(self) -> str:
        """The separator used to store to group data into subfolders."""
        return "/"

    @abc.abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log the passed metrics.

        Args:
            metrics (dict[str, float]): Dictionary of metrics to log.
            step (Optional[int], optional): The current step, defaults
                to None.
        """

    @abc.abstractmethod
    def log_hyperparams(
        self, params: dict[str, Any] | argparse.Namespace, *args: Any, **kwargs: Any
    ) -> None:
        """Log the passed hyperparameters.

        Args:
            params (Union[dict[str, Any], argparse.Namespace]): A `dict`
                or :class:`~argparse.Namespace` containing
                hyperparameters.
            *args: Additional optional args.
            **kwargs: Additional optional kwargs.
        """

    def log_graph(self, model: Callable, *args, **kwargs) -> None:  # noqa: B027
        """Log the model graph.

        Args:
            model (Callable): The model function.
            *args: The args to pass to the model.
            **kwargs: The kwargs to pass to the model
        """

    def save(self) -> None:  # noqa: B027
        """Save the log data."""

    def finalize(self, status: str) -> None:
        """Do postprocessing to be done at the end of an experiment.

        Args:
            status (str): A status string indicating the outcome of the
                experiment.
        """
        self.save()

    def after_save_checkpoint(self, checkpoint_listener: "ModelCheckpoint") -> None:  # noqa: B027
        """Called after model checkpoint listener saves a new checkpoint.

        Args:
            checkpoint_listener ("reax.listeners.ModelCheckpoint"): The
                model checkpoint listener instance.
        """

    @contextlib.contextmanager
    def open_artifact(self, path: str) -> Iterator[IO[bytes]]:
        """Open an artifact file for (binary) writing, the logger decides where it is stored.

        Example::

            with logger.open_artifact("plots/loss.svg") as file:
                fig.savefig(file, format="svg")

        The default implementation writes to ``path`` relative to :attr:`log_dir`.

        Args:
            path (str): A relative, ``/`` separated, path identifying the artifact.

        Raises:
            NotImplementedError: If the logger doesn't support storing artifacts.
        """
        with _open_local_artifact(self, path) as file:
            yield file

    def log_figure(
        self, name: str, figure: Any, step: int | None = None, *, close: bool = False
    ) -> None:
        """Log a figure or image.

        Supported types include ``matplotlib`` and ``plotly`` figures, ``PIL`` images, ``numpy``
        and ``jax`` arrays of shape ``(H, W[, C])`` (``uint8``, or floats in ``[0, 1]``) and
        PNG/JPEG encoded ``bytes``.  Further types can be supported using
        :func:`reax.loggers.register_figure_converter`.

        The default implementation stores the figure, as an image, using :meth:`open_artifact`.

        Args:
            name (str): The figure name, use :attr:`group_separator` to create groups.
            figure (Any): The figure to log.
            step (Optional[int], optional): The current step, defaults to None.
            close (bool, optional): Close the figure after logging (where supported, e.g.
                ``matplotlib``), defaults to False.
        """
        _default_log_figure(self, name, figure, step, close=close)


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

    @override
    def log_figure(self, *args: Any, **kwargs: Any) -> None:
        pass

    @override
    @contextlib.contextmanager
    def open_artifact(self, path: str) -> Iterator[IO[bytes]]:
        yield io.BytesIO()

    def __getitem__(self, idx: int) -> "DummyLogger":
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
        self, params: dict[str, Any] | argparse.Namespace, *args: Any, **kwargs: Any
    ) -> None:
        """Log hyperparams."""
        return self._log_hyperparams(params, *args, **kwargs)

    @rank_zero.rank_zero_only
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
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

    def log_figure(
        self, name: str, figure: Any, step: int | None = None, *, close: bool = False
    ) -> None:
        """Log a figure (see :meth:`Logger.log_figure`)."""
        if rank_zero.rank_zero_only.rank > 0:
            if close:
                _figures.close_figure(figure)
            return None

        return self._log_figure(name, figure, step, close=close)

    @contextlib.contextmanager
    def open_artifact(self, path: str) -> Iterator[IO[bytes]]:
        """Open an artifact for writing (see :meth:`Logger.open_artifact`).

        On ranks other than zero, the data written is discarded.
        """
        if rank_zero.rank_zero_only.rank > 0:
            yield io.BytesIO()
            return

        with self._open_artifact(path) as file:
            yield file

    @property
    def _experiment(self) -> Exp | None:
        """Experiment function."""
        return None

    @abc.abstractmethod
    def _log_hyperparams(
        self, params: dict[str, Any] | argparse.Namespace, *args: Any, **kwargs: Any
    ) -> None:
        """Log hyperparams implementation."""

    @abc.abstractmethod
    def _log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics implementation."""

    def _log_graph(self, model: Callable, *args, **kwargs) -> None:
        """Log graph implementation."""

    def _log_figure(
        self, name: str, figure: Any, step: int | None = None, *, close: bool = False
    ) -> None:
        """Log figure implementation."""
        _default_log_figure(self, name, figure, step, close=close)

    def _open_artifact(self, path: str) -> contextlib.AbstractContextManager[IO[bytes]]:
        """Open artifact implementation."""
        return _open_local_artifact(self, path)

    def _save(self) -> None:
        """Save implementation."""

    def _finalize(self, status: str) -> None:
        """Finalize implementation."""
        self._save()


def _check_artifact_path(path: str) -> str:
    """Make sure that the artifact path is relative and stays within the logger's directory."""
    norm = posixpath.normpath(path.replace("\\", "/"))
    if not path or posixpath.isabs(norm) or norm == "." or norm.split("/")[0] == "..":
        raise ValueError(f"Artifact paths must be relative and non-empty, got {path!r}")
    return norm


@contextlib.contextmanager
def _open_local_artifact(logger: Any, path: str) -> Iterator[IO[bytes]]:
    """Open an artifact relative to the logger's ``log_dir``."""
    path = _check_artifact_path(path)
    log_dir = logger.log_dir
    if log_dir is None:
        raise NotImplementedError(
            f"{type(logger).__name__} does not save locally, so it cannot store artifacts"
        )

    fs, root = fsspec.core.url_to_fs(log_dir)
    full_path = posixpath.join(root.rstrip("/"), path)
    fs.makedirs(posixpath.dirname(full_path), exist_ok=True)
    with fs.open(full_path, "wb") as file:
        yield file


def _default_log_figure(
    logger: Any, name: str, figure: Any, step: int | None, *, close: bool
) -> None:
    """Convert the figure to an image and save it using the logger's ``open_artifact()``."""
    try:
        data = _figures.to_figure_data(figure)
        path = _figures.figure_path(name, step, data.fmt, logger.group_separator)
        with logger.open_artifact(path) as file:
            file.write(data.data)
    finally:
        if close:
            _figures.close_figure(figure)


def rank_zero_experiment(fn: Callable) -> Callable:
    """Returns the real experiment on rank 0 and otherwise the _DummyExperiment."""

    @functools.wraps(fn)
    def experiment(self: Logger) -> Any | _DummyExperiment:
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
