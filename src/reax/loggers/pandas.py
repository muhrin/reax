import argparse
import os
from typing import Any, Callable, Final, Mapping, Optional, Union

import fsspec
import jax.typing
from lightning_utilities.core import rank_zero
import numpy as np
import pandas as pd
from typing_extensions import override

from reax import saving

from . import _utils, logger

__all__ = ("PandasLogger",)

DEFAULT_FORMAT = "json"
Path = Union[str, bytes, os.PathLike]


class PandasLogger(logger.WithDdp["ExperimentWriter"], logger.Logger):
    """Log results in a pandas dataframe."""

    LOGGER_JOIN_CHAR: Final[str] = "-"

    def __init__(
        self,
        save_dir: Optional[Path] = None,
        name: Optional[str] = "reax_logs",
        *,
        fmt: str = DEFAULT_FORMAT,
        version: Optional[Union[int, str]] = None,
        prefix: str = "",
        flush_logs_every_n_steps: int = 100,
    ):
        super().__init__()
        self._root_dir = None if save_dir is None else os.fspath(save_dir)
        self._name = name or ""
        self._fmt = fmt
        self._version = version
        self._prefix = prefix
        self._flush_logs_every_n_step = flush_logs_every_n_steps

        self._exp: Optional["ExperimentWriter"] = None
        self._fs: fsspec.AbstractFileSystem = fsspec.url_to_fs(self.root_dir)[0]

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
    def root_dir(self) -> Optional[str]:
        """Root dir."""
        return self._root_dir

    @property
    @override
    def log_dir(self) -> Optional[str]:
        """Log dir."""
        if self.root_dir is None:
            return None

        # Use this scheme to generate some kind of standard logging path
        version = self.version if isinstance(self.version, str) else f"version_{self.version}"
        return os.path.join(self._root_dir, self.name, version)

    @property
    def dataframe(self) -> pd.DataFrame:
        """Get the dataframe from the current experiment."""
        return self.experiment.dataframe

    @property
    def _experiment(self) -> "ExperimentWriter":
        """Experiment function."""
        if self._exp is None:
            self._exp = ExperimentWriter(log_dir=self.log_dir, fmt=self._fmt)

        return self._exp

    @property
    @override
    def save_dir(self) -> Optional[str]:
        """Save dir."""
        return self._root_dir

    @override
    def _log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        """Log metrics."""
        metrics = _utils.add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)

        if step is None:
            step = len(self.experiment.metrics)

        self.experiment.log_metrics(metrics, step)
        if (step + 1) % self._flush_logs_every_n_step == 0:
            self.save()

    @override
    def _log_hyperparams(
        # pylint: disable=arguments-differ
        self,
        params: Union[dict[str, Any], argparse.Namespace],
    ) -> None:
        """Log hyperparams."""
        params = _utils.convert_params(params)
        self.experiment.log_hparams(params)

    @override
    def _save(self) -> None:
        """Save function."""
        self.experiment.save()

    @override
    def _finalize(self, status: str) -> None:
        """Finalize function."""
        if self._experiment is not None:
            return

        self.save()

    def _get_next_version(self) -> int:
        """Get next version."""
        if self.root_dir is None:
            return 0

        version_root = os.path.join(self._root_dir, self.name)

        if not self._fs.isdir(version_root):
            # Doesn't exist yet
            return 0
        try:
            listdir_info = self._fs.listdir(version_root)
        except OSError:
            return 0

        existing_versions = []
        for info in listdir_info:
            name = info["name"]
            name = os.path.join(version_root, name)
            basename = os.path.basename(name)
            if self._fs.isdir(name) and basename.startswith("version_"):
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


class ExperimentWriter:
    r"""Pandas experiment write."""

    DEFAULT_BASENAME = "metrics"
    HPARAMS_FILENAME = "hparams.yaml"

    def __init__(self, log_dir: Optional[str], fmt: str = DEFAULT_FORMAT):
        try:
            getattr(pd.DataFrame, f"to_{fmt}")
        except AttributeError:
            supported_formats = [
                name[len("to_") :] for name in dir(pd.DataFrame) if name.startswith("to_")
            ]
            raise ValueError(
                f"Format '{fmt}' is not supported.  Try one of {','.join(supported_formats)}'"
            ) from None

        self._log_dir = log_dir
        self._fmt = fmt

        self._rows: list[dict[str, float]] = []

        if self._log_dir is None:
            self._metrics_file_path = None
            self._fs = None
        else:
            filename = f"{self.DEFAULT_BASENAME}.{DEFAULT_FORMAT}"
            self._metrics_file_path = os.path.join(self._log_dir, filename)
            self._fs = fsspec.url_to_fs(log_dir)[0]
            self._check_log_dir_exists()
            self._fs.makedirs(self._log_dir, exist_ok=True)

        self.hparams: dict[str, Any] = {}

    @property
    def log_dir(self) -> Optional[str]:
        """Log dir."""
        return self._log_dir

    @property
    def metrics(self):
        """Metrics function."""
        return self._rows

    @property
    def metrics_file_path(self) -> Optional[str]:
        """Metrics file path."""
        return self._metrics_file_path

    @property
    def dataframe(self) -> pd.DataFrame:
        """Get the experiment's dataframe."""
        return pd.DataFrame(self._rows)

    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics."""

        def _handle_value(value: Union[jax.Array, np.ndarray, Any]) -> Any:
            """Handle value."""
            if isinstance(value, (jax.Array, np.ndarray)):
                return value.item()
            return value

        if step is None:
            step = len(self._rows)

        metrics = jax.tree.map(_handle_value, metrics)
        metrics["step"] = step
        self._rows.append(metrics)

    def log_hparams(self, params: dict[str, Any]) -> None:
        """Record hparams."""
        self.hparams.update(params)

    def save(self) -> None:
        """Save recorded metrics into files."""
        if self._log_dir is None:
            return

        hparams_file = os.path.join(self._log_dir, self.HPARAMS_FILENAME)
        saving.save_hparams_to_yaml(hparams_file, self.hparams, use_omegaconf=False)

        if not self._rows:
            return

        dframe = self.dataframe
        save_method: Callable = getattr(dframe, f"to_{self._fmt}")
        save_method(self._metrics_file_path)

    def _check_log_dir_exists(self) -> None:
        """Check log dir exists."""
        if self._fs.exists(self.log_dir) and self._fs.listdir(self.log_dir):
            rank_zero.rank_zero_warn(
                f"Experiment logs directory {self.log_dir} exists and is not empty."
                " Previous log files in this directory will be deleted when the new ones are saved!"
            )
            if self._fs.isfile(self.metrics_file_path):
                self._fs.rm_file(self.metrics_file_path)
