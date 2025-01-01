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
    LOGGER_JOIN_CHAR: Final[str] = "-"
    NAME_HPARAMS_FILE: Final[str] = "hparams.yaml"

    def __init__(
        self,
        log_dir: typing.Path,
        name: Optional[str] = "reax_logs",
        version: Optional[Union[int, str]] = None,
        log_graph: bool = False,
        default_hp_metric: bool = True,
        prefix: str = "",
        sub_dir: Optional[typing.Path] = None,
        **kwargs: Any,
    ):
        """Init function."""
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
        """Root dir."""
        return os.path.join(self._root_dir, self.name)

    @property
    @override
    def log_dir(self) -> str:
        """Log dir."""
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
        """Save dir."""
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
        """Log hyperparams."""
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
