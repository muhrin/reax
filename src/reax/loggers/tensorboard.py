import argparse
import os
from typing import Any, Callable, Final, Mapping, Optional, Union

import jax.typing
import omegaconf
import tensorboardX
from typing_extensions import override

from . import _utils, logger

__all__ = ("TensorBoardLogger",)


class TensorBoardLogger(logger.WithDdp["tensorboardX.SummaryWriter"], logger.Logger):
    LOGGER_JOIN_CHAR: Final[str] = "-"
    NAME_HPARAMS_FILE: Final[str] = "hparams.yaml"

    def __init__(
        self,
        root_dir: os.PathLike,
        name: Optional[str] = "reax_logs",
        version: Optional[Union[int, str]] = None,
        log_graph: bool = False,
        default_hp_metric: bool = True,
        prefix: str = "",
        sub_dir: Optional[os.PathLike] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self._root_dir = os.fspath(root_dir)
        self._name = name or ""
        self._version = version
        self._sub_dir = None if sub_dir is None else os.fspath(sub_dir)

        self._default_hp_metric = default_hp_metric
        self._prefix = prefix
        # self._fs = get_filesystem(root_dir)

        self._exp: Optional["tensorboardX.SummaryWriter"] = None
        self._kwargs = kwargs
        self._log_graph = log_graph
        self.hparams: Union[dict[str, Any], argparse.Namespace] = {}

    @property
    @override
    def name(self) -> str:
        return self._name

    @property
    @override
    def version(self) -> Union[int, str]:
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    @property
    @override
    def root_dir(self) -> str:
        return os.path.join(self._root_dir, self.name)

    @property
    @override
    def log_dir(self) -> str:
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
        return self._sub_dir

    @property
    @override
    def save_dir(self) -> str:
        return self._root_dir

    @property
    def _experiment(self) -> "tensorboardX.SummaryWriter":
        """
        Get the tensorboard object.
        """
        if self._exp is not None:
            return self._exp

        if self.root_dir:
            os.makedirs(self.root_dir, exist_ok=True)

        self._exp = tensorboardX.SummaryWriter(log_dir=self.log_dir, **self._kwargs)
        return self._exp

    # @override
    def _log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        metrics = _utils.add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)

        for key, val in metrics.items():
            if isinstance(val, jax.typing.ArrayLike):
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

    # @override
    def _log_hyperparams(
        # pylint: disable=arguments-differ
        self,
        params: Union[dict[str, Any], argparse.Namespace],
        metrics: Optional[dict[str, Any]] = None,
    ) -> None:
        params = _utils.convert_params(params)

        # store params to output
        if isinstance(params, omegaconf.Container):
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
    def _log_graph(self, model: Callable, *args, **kwargs) -> None:  # type: ignore[override]
        if not self._log_graph:
            return

        # TODO: Check this
        self.experiment.add_graph(model, *args, **kwargs)

    @override
    def _save(self) -> None:
        self.experiment.flush()
        # dir_path = self.log_dir
        #
        # # prepare the file path
        # hparams_file = os.path.join(dir_path, self.NAME_HPARAMS_FILE)
        #
        # # save the metatags file if it doesn't exist and the log directory exists
        # if os.path.isdir(dir_path) and not os.path.isfile(hparams_file):
        #     save_hparams_to_yaml(hparams_file, self.hparams)

    @override
    def _finalize(self, status: str) -> None:
        if status == "success":
            # saving hparams happens independent of experiment manager
            self.save()

        if self._exp is not None:
            self.experiment.flush()
            self.experiment.close()

    # @override
    def _get_next_version(self) -> int:
        root_dir = self.root_dir

        try:
            listdir_info = os.listdir(root_dir)
        except OSError:
            return 0

        existing_versions = []
        for listing in listdir_info:
            name = listing["name"]
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
        params = _utils.sanitize_params(params)
        # logging of arrays with dimension > 1 is not supported, sanitize as string
        return {
            key: str(val) if hasattr(val, "ndim") and val.ndim > 1 else val
            for key, val in params.items()
        }
