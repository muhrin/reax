import copy
import logging
import os
import pathlib
import re
from typing import TYPE_CHECKING, Literal, Optional, Union

import jax
from jax import numpy as jnp
from typing_extensions import override

from reax import exceptions, stages

from . import checkpointer

if TYPE_CHECKING:
    import reax

_LOGGER = logging.getLogger(__name__)

__all__ = ("ModelCheckpoint",)

CHECKPOINTS_DIR = "checkpoints"


class ModelCheckpoint(checkpointer.Checkpointer):

    CHECKPOINT_JOIN_CHAR = "-"
    CHECKPOINT_EQUALS_CHAR = "="
    CHECKPOINT_NAME_LAST = "last"
    FILE_EXTENSION = ".ckpt"
    STARTING_VERSION = 1

    def __init__(
        self,
        dirpath: Optional[Union[str, pathlib.Path]] = None,
        filename: Optional[str] = None,
        monitor: Optional[str] = None,
        mode: Literal["min", "max"] = "min",
        auto_insert_metric_name: bool = True,
        every_n_train_steps: Optional[int] = None,
        every_n_epochs: Optional[int] = None,
        save_on_train_epoch_end: Optional[bool] = None,
        save_last: Optional[bool] = None,
        save_top_k: int = 1,
        enable_version_counter: bool = True,
    ):
        super().__init__()
        self._monitor = monitor
        self._mode = mode
        self._auto_insert_metric_name = auto_insert_metric_name
        self._every_n_epochs = every_n_epochs
        self._save_on_train_epoch_end = save_on_train_epoch_end
        self._enable_version_counter = enable_version_counter
        self._save_last = save_last
        self._save_top_k = save_top_k
        self._dirpath: Optional[str] = (
            os.path.realpath(os.path.expanduser(dirpath)) if dirpath else dirpath
        )
        self.filename: Optional[str] = filename
        self.__init_triggers(every_n_train_steps, every_n_epochs)

        # State
        self.last_model_path = ""
        self._last_checkpoint_saved = ""
        self._best_model_path = ""
        self._best_k_models: dict[str, jax.Array] = {}
        self._kth_best_model_path = ""
        # Keep track of the global number of optimizer steps
        self._last_global_step_saved = 0

    def __init_triggers(
        self,
        every_n_train_steps: Optional[int],
        every_n_epochs: Optional[int],
        # train_time_interval: Optional[timedelta],
    ) -> None:
        # Default to running once after each validation epoch if neither
        # every_n_train_steps nor every_n_epochs is set
        if every_n_train_steps is None and every_n_epochs is None:
            every_n_epochs = 1
            every_n_train_steps = 0
            _LOGGER.debug(
                "Both every_n_train_steps and every_n_epochs are not set. Setting every_n_epochs=1"
            )
        else:
            every_n_epochs = every_n_epochs or 0
            every_n_train_steps = every_n_train_steps or 0

        # self._train_time_interval: Optional[timedelta] = train_time_interval
        self._every_n_epochs: int = every_n_epochs
        self._every_n_train_steps: int = every_n_train_steps

    def __init_monitor_mode(self, mode: str) -> None:
        # TODO: Call this where necessary
        jnp_info = jnp.array(float("inf" if self._mode == "min" else "-inf"))
        mode_dict = {"min": (jnp_info, "min"), "max": (-jnp_info, "max")}

        if mode not in mode_dict:
            raise exceptions.MisconfigurationException(
                f"`mode` can be {', '.join(mode_dict.keys())} but got {mode}"
            )

        self._kth_value, self._mode = mode_dict[mode]

    @override
    def on_train_epoch_end(self, trainer: "reax.Trainer", stage: "reax.stages.Train") -> None:
        metrics = self._monitor_candidates(stage, trainer)
        if self._should_save_on_train_epoch_end(stage):
            if isinstance(stage, stages.Train):
                self._do_save(trainer, metrics)

    @override
    def on_validation_epoch_end(
        self, trainer: "reax.Trainer", stage: "reax.stages.Validate"
    ) -> None:
        metrics = self._monitor_candidates(stage, trainer)
        if not self._should_save_on_train_epoch_end(stage):
            if isinstance(stage, stages.Validate):
                self._do_save(trainer, metrics)

    @property
    def best_model_path(self) -> str:
        return self._best_model_path

    def _monitor_candidates(
        self, stage: "reax.stages.EpochStage", trainer: "reax.Trainer"
    ) -> dict[str, jax.Array]:
        monitor_candidates: dict = copy.deepcopy(stage.callback_metrics)
        monitor_candidates.setdefault("epoch", trainer.current_epoch)
        monitor_candidates.setdefault("step", trainer.global_updates)
        return monitor_candidates

    def check_monitor_top_k(self, current: Optional[jax.Array] = None) -> bool:
        if current is None:
            return False

        if self._save_top_k == -1:
            return True

        less_than_k_models = len(self._best_k_models) < self._save_top_k
        if less_than_k_models:
            return True

        monitor_op = {"min": jnp.less, "max": jnp.greater}[self._mode]
        should_update_best_and_save = monitor_op(
            current, self._best_k_models[self._kth_best_model_path]
        )

        return should_update_best_and_save

    def _do_save(self, trainer: "reax.Trainer", monitor_candidates: dict[str, jax.Array]):
        if self._every_n_epochs >= 1 and (trainer.current_epoch + 1) % self._every_n_epochs == 0:
            self._save_topk_checkpoint(trainer, monitor_candidates)

        self._save_last_checkpoint(trainer, monitor_candidates)

    def _should_save_on_train_epoch_end(self, stage: stages.EpochStage) -> bool:
        if self._save_on_train_epoch_end is not None:
            # Do whatever the user asked for
            return self._save_on_train_epoch_end

        if (
            stage.parent is not None
            and isinstance(stage.parent, stages.FitEpoch)
            and stage.parent.validate is None
        ):
            # There is no validation, so save on train end
            return True

        return False

    def _save_topk_checkpoint(
        self,
        trainer: "reax.Trainer",
        monitor_candidates: dict[str, jax.Array],
    ) -> None:
        if self._save_top_k == 0:
            return

        if self._monitor is None:
            self._save_none_monitor_checkpoint(trainer, monitor_candidates)
        else:
            self._save_monitor_checkpoint(trainer, monitor_candidates)

    def _save_last_checkpoint(
        self, trainer: "reax.Trainer", monitor_candidates: dict[str, jax.Array]
    ) -> None:
        if not self._save_last:
            return

        filepath = self._get_checkpoint_filepath(
            trainer, monitor_candidates, self.CHECKPOINT_NAME_LAST
        )

        if self._enable_version_counter:
            version_cnt = self.STARTING_VERSION
            while os.path.exists(filepath) and filepath != self.last_model_path:
                filepath = self._get_checkpoint_filepath(
                    trainer, monitor_candidates, self.CHECKPOINT_NAME_LAST, version=version_cnt
                )
                version_cnt += 1

        # set the last model path before saving because it will be part of the state.
        previous, self.last_model_path = self.last_model_path, filepath
        self._save_checkpoint(trainer, filepath)

        if previous and self._should_remove_checkpoint(trainer, previous, filepath):
            os.unlink(previous)

    def _save_monitor_checkpoint(
        self, trainer: "reax.Trainer", monitor_candidates: dict[str, jax.Array]
    ) -> None:
        assert self._monitor is not None
        current = monitor_candidates.get(self._monitor)

        if self.check_monitor_top_k(current):
            assert current is not None
            self._update_best_and_save(current, trainer, monitor_candidates)

    def _save_none_monitor_checkpoint(
        self, trainer: "reax.Trainer", monitor_candidates: dict[str, jax.Array]
    ) -> None:
        filepath = self._get_metric_interpolated_filepath_name(
            trainer, monitor_candidates, self._best_model_path
        )
        # set the best model path before saving because it will be part of the state.
        previous, self._best_model_path = self._best_model_path, filepath
        self._save_checkpoint(trainer, filepath)

        if (
            self._save_top_k == 1
            and previous
            and self._should_remove_checkpoint(trainer, previous, filepath)
        ):
            os.unlink(previous)

    def _update_best_and_save(
        self, current: jax.Array, trainer: "reax.Trainer", monitor_candidates: dict[str, jax.Array]
    ) -> None:
        k = len(self._best_k_models) + 1 if self._save_top_k == -1 else self._save_top_k

        del_filepath = None
        if len(self._best_k_models) == k and k > 0:
            del_filepath = self._kth_best_model_path
            self._best_k_models.pop(del_filepath)

        # do not save nan, replace with +/- inf
        if isinstance(current, jax.Array) and jnp.isnan(current):
            current = jnp.array(float("inf" if self._mode == "min" else "-inf"))

        filepath = self._get_metric_interpolated_filepath_name(
            trainer, monitor_candidates, del_filepath
        )

        # save the current score
        self.current_score = current
        self._best_k_models[filepath] = current

        if len(self._best_k_models) == k:
            # monitor dict has reached k elements
            _op = max if self._mode == "min" else min
            self._kth_best_model_path = _op(self._best_k_models, key=self._best_k_models.get)  # type: ignore[arg-type]
            self._kth_value = self._best_k_models[self._kth_best_model_path]

        _op = min if self._mode == "min" else max
        self._best_model_path = _op(self._best_k_models, key=self._best_k_models.get)  # type: ignore[arg-type]
        self._best_model_score = self._best_k_models[self._best_model_path]

        self._save_checkpoint(trainer, filepath)

        if del_filepath and self._should_remove_checkpoint(trainer, del_filepath, filepath):
            os.unlink(del_filepath)

    def format_checkpoint_name(
        self,
        metrics: dict[str, jax.Array],
        filename: Optional[str] = None,
        version: Optional[int] = None,
    ) -> str:
        """Generate a filename using a standard format."""
        filename = filename or self.filename
        filename = self._format_checkpoint_name(
            filename, metrics, auto_insert_metric_name=self._auto_insert_metric_name
        )

        if version is not None:
            filename = self.CHECKPOINT_JOIN_CHAR.join((filename, f"v{version}"))

        ckpt_name = f"{filename}{self.FILE_EXTENSION}"
        return os.path.join(self._dirpath, ckpt_name) if self._dirpath else ckpt_name

    def _get_checkpoint_filepath(
        self,
        trainer: "reax.Trainer",
        metrics: dict[str, jax.Array],
        filename: Optional[str] = None,
        version: Optional[int] = None,
    ) -> str:
        filepath = self.format_checkpoint_name(metrics, filename, version)
        return os.path.join(self._get_dirpath(trainer), filepath)

    def _get_dirpath(self, trainer: "reax.Trainer") -> str:
        if self._dirpath is not None:
            return self._dirpath
        elif trainer.default_root_dir is not None:
            return trainer.default_root_dir

        return ""

    def _get_metric_interpolated_filepath_name(
        self,
        trainer: "reax.Trainer",
        monitor_candidates: dict[str, jax.Array],
        del_filepath: Optional[str] = None,
    ) -> str:
        filepath = self._get_checkpoint_filepath(trainer, monitor_candidates)

        if self._enable_version_counter:
            version_cnt = self.STARTING_VERSION
            while os.path.exists(filepath) and filepath != del_filepath:
                filepath = self._get_checkpoint_filepath(
                    trainer, monitor_candidates, version=version_cnt
                )
                version_cnt += 1

        return filepath

    def _format_checkpoint_name(
        self,
        filename: Optional[str],
        metrics: dict[str, jax.Array],
        prefix: str = "",
        auto_insert_metric_name: bool = True,
    ) -> str:
        if not filename:
            # filename is not set, use default name
            filename = "{epoch}" + self.CHECKPOINT_JOIN_CHAR + "{step}"

        # check and parse user passed keys in the string
        groups = re.findall(r"(\{.*?)[:\}]", filename)

        # sort keys from longest to shortest to avoid replacing substring
        # eg: if keys are "epoch" and "epoch_test", the latter must be replaced first
        groups = sorted(groups, key=lambda x: len(x), reverse=True)

        for group in groups:
            name = group[1:]

            if auto_insert_metric_name:
                filename = filename.replace(group, name + self.CHECKPOINT_EQUALS_CHAR + "{" + name)

            # support for dots: https://stackoverflow.com/a/7934969
            filename = filename.replace(group, f"{{0[{name}]")

            if name not in metrics:
                metrics[name] = jnp.array(0)

        filename = filename.format(metrics)

        if prefix:
            filename = self.CHECKPOINT_JOIN_CHAR.join([prefix, filename])

        return filename

    def _save_checkpoint(self, trainer: "reax.Trainer", filepath: str) -> None:
        trainer.save_checkpoint(filepath)
        self._last_global_step_saved = trainer.global_updates
        self._last_checkpoint_saved = filepath

    def _should_remove_checkpoint(
        self, trainer: "reax.Trainer", previous: str, current: str
    ) -> bool:
        """Checks if the previous checkpoint should be deleted.

        A checkpoint won't be deleted if any of the cases apply:
        - The previous checkpoint is the same as the current checkpoint (means the old was already overwritten by new)
        - The previous checkpoint is not in the current checkpoint directory and the filesystem is local
        - The previous checkpoint is the checkpoint the Trainer resumed from and the filesystem is local

        """
        if previous == current:
            return False

        previous = pathlib.Path(previous).absolute()
        dirpath = pathlib.Path(self._get_dirpath(trainer)).absolute()
        return dirpath in previous.parents
