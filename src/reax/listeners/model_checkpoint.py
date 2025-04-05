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
# https://github.com/Lightning-AI/pytorch-lightning/blob/9177ec09caadcf88859e1f1e3e10a18e8832069a/src/lightning/pytorch/callbacks/model_checkpoint.py
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

import copy
import datetime
import logging
import os
import pathlib
import re
import time
from typing import TYPE_CHECKING, Final, Literal, Optional, Union
import weakref

import jax
from jax import numpy as jnp
from typing_extensions import override

from reax import exceptions, stages
from reax.lightning import rank_zero

from . import checkpointer

if TYPE_CHECKING:
    import reax

_LOGGER = logging.getLogger(__name__)

__all__ = ("ModelCheckpoint",)

CHECKPOINTS_DIR = "checkpoints"
PathType = Union[str, pathlib.Path]


class ModelCheckpoint(checkpointer.Checkpointer):
    r"""Save the model periodically by monitoring a quantity.

    Every metric logged with :meth:`~reax.Module.log` or :meth:`~reax.Module.log_dict` is a
    candidate for the monitor key. For more information, see :ref:`checkpointing`.

    After training finishes, use :attr:`best_model_path` to retrieve the path to the
    best checkpoint file and :attr:`best_model_score` to retrieve its score.

    Example::

        >>> from reax import Trainer
        >>> from reax.listeners import ModelCheckpoint

        # saves checkpoints to 'my/path/' at every epoch
        >>> checkpoint_listener = ModelCheckpoint(dirpath='my/path/')
        >>> trainer = Trainer(listeners=[checkpoint_listener])

        # save epoch and val_loss in name
        # saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
        >>> checkpoint_listener = ModelCheckpoint(
        ...     monitor='val_loss',
        ...     dirpath='my/path/',
        ...     filename='sample-mnist-{epoch:02d}-{val_loss:.2f}'
        ... )

        # save epoch and val_loss in name, but specify the formatting yourself (e.g. to avoid
        # problems with Tensorboard
        # or Neptune, due to the presence of characters like '=' or '/')
        # saves a file like: my/path/sample-mnist-epoch02-val_loss0.32.ckpt
        >>> checkpoint_listener = ModelCheckpoint(
        ...     monitor='val/loss',
        ...     dirpath='my/path/',
        ...     filename='sample-mnist-epoch{epoch:02d}-val_loss{val/loss:.2f}',
        ...     auto_insert_metric_name=False
        ... )

        # retrieve the best checkpoint after training
        checkpoint_listener = ModelCheckpoint(dirpath='my/path/')
        trainer = Trainer(listeners=[checkpoint_listener])
        model = ...
        trainer.fit(model)
        checkpoint_listener.best_model_path

    .. tip:: Saving and restoring multiple checkpoint listeners at the same time is supported under
        variation in the following arguments:

        *monitor, mode, every_n_train_steps, every_n_epochs, train_time_interval*

        Read more: :ref:`Persisting Listener State <extensions/listeners_state:save listener state>`
    """

    CHECKPOINT_JOIN_CHAR: Final[str] = "-"
    CHECKPOINT_EQUALS_CHAR: Final[str] = "="
    CHECKPOINT_NAME_LAST: Final[str] = "last"
    FILE_EXTENSION: Final[str] = ".ckpt"
    STARTING_VERSION: Final[int] = 1

    def __init__(
        self,
        dirpath: Optional[Union[str, pathlib.Path]] = None,
        filename: Optional[str] = None,
        monitor: Optional[str] = None,
        *,
        verbose: bool = False,
        mode: Literal["min", "max"] = "min",
        auto_insert_metric_name: bool = True,
        every_n_train_steps: Optional[int] = None,
        train_time_interval: Optional[datetime.timedelta] = None,
        every_n_epochs: Optional[int] = None,
        save_on_train_epoch_end: Optional[bool] = None,
        save_last: Optional[bool] = None,
        save_top_k: int = 1,
        save_weights_only: bool = False,
        enable_version_counter: bool = True,
    ):
        """Init function. :param dirpath: Directory to save the model file.

            Example::

            # custom path
            # saves a file like: my/path/epoch=0-step=10.ckpt
            >>> checkpoint_listener = ModelCheckpoint(dirpath='my/path/')


            specified by :class:`~reax.Trainer`'s
        :paramref:`~reax.Trainer.default_root_dir` argument, and if the Trainer uses a logger, the
            path will also contain logger name and version, defaults to None.
        :type dirpath: Optional[Union[str, pathlib.Path]], optional
        :param filename: Checkpoint filename. Can contain named formatting options to be autofilled.

            Example::

            # save any arbitrary metrics like `val_loss`, etc. in name
            # saves a file like: my/path/epoch=2-val_loss=0.02-other_metric=0.03.ckpt
            >>> checkpoint_listener = ModelCheckpoint(
            ...     dirpath='my/path',
            ...     filename='{epoch}-{val_loss:.2f}-{other_metric:.2f}'
            ... )


            the number of finished epoch and optimizer steps respectively, defaults to None.
        :type filename: Optional[str], optional
        :param monitor: Quantity to monitor, defaults to None.
        :type monitor: Optional[str], optional
        :param verbose: verbosity mode. Default: ``False``.
        :param verbose: Verbosity mode, defaults to ``False``.
        :param save_last: When ``True``, saves a `last.ckpt` copy whenever a checkpoint file gets
            saved. Can be set to ``'link'`` on a local filesystem to create a symbolic link. This
            allows accessing the latest checkpoint in a deterministic manner, defaults to None.
        :type save_last: Optional[bool], optional
        :param save_top_k: If ``save_top_k == k``,
            the best k models according to the quantity monitored will be saved.
            If ``save_top_k == 0``, no models are saved.
            If ``save_top_k == -1``, all models are saved.
            Please note that the monitors are checked every ``every_n_epochs`` epochs.
            If ``save_top_k >= 2`` and the listener is called multiple times inside an epoch, and
            the filename remains unchanged, the name of the saved file will be appended with a
            version count starting with ``v1`` to avoid collisions unless
            ``enable_version_counter`` is set to False. The version counter is unrelated to the
            top-k ranking of the checkpoint, and we recommend formatting the filename to include
            the monitored metric to avoid collisions, defaults to 1.
        :type save_top_k: int, optional
        :param save_weights_only: if ``True``, then only the model's weights will be saved.
            Otherwise, the optimizer states, lr-scheduler states, etc are added in the checkpoint
            too.
        :param mode: One of {min, max}.
            If ``save_top_k != 0``, the decision to overwrite the current save file is made
            based on either the maximization or the minimization of the monitored quantity.
            For ``'val_acc'``, this should be ``'max'``, for ``'val_loss'`` this should be
            ``'min'``, etc., defaults to "min".
        :type mode: Literal["min", "max"], optional
        :param auto_insert_metric_name: When ``True``, the checkpoints filenames will contain the
            metric name. For example, ``filename='checkpoint_{epoch:02d}-{acc:02.0f}`` with epoch
            ``1`` and acc ``1.12`` will resolve to ``checkpoint_epoch=01-acc=01.ckpt``.
            Is useful to set it to ``False`` when metric names contain ``/`` as this will result in
            extra folders. For example,
            ``filename='epoch={epoch}-step={step}-val_acc={val/acc:.2f}',
            auto_insert_metric_name=False``, defaults to True.
        :type auto_insert_metric_name: bool, optional
        :param save_weights_only: If ``True``, then only the model's weights will be saved.
            Otherwise, the optimizer states, lr-scheduler states, etc. are added in the
            checkpoint too.
        :param every_n_train_steps: Number of training steps between checkpoints.
            If ``every_n_train_steps == None or every_n_train_steps == 0``, we skip saving during
            training.
            To disable, set ``every_n_train_steps = 0``. This value must be ``None`` or
            non-negative.
            This must be mutually exclusive with ``train_time_interval`` and ``every_n_epochs``,
            defaults to None.
        :type every_n_train_steps: Optional[int], optional
        :param train_time_interval: Checkpoints are monitored at the specified time interval.
            For all practical purposes, this cannot be smaller than the amount
            of time it takes to process a single training batch. This is not
            guaranteed to execute at the exact time specified, but should be close.
            This must be mutually exclusive with ``every_n_train_steps`` and ``every_n_epochs``.
        :param every_n_epochs: Number of epochs between checkpoints.
            This value must be ``None`` or non-negative.
            To disable saving top-k checkpoints, set ``every_n_epochs = 0``.
            This argument does not impact the saving of ``save_last=True`` checkpoints.
            If all of ``every_n_epochs``, ``every_n_train_steps`` and
            ``train_time_interval`` are ``None``, we save a checkpoint at the end of every epoch
            (equivalent to ``every_n_epochs = 1``).
            If ``every_n_epochs == None`` and either ``every_n_train_steps != None`` or
            ``train_time_interval != None``,
            saving at the end of each epoch is disabled
            (equivalent to ``every_n_epochs = 0``).
            This must be mutually exclusive with ``every_n_train_steps`` and
            ``train_time_interval``.
            Setting both ``ModelCheckpoint(..., every_n_epochs=V, save_on_train_epoch_end=False)``
            and ``Trainer(max_epochs=N, check_val_every_n_epoch=M)``
            will only save checkpoints at epochs 0 < E <= N where both values for
            ``every_n_epochs`` and ``check_val_every_n_epoch`` evenly divide E, defaults to None.
        :type every_n_epochs: Optional[int], optional
        :param save_on_train_epoch_end: Whether to run checkpointing at the end of the training
            epoch.
            If this is ``False``, then the check runs at the end of the validation, defaults to
            None.
        :type save_on_train_epoch_end: Optional[bool], optional
        :param enable_version_counter: Whether to append a version to the existing file name.
            If this is ``False``, then the checkpoint files will be overwritten, defaults to True.
        :type enable_version_counter: bool, optional
        :raises MisconfigurationException:
            If ``save_top_k`` is smaller than ``-1``,
            if ``monitor`` is ``None`` and ``save_top_k`` is none of ``None``, ``-1``, and ``0``, or
            if ``mode`` is none of ``"min"`` or ``"max"``.
        :raises ValueError:
            If ``trainer.save_checkpoint`` is ``None``.

        Note:
            For extra customization, ModelCheckpoint includes the following attributes:

            - ``CHECKPOINT_JOIN_CHAR = "-"``
            - ``CHECKPOINT_EQUALS_CHAR = "="``
            - ``CHECKPOINT_NAME_LAST = "last"``
            - ``FILE_EXTENSION = ".ckpt"``
            - ``STARTING_VERSION = 1``

            For example, you can change the default last checkpoint name by doing
            ``checkpoint_listener.CHECKPOINT_NAME_LAST = "{epoch}-last"``

            If you want to checkpoint every N hours, every M train batches, and/or every K val
            epochs, then you should create multiple ``ModelCheckpoint`` listeners.

            If the checkpoint's ``dirpath`` changed from what it was before while resuming the
            training, only ``best_model_path`` will be reloaded and a warning will be issued.
        """
        super().__init__()
        every_n_train_steps, every_n_epochs, train_time_interval = self.__init_triggers(
            every_n_train_steps, every_n_epochs, train_time_interval
        )
        # Params
        self._monitor: Final[Optional[str]] = monitor
        self._verbose: Final[bool] = verbose
        self._mode: Final[Literal["min", "max"]] = mode
        self._auto_insert_metric_name: Final[bool] = auto_insert_metric_name
        self._save_on_train_epoch_end: Final[Optional[int]] = save_on_train_epoch_end
        self._enable_version_counter: Final[bool] = enable_version_counter
        self._save_last: Final[Optional[bool]] = save_last
        self._save_top_k: Final[int] = save_top_k
        self._save_weights_only: Final[bool] = save_weights_only
        self.filename: Optional[str] = filename
        self._every_n_train_steps: Final[int] = every_n_train_steps
        self._train_time_interval: Final[Optional[datetime.timedelta]] = train_time_interval
        self._every_n_epochs: Final[int] = every_n_epochs

        # State
        self._kth_value: jax.Array = self.__init_monitor_mode(mode)
        self._dirpath: Optional[str] = (
            os.path.realpath(os.path.expanduser(dirpath)) if dirpath else dirpath
        )
        self.last_model_path = ""
        self._last_time_checked: Optional[float] = None
        self._current_score: Optional[jax.Array] = None
        self._last_checkpoint_saved = ""
        self._best_model_path = ""
        self._best_k_models: dict[str, jax.Array] = {}
        self._kth_best_model_path = ""
        self._best_model_score: Optional[jax.Array] = None
        # Keep track of the global number of optimizer steps
        self._last_global_step_saved = 0

    @staticmethod
    def __init_triggers(
        every_n_train_steps: Optional[int],
        every_n_epochs: Optional[int],
        train_time_interval: Optional[datetime.timedelta],
    ) -> [int, int, Optional[datetime.timedelta]]:
        """Init triggers."""
        # Default to running once after each validation epoch if neither
        # every_n_train_steps nor every_n_epochs is set
        if every_n_train_steps is None and every_n_epochs is None and train_time_interval is None:
            every_n_epochs = 1
            every_n_train_steps = 0
            _LOGGER.debug(
                "Both every_n_train_steps and every_n_epochs are not set. Setting every_n_epochs=1"
            )
        else:
            every_n_epochs = every_n_epochs or 0
            every_n_train_steps = every_n_train_steps or 0

        return every_n_train_steps, every_n_epochs, train_time_interval

    @staticmethod
    def __init_monitor_mode(mode: str) -> jax.Array:
        """Init monitor mode."""
        jnp_info = jnp.array(float("inf" if mode == "min" else "-inf"))
        mode_dict = {"min": jnp_info, "max": -jnp_info}

        if mode not in mode_dict:
            raise exceptions.MisconfigurationException(
                f"`mode` can be {', '.join(mode_dict.keys())} but got {mode}"
            )

        return mode_dict[mode]

    @property
    def dirpath(self) -> Optional[str]:
        """Dirpath function."""
        return self._dirpath

    @property
    def best_model_path(self) -> str:
        """Best model path."""
        return self._best_model_path

    @property
    def best_model_score(self) -> Optional[jax.Array]:
        """Best model score."""
        return self._best_model_score

    @property
    def save_top_k(self) -> int:
        """Save top k."""
        return self._save_top_k

    @override
    def setup(self, trainer: "reax.Trainer", stage: "reax.Stage", /) -> None:
        """Setup function."""
        dirpath = self.__resolve_ckpt_dir(trainer)
        dirpath = trainer.strategy.broadcast(dirpath)
        self._dirpath = dirpath
        if trainer.is_global_zero and stage == "fit":
            self.__warn_if_dir_not_empty(self.dirpath)

    @override
    def on_fit_start(self, trainer: "reax.Trainer", stage: "reax.stages.Fit", /) -> None:
        self._last_time_checked = time.monotonic()

    @override
    def on_train_batch_end(
        self, trainer: "reax.Trainer", stage: "reax.stages.Train", /, *_
    ) -> None:
        """Save checkpoint on train batch end if we meet the criteria for `every_n_train_steps`"""
        if self._should_skip_saving_checkpoint(trainer, stage):
            return
        skip_batch = self._every_n_train_steps < 1 or (
            trainer.global_updates % self._every_n_train_steps != 0
        )

        train_time_interval = self._train_time_interval
        skip_time = True
        now = time.monotonic()
        if train_time_interval:
            prev_time_check = self._last_time_checked
            skip_time = (
                prev_time_check is None
                or (now - prev_time_check) < train_time_interval.total_seconds()
            )
            # in case we have time differences across ranks
            # broadcast the decision on whether to checkpoint from rank 0 to avoid possible hangs
            skip_time = trainer.strategy.broadcast(skip_time)

        if skip_batch and skip_time:
            return
        if not skip_time:
            self._last_time_checked = now

        monitor_candidates = self._monitor_candidates(stage, trainer)
        self._save_topk_checkpoint(trainer, monitor_candidates)
        self._save_last_checkpoint(trainer, monitor_candidates)

    @override
    def on_train_epoch_end(self, trainer: "reax.Trainer", stage: "reax.stages.Train", /) -> None:
        """On train epoch end."""
        if not self._should_skip_saving_checkpoint(
            trainer, stage
        ) and self._should_save_on_train_epoch_end(stage):
            if isinstance(stage, stages.Train):
                self._do_save(trainer, self._monitor_candidates(stage, trainer))

    @override
    def on_validation_epoch_end(
        self, trainer: "reax.Trainer", stage: "reax.stages.Validate", /
    ) -> None:
        """On validation epoch end."""
        if not self._should_skip_saving_checkpoint(
            trainer, stage
        ) and not self._should_save_on_train_epoch_end(stage):
            if isinstance(stage, stages.Validate):
                self._do_save(trainer, self._monitor_candidates(stage, trainer))

    def _monitor_candidates(
        self, stage: "reax.stages.EpochStage", trainer: "reax.Trainer", /
    ) -> dict[str, jax.Array]:
        """Monitor candidates."""
        monitor_candidates: dict = copy.deepcopy(stage.listener_metrics)
        monitor_candidates.setdefault("epoch", trainer.current_epoch)
        monitor_candidates.setdefault("step", trainer.global_updates)
        return monitor_candidates

    def check_monitor_top_k(self, current: Optional[jax.Array] = None) -> bool:
        """Check monitor top k."""
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
        """Do save."""
        if self._every_n_epochs >= 1 and (trainer.current_epoch + 1) % self._every_n_epochs == 0:
            self._save_topk_checkpoint(trainer, monitor_candidates)

        self._save_last_checkpoint(trainer, monitor_candidates)

    def _should_skip_saving_checkpoint(
        self, trainer: "reax.Trainer", stage: "reax.stages.EpochStage"
    ) -> bool:
        return (
            bool(stage.fast_dev_run)  # disable checkpointing with fast_dev_run
            # don't save anything during non-fit
            or not stage.enable_checkpointing
            or trainer.sanity_checking  # don't save anything during sanity check
            # already saved at the last step
            or self._last_global_step_saved == trainer.global_updates
        )

    def _should_save_on_train_epoch_end(self, stage: stages.EpochStage) -> bool:
        """Should save on train epoch end."""
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
        """Save topk checkpoint."""
        if self._save_top_k == 0:
            return

        if self._monitor is None:
            self._save_none_monitor_checkpoint(trainer, monitor_candidates)
        else:
            self._save_monitor_checkpoint(trainer, monitor_candidates)

    def _save_last_checkpoint(
        self, trainer: "reax.Trainer", monitor_candidates: dict[str, jax.Array]
    ) -> None:
        """Save last checkpoint."""
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
        """Save monitor checkpoint."""
        assert self._monitor is not None
        current = monitor_candidates.get(self._monitor)

        if self.check_monitor_top_k(current):
            assert current is not None
            self._update_best_and_save(current, trainer, monitor_candidates)
        elif self._verbose:
            epoch = monitor_candidates["epoch"]
            step = monitor_candidates["step"]
            rank_zero.rank_zero_info(
                f"Epoch {epoch:d}, global step {step:d}: {self._monitor!r} was not in top "
                f"{self.save_top_k}"
            )

    def _save_none_monitor_checkpoint(
        self, trainer: "reax.Trainer", monitor_candidates: dict[str, jax.Array]
    ) -> None:
        """Save none monitor checkpoint."""
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
        """Update best and save."""
        k = len(self._best_k_models) + 1 if self._save_top_k == -1 else self._save_top_k

        del_filepath = None
        if len(self._best_k_models) == k and k > 0:
            del_filepath = self._kth_best_model_path
            self._best_k_models.pop(del_filepath)

        # do not save nan, replace with +/- inf
        if isinstance(current, jax.Array) and jnp.isnan(current):
            current = jnp.array(float("inf" if self._mode == "min" else "-inf"))

        filepath: Final[str] = self._get_metric_interpolated_filepath_name(
            trainer, monitor_candidates, del_filepath
        )

        # save the current score
        self._current_score = current
        self._best_k_models[filepath] = current

        if len(self._best_k_models) == k:
            # monitor dict has reached k elements
            _op = max if self._mode == "min" else min
            self._kth_best_model_path = _op(self._best_k_models, key=self._best_k_models.get)
            self._kth_value = self._best_k_models[self._kth_best_model_path]

        _op = min if self._mode == "min" else max
        self._best_model_path = _op(self._best_k_models, key=self._best_k_models.get)
        self._best_model_score = self._best_k_models[self._best_model_path]

        if self._verbose:
            epoch = monitor_candidates["epoch"]
            step = monitor_candidates["step"]
            rank_zero.rank_zero_info(
                f"Epoch {epoch:d}, global step {step:d}: {self._monitor!r} reached {current:0.5f}"
                f" (best {self.best_model_score:0.5f}), saving model to {filepath!r} as top {k}"
            )
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
        """Get checkpoint filepath."""
        filepath = self.format_checkpoint_name(metrics, filename, version)
        return os.path.join(self._get_dirpath(trainer), filepath)

    def _get_dirpath(self, trainer: "reax.Trainer") -> str:
        """Get dirpath."""
        if self._dirpath is not None:
            return self._dirpath

        if trainer.default_root_dir is not None:
            return trainer.default_root_dir

        return ""

    def _get_metric_interpolated_filepath_name(
        self,
        trainer: "reax.Trainer",
        monitor_candidates: dict[str, jax.Array],
        del_filepath: Optional[str] = None,
    ) -> str:
        """Get metric interpolated filepath name."""
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
        """Format checkpoint name."""
        if not filename:
            # filename is not set, use default name
            filename = "{epoch}" + self.CHECKPOINT_JOIN_CHAR + "{step}"

        # check and parse user passed keys in the string
        groups = re.findall(r"(\{.*?)[:\}]", filename)

        # sort keys from longest to shortest to avoid replacing substring
        # eg: if keys are "epoch" and "epoch_test", the latter must be replaced first
        groups = sorted(groups, key=len, reverse=True)

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
        """Save checkpoint."""
        trainer.save_checkpoint(filepath, self._save_weights_only)
        self._last_global_step_saved = trainer.global_updates
        self._last_checkpoint_saved = filepath

        # notify loggers
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(weakref.proxy(self))

    def _should_remove_checkpoint(
        self, trainer: "reax.Trainer", previous: str, current: str
    ) -> bool:
        """Checks if the previous checkpoint should be deleted.

        A checkpoint won't be deleted if any of the cases apply:
        - The previous checkpoint is the same as the current checkpoint (means the old was already
            overwritten by new)
        - The previous checkpoint is not in the current checkpoint directory and the filesystem is
            local
        - The previous checkpoint is the checkpoint the Trainer resumed from and the filesystem is
            local
        """
        if previous == current:
            return False

        previous = pathlib.Path(previous).absolute()
        dirpath = pathlib.Path(self._get_dirpath(trainer)).absolute()
        return dirpath in previous.parents

    def __warn_if_dir_not_empty(self, dirpath: PathType) -> None:
        """Warn if dir not empty."""
        dirpath = pathlib.Path(dirpath)

        if self.save_top_k != 0 and dirpath.is_dir() and any(dirpath.iterdir()):
            rank_zero.rank_zero_warn(f"Checkpoint directory {dirpath} exists and is not empty.")

    def __resolve_ckpt_dir(self, trainer: "reax.Trainer") -> PathType:
        """Determines model checkpoint save directory at runtime.

                Reference attributes from the trainer's logger to
        determine where to save checkpoints. The path for saving weights is set in this priority:

                1.  The ``ModelCheckpoint``'s ``dirpath`` if passed in
                2.  The ``Logger``'s ``log_dir`` if the trainer has loggers
                3.  The ``Trainer``'s ``default_root_dir`` if the trainer has no loggers

                The path gets extended with subdirectory "checkpoints".
        """
        if self.dirpath is not None:
            # short circuit if dirpath was passed to ModelCheckpoint
            return self.dirpath

        if len(trainer.loggers) > 0:
            if trainer.loggers[0].save_dir is not None:
                save_dir = trainer.loggers[0].save_dir
            else:
                save_dir = trainer.default_root_dir
            name = trainer.loggers[0].name
            version = trainer.loggers[0].version
            version = version if isinstance(version, str) else f"version_{version}"
            ckpt_path = os.path.join(save_dir, str(name), version, "checkpoints")
        else:
            # if no loggers, use default_root_dir
            ckpt_path = os.path.join(trainer.default_root_dir, "checkpoints")

        return ckpt_path
