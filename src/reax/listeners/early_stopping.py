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
# https://github.com/Lightning-AI/pytorch-lightning/blob/0324a20f00235c7a10a235a44326811ba42b6ae4/src/lightning/pytorch/callbacks/early_stopping.py
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
r"""
Early Stopping
^^^^^^^^^^^^^^

Monitor a metric and stop training when it stops improving.

"""
from collections.abc import Callable
import logging
from typing import TYPE_CHECKING, Final, Literal, Optional

import beartype
import jax.numpy as jnp
import jax.typing
import jaxtyping as jt
from typing_extensions import override

from reax import hooks, stages
from reax.lightning import rank_zero

if TYPE_CHECKING:
    import reax

_LOGGER = logging.getLogger(__name__)

__all__ = ("EarlyStopping",)

MonitorOp = Callable[[jax.typing.ArrayLike, jax.typing.ArrayLike], jax.typing.ArrayLike]


class EarlyStopping(hooks.TrainerListener):
    r"""Monitor a metric and stop training when it stops improving.

    :param monitor: Quantity to be monitored.
    :type monitor: str
    :param min_delta: Minimum change in the monitored quantity to qualify as an improvement, i.e.
        an absolute change of less than or equal to ``min_delta``, will count as no improvement,
        defaults to 0.0.
    :type min_delta: float, optional
    :param patience: Number of checks with no improvement after which training will be stopped.
        Under the default configuration, one check happens after every training epoch.
        However, the frequency of validation can be modified by setting various parameters on the
        ``Trainer``, for example ``check_val_every_n_epoch`` and ``val_check_interval``.

        .. note::

        It must be noted that the patience parameter counts the number of validation checks with
        no improvement, and not the number of training epochs. Therefore, with parameters
        ``check_val_every_n_epoch=10`` and ``patience=3``, the trainer will perform at least 40
        training epochs before being stopped, defaults to 3.
    :type patience: int, optional
    :param verbose: Verbosity mode, defaults to False.
    :type verbose: bool, optional
    :param mode: One of ``'min'``, ``'max'``. In ``'min'`` mode, training will stop when the
        quantity monitored has stopped decreasing and in ``'max'`` mode it will stop when the
        quantity monitored has stopped increasing, defaults to "min".
    :type mode: Literal["min", "max"], optional
    :param strict: Whether to crash the training if `monitor` is not found in the validation
        metrics, defaults to True.
    :type strict: bool, optional
    :param check_finite: When set ``True``, stops training when the monitor becomes NaN or infinite,
        defaults to True.
    :type check_finite: bool, optional
    :param stopping_threshold: Stop training immediately once the monitored quantity reaches this
        threshold, defaults to None.
    :type stopping_threshold: Optional[float], optional
    :param divergence_threshold: Stop training as soon as the monitored quantity becomes worse than
        this threshold, defaults to None.
    :type divergence_threshold: Optional[float], optional
    :param check_on_train_epoch_end: Whether to run early stopping at the end of the training epoch.
        If this is ``False``, then the check runs at the end of the validation, defaults to None.
    :type check_on_train_epoch_end: bool, optional
    :param log_rank_zero_only: When set ``True``, logs the status of the early stopping listener
        only for rank 0 process, defaults to False.
    :type log_rank_zero_only: bool, optional
    :raises MisconfigurationException: If ``mode`` is none of ``"min"`` or ``"max"``.
    :raises RuntimeError: If the metric ``monitor`` is not available.

    Example::

        >>> from reax import Trainer
        >>> from reax.listeners import EarlyStopping
        >>> early_stopping = EarlyStopping('val_loss')
        >>> trainer = Trainer(listeners=[early_stopping])
    """

    mode_dict: dict[str, MonitorOp] = {"min": jnp.less, "max": jnp.greater}
    order_dict = {"min": "<", "max": ">"}

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        monitor: str,
        *,
        min_delta: float = 0.0,
        patience: int = 3,
        verbose: bool = False,
        mode: Literal["min", "max"] = "min",
        strict: bool = True,
        check_finite: bool = True,
        stopping_threshold: Optional[float] = None,
        divergence_threshold: Optional[float] = None,
        check_on_train_epoch_end: bool = None,
        log_rank_zero_only: bool = False,
    ):
        # Params
        self._monitor: Final[str] = monitor
        self._patience: Final[int] = patience
        self._verbose: Final[bool] = verbose
        self._mode: Final[Literal["min", "max"]] = mode
        self._strict: Final[bool] = strict
        self._check_finite = check_finite
        self._stopping_threshold = stopping_threshold
        self._divergence_threshold = divergence_threshold
        self._wait_count = 0
        self.__check_on_train_epoch_end = check_on_train_epoch_end
        self._log_rank_zero_only = log_rank_zero_only
        self._min_delta: Final[float] = min_delta if mode == "max" else -min_delta

        # State
        self._check_on_train_epoch_end = check_on_train_epoch_end
        self._best_score = float("inf")
        self._wait_count: int = 0
        self._stopped_epoch = 0

    @property
    def monitor_op(self) -> MonitorOp:
        """Monitor op."""
        return self.mode_dict[self._mode]

    @property
    def best_score(self) -> float:
        """Best score."""
        return self._best_score

    @property
    def stopped_epoch(self) -> int:
        return self._stopped_epoch

    def _should_skip_check(self, trainer: "reax.Trainer") -> bool:
        """Should skip check."""
        return not isinstance(trainer.stage, stages.Fit)

    @override
    def on_fit_start(self, trainer: "reax.Trainer", stage: "reax.stages.Fit", /) -> None:
        """On fit start."""
        if self.__check_on_train_epoch_end is None:
            # if the user runs validation multiple times per training epoch or multiple training
            # epochs without validation, then we run after validation instead of on train epoch end
            self._check_on_train_epoch_end = (
                stage.val_check_interval == 1.0 and stage.check_val_every_n_epoch == 1
            )

    @override
    def on_train_epoch_end(self, trainer: "reax.Trainer", stage: "reax.stages.Train", /) -> None:
        """On train epoch end."""
        if not self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return
        self._run_early_stopping_check(trainer, stage)

    @override
    def on_validation_end(self, trainer: "reax.Trainer", stage: "reax.stages.Validate", /) -> None:
        """On validation end."""
        if self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return
        self._run_early_stopping_check(trainer, stage)

    def _run_early_stopping_check(
        self, trainer: "reax.Trainer", stage: "reax.stages.EpochStage"
    ) -> None:
        """Check the early stopping condition and tell the Trainer to stop if needed."""
        logs = trainer.listener_metrics

        if stage.fast_dev_run or not self._validate_condition_metric(logs):
            # disable early_stopping with fast_dev_run or short circuit if metric not present
            return

        current = jnp.squeeze(logs[self._monitor])
        should_stop, reason = self._evaluate_stopping_criteria(current)

        # stop every ddp process if any world process decides to stop
        # should_stop = trainer.strategy.reduce_boolean_decision(should_stop, all=False)
        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop:
            self._stopped_epoch = trainer.current_epoch
        if reason and self._verbose:
            self._log_info(trainer, reason, self._log_rank_zero_only)

    def _validate_condition_metric(self, logs: dict[str, jax.typing.ArrayLike]) -> bool:
        """Validate condition metric."""
        monitor_val = logs.get(self._monitor)

        error_msg = (
            f"Early stopping conditioned on metric `{self._monitor}` which is not available."
            " Pass in or modify your `EarlyStopping` listener to use any of the following:"
            f' `{"`, `".join(list(logs.keys()))}`'
        )

        if monitor_val is None:
            if self._strict:
                raise RuntimeError(error_msg)
            if self._verbose > 0:
                rank_zero.rank_zero_warn(error_msg, category=RuntimeWarning)

            return False

        return True

    def _evaluate_stopping_criteria(
        self, current: jax.typing.ArrayLike
    ) -> tuple[bool, Optional[str]]:
        """Evaluate stopping criteria."""
        should_stop = False
        reason = None
        if self._check_finite and not jnp.isfinite(current):
            should_stop = True
            reason = (
                f"Monitored metric {self._monitor} = {current} is not finite."
                f" Previous best value was {self.best_score:.3f}. Signaling Trainer to stop."
            )
        elif self._stopping_threshold is not None and self.monitor_op(
            current, self._stopping_threshold
        ):
            should_stop = True
            reason = (
                f"Stopping threshold reached: {self._monitor} = {current} "
                f"{self.order_dict[self._mode]} {self._stopping_threshold}. "
                f"Signaling Trainer to stop."
            )
        elif self._divergence_threshold is not None and self.monitor_op(
            -current,
            -self._divergence_threshold,  # pylint: disable=invalid-unary-operand-type
        ):
            should_stop = True
            reason = (
                f"Divergence threshold reached: {self._monitor} = {current} "
                f"{self.order_dict[self._mode]} {self._divergence_threshold}."
                " Signaling Trainer to stop."
            )
        elif self.monitor_op(current - self._min_delta, self.best_score):
            should_stop = False
            reason = self._improvement_message(current)
            self._best_score = current
            self._wait_count = 0
        else:
            self._wait_count += 1
            if self._wait_count >= self._patience:
                should_stop = True
                reason = (
                    f"Monitored metric {self._monitor} did not improve in the last "
                    f"{self._wait_count} records. "
                    f"Best score: {self.best_score:.3f}. Signaling Trainer to stop."
                )

        return should_stop, reason

    def _improvement_message(self, current: jax.typing.ArrayLike) -> str:
        """Formats a log message that informs the user about an improvement in the monitored
        score."""
        if jnp.isfinite(self.best_score):
            msg = (
                f"Metric {self._monitor} improved by {abs(self.best_score - current):.3f} >="
                f" min_delta = {abs(self._min_delta)}. New best score: {current:.3f}"
            )
        else:
            msg = f"Metric {self._monitor} improved. New best score: {current:.3f}"
        return msg

    @staticmethod
    def _log_info(trainer: "reax.Trainer", message: str, log_rank_zero_only: bool) -> None:
        """Log info."""
        rank = trainer.global_rank if trainer.world_size > 1 else None
        message = rank_zero.rank_prefixed_message(message, rank)
        if rank is None or not log_rank_zero_only or rank == 0:
            _LOGGER.info(message)
