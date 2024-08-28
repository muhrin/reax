from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import jax.numpy as jnp
import jax.typing

from reax import hooks

if TYPE_CHECKING:
    import reax

__all__ = ("EarlyStopping",)

MonitorOp = Callable[[jax.typing.ArrayLike, jax.typing.ArrayLike], jax.typing.ArrayLike]


class EarlyStopping(hooks.TrainerListener):
    """
    Early stopping will monitor validation metrics and attempt to stop the trainer if the
    `monitor` metric increases for `patience` epochs or more.
    """

    mode_dict: dict[str, MonitorOp] = {"min": jnp.less, "max": jnp.greater}

    def __init__(
        self,
        monitor: str,
        min_delta: float = 0.0,
        patience: int = 3,
        mode: Literal["min", "max"] = "min",
    ):
        self._monitor = monitor
        self._min_delta = min_delta
        self._patience = patience
        self._mode = mode

        self._best_score = float("inf")
        self._wait_count: int = 0

    @property
    def monitor_op(self) -> MonitorOp:
        return self.mode_dict[self._mode]

    def on_epoch_ending(  # pylint: disable=unused-argument
        self, trainer: "reax.Trainer", stage: "reax.stages.EpochStage", metrics: dict
    ) -> None:
        current = metrics.get(f"validation.{self._monitor}")
        if current is None:
            return

        if self.monitor_op(  # pylint: disable=not-callable
            current - self._min_delta, self._best_score
        ):
            self._best_score = current
            self._wait_count = 0
        else:
            self._wait_count += 1
            if self._wait_count >= self._patience:
                # Ask the trainer to stop
                trainer.should_stop = True
