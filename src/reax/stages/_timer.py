import datetime
import logging
import re
import time
from typing import TYPE_CHECKING, Any, Final

from typing_extensions import override

from . import common
from .. import exceptions

if TYPE_CHECKING:
    import reax

_LOGGER = logging.getLogger(__name__)


__all__ = ("StageTimer",)


class StageTimer(common.StageListener):
    def __init__(self, max_time: str | datetime.timedelta | dict[str, int] = None):
        # Params
        self._max_time: Final[float] = self._init_max_time(max_time)

        # State
        self._start_time: float | None = None

    @staticmethod
    def _init_max_time(duration) -> float:
        if isinstance(duration, str):
            duration_match = re.fullmatch(r"(\d+):(\d\d):(\d\d):(\d\d)", duration.strip())
            if not duration_match:
                raise exceptions.MisconfigurationException(
                    f"`max_time={duration!r}` is not a valid duration. "
                    "Expected a string in the format DD:HH:MM:SS."
                )
            duration = datetime.timedelta(
                days=int(duration_match.group(1)),
                hours=int(duration_match.group(2)),
                minutes=int(duration_match.group(3)),
                seconds=int(duration_match.group(4)),
            )
        elif isinstance(duration, dict):
            duration = datetime.timedelta(**duration)

        return duration.total_seconds()

    def start(self) -> bool:
        """Manually start the stage timer"""
        if self._start_time is not None:
            return False

        self._start_time = time.monotonic()
        return True

    def start_time(self) -> float | None:
        return self._start_time

    def time_elapsed(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.monotonic() - self._start_time

    def time_remaining(self) -> float:
        return self._max_time - self.time_elapsed()

    @override
    def on_stage_start(self, stage: "reax.Stage", /):
        """Called when the stage is started."""
        super().on_stage_started(stage)
        if self._start_time is None:
            self._start_time = time.monotonic()
        self._stop_check(stage)

    @override
    def on_stage_iter_ended(self, stage: "reax.Stage", step: int, outputs: Any, /):
        """Called when the stage iteration ended."""
        super().on_stage_iter_ended(stage, step, outputs)
        self._stop_check(stage)

    def _stop_check(self, stage: "reax.Stage"):
        elapsed = self.time_elapsed()
        if elapsed > self._max_time:
            stage.stop("Max time elapsed")
            _LOGGER.info(
                "Time limit reached. Elapsed time is %.2f. Signaling Trainer to stop.", elapsed
            )
