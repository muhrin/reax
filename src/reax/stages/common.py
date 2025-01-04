from typing import TYPE_CHECKING, Any, Callable, TypedDict, Union, cast

import jax

from .. import data
from ..utils import events

if TYPE_CHECKING:
    import reax


__all__ = ("MetricResults", "StageListener")

StageEvents = events.EventGenerator["StageListener"]


class StageListener:
    def on_stage_starting(self, stage: "reax.Stage", /):
        """The stage is about to start."""

    def on_stage_started(self, stage: "reax.Stage", /):
        """The stage has started, all initialisation if complete."""

    def on_stage_iter_starting(self, stage: "reax.Stage", step: int, /):
        """The stage is about to start an iteration."""

    def on_stage_iter_ending(self, stage: "reax.Stage", step: int, outputs: Any, /):
        """The stage just finished processing an iteration."""

    def on_stage_ending(self, stage: "reax.Stage", /):
        """The stage is about to finish."""

    def on_stage_ended(self, stage: "reax.Stage", /):
        """The stage has ended.

        It will not be mutated after this point until it is starting again.
        """


class MetricResults(TypedDict):
    callback: dict[str, jax.Array]
    log: dict[str, jax.Array]
    pbar: dict[str, float]


class Stopper:
    """Class used by loops to manage their stop conditions."""

    def __init__(self):
        """Init function."""
        self._stop_requested = False
        self._conditions: list[Callable[[], bool]] = []

    @property
    def stop_requested(self) -> bool:
        """Stop requested."""
        return self._stop_requested

    @property
    def can_stop(self):
        """Can stop."""
        return all(condition() for condition in self._conditions)

    def do_stop(self) -> bool:
        """Do stop."""
        return self._stop_requested and self.can_stop

    def set(self):
        """Set function."""
        self._stop_requested = True

    def get(self) -> bool:
        """Get function."""
        return self._stop_requested

    def add_condition(self, condition: Callable[[], bool]) -> None:
        """Add condition."""
        self._conditions.append(condition)


def batches_limit(
    batch_limit: Union[int, float], dataloader: "reax.DataLoader"
) -> Union[int, float]:
    """Return a maximum number of batches given a dataloader and an optional batches limit.

    If the dataloader has fewer entries than the batch limit, then this will be used, otherwise
    the batches limit.

    .. note:: Will return `float("inf")` if there is no limit.
    :param batch_limit: The batches limit.
    :type batch_limit: Union[int, float]
    :param dataloader: The dataloader.
    :type dataloader: "reax.DataLoader"
    :return: The maximum number of batches.
    :rtype: Union[int, float]
    """
    dataloader_size = data.sized_len(dataloader)
    if isinstance(batch_limit, int):
        if dataloader_size is None:
            return batch_limit

        return min(batch_limit, dataloader_size)

    if isinstance(batch_limit, float):
        if batch_limit == float("inf") or batch_limit == 1.0:
            if dataloader_size is not None:
                return dataloader_size
            return float("inf")

        if dataloader_size is not None:
            # batch_limit is a finite float and we have a dataloader size
            batch_limit = cast(float, batch_limit)
            return int(round(batch_limit * dataloader_size))

        raise ValueError(
            f"Cannot determine number of batches from dataloader and batch_limit is "
            f"{batch_limit}"
        )

    # We can't say anything other than just 'go to the end'
    return float("inf")
