from typing import TYPE_CHECKING

import fsspec
import fsspec.utils

from .. import exceptions, hooks
from .. import listeners as listeners_

if TYPE_CHECKING:
    import reax


def init_progress_bar(
    listeners: list[hooks.TrainerListener], enable_progress_bar: bool = True
) -> listeners_.ProgressBar | None:
    progress_bars = [
        listener for listener in listeners if isinstance(listener, listeners_.progress.ProgressBar)
    ]
    if len(progress_bars) > 1:
        raise exceptions.MisconfigurationException(
            "You added multiple progress bar listeners to the Trainer, but currently only one"
            " progress bar is supported."
        )
    if len(progress_bars) == 1:
        # the user specified the progress bar in the listeners list
        # so the trainer doesn't need to provide a default one
        if enable_progress_bar:
            return None

        # otherwise the user specified a progress bar listener but also
        # elected to disable the progress bar with the trainer flag
        progress_bar_listener = progress_bars[0]
        raise exceptions.MisconfigurationException(
            "Trainer was configured with `enable_progress_bar=False`"
            f" but found `{progress_bar_listener.__class__.__name__}` in listeners list."
        )

    if enable_progress_bar:
        return listeners_.progress.TqdmProgressBar()

    return None


def is_local_file_protocol(path: "reax.types.Path") -> bool:
    """Is local file protocol."""
    return fsspec.utils.get_protocol(str(path)) == "file"


def reorder_listeners(
    listeners: "list[reax.TrainerListener]",
) -> "list[reax.TrainerListener]":
    """Moves all the tuner specific listeners at the beginning of the list and all the
    `ModelCheckpoint` listeners to the end of the list. The sequential order within the group of
    checkpoint listeners is preserved, as well as the order of all other listeners.

    Args:
        listeners: A list of listeners.

    Return:
        A new list in which the first elements are tuner specific listeners and last elements are
        ModelCheckpoints if there were any present in the input.

    """
    tuner_listeners: list[reax.TrainerListener] = []
    other_listeners: list[reax.TrainerListener] = []
    checkpoint_listeners: list[reax.TrainerListener] = []

    for cb in listeners:
        # if isinstance(cb, (BatchSizeFinder, LearningRateFinder)):
        #     tuner_listeners.append(cb)
        if isinstance(cb, listeners_.Checkpointer):
            checkpoint_listeners.append(cb)
        else:
            other_listeners.append(cb)

    return tuner_listeners + other_listeners + checkpoint_listeners
