from typing import Any, Optional

from reax import hooks

__all__ = ("ProgressBar",)


class ProgressBar(hooks.TrainerListener):
    r"""Base class for progress bars"""

    def __init__(self) -> None:
        self._current_eval_dataloader_idx: Optional[int] = None

    def disable(self) -> None:
        """You should provide a way to disable the progress bar."""
        raise NotImplementedError

    def enable(self) -> None:
        """You should provide a way to enable the progress bar.

        The :class:`~lightning.pytorch.trainer.trainer.Trainer` will call this in e.g. pre-training
        routines like the :ref:`learning rate finder <advanced/training_tricks:Learning Rate Finder>`.
        to temporarily enable and disable the training progress bar.

        """
        raise NotImplementedError

    def print(self, *args: Any, **kwargs: Any) -> None:
        """Subclasses should provide a way to print without breaking the progress bar."""
        print(*args, **kwargs)
