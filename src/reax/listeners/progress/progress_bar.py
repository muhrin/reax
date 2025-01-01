from typing import Any, Optional

from reax import hooks

__all__ = ("ProgressBar",)


class ProgressBar(hooks.TrainerListener):
    r"""Base class for progress bars."""

    def __init__(self) -> None:
        self._current_eval_dataloader_idx: Optional[int] = None

    def disable(self) -> None:
        """Disable the progress bar."""
        raise NotImplementedError

    def enable(self) -> None:
        """Enable the progress bar."""
        raise NotImplementedError

    def print(self, *args: Any, **kwargs: Any) -> None:
        """Print the progress bar status."""
        print(*args, **kwargs)
