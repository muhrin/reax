from typing import TYPE_CHECKING, Optional

from typing_extensions import deprecated

from reax import listeners as listeners_

if TYPE_CHECKING:
    import reax


class TrainerDeprecatedMixin:
    """Mixin that enables functionality that is deprecated in REAX but may be closer to the way
    lightning works"""

    # region Deprecated

    @property
    @deprecated("REAX uses the term 'update' instead of 'step', please use `.global_updates`")
    def global_step(self) -> int:
        """Get the global number of optimizer updates."""
        return self.global_updates

    @property
    @deprecated(
        "REAX uses the term 'listener' instead of 'callback, please use `.listener_metrics`"
    )
    def callback_metrics(self) -> dict:
        """Callback metrics."""
        # Kept for compatibility with Lightning
        return self.listener_metrics

    @property
    @deprecated(
        "REAX uses the term 'listener' instead of 'callback, please use `.progress_bar_listeners`"
    )
    def progress_bar_callbacks(self) -> list["reax.listeners.ProgressBar"]:
        """The first :class:`~reax.listeners.ProgressBar` listener in the
        Trainer.listeners list, or ``None`` if it doesn't exist."""
        return self.progress_bar_listeners

    @property
    @deprecated(
        "REAX uses the term 'listener' instead of 'callback, please use `.progress_bar_callback`"
    )
    def progress_bar_callback(self) -> Optional["reax.listeners.ProgressBar"]:
        """The first :class:`~reax.listeners.ProgressBar` listener in the
        Trainer.listeners list, or ``None`` if it doesn't exist."""
        return self.progress_bar_listener

    @property
    @deprecated(
        "REAX uses the term 'listener' instead of 'callback, please use `.checkpoint_listeners`"
    )
    def checkpoint_callbacks(self) -> list["reax.listeners.Checkpointer"]:
        """A list of all instances of :class:`~reax.listeners.model_checkpoint.ModelCheckpoint`
        found in the Trainer.listeners list."""
        # Kept for compatibility with Lightning
        return self.checkpoint_listeners

    @property
    @deprecated(
        "REAX uses the term 'listener' instead of 'callback, please use `.checkpoint_listener`"
    )
    def checkpoint_callback(self) -> Optional["reax.listeners.Checkpointer"]:
        """The first :class:`~reax.listeners.model_checkpoint.ModelCheckpoint` callback in the
        Trainer.listeners list, or ``None`` if it doesn't exist."""
        # Kept for compatibility with Lightning
        return self.checkpoint_listener

    @property
    @deprecated(
        "REAX uses the term 'listener' instead of 'callback, please use `.early_stopping_listener`"
    )
    def early_stopping_callback(self) -> Optional[listeners_.EarlyStopping]:
        """The first :class:`~reax.listeners.early_stopping.EarlyStopping` listener in the
        Trainer.callbacks list, or ``None`` if it doesn't exist."""
        return self.early_stopping_listener

    @property
    @deprecated(
        "REAX uses the term 'listener' instead of 'callback, please use `.early_stopping_listeners`"
    )
    def early_stopping_callbacks(self) -> list[listeners_.EarlyStopping]:
        """A list of all instances of :class:`~reax.listeners.early_stopping.EarlyStopping` found in
        the Trainer.callbacks list."""
        return self.early_stopping_listeners

    # endregion
