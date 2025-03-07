from typing import TYPE_CHECKING, Optional

from lightning_utilities.core import rank_zero

if TYPE_CHECKING:
    import reax


class _CheckpointConnector:
    def __init__(self, trainer: "reax.Trainer") -> None:
        """Init function."""
        self._trainer = trainer

    def _get_checkpoint_path(self) -> Optional[str]:
        """Get checkpoint path."""
        checkpointers = self._trainer.checkpoint_listeners
        if not checkpointers:
            return None

        if len(checkpointers) > 1:
            fn = self._get_checkpoint_path
            rank_zero.rank_zero_warn(
                f'`.{fn}(ckpt_path="best")` found Trainer with multiple `ModelCheckpoint`'
                " callbacks. The best checkpoint path from first checkpoint callback will be used."
            )

        checkpointer = checkpointers[0]
        return getattr(checkpointer, "best_model_path", None)
