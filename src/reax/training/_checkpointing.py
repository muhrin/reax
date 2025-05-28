import abc
import logging
import os
from typing import TYPE_CHECKING, Any, Final

import flax.serialization
from typing_extensions import override

if TYPE_CHECKING:
    import reax


__all__ = (
    "get_default_checkpointing",
    "Checkpointing",
    "CheckpointDict",
    "MsgpackCheckpointing",
)

_LOGGER = logging.getLogger(__name__)

CheckpointDict = dict[str, Any]

PARAMS: Final[str] = "parameters"
REAX_VERSION: Final[str] = "reax_version"
GLOBAL_STEP: Final[str] = "global_step"
EPOCH: Final[str] = "epoch"


def dump_checkpoint(module: "reax.Module", weights_only: bool = False) -> CheckpointDict:
    """Creating a model checkpoint dictionary object from various component states."""
    import reax

    trainer = module.trainer

    checkpoint = {
        EPOCH: trainer.current_epoch,
        GLOBAL_STEP: trainer.global_updates,
        REAX_VERSION: reax.__version__,
        PARAMS: module.parameters(),
    }
    if not weights_only:
        checkpoint["state_dict"] = module.state_dict()

    return checkpoint


class Checkpointing(abc.ABC):
    @abc.abstractmethod
    def save(self, checkpoint: CheckpointDict, filepath: str):
        """Save the checkpoint dictionary to the given path"""

    @abc.abstractmethod
    def load(self, filepath: str) -> CheckpointDict:
        """Load the checkpoint dictionary from the given path"""


class MsgpackCheckpointing(Checkpointing):
    @override
    def save(self, checkpoint: CheckpointDict, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as file:
            file.write(flax.serialization.msgpack_serialize(checkpoint))

    @override
    def load(self, filepath: str) -> CheckpointDict:
        with open(filepath, "rb") as file:
            return flax.serialization.msgpack_restore(file.read())


def get_default_checkpointing() -> Checkpointing:
    return MsgpackCheckpointing()
