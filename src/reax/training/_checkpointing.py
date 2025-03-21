import logging
import os
import pickle  # nosec
from typing import TYPE_CHECKING

from reax import typing

if TYPE_CHECKING:
    import reax


_LOGGER = logging.getLogger(__name__)


def save_checkpoint(module: "reax.Module", filepath: typing.Path, weights_only: bool = True):
    ckpt = dump_checkpoint(module, weights_only=weights_only)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as file:
        pickle.dump(ckpt, file)


def load_checkpoint(module: "reax.Module", filepath: typing.Path, weights_only: bool = True):
    with open(filepath, "rb") as file:
        ckpt = pickle.load(file)  # nosec: B301
        module.set_parameters(ckpt["parameters"])
        if not weights_only:
            module.load_state(ckpt["state_dict"])

    return module


def dump_checkpoint(module: "reax.Module", weights_only: bool = False) -> dict:
    """Creating a model checkpoint dictionary object from various component states."""
    import reax

    trainer = module.trainer

    checkpoint = {
        "epoch": trainer.current_epoch,
        "global_step": trainer.global_updates,
        "reax_version": reax.__version__,
        "parameters": module.parameters(),
    }
    if not weights_only:
        checkpoint["state_dict"] = module.state_dict()

    return checkpoint
