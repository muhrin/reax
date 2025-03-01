import os
import random
from typing import Optional, Union

import jax
import numpy

__all__ = ("seed_everything", "Generator")


def seed_everything(seed: Optional[int], workers: bool = False):
    """Seed everything."""
    numpy.random.seed(seed)
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass

    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"

    return seed


class Generator:
    """Random number generator utility class"""

    def __init__(self, seed: Union[int, jax.typing.ArrayLike] = None, key: jax.Array = None):
        if seed is not None:
            self._key = jax.random.key(seed)
        elif key is not None:
            self._key = key
        else:
            self._key = jax.random.key(0)

    def make_key(self, num: Union[int, tuple[int, ...]] = 1) -> jax.Array:
        """Make onr or more random keys, updating the internal state"""
        self._key, subkey = jax.random.split(self._key, num + 1)
        return subkey
