import os
import random

import numpy

__all__ = ("seed_everything",)


def seed_everything(seed: int | None, workers: bool = False):
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
