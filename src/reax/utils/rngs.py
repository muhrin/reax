import random
from typing import Optional

import numpy


def seed_everything(seed: Optional[int]):
    """Seed everything."""
    numpy.random.seed(seed)
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
