import math
from typing import Optional, Union

__all__ = ("convert_inf",)


def convert_inf(x: Optional[Union[int, float]]) -> Optional[Union[int, float]]:
    """For code that doesn't support inf/nan values such as tqdm -> convert these to `None`."""
    if x is None or math.isinf(x) or math.isnan(x):
        return None
    return x
