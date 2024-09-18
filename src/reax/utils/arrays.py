from typing import Union

import jax
import jax.numpy as jnp
import jax.typing
import numpy as np


def to_base(array: jax.typing.ArrayLike) -> Union[int, float, list]:
    if jnp.isscalar(array):
        return array.item()

    return array.tolist()


def to_scalar(array: jax.typing.ArrayLike) -> Union[float, int]:
    """Convert an array to a python scalar type"""
    if isinstance(array, (np.ndarray, jax.Array)):
        return array.item()

    return array
