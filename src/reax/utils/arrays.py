from typing import Union

import jax.numpy as jnp
import jax.typing


def to_base(array: jax.typing.ArrayLike) -> Union[int, float, list]:
    if jnp.isscalar(array):
        return array.item()

    return array.tolist()
