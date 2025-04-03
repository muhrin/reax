import types
from typing import Union

import jax
import jax.numpy as jnp
import jax.typing
import numpy as np


def to_base(array: jax.typing.ArrayLike) -> Union[int, float, list]:
    """To base."""
    if isinstance(array, (int, float)):
        return array

    if isinstance(array, list):
        return array

    if jnp.isscalar(array):
        return array.item()

    return array.tolist()


def to_scalar(array: jax.typing.ArrayLike) -> Union[float, int]:
    """Convert an array to a python scalar type."""
    if isinstance(array, (np.ndarray, jax.Array)):
        return array.item()

    return array


def infer_backend(pytree) -> types.ModuleType:
    """Try to infer a backend from the passed pytree (numpy or jax.numpy)"""
    any_numpy = any(isinstance(x, np.ndarray) for x in jax.tree_util.tree_leaves(pytree))
    any_jax = any(isinstance(x, jax.Array) for x in jax.tree_util.tree_leaves(pytree))
    if any_numpy and any_jax:
        raise ValueError("Cannot mix numpy and jax arrays")

    if any_numpy:
        return np

    if any_jax:
        return jnp

    return jnp
