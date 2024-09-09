import argparse
import dataclasses
from typing import Any, Mapping, Optional, Union

import jax.numpy as jnp
import jax.typing


def convert_params(params: Optional[Union[dict[str, Any], argparse.Namespace]]) -> dict[str, Any]:
    """Ensure parameters are a dict or convert to dict if necessary.

    Args:
        params: Target to be converted to a dictionary

    Returns:
        params as a dictionary

    """
    # in case converting from namespace
    if isinstance(params, argparse.Namespace):
        params = vars(params)

    if params is None:
        params = {}

    return params


def sanitize_params(params: dict[str, Any]) -> dict[str, Any]:
    for key in params:
        # JAX/numpy scalars to python base types
        if isinstance(params[key], (jnp.bool_, jnp.integer, jnp.floating)):
            params[key] = params[key].item()
        elif type(params[key]) not in [bool, int, float, str, jax.typing.ArrayLike]:
            params[key] = str(params[key])

    return params


def flatten_dict(
    params: Mapping[Any, Any], delimiter: str = "/", parent_key: str = ""
) -> dict[str, Any]:
    """Flatten hierarchical dict, e.g. ``{'a': {'b': 'c'}} -> {'a/b': 'c'}``.

    Args:
        params: Dictionary containing the hyperparameters
        delimiter: Delimiter to express the hierarchy. Defaults to ``'/'``.

    Returns:
        Flattened dict.

    Examples:
        >>> flatten_dict({'a': {'b': 'c'}})
        {'a/b': 'c'}
        >>> flatten_dict({'a': {'b': 123}})
        {'a/b': 123}
        >>> flatten_dict({5: {'a': 123}})
        {'5/a': 123}

    """
    result: dict[str, Any] = {}
    for key, val in params.items():
        new_key = parent_key + delimiter + str(key) if parent_key else str(key)
        if dataclasses.is_dataclass(val):
            val = dataclasses.asdict(val)
        elif isinstance(val, argparse.Namespace):
            val = vars(val)

        if isinstance(val, Mapping):
            result = {**result, **flatten_dict(val, parent_key=new_key, delimiter=delimiter)}
        else:
            result[new_key] = val

    return result


def add_prefix(
    metrics: Mapping[str, Union[jax.typing.ArrayLike, float]], prefix: str, separator: str
) -> Mapping[str, Union[jax.typing.ArrayLike, float]]:
    """Insert prefix before each key in a dict, separated by the separator.

    Args:
        metrics: Dictionary with metric names as keys and measured quantities as values
        prefix: Prefix to insert before each key
        separator: Separates prefix and original key name

    Returns:
        Dictionary with prefix and separator inserted before each key

    """
    if not prefix:
        return metrics

    return {f"{prefix}{separator}{key}": val for key, val in metrics.items()}
