import argparse
from collections.abc import Mapping
import dataclasses
import pathlib
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import jax.typing

if TYPE_CHECKING:
    import reax


def convert_params(params: dict[str, Any] | argparse.Namespace | None) -> dict[str, Any]:
    """Ensure parameters are a dict or convert to dict if necessary.

    Args:
        params (dict[str, Any] | argparse.Namespace | None): Object to
            be converted to `dict`.
    """
    # in case converting from namespace
    if isinstance(params, argparse.Namespace):
        params = vars(params)

    if params is None:
        params = {}

    return params


def sanitize_params(params: dict[str, Any]) -> dict[str, Any]:
    """Sanitize params."""
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
        parent_key (str, optional): defaults to "".
        params (Mapping[Any, Any]): The mapping containing
            hyperparameters.
        delimiter (str, optional): The delimiter to express hierarchy,
            defaults to "/".

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
    metrics: Mapping[str, jax.typing.ArrayLike | float], prefix: str, separator: str
) -> Mapping[str, jax.typing.ArrayLike | float]:
    """Insert prefix before each key in a dict, separated by the separator.

    Args:
        metrics (Mapping[str, jax.typing.ArrayLike | float]): Dictionary
            with metric names as keys and measured quantities as values.
        prefix (str): Prefix to insert before each key.
        separator (str): Separates prefix and original key name.
    :return s: Dictionary with prefix and separator inserted before each key.
    :rtype s: Mapping[str, jax.typing.ArrayLike | float]
    """
    if not prefix:
        return metrics

    return {f"{prefix}{separator}{key}": val for key, val in metrics.items()}


def scan_checkpoints(
    checkpoint_listener: "reax.listeners.ModelCheckpoint", logged_model_time: dict
) -> list[tuple[float, str, float, str]]:
    """Return the checkpoints to be logged.

    Args:
        checkpoint_listener ("reax.listeners.ModelCheckpoint"):
            Checkpoint listener reference.
        logged_model_time (dict): Dictionary containing the logged model
            times.
    """
    # get checkpoints to be saved with associated score
    checkpoints = {}
    if hasattr(checkpoint_listener, "last_model_path") and hasattr(
        checkpoint_listener, "current_score"
    ):
        checkpoints[checkpoint_listener.last_model_path] = (
            checkpoint_listener.current_score,
            "latest",
        )

    if hasattr(checkpoint_listener, "best_model_path") and hasattr(
        checkpoint_listener, "best_model_score"
    ):
        checkpoints[checkpoint_listener.best_model_path] = (
            checkpoint_listener.best_model_score,
            "best",
        )

    if hasattr(checkpoint_listener, "best_k_models"):
        for key, value in checkpoint_listener.best_k_models.items():
            checkpoints[key] = (value, "best_k")

    checkpoints = sorted(
        (pathlib.Path(path).stat().st_mtime, path, s, tag)
        for path, (s, tag) in checkpoints.items()
        if pathlib.Path(path).is_file()
    )
    checkpoints = [
        ckpt
        for ckpt in checkpoints
        if ckpt[1] not in logged_model_time or logged_model_time[ckpt[1]] < ckpt[0]
    ]
    return checkpoints
