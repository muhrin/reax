import argparse
import dataclasses
import pathlib
from typing import TYPE_CHECKING, Any, Mapping, Optional, Union

import jax.numpy as jnp
import jax.typing

if TYPE_CHECKING:
    import reax


def convert_params(params: Optional[Union[dict[str, Any], argparse.Namespace]]) -> dict[str, Any]:
    """Ensure parameters are a dict or convert to dict if necessary.

    :param params: Object to be converted to `dict`.
    :type params: Optional[Union[dict[str, Any], argparse.Namespace]]
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

    :param parent_key: defaults to "".
    :type parent_key: str, optional
    :param params: The mapping containing hyperparameters.
    :type params: Mapping[Any, Any]
    :param delimiter: The delimiter to express hierarchy, defaults to "/".
    :type delimiter: str, optional

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

    :param metrics: Dictionary with metric names as keys and measured quantities as values.
    :type metrics: Mapping[str, Union[jax.typing.ArrayLike, float]]
    :param prefix: Prefix to insert before each key.
    :type prefix: str
    :param separator: Separates prefix and original key name.
    :type separator: str
    :return s: Dictionary with prefix and separator inserted before each key.
    :rtype s: Mapping[str, Union[jax.typing.ArrayLike, float]]
    """
    if not prefix:
        return metrics

    return {f"{prefix}{separator}{key}": val for key, val in metrics.items()}


def scan_checkpoints(
    checkpoint_listener: "reax.listeners.ModelCheckpoint", logged_model_time: dict
) -> list[tuple[float, str, float, str]]:
    """Return the checkpoints to be logged.

    :param checkpoint_listener: Checkpoint listener reference.
    :type checkpoint_listener: "reax.listeners.ModelCheckpoint"
    :param logged_model_time: Dictionary containing the logged model times.
    :type logged_model_time: dict
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
