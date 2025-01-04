import argparse
import contextlib
import copy
import enum
import os
from typing import Any, Union
import warnings

import fsspec
from lightning_utilities.core import apply_func
import yaml

try:
    import omegaconf

    _OMEGACONF_AVAILABLE = True
except ImportError:
    _OMEGACONF_AVAILABLE = False

from reax import typing
from reax.lightning import rank_zero

__all__ = "save_hparams_to_yaml", "load_hparams_from_yaml"


def save_hparams_to_yaml(
    config_yaml: typing.Path, hparams: Union[dict, argparse.Namespace], use_omegaconf: bool = True
) -> None:
    """Save the hparams to a yaml file.

    :param config_yaml: The path to save the hparams to.
    :type config_yaml: typing.Path
    :param hparams: The hparams to save.
    :type hparams: Union[dict, argparse.Namespace]
    :param use_omegaconf: If `omegaconf` is available and `use_omegaconf=True`, the hparams will be
        converted to `omegaconf.DictConfig` if possible, defaults to True.
    :type use_omegaconf: bool, optional
    """
    fs = fsspec.url_to_fs(config_yaml)[0]
    if not fs.isdir(os.path.dirname(config_yaml)):
        raise RuntimeError(f"Missing folder: {os.path.dirname(config_yaml)}.")

    # convert Namespace to dict
    if isinstance(hparams, argparse.Namespace):
        hparams = vars(hparams)

    # saving with OmegaConf objects
    if _OMEGACONF_AVAILABLE and use_omegaconf:
        # deepcopy: hparams from user shouldn't be resolved
        hparams = copy.deepcopy(hparams)
        hparams = apply_func.apply_to_collection(
            hparams, omegaconf.DictConfig, omegaconf.OmegaConf.to_container, resolve=True
        )
        with fs.open(config_yaml, "w", encoding="utf-8") as fp:
            try:
                omegaconf.OmegaConf.save(hparams, fp)
                return
            except (omegaconf.UnsupportedValueType, omegaconf.ValidationError):
                pass

    if not isinstance(hparams, dict):
        raise TypeError("hparams must be dictionary")

    hparams_allowed = {}
    # drop parameters which contain some strange datatypes as fsspec
    for key, value in hparams.items():
        try:
            value = value.name if isinstance(value, enum.Enum) else value
            yaml.dump(value)
        except (TypeError, ValueError):
            warnings.warn(
                f"Skipping '{key}' parameter because it is not possible to safely dump to YAML."
            )
            hparams[key] = type(value).__name__
        else:
            hparams_allowed[key] = value

    # saving the standard way
    with fs.open(config_yaml, "w", newline="") as fp:
        yaml.dump(hparams_allowed, fp)


def load_hparams_from_yaml(
    config_yaml: typing.Path, use_omegaconf: bool = True
) -> Union[dict[str, Any], "omegaconf.DictConfig"]:
    """Load hparams from a file.

    :param config_yaml: Path to the yaml file to be loaded.
    :type config_yaml: typing.Path
    :param use_omegaconf: If `omegaconf` is available and `use_omegaconf=True`, the hparams will
        be converted to a `omegaconf.DictConfig` if possible, defaults to True.
    :type use_omegaconf: bool, optional
    """
    fs: fsspec.AbstractFileSystem = fsspec.url_to_fs(config_yaml)[0]
    if not fs.exists(config_yaml):
        rank_zero.rank_zero_warn(f"Missing Tags: {config_yaml}.", category=RuntimeWarning)
        return {}

    with fs.open(config_yaml, "r") as fp:
        hparams = yaml.full_load(fp)

    if _OMEGACONF_AVAILABLE and use_omegaconf:
        with contextlib.suppress(omegaconf.UnsupportedValueType, omegaconf.ValidationError):
            return omegaconf.OmegaConf.create(hparams)

    return hparams
