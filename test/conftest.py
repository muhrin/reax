import os

import jax
from jax import config, random
import pytest
import tqdm

import reax.loggers.pandas


@pytest.fixture
def rng_key():
    return random.PRNGKey(0)


@pytest.fixture
def default_logger(tmp_path) -> reax.loggers.pandas.PandasLogger:
    """A default logger that can be used for testing"""
    return reax.loggers.pandas.PandasLogger(tmp_path)


@pytest.fixture
def test_trainer(tmp_path):
    return reax.Trainer(default_root_dir=tmp_path)


@pytest.fixture(autouse=True)
def reax_config():
    # Some numerical test rely on checking against numpy implementations and np will default to 64-bit,
    # so do that too
    jax.config.update("jax_enable_x64", True)
