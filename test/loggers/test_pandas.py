import os
from unittest import mock

import fsspec
import jax.numpy as jnp
import numpy as np
import pytest

import reax
from reax import demos, loggers
import reax.loggers.pandas


def test_automatic_versioning(tmp_path):
    """Verify that automatic versioning works."""
    (tmp_path / "exp" / "version_0").mkdir(parents=True)
    (tmp_path / "exp" / "version_1").mkdir()
    (tmp_path / "exp" / "version_nonumber").mkdir()
    (tmp_path / "exp" / "other").mkdir()
    print(tmp_path)

    logger = loggers.pandas.PandasLogger(save_dir=tmp_path, name="exp")

    assert logger.version == 2


def test_manual_versioning(tmp_path):
    """Verify that manual versioning works."""
    root_dir = tmp_path / "exp"
    (root_dir / "version_0").mkdir(parents=True)
    (root_dir / "version_1").mkdir()
    (root_dir / "version_2").mkdir()

    logger = loggers.pandas.PandasLogger(save_dir=tmp_path, name="exp", version=1)

    assert logger.version == 1


def test_manual_versioning_file_exists(tmp_path):
    """Test that a warning is emitted and existing files get overwritten."""

    # Simulate an existing 'version_0' from a previous run
    (tmp_path / "exp" / "version_0").mkdir(parents=True)
    previous_metrics_file = (
        tmp_path / "exp" / "version_0" / f"metrics.{loggers.pandas.DEFAULT_FORMAT}"
    )
    previous_metrics_file.touch()

    logger = loggers.pandas.PandasLogger(save_dir=tmp_path, name="exp", version=0)
    assert previous_metrics_file.exists()
    with pytest.warns(UserWarning, match="Experiment logs directory .* exists and is not empty"):
        _ = logger.experiment
    assert not previous_metrics_file.exists()


def test_named_version(tmp_path):
    """Verify that manual versioning works for string versions, e.g. '2020-02-05-162402'."""
    exp_name = "exp"
    (tmp_path / exp_name).mkdir()
    expected_version = "2020-02-05-162402"

    logger = loggers.pandas.PandasLogger(save_dir=tmp_path, name=exp_name, version=expected_version)
    logger.log_hyperparams({"a": 1, "b": 2})
    logger.save()
    assert logger.version == expected_version
    assert os.listdir(tmp_path / exp_name) == [expected_version]
    assert os.listdir(tmp_path / exp_name / expected_version)


@pytest.mark.parametrize("name", ["", None])
def test_no_name(tmp_path, name):
    """Verify that None or empty name works."""
    logger = loggers.pandas.PandasLogger(save_dir=tmp_path, name=name)
    logger.save()
    assert os.path.normpath(logger.root_dir) == str(
        tmp_path
    )  # use os.path.normpath to handle trailing /
    assert os.listdir(tmp_path / "version_0")


def test_log_hyperparams(tmp_path):
    logger = loggers.pandas.PandasLogger(tmp_path)
    hparams = {
        "float": 0.3,
        "int": 1,
        "string": "abc",
        "bool": True,
        "dict": {"a": {"b": "c"}},
        "list": [1, 2, 3],
    }
    logger.log_hyperparams(hparams)
    logger.save()

    path_yaml = os.path.join(logger.log_dir, loggers.pandas.ExperimentWriter.HPARAMS_FILENAME)
    params = reax.load_hparams_from_yaml(path_yaml)
    assert all(n in params for n in hparams)


def test_csv_logger_remotefs():
    logger = loggers.pandas.PandasLogger(save_dir="memory://test_fit_csv_logger_remotefs")
    fs, _ = fsspec.core.url_to_fs("memory://test_fit_csv_logger_remotefs")
    exp = logger.experiment
    exp.log_metrics({"loss": 0.1})
    exp.save()
    assert fs.isfile(logger.experiment.metrics_file_path)


def test_flush_n_steps(tmp_path):
    logger = loggers.pandas.PandasLogger(tmp_path, flush_logs_every_n_steps=2)
    metrics = {
        "float": 0.3,
        "int": 1,
        "FloatArray": jnp.array(0.1),
        "IntArray": jnp.array(1),
    }
    logger.save = mock.MagicMock()
    logger.log_metrics(metrics, step=0)

    logger.save.assert_not_called()
    logger.log_metrics(metrics, step=1)
    logger.save.assert_called_once()


def test_formats(tmp_path):
    with pytest.raises(ValueError):
        logger = loggers.pandas.PandasLogger(tmp_path, fmt="invalid")
        # We have to ask for the experiment as the format is checked lazily
        print(logger.experiment)

    logger = loggers.pandas.PandasLogger(tmp_path, fmt="json")
    # We have to ask for the experiment as the format is checked lazily
    print(logger.experiment)


def test_dataframe_content(tmp_path):
    """
    Test that log graph works with both model.example_input_array and if array is passed externally.
    """
    logger = loggers.pandas.PandasLogger(tmp_path)
    df = logger.dataframe
    assert df.empty

    logger.log_metrics({"training.loss": 10.0, "epoch": 0}, step=0)
    logger.log_metrics({"training.loss": 8.0, "epoch": 0}, step=1)
    logger.log_metrics({"validation.loss": 12.0, "epoch": 0}, step=0)
    logger.log_metrics({"training.loss": 7.0, "epoch": 1}, step=0)
    df = logger.dataframe

    # Training loss column
    assert np.allclose(
        df["training.loss"].to_numpy(), [10.0, 8.0, float("nan"), 7.0], equal_nan=True
    )

    # Validation loss column
    assert np.allclose(
        df["validation.loss"].to_numpy(),
        [float("nan"), float("nan"), 12.0, float("nan")],
        equal_nan=True,
    )

    # Steps
    assert all(df["step"].to_numpy() == [0, 1, 0, 0])
    # Epoch
    assert all(df["epoch"].to_numpy() == [0, 0, 0, 1])
