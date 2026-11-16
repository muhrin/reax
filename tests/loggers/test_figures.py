import io
import os
from unittest import mock

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import pytest  # noqa: E402

import reax  # noqa: E402
from reax import demos, loggers  # noqa: E402
from reax.lightning import rank_zero  # noqa: E402
from reax.loggers import _figures  # noqa: E402
from reax.loggers import mlflow as reax_mlflow  # noqa: E402
from reax.loggers import pandas as reax_pandas  # noqa: E402


@pytest.fixture
def mpl_fig():
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.plot([0, 1], [0, 1])
    yield fig
    plt.close(fig)


def _png_shape(data: bytes) -> tuple:
    return np.asarray(Image.open(io.BytesIO(data))).shape


def test_matplotlib_converter(mpl_fig):
    data = _figures.to_figure_data(mpl_fig)
    assert data.fmt == "png"
    assert data.data.startswith(_figures.PNG_MAGIC)


def test_pil_converter():
    data = _figures.to_figure_data(Image.new("RGB", (4, 3)))
    assert data.fmt == "png"
    assert _png_shape(data.data) == (3, 4, 3)


@pytest.mark.parametrize(
    "array, expected_shape",
    [
        (np.zeros((3, 4), dtype=np.uint8), (3, 4)),
        (np.ones((3, 4, 1), dtype=np.float32), (3, 4)),
        (np.full((3, 4, 3), 0.5, dtype=np.float64), (3, 4, 3)),
        (jnp.ones((3, 4, 4)), (3, 4, 4)),
    ],
)
def test_array_converter(array, expected_shape):
    data = _figures.to_figure_data(array)
    assert data.fmt == "png"
    assert _png_shape(data.data) == expected_shape


def test_array_converter_float_scaling():
    data = _figures.to_figure_data(np.array([[0.0, 1.0, 2.0]]))
    assert np.asarray(Image.open(io.BytesIO(data.data))).tolist() == [[0, 255, 255]]


def test_array_converter_bad_shape():
    with pytest.raises(ValueError):
        _figures.to_figure_data(np.zeros((2, 2, 2)))


def test_bytes_converter():
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2)).save(buffer, format="JPEG")
    assert _figures.to_figure_data(buffer.getvalue()).fmt == "jpeg"

    buffer = io.BytesIO()
    Image.new("RGB", (2, 2)).save(buffer, format="PNG")
    assert _figures.to_figure_data(buffer.getvalue()).fmt == "png"

    with pytest.raises(ValueError):
        _figures.to_figure_data(b"not an image")


def test_repr_png_converter():
    class Thing:
        def _repr_png_(self):
            return _figures.PNG_MAGIC + b"rest"

    assert _figures.to_figure_data(Thing()).data.startswith(_figures.PNG_MAGIC)


def test_unknown_type():
    with pytest.raises(TypeError, match="register_figure_converter"):
        _figures.to_figure_data(object())


def test_user_converter_takes_precedence(monkeypatch):
    monkeypatch.setattr(_figures, "_CONVERTERS", list(_figures._CONVERTERS))
    custom = _figures.FigureData(b"custom", "png")
    loggers.register_figure_converter(lambda obj: isinstance(obj, np.ndarray), lambda _: custom)
    assert _figures.to_figure_data(np.zeros((2, 2))) is custom


def test_figure_path():
    assert _figures.figure_path("val/parity", 3, "png") == "figures/val/parity/step_00000003.png"
    assert _figures.figure_path("loss", None, "png") == "figures/loss.png"
    assert _figures.figure_path("a.b", 1, "png", sep=".") == "figures/a/b/step_00000001.png"
    with pytest.raises(ValueError):
        _figures.figure_path("/", 1, "png")


@pytest.mark.parametrize("logger_cls", [loggers.CsvLogger, reax_pandas.PandasLogger])
def test_file_loggers(tmp_path, logger_cls, mpl_fig):
    logger = logger_cls(tmp_path)
    logger.log_figure("val/parity", mpl_fig, step=3)
    logger.log_figure("image", np.zeros((2, 2), dtype=np.uint8))

    log_dir = logger.log_dir
    assert os.path.isfile(os.path.join(log_dir, "figures", "val", "parity", "step_00000003.png"))
    assert os.path.isfile(os.path.join(log_dir, "figures", "image.png"))

    with logger.open_artifact("plots/fig.svg") as file:
        mpl_fig.savefig(file, format="svg")
    assert os.path.isfile(os.path.join(log_dir, "plots", "fig.svg"))


@pytest.mark.parametrize("path", ["", "/abs/path.png", "../escape.png", "a/../../b.png"])
def test_open_artifact_bad_path(tmp_path, path):
    logger = loggers.CsvLogger(tmp_path)
    with pytest.raises(ValueError):
        with logger.open_artifact(path):
            pass


def test_close(tmp_path, mpl_fig):
    logger = loggers.CsvLogger(tmp_path)
    logger.log_figure("fig", mpl_fig)
    assert plt.fignum_exists(mpl_fig.number)
    logger.log_figure("fig", mpl_fig, close=True)
    assert not plt.fignum_exists(mpl_fig.number)


def test_non_zero_rank(tmp_path, mpl_fig, monkeypatch):
    monkeypatch.setattr(rank_zero.rank_zero_only, "rank", 1)
    logger = reax_pandas.PandasLogger(tmp_path)
    logger.log_figure("fig", mpl_fig)
    with logger.open_artifact("file.txt") as file:
        file.write(b"discarded")
    assert not os.path.exists(os.path.join(tmp_path, "lightning_logs"))


def test_dummy_logger(mpl_fig):
    logger = loggers.logger.DummyLogger()
    logger.log_figure("fig", mpl_fig)
    with logger.open_artifact("file.txt") as file:
        file.write(b"discarded")


def test_tensorboard(tmp_path, mpl_fig):
    logger = loggers.TensorBoardLogger(tmp_path, prefix="pre")
    with (
        mock.patch.object(logger.experiment, "add_figure") as add_figure,
        mock.patch.object(logger.experiment, "add_image") as add_image,
    ):
        logger.log_figure("val/fig", mpl_fig, step=2)
        add_figure.assert_called_once_with("pre-val/fig", mpl_fig, 2, close=False)

        logger.log_figure("img", np.zeros((3, 4, 3), dtype=np.uint8), step=5)
        args, kwargs = add_image.call_args
        assert args[0] == "pre-img"
        assert args[1].shape == (3, 4, 3)
        assert args[2] == 5
        assert kwargs == {"dataformats": "HWC"}

    # Check that it really works with tensorboardX
    logger.log_figure("real", mpl_fig, step=1)
    logger.log_figure("real_img", np.zeros((3, 4), dtype=np.uint8), step=1)
    logger.save()

    with logger.open_artifact("extra/file.bin") as file:
        file.write(b"data")
    assert os.path.isfile(os.path.join(logger.log_dir, "extra", "file.bin"))


def test_mlflow(tmp_path, mpl_fig):
    if not reax_mlflow._MLFLOW_AVAILABLE:
        pytest.skip("requires mlflow")

    logger = reax_mlflow.MlflowLogger("test", save_dir=str(tmp_path))
    client, run_id = logger.experiment, logger.run_id

    def artifacts(path=None):
        return {info.path for info in client.list_artifacts(run_id, path)}

    with logger.open_artifact("plots/fig.svg") as file:
        mpl_fig.savefig(file, format="svg")
    assert "plots/fig.svg" in artifacts("plots")

    logger.log_figure("val/parity", mpl_fig)
    assert "figures/val/parity.png" in artifacts("figures/val")

    logger.log_figure("parity", mpl_fig, step=4)
    assert "images" in artifacts()


def test_module_log_figure(tmp_path):
    received = []

    class RecordingLogger(loggers.CsvLogger):
        def log_figure(self, name, figure, step=None, *, close=False):
            received.append((self, name, step, close))
            super().log_figure(name, figure, step, close=close)

    class FigureModel(demos.BoringModel):
        def on_validation_epoch_end(self, *args, **kwargs):
            self.figure = Image.new("RGB", (2, 2))
            self.log_figure("val/fig", self.figure)

    logger1 = RecordingLogger(tmp_path, name="one")
    logger2 = RecordingLogger(tmp_path, name="two")
    trainer = reax.Trainer(default_root_dir=tmp_path, logger=[logger1, logger2])
    trainer.fit(FigureModel(), max_epochs=1, limit_train_batches=2, limit_val_batches=1)

    step = trainer.global_updates
    assert (logger1, "val/fig", step, False) in received
    assert (logger2, "val/fig", step, False) in received
    for logger in (logger1, logger2):
        assert os.path.isfile(
            os.path.join(logger.log_dir, "figures", "val", "fig", f"step_{step:08d}.png")
        )


def test_module_log_figure_without_trainer(mpl_fig):
    with pytest.warns(UserWarning, match="log_figure"):
        demos.BoringModel().log_figure("fig", mpl_fig, close=True)
    assert not plt.fignum_exists(mpl_fig.number)
