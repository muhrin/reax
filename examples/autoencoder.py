import functools
from os import path
import sys
from typing import Any

try:
    import accimage
except ImportError:
    accimage = None
import PIL
from flax import linen
import jax
from lightning.pytorch.demos.mnist_datamodule import MNIST
import numpy as np
import optax
import torch
from torch.utils.data import random_split

import reax

DATASETS_PATH = path.join(path.dirname(__file__), ".", "datasets")


class Autoencoder(linen.Module):
    hidden_dim: int = 64

    def setup(self):
        super().__init__()
        # pylint: disable=attribute-defined-outside-init
        self.encoder = linen.Sequential([linen.Dense(self.hidden_dim), linen.relu, linen.Dense(3)])
        self.decoder = linen.Sequential(
            [linen.Dense(self.hidden_dim), linen.relu, linen.Dense(28 * 28)]
        )

    def __call__(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class ReaxAutoEncoder(reax.Module):
    def __init__(self, hidden_dim: int = 64, learning_rate=10e-3):
        super().__init__()
        self.autoencoder = Autoencoder(hidden_dim=hidden_dim)
        self._learning_rate = learning_rate

    def configure_model(self, _stage: "reax.Stage", batch: Any, /) -> None:
        if self.parameters() is None:
            inputs = self._prepare_batch(batch)
            params = self.autoencoder.init(self.rng_key(), inputs)
            self.set_parameters(params)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x):
        return self.autoencoder.apply(self.parameters(), x)

    def training_step(self, batch, batch_idx, /):
        x = self._prepare_batch(batch)
        loss, grads = jax.value_and_grad(self.loss_fn, argnums=0)(
            self.parameters(), x, self.autoencoder
        )

        self.log("train_loss", loss, on_step=True, prog_bar=True)
        return loss, grads

    def validation_step(self, batch, batch_idx: int, /):
        x = self._prepare_batch(batch)
        loss = self.loss_fn(self.parameters(), x, self.autoencoder)
        self.log("val_loss", loss, on_step=True, prog_bar=True)

    def test_step(self, batch, batch_idx: int, /):
        x = self._prepare_batch(batch)
        loss = self.loss_fn(self.parameters(), x, self.autoencoder)
        self.log("test_loss", loss, on_step=True, prog_bar=True)

    @staticmethod
    @functools.partial(jax.jit, static_argnums=2)
    def loss_fn(params, x, model):
        predictions = model.apply(params, x)
        return optax.losses.squared_error(predictions, x).mean()

    def configure_optimizers(self):
        opt = optax.adam(learning_rate=self._learning_rate)
        state = opt.init(self.parameters())
        return opt, state

    @staticmethod
    def _prepare_batch(batch):
        x, _ = batch
        return x.reshape(x.shape[0], -1)


def _is_pil_image(img: Any) -> bool:
    if accimage is not None:
        return isinstance(img, (PIL.Image.Image, accimage.Image))

    return isinstance(img, PIL.Image.Image)


def get_image_num_channels(img: Any) -> int:
    if _is_pil_image(img):
        if hasattr(img, "getbands"):
            return len(img.getbands())

        return img.channels
    raise TypeError(f"Unexpected type {type(img)}")


def to_array(pic) -> np.ndarray:
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor. This function does not support
    torchscript.

    See :class:`~torchvision.transforms.ToTensor` for more details.

    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """

    # handle PIL Image
    mode_to_nptype = {
        "I": np.int32,
        "I;16" if sys.byteorder == "little" else "I;16B": np.int16,
        "F": np.float32,
    }
    img = np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True)

    if pic.mode == "1":
        img = 255 * img
    img = img.reshape((pic.size[1], pic.size[0], get_image_num_channels(pic)))
    # put it from HWC to CHW format
    img = img.transpose((2, 0, 1))
    if img.dtype == np.uint8:
        return np.divide(img, 255, dtype=np.float32)

    return img


class MyDataModule(reax.DataModule):
    def __init__(self, batch_size: int = 64):
        super().__init__()
        dataset = MNIST(DATASETS_PATH, train=True, download=True, transform=to_array)
        self.mnist_test = MNIST(DATASETS_PATH, train=False, download=True, transform=to_array)
        self.mnist_train, self.mnist_val = random_split(
            dataset, [55000, 5000], generator=torch.Generator().manual_seed(42)
        )
        self.batch_size = batch_size

    def train_dataloader(self):
        return reax.data.ReaxDataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return reax.data.ReaxDataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return reax.data.ReaxDataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return reax.data.ReaxDataLoader(self.mnist_test, batch_size=self.batch_size)


if __name__ == "__main__":
    rng_key = jax.random.key(0)

    # setup data
    datamodule = MyDataModule()
    autoencoder = ReaxAutoEncoder()

    trainer = reax.Trainer()
    trainer.fit(autoencoder, datamodule=datamodule, max_epochs=10)  # pylint: disable=not-callable
    trainer.test(autoencoder, datamodule=datamodule)  # pylint: disable=not-callable
