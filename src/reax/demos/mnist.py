import array
from collections.abc import Sequence
import gzip
import os
from os import path
import pathlib
import struct
from typing import Any, Final
import urllib.error
import urllib.request

import jax
import numpy as np
from typing_extensions import override

import reax

Dataset = Any


class MnistDataset(Sequence[tuple[jax.Array, jax.Array]]):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        data_dir (str or ``pathlib.Path``): Root directory of dataset where
            ``MNIST/raw/train-images-idx3-ubyte``
            and  ``MNIST/raw/t10k-images-idx3-ubyte`` exist.
        train (bool, optional): If True, creates dataset from ``train-images-idx3-ubyte``,
            otherwise from ``t10k-images-idx3-ubyte``.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    MIRRORS: Final[tuple[str, ...]] = (
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
        "https://storage.googleapis.com/cvdf-datasets/mnist/",
        "http://yann.lecun.com/exdb/mnist/",
    )

    RESOURCES: Final[tuple[tuple[str, str]]] = (
        ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
    )

    def __init__(
        self,
        data_dir: str | pathlib.Path = "data/",
        train: bool = True,
        download: bool = False,
    ) -> None:
        # Params
        self.root: Final[str] = data_dir
        self.train: Final[bool] = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        # State
        self.data = self._load_data()

    def _load_data(self) -> reax.data.ArrayDataset:
        image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte.gz"
        label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte.gz"

        return reax.data.ArrayDataset(
            self.parse_images(path.join(self.downloads_dir, image_file)),
            self.parse_labels(path.join(self.downloads_dir, label_file)),
        )

    def __getitem__(self, index: int) -> tuple[jax.Array, jax.Array]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)

    @property
    def downloads_dir(self) -> str:
        return os.path.join(self.root, self.__class__.__name__)

    def download(self) -> None:
        """Download the MNIST data if it doesn't exist already."""
        if self._check_exists():
            return

        os.makedirs(self.downloads_dir, exist_ok=True)

        # download files
        for filename, _md5 in self.RESOURCES:
            errors = []
            for mirror in self.MIRRORS:
                url = f"{mirror}{filename}"
                try:
                    self._do_download(url, filename=filename)
                except urllib.error.URLError as e:
                    errors.append(e)
                    continue
                break
            else:
                s = f"Error downloading {filename}:\n"
                for mirror, err in zip(self.MIRRORS, errors):
                    s += f"Tried {mirror}, got:\n{str(err)}\n"
                raise RuntimeError(s)

    def _check_exists(self) -> bool:
        return all(
            os.path.exists(os.path.join(self.downloads_dir, os.path.basename(url)))
            for url, _ in self.RESOURCES
        )

    def _do_download(self, url: str, filename: str):
        """Download the file at the URL to our data dir."""
        save_dir = self.downloads_dir
        if not path.exists(save_dir):
            os.makedirs(save_dir)

        out_file = path.join(save_dir, filename)
        if not path.isfile(out_file):
            urllib.request.urlretrieve(url, out_file)  # nosec
            print(f"downloaded {url} to {save_dir}")

    @staticmethod
    def parse_labels(filename) -> np.ndarray:
        with gzip.open(filename, "rb") as fh:
            _ = struct.unpack(">II", fh.read(8))
            labels = np.array(array.array("B", fh.read()), dtype=np.uint8)

            # Create a one-hot encoding of x of size k
            labels = np.array(labels[:, None] == np.arange(10), dtype=np.int32)
            return labels

    @staticmethod
    def parse_images(filename) -> np.ndarray:
        with gzip.open(filename, "rb") as fh:
            _, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            # pylint: disable=too-many-function-args
            img = np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(
                num_data, rows, cols
            )
            # Flatten all but the first dimension of an ndarray
            return np.reshape(img, (img.shape[0], -1)) / np.float32(255.0)


class MnistDataModule(reax.DataModule):
    """`REAX DataModule` for the MNIST dataset.

    The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set
    of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been
    size-normalized and centered in a fixed-size image. The original black and white images from
    NIST were size normalized to fit in a 20x20 pixel box while preserving their aspect ratio. The
    resulting images contain grey levels as a result of the anti-aliasing technique used by the
    normalization algorithm. the images were centered in a 28x28 image by computing the center of
    mass of the pixels, and translating the image so as to position this point at the center of the
    28x28 field.

    A `reax.DataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.
    """

    mirrors = (
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
        "https://storage.googleapis.com/cvdf-datasets/mnist/",
        "http://yann.lecun.com/exdb/mnist/",
    )

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 0,
        download: bool = True,
    ) -> None:
        """Initialize a `MnistDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to
            `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        """
        super().__init__()

        # Params
        self._data_dir: Final[str] = data_dir
        self._train_val_test_split: Final[tuple[int, int, int]] = train_val_test_split
        self._batch_size: Final[int] = batch_size
        self._num_workers: Final[int] = num_workers
        self._download: Final[bool] = download

        # State
        self.batch_size_per_device = batch_size
        self.data_train: Dataset | None = None
        self.data_val: Dataset | None = None
        self.data_test: Dataset | None = None

    @property
    def num_classes(self) -> int:
        """Get the number of classes.

        :return: The number of MNIST classes (10).
        """
        return 10

    @property
    def downloads_dir(self) -> str:
        return os.path.join(self._data_dir, "MNIST", "raw")

    @override
    def prepare_data(self) -> None:
        """Download data if needed. REAX ensures that `self.prepare_data()` is called only within a
        single process on CPU, so you can safely add your downloading logic within. In case of
        multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        if self._download:
            for filename in [
                "train-images-idx3-ubyte.gz",
                "train-labels-idx1-ubyte.gz",
                "t10k-images-idx3-ubyte.gz",
                "t10k-labels-idx1-ubyte.gz",
            ]:
                self._do_download(self.mirrors[0] + filename, filename)

    @override
    def setup(self, stage: "reax.Stage", /) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by REAX before `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`, so be careful not to execute things like random
        split twice! Also, it is called after `self.prepare_data()` and there is a barrier in
        between which ensures that all the processes proceed to `self.setup()` once the data is
        prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        Defaults to ``None``.
        """
        # TODO: Divide batch size by the number of devices.
        # if self.trainer is not None:
        #     if self._batch_size % self.trainer.world_size != 0:
        #         raise RuntimeError(
        #             f"Batch size ({self._batch_size}) is not divisible by the number of devices "
        #             f"({self.trainer.world_size})."
        #         )
        #     self.batch_size_per_device = self._batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = reax.data.ArrayDataset(
                self.parse_images(path.join(self.downloads_dir, "train-images-idx3-ubyte.gz")),
                self.parse_labels(path.join(self.downloads_dir, "train-labels-idx1-ubyte.gz")),
            )

            testset = reax.data.ArrayDataset(
                self.parse_images(path.join(self.downloads_dir, "t10k-images-idx3-ubyte.gz")),
                self.parse_labels(path.join(self.downloads_dir, "t10k-labels-idx1-ubyte.gz")),
            )

            dataset = reax.data.ConcatDataset([trainset, testset])
            self.data_train, self.data_val, self.data_test = reax.data.random_split(
                stage.rngs,
                dataset=dataset,
                lengths=self._train_val_test_split,
            )

    @override
    def train_dataloader(self) -> reax.DataLoader[jax.Array, jax.Array]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return reax.data.ReaxDataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            # num_workers=self._num_workers,
            shuffle=True,
        )

    @override
    def val_dataloader(self) -> reax.DataLoader[jax.Array, jax.Array]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return reax.data.ReaxDataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            # num_workers=self._num_workers,
            shuffle=False,
        )

    @override
    def test_dataloader(self) -> reax.DataLoader[jax.Array, jax.Array]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return reax.data.ReaxDataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            # num_workers=self._num_workers,
            shuffle=False,
        )

    def _do_download(self, url: str, filename: str):
        """Download the file at the URL to our data dir."""
        save_dir = self.downloads_dir
        if not path.exists(save_dir):
            os.makedirs(save_dir)

        out_file = path.join(save_dir, filename)
        if not path.isfile(out_file):
            urllib.request.urlretrieve(url, out_file)  # nosec
            print(f"downloaded {url} to {save_dir}")

    @staticmethod
    def parse_labels(filename) -> np.ndarray:
        with gzip.open(filename, "rb") as fh:
            _ = struct.unpack(">II", fh.read(8))
            labels = np.array(array.array("B", fh.read()), dtype=np.uint8)

            # Create a one-hot encoding of x of size k
            labels = np.array(labels[:, None] == np.arange(10), dtype=np.int32)
            return labels

    @staticmethod
    def parse_images(filename) -> np.ndarray:
        with gzip.open(filename, "rb") as fh:
            _, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            # pylint: disable=too-many-function-args
            img = np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(
                num_data, rows, cols
            )
            # Flatten all but the first dimension of an ndarray
            return np.reshape(img, (img.shape[0], -1)) / np.float32(255.0)


if __name__ == "__main__":
    _ = MnistDataModule()
