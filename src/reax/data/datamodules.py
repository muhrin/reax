from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Optional, Union

from flax import nnx
from typing_extensions import override

from . import _datasources, _loaders

if TYPE_CHECKING:
    import reax

__all__ = ("DataModule",)

Dataset = Any


class DataModule(_datasources.DataSource):
    def __init__(self):
        super().__init__()
        self._rngs = nnx.Rngs(0)

    @property
    def rngs(self) -> nnx.Rngs:
        return self._rngs

    @rngs.setter
    def rngs(self, rngs: nnx.Rngs):
        self._rngs = rngs

    @classmethod
    def from_datasets(
        cls,
        train_dataset: Optional[Union[Dataset, Iterable[Dataset]]] = None,
        val_dataset: Optional[Union[Dataset, Iterable[Dataset]]] = None,
        test_dataset: Optional[Union[Dataset, Iterable[Dataset]]] = None,
        predict_dataset: Optional[Union[Dataset, Iterable[Dataset]]] = None,
        *,
        batch_size: int = 1,
    ) -> "DataModule":
        """From datasets."""
        return FromDatasets(
            train_dataset, val_dataset, test_dataset, predict_dataset, batch_size=batch_size
        )


class FromDatasets(DataModule):
    def __init__(
        self,
        train_dataset: Optional[Union[Dataset, Iterable[Dataset]]] = None,
        val_dataset: Optional[Union[Dataset, Iterable[Dataset]]] = None,
        test_dataset: Optional[Union[Dataset, Iterable[Dataset]]] = None,
        predict_dataset: Optional[Union[Dataset, Iterable[Dataset]]] = None,
        *,
        batch_size: int = 1,
    ):
        """Init function."""
        super().__init__()
        self._train_dataset = train_dataset
        self._val_dataset = val_dataset
        self._test_dataset = test_dataset
        self._predict_dataset = predict_dataset
        self._batch_size = batch_size

    @override
    def train_dataloader(self) -> "reax.DataLoader":
        """Train dataloader."""
        return _loaders.ReaxDataLoader(self._train_dataset, batch_size=self._batch_size)

    @override
    def val_dataloader(self) -> "reax.DataLoader":
        """Val dataloader."""
        return _loaders.ReaxDataLoader(self._val_dataset, batch_size=self._batch_size)

    @override
    def test_dataloader(self) -> "reax.DataLoader":
        """Test dataloader."""
        return _loaders.ReaxDataLoader(self._test_dataset, batch_size=self._batch_size)

    @override
    def predict_dataloader(self) -> "reax.DataLoader":
        """Predict dataloader."""
        return _loaders.ReaxDataLoader(self._predict_dataset, batch_size=self._batch_size)
