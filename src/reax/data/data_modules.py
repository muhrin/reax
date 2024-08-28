from typing import TYPE_CHECKING, Any, Iterable, Optional, Union

from . import _loaders

if TYPE_CHECKING:
    import reax

__all__ = ("DataModule",)

Dataset = Any


class DataModule:
    @classmethod
    def from_datasets(
        cls,
        train_dataset: Optional[Union[Dataset, Iterable[Dataset]]] = None,
        val_dataset: Optional[Union[Dataset, Iterable[Dataset]]] = None,
        test_dataset: Optional[Union[Dataset, Iterable[Dataset]]] = None,
        predict_dataset: Optional[Union[Dataset, Iterable[Dataset]]] = None,
        batch_size: int = 1,
    ) -> "DataModule":
        return FromDatasets(
            train_dataset, val_dataset, test_dataset, predict_dataset, batch_size=batch_size
        )

    def train_dataloader(self) -> "reax.DataLoader":
        raise NotImplementedError("DataModule.train_dataloaders has not been implemented")

    def val_dataloader(self) -> "reax.DataLoader":
        raise NotImplementedError("DataModule.val_dataloaders has not been implemented")

    def test_dataloader(self) -> "reax.DataLoader":
        raise NotImplementedError("DataModule.test_dataloaders has not been implemented")

    def predict_dataloader(self) -> "reax.DataLoader":
        raise NotImplementedError("DataModule.predict_dataloaders has not been implemented")


class FromDatasets(DataModule):
    def __init__(
        self,
        train_dataset: Optional[Union[Dataset, Iterable[Dataset]]] = None,
        val_dataset: Optional[Union[Dataset, Iterable[Dataset]]] = None,
        test_dataset: Optional[Union[Dataset, Iterable[Dataset]]] = None,
        predict_dataset: Optional[Union[Dataset, Iterable[Dataset]]] = None,
        batch_size: int = 1,
    ):
        self._train_dataset = train_dataset
        self._val_dataset = val_dataset
        self._test_dataset = test_dataset
        self._predict_dataset = predict_dataset
        self._batch_size = batch_size

    def train_dataloader(self) -> "reax.DataLoader":
        return _loaders.ReaxDataLoader(self._train_dataset, batch_size=self._batch_size)

    def val_dataloader(self) -> "reax.DataLoader":
        return _loaders.ReaxDataLoader(self._val_dataset, batch_size=self._batch_size)

    def test_dataloader(self) -> "reax.DataLoader":
        return _loaders.ReaxDataLoader(self._test_dataset, batch_size=self._batch_size)

    def predict_dataloader(self) -> "reax.DataLoader":
        return _loaders.ReaxDataLoader(self._predict_dataset, batch_size=self._batch_size)
