from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from typing_extensions import override

from . import _datasources, _loaders

if TYPE_CHECKING:
    import reax

__all__ = ("DataModule",)

Dataset = Any

_T_co = TypeVar("_T_co", covariant=True)
U = TypeVar("U")


class DataModule(Generic[_T_co, U], _datasources.DataSource[_T_co, U]):
    """Encapsulates all data-related logic: downloading, splitting, and batch construction.

    A :class:`DataModule` is the primary interface for providing data to a
    :class:`reax.Module` via the :class:`reax.Trainer`.

    Attributes:
        rngs: The :class:`flax.nnx.Rngs` shared with the engine for reproducible splits.
    """

    @classmethod
    def from_datasets(
        cls,
        train_dataset: Dataset | Iterable[Dataset] | None = None,
        val_dataset: Dataset | Iterable[Dataset] | None = None,
        test_dataset: Dataset | Iterable[Dataset] | None = None,
        predict_dataset: Dataset | Iterable[Dataset] | None = None,
        *,
        batch_size: int = 1,
    ) -> "DataModule[_T_co, U]":
        """Create a :class:`FromDatasets` data module from in-memory datasets.

        Args:
            train_dataset: Training data (array, ``array.array``, or list of them).
            val_dataset: Validation data.
            test_dataset: Test data.
            predict_dataset: Prediction data.
            batch_size: Global batch size.
        """
        return FromDatasets(
            train_dataset, val_dataset, test_dataset, predict_dataset, batch_size=batch_size
        )


class FromDatasets(DataModule[_T_co, U], Generic[_T_co, U]):
    def __init__(
        self,
        train_dataset: Dataset | Iterable[Dataset] | None = None,
        val_dataset: Dataset | Iterable[Dataset] | None = None,
        test_dataset: Dataset | Iterable[Dataset] | None = None,
        predict_dataset: Dataset | Iterable[Dataset] | None = None,
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
    def train_dataloader(self) -> "reax.DataLoader[_T_co, U]":
        """Train dataloader."""
        return _loaders.ReaxDataLoader(self._train_dataset, batch_size=self._batch_size)

    @override
    def val_dataloader(self) -> "reax.DataLoader[_T_co, U]":
        """Val dataloader."""
        return _loaders.ReaxDataLoader(self._val_dataset, batch_size=self._batch_size)

    @override
    def test_dataloader(self) -> "reax.DataLoader[_T_co, U]":
        """Test dataloader."""
        return _loaders.ReaxDataLoader(self._test_dataset, batch_size=self._batch_size)

    @override
    def predict_dataloader(self) -> "reax.DataLoader[_T_co, U]":
        """Predict dataloader."""
        return _loaders.ReaxDataLoader(self._predict_dataset, batch_size=self._batch_size)
