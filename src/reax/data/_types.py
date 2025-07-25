import abc
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Generic, TypeVar, Union

if TYPE_CHECKING:
    import reax

__all__ = "Sampler", "DataLoader", "Dataset", "CollateFn"

_T_co = TypeVar("_T_co", covariant=True)
U = TypeVar("U")
IdxT = TypeVar("IdxT")

Dataset = Union[Iterable[_T_co], Sequence[_T_co]]
Sampler = Iterable[IdxT]
CollateFn = Callable[[Sequence[_T_co]], U]


class DataLoader(Generic[_T_co, U], Iterable[U], abc.ABC):
    def __len__(self) -> int:
        return len(self.sampler)

    @property
    @abc.abstractmethod
    def dataset(self) -> "reax.data.Dataset[_T_co]":
        """The dataset being loaded."""

    @property
    @abc.abstractmethod
    def sampler(self) -> "reax.data.Sampler":
        """Access the index sampler used by the dataloader"""

    @abc.abstractmethod
    def with_new_sampler(self, sampler: "reax.data.Sampler") -> "DataLoader[_T_co, U]":
        """Recreate the loader with the given index sampler"""
