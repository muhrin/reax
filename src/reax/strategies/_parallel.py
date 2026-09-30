import abc
from collections.abc import Iterator
from typing import TYPE_CHECKING, TypeVar

import beartype
import jaxtyping as jt
from typing_extensions import override

from .. import data as data_
from . import _strategies

if TYPE_CHECKING:
    import reax


_T_co = TypeVar("_T_co", covariant=True)
_U = TypeVar("_U")
_IdxEv = TypeVar("_IdxEv")


__all__ = ("ParallelStrategy",)


class ParallelStrategy(_strategies.Strategy, abc.ABC):
    @property
    @abc.abstractmethod
    def process_index(self) -> int:
        """Return the global rank of the current process."""

    @property
    @abc.abstractmethod
    def process_count(self) -> int:
        """Return the total number of ranks."""

    @override
    def setup_dataloader(self, data: "reax.DataLoader[_T_co, _U]") -> "reax.DataLoader[_T_co, _U]":
        if not isinstance(data, data_.DataLoader):
            data = super().setup_dataloader(data)

        # The new DistributedSampler wraps the existing sampler cleanly
        sampler = data_.samplers.DistributedSampler(
            inner=data.sampler,
            num_replicas=self.process_count,
            process_index=self.process_index,
        )

        data = data.with_new_sampler(sampler)
        return data


class ParallelDataLoader(data_.DataLoader[_T_co, _U]):
    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        dataset: "reax.data.Dataset[_T_co]",
        sampler: data_.samplers.DistributedSampler,
        fetcher: "fetchers._BaseFetcher[_T_co, _U]",
    ):
        self._dataset = dataset
        self._sampler = sampler
        self._fetcher = fetcher

    @override
    @property
    def dataset(self) -> "reax.data.Dataset[_T_co]":
        return self._dataset

    @override
    @property
    def sampler(self):
        return self._sampler

    def __iter__(self) -> Iterator[_U]:
        """Iter function."""
        try:
            for indices in self.sampler:
                yield self._fetcher.fetch(indices)
        except StopIteration:
            pass

    def with_new_sampler(self, sampler: "reax.data.Sampler") -> "ParallelDataLoader[_T_co, _U]":
        return ParallelDataLoader(dataset=self._dataset, sampler=sampler, fetcher=self._fetcher)
