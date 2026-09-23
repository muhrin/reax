import abc
from collections.abc import Callable, Iterable, Sequence
import contextlib
import functools
from typing import Generic, TypeVar

import jax

from . import _types

_T_co = TypeVar("_T_co", covariant=True)
_U = TypeVar("_U")
CollateFn = Callable[[Sequence[_T_co]], _U]


class _BaseFetcher(Generic[_T_co, _U]):
    def __init__(self, dataset: _types.Dataset[_T_co], collate_fn: CollateFn):
        """Init function."""
        self._dataset = dataset
        self._collate_fn = collate_fn

    @abc.abstractmethod
    def fetch(self, possibly_batched_index) -> _U:
        """Fetch the batch."""


class _IterableFetcher(_BaseFetcher, Generic[_T_co, _U]):
    def __init__(self, dataset: _types.Dataset[_T_co], collate_fn: CollateFn):
        """Init function."""
        super().__init__(dataset, collate_fn)
        self._iter = iter(dataset)
        self._ended = False

    def fetch(self, possibly_batched_index) -> _U:
        """Fetch function."""
        if self._ended:
            raise StopIteration

        data = []
        for _ in possibly_batched_index:
            try:
                data.append(next(self._iter))
            except StopIteration:
                self._ended = True
                break

        # If we didn't fetch any data, then we need to stop iteration
        if len(data) == 0:
            raise StopIteration

        return self._collate_fn(data)


class _MapFetcher(_BaseFetcher, Generic[_T_co, _U]):
    def fetch(self, possibly_batched_index) -> _U:
        """Fetch function."""
        if hasattr(self._dataset, "__getitems__") and self._dataset.__getitems__:
            data = self._dataset.__getitems__(possibly_batched_index)
        else:
            data = [self._dataset[idx] for idx in possibly_batched_index]

        return self._collate_fn(data)


@functools.singledispatch
def _create_fetcher(dataset: _types.Dataset[_T_co], collate_fn: CollateFn) -> _BaseFetcher:
    """Create fetcher (internal singledispatch)."""
    raise TypeError(f"Unsupported type {type(dataset).__name__}")


with contextlib.suppress(ImportError):
    import torch.utils.data

    @_create_fetcher.register(torch.utils.data.IterableDataset)
    def create_torch_iterable_dataset_fetcher(
        dataset: torch.utils.data.IterableDataset[_T_co], collate_fn: CollateFn
    ):
        """Create torch iterable dataset fetcher."""
        return create_iterable_fetcher(dataset, collate_fn)

    @_create_fetcher.register(torch.utils.data.Dataset)
    def create_torch_dataset_fetcher(
        dataset: torch.utils.data.Dataset[_T_co], collate_fn: CollateFn
    ):
        """Create torch dataset fetcher."""
        return create_sequence_fetcher(dataset, collate_fn)


@_create_fetcher.register(Sequence)
def create_sequence_fetcher(
    dataset: Sequence[_T_co], collate_fn: CollateFn[_T_co, _U]
) -> _MapFetcher:
    """Create sequence fetcher."""
    return _MapFetcher(dataset, collate_fn)


@_create_fetcher.register(Iterable)
def create_iterable_fetcher(
    dataset: Iterable[_T_co], collate_fn: CollateFn
) -> _IterableFetcher[_T_co, _U]:
    """Create iterable fetcher."""
    return _IterableFetcher(dataset, collate_fn)


def create_fetcher(
    dataset: _types.Dataset[_T_co], collate_fn: CollateFn[_T_co, _U]
) -> _BaseFetcher[_T_co, _U]:
    """Create fetcher.

    A ``jax.Array`` is registered as a (concrete) ``Iterable`` but not a ``Sequence``, and
    CPython's singledispatch C3 linearization places the ``Iterable`` abstract base class before
    the concrete ``jax.Array`` in the composed MRO. Registering a ``jax.Array`` handler therefore
    would never be selected and the array would be handled by the iterable fetcher, which ignores
    the sampler indices and yields the dataset sequentially. We disambiguate with an explicit
    ``isinstance`` check (mirroring the equivalent special case in ``create_sampler``) so the array
    is treated as a ``_MapFetcher`` and the sampler indices are actually used.
    """
    if isinstance(dataset, jax.Array):
        return _MapFetcher(dataset, collate_fn)

    return _create_fetcher(dataset, collate_fn)
