from collections.abc import Iterable, Iterator
from typing import Optional, TypeVar, Union

import beartype
import jax
import jaxtyping as jt
import numpy as np

from . import _types, collate, fetchers, samplers

__all__ = ("ArrayLoader", "CachingLoader", "ReaxDataLoader")


T = TypeVar("T")
ArrayOrArrayTuple = Union[jax.typing.ArrayLike, tuple[jax.typing.ArrayLike, ...]]


def _single_or_value(value: tuple[T, ...], to_test=None) -> Union[T, tuple[T, ...]]:
    if to_test is None:
        to_test = value
    if len(to_test) > 1:
        return tuple(value)
    return value[0]


class ReaxDataLoader(_types.DataLoader):
    """
    A general-purpose data loader provided by REAX that is suitable for most situations.
    """

    def __init__(
        self,
        dataset: _types.Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        collate_fn: Optional[_types.CollateFn] = None,
    ):
        self._batch_size = batch_size
        self._dataset = dataset
        self._index_sampler = samplers.create_sampler(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )

        if collate_fn is None:
            collate_fn = collate.get_default_collator().collate

        self._fetcher = fetchers.create_fetcher(dataset, collate_fn=collate_fn)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def __len__(self) -> int:
        return len(self._index_sampler)

    def __iter__(self):
        for indices in self._index_sampler:
            yield self._fetcher.fetch(indices)


class ArrayLoader(Iterable[ArrayOrArrayTuple]):
    """A dataset of arrays"""

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        arrays: ArrayOrArrayTuple,
        batch_size: int = 1,
        shuffle=False,
    ):
        if isinstance(arrays, tuple):
            if not all(arrays[0].shape[0] == array.shape[0] for array in arrays):
                raise ValueError("Size mismatch between tensors")
            first_array = arrays[0]
        else:
            if not isinstance(arrays, (jax.Array, np.ndarray)):
                raise TypeError(f"Expected array or tuple of arrays, got {type(arrays).__name__}")
            first_array = arrays
        self._arrays = arrays

        self._sampler: _types.Sampler[list[int]] = samplers.create_sequence_sampler(
            first_array, batch_size=batch_size, shuffle=shuffle
        )

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __iter__(self) -> Iterator[ArrayOrArrayTuple]:
        for idx in self._sampler:
            idx = np.asarray(idx)
            if isinstance(self._arrays, tuple):
                yield tuple(array.take(idx) for array in self._arrays)
            else:
                yield self._arrays.take(idx)

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __len__(self) -> int:
        return len(self._sampler)

    def first(self) -> ArrayOrArrayTuple:
        return next(iter(self))


class CachingLoader(Iterable):
    """
    Caching loader is useful, for example, if you don't want to shuffle data every time but at
    some interval defined by `repeat_every`.  This means you need to have enough memory to
    accommodate all the data.
    """

    def __init__(self, loader: _types.DataLoader, reset_every: int):
        self._loader = loader
        self._reset_every = reset_every
        self._time_since_reset = 0
        self._cache = None

    def __iter__(self):
        if self._cache:
            yield from self._cache
        else:
            # Have to pull from the loader
            cache = []
            for entry in self._loader:
                yield entry
                cache.append(entry)
            self._cache = cache

        self._time_since_reset += 1
        # Check if we should clear the cache
        if self._time_since_reset == self._reset_every:
            self._cache = []
            self._time_since_reset = 0
