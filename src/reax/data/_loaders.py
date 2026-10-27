from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Final, TypeVar

import beartype
import jax
import jaxtyping as jt
import numpy as np
from typing_extensions import override

from . import _types, collate, fetchers, samplers

if TYPE_CHECKING:
    import reax

__all__ = "ArrayLoader", "CachingLoader", "ReaxDataLoader", "FetcherDataLoader", "DeviceDataLoader"


T = TypeVar("T")
_U = TypeVar("_U")
_T_co = TypeVar("_T_co", covariant=True)
Array = jax.Array | np.ndarray
ArrayOrArrayTuple = Array | tuple[Array, ...]


def _single_or_value(value: tuple[T, ...], to_test=None) -> T | tuple[T, ...]:
    """Single or value."""
    if to_test is None:
        to_test = value
    if len(to_test) > 1:
        return tuple(value)
    return value[0]


class FetcherDataLoader(_types.DataLoader[_T_co, _U]):
    """A data loader that uses a fetcher to get samples"""

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        dataset: "reax.data.Dataset[_T_co]",
        sampler: _types.Sampler,
        collate_fn: _types.CollateFn,
    ):
        # Params
        self._collate_fn: Final[_types.CollateFn] = collate_fn

        # State
        self._dataset = dataset
        self._sampler = sampler

    @override
    @property
    def dataset(self) -> "reax.data.Dataset[_T_co]":
        return self._dataset

    @override
    @property
    def sampler(self) -> "reax.data.Sampler":
        return self._sampler

    def __iter__(self) -> Iterator[_U]:
        """Iter function."""
        fetcher = fetchers.create_fetcher(self._dataset, collate_fn=self._collate_fn)
        try:
            for indices in self.sampler:
                if not indices:
                    continue
                yield fetcher.fetch(indices)
        except StopIteration:
            pass

    def with_new_sampler(self, sampler: "reax.data.Sampler") -> "FetcherDataLoader[_T_co, _U]":
        return FetcherDataLoader(
            dataset=self._dataset, sampler=sampler, collate_fn=self._collate_fn
        )


class ReaxDataLoader(FetcherDataLoader[_T_co, _U]):
    """A general-purpose data loader provided by REAX that is suitable for most situations."""

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        dataset: "reax.data.Dataset[_T_co]",
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: "reax.data.Sampler | None" = None,
        collate_fn: _types.CollateFn | None = None,
    ):
        sampler = samplers.create_sampler(
            dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler
        )

        if collate_fn is None:
            collate_fn = collate.get_default_collator().collate

        super().__init__(dataset, sampler, collate_fn)


class ArrayLoader(_types.DataLoader[Any, ArrayOrArrayTuple]):
    """A more efficient loader for array datasets that uses .take() to batch over slices of an
    array dataset.
    """

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        arrays: ArrayOrArrayTuple,
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: "reax.data.Sampler | None" = None,
        *,
        _batched_sampler: "reax.data.Sampler | None" = None,
    ):
        """Init function.

        Args:
            arrays: The array, or tuple of equal-length arrays, to load from.
            batch_size: Number of rows per batch.
            shuffle: Whether to shuffle the rows.
            sampler: Optional per-row index sampler. It is batched using ``batch_size``.
            _batched_sampler: Internal, used by :meth:`with_new_sampler`. An already batched
                sampler that is used as-is.
        """
        # Params
        self._batch_size: Final[int] = batch_size
        self._shuffle: Final[bool] = shuffle

        if isinstance(arrays, tuple):
            if not all(arrays[0].shape[0] == array.shape[0] for array in arrays):
                raise ValueError("Size mismatch between tensors")
            first_array = arrays[0]
        else:
            if not isinstance(arrays, (jax.Array, np.ndarray)):
                raise TypeError(f"Expected array or tuple of arrays, got {type(arrays).__name__}")
            first_array = arrays

        # State
        self._arrays: Final[ArrayOrArrayTuple] = arrays
        self._sampler: reax.data.Sampler
        if _batched_sampler is not None:
            self._sampler = _batched_sampler
        else:
            self._sampler = samplers.create_sampler(
                first_array, batch_size=batch_size, shuffle=shuffle, sampler=sampler
            )

    def __iter__(self) -> Iterator[ArrayOrArrayTuple]:
        """Iter function."""
        for indices in self._sampler:
            if len(indices) == 0:
                continue  # Padding entry from a DistributedSampler
            idx = np.asarray(indices, dtype=np.intp)
            if isinstance(self._arrays, tuple):
                yield tuple(array.take(idx, axis=0) for array in self._arrays)
            else:
                yield self._arrays.take(idx, axis=0)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def shuffle(self) -> bool:
        return self._shuffle

    @override
    @property
    def dataset(self) -> ArrayOrArrayTuple:
        return self._arrays

    @override
    @property
    def sampler(self) -> "reax.data.Sampler":
        return self._sampler

    def first(self) -> ArrayOrArrayTuple:
        """First function."""
        return next(iter(self))

    @override
    def with_new_sampler(self, sampler: "reax.data.Sampler") -> "ArrayLoader":
        # Like the other loaders, ``sampler`` replaces ``self.sampler`` (which is already batched)
        return ArrayLoader(
            arrays=self._arrays,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            _batched_sampler=sampler,
        )


class CachingLoader(_types.DataLoader[_T_co, _U]):
    """Caching loader is useful, for example, if you don't want to shuffle data every time but at
    some interval defined by ``reset_every``.

    This means you need to have enough memory to accommodate all the data.
    """

    def __init__(self, loader: "reax.data.DataLoader[_T_co, _U]", reset_every: int):
        self._loader = loader
        self._reset_every = reset_every
        self._time_since_reset = 0
        self._cache: list[_U] | None = None

    def __iter__(self) -> Iterator[_U]:
        """Iter function."""
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

    @override
    @property
    def dataset(self) -> "reax.data.Dataset[_T_co]":
        return self._loader.dataset

    @override
    @property
    def sampler(self) -> "reax.data.Sampler":
        return self._loader.sampler

    @override
    def with_new_sampler(self, sampler: "reax.data.Sampler") -> "CachingLoader[_T_co, _U]":
        return CachingLoader(self._loader.with_new_sampler(sampler), reset_every=self._reset_every)


class DeviceDataLoader(_types.DataLoader[_T_co, _U]):
    """A loader that wraps an existing dataloader the puts data onto the specified device."""

    def __init__(self, loader: "reax.DataLoader[_T_co, _U]", device: jax.Device):
        self._loader = loader
        self._device = device

    def __iter__(self) -> Iterator[_U]:
        for data in self._loader:
            yield jax.device_put(data, device=self._device)

    @property
    def parent(self) -> "reax.data.DataLoader[_T_co, _U]":
        """Access the original loader that this one wraps"""
        return self._loader

    @override
    @property
    def dataset(self) -> "reax.data.Dataset[_T_co]":
        return self._loader.dataset

    @override
    @property
    def sampler(self) -> "reax.data.Sampler":
        return self._loader.sampler

    @override
    def with_new_sampler(self, sampler: "reax.data.Sampler") -> "DeviceDataLoader[_T_co, _U]":
        return DeviceDataLoader(self._loader.with_new_sampler(sampler), device=self._device)
