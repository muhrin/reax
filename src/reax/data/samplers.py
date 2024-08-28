from collections.abc import Iterable, Iterator, Sequence
import contextlib
import functools
import itertools
from typing import Hashable, TypeVar, Union

import numpy as np

from . import _types

__all__ = "SequentialSampler", "RandomSampler", "BatchSampler", "IterableSampler"

T_co = TypeVar("T_co", covariant=True)
IdxT = TypeVar("IdxT", bound=Hashable)


class SequentialSampler(_types.Sampler[int]):
    """
    Sequentially sample integers index samples up to a given `length`.  Equivalent to
    `range(length)`
    """

    def __init__(self, length: int) -> None:
        if not isinstance(length, int):
            raise TypeError("Length must be an integer")

        self._length = length

    def __iter__(self) -> Iterator[int]:
        return iter(range(self._length))

    def __len__(self) -> int:
        return self._length


class RandomSampler(_types.Sampler[int]):
    """
    Randomly sample integer index up to a given `length` with possible `replacements`

    """

    SAMPLE_SIZE = 32  # Used to control the number of samples we generate internally at once

    def __init__(self, length: int, replacements: bool = False, num_samples: int = None):
        self._length = length
        self._replacements = replacements
        self._num_samples = num_samples

    @property
    def num_samples(self) -> int:
        if self._num_samples is None:
            return self._length

        return self._num_samples  # Fixed number of samples

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        total = self._length

        if self._replacements:
            for _ in range(self.num_samples // self.SAMPLE_SIZE):
                yield from np.random.randint(0, high=total, size=(self.SAMPLE_SIZE,)).tolist()
            yield from np.random.randint(
                0, high=total, size=(self.num_samples % self.SAMPLE_SIZE,)
            ).tolist()
        else:
            for _ in range(self.num_samples // total):
                yield from np.random.permutation(total).tolist()
            yield from np.random.permutation(total).tolist()[: self.num_samples % total]


class BatchSampler(_types.Sampler[list[IdxT]]):
    """Sample batches of indexes from a given sampler"""

    def __init__(self, sampler: _types.Sampler[IdxT], batch_size: int, drop_last: bool) -> None:
        self._sampler = sampler
        self._batch_size = batch_size
        self._drop_last = drop_last

    def __iter__(self) -> Iterator[list[IdxT]]:
        if self._drop_last:
            sampler_iter = iter(self._sampler)
            while True:
                try:
                    batch = [next(sampler_iter) for _ in range(self._batch_size)]
                    yield batch
                except StopIteration:
                    break
        else:
            batch = [0] * self._batch_size
            idx_in_batch = 0
            for idx in self._sampler:
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx_in_batch == self._batch_size:
                    yield batch
                    idx_in_batch = 0
                    batch = [0] * self._batch_size
            if idx_in_batch > 0:
                yield batch[:idx_in_batch]

    def __len__(self) -> int:
        if self._drop_last:
            return len(self._sampler) // self._batch_size

        return (len(self._sampler) + self._batch_size - 1) // self._batch_size


class IterableSampler(_types.Sampler[None]):
    def __iter__(self) -> Iterator[None]:
        yield from itertools.repeat(None)


@functools.singledispatch
def create_sampler(
    dataset: _types.Dataset[T_co],
    batch_size: int = 1,
    replacements: bool = False,
    shuffle: bool = False,
) -> _types.Sampler:
    raise TypeError(f"Unsupported type {type(dataset).__name__}")


with contextlib.suppress(ImportError):
    import torch.utils.data

    @create_sampler.register(torch.utils.data.IterableDataset)
    def create_torch_iterable_dataset_sampler(
        dataset: torch.utils.data.IterableDataset[T_co],
        batch_size: int = 1,
        replacements: bool = False,
        shuffle: bool = False,
    ):
        return create_iterable_sampler(
            dataset, batch_size=batch_size, replacements=replacements, shuffle=shuffle
        )

    @create_sampler.register(torch.utils.data.Dataset)
    def create_torch_dataset_sampler(
        dataset: torch.utils.data.Dataset[T_co],
        batch_size: int = 1,
        replacements: bool = False,
        shuffle: bool = False,
    ):
        return create_sequence_sampler(
            dataset, batch_size=batch_size, replacements=replacements, shuffle=shuffle
        )


@create_sampler.register(Sequence)
def create_sequence_sampler(
    dataset: Sequence[T_co],
    batch_size: int = 1,
    replacements: bool = False,
    shuffle: bool = False,
) -> _types.Sampler[list[IdxT]]:
    if shuffle:
        sampler = RandomSampler(len(dataset), replacements=replacements)
    else:
        sampler = SequentialSampler(len(dataset))

    return BatchSampler(sampler, batch_size, False)


@create_sampler.register(Iterable)
def create_iterable_sampler(
    dataset: Iterable[T_co],
    batch_size: int = 1,
    replacements: bool = False,
    shuffle: bool = False,
) -> Union[_types.Sampler[None], _types.Sampler[list[None]]]:
    if shuffle:
        raise ValueError(
            f"``shuffle=True`` is not supported with dataset type {type(dataset).__name__} which "
            f"does not support random access"
        )
    if replacements:
        raise ValueError(
            f"``replacements=True`` is not supported with dataset type {type(dataset).__name__} "
            f"which does not support random access"
        )

    sampler = IterableSampler()
    if batch_size == 1:
        return sampler

    return BatchSampler(sampler, batch_size, False)


def create_batch_sampler(
    dataset: Sequence[T_co],
    batch_size: int = 1,
    replacements: bool = False,
    shuffle: bool = False,
) -> BatchSampler[int]:
    if shuffle:
        sampler = RandomSampler(len(dataset), replacements=replacements)
    else:
        sampler = SequentialSampler(len(dataset))

    return BatchSampler(sampler, batch_size, False)
