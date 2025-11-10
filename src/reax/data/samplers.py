from collections.abc import Hashable, Iterable, Iterator, Sequence
import contextlib
import functools
import itertools
import math
from typing import TYPE_CHECKING, Final, TypeVar

import beartype
import jax
import jaxtyping as jt
import numpy as np

from . import _types, datasets

if TYPE_CHECKING:
    import reax

__all__ = (
    "SequentialSampler",
    "RandomSampler",
    "BatchSampler",
    "IterableSampler",
    "DistributedSampler",
)

_T_co = TypeVar("_T_co", covariant=True)
_IdxT = TypeVar("_IdxT", bound=Hashable)


class SequentialSampler(_types.Sampler[int]):
    """Sequentially sample integers index samples up to a given `length`.

    Equivalent to `range(length)`.
    """

    def __init__(self, length: int) -> None:
        """Init function."""
        if not isinstance(length, int):
            raise TypeError("Length must be an integer")

        self._length: Final[int] = length

    def __iter__(self) -> Iterator[int]:
        """Iter function."""
        return iter(range(self._length))

    def __len__(self) -> int:
        """Len function."""
        return self._length


class RandomSampler(_types.Sampler[int]):
    """Randomly sample integer index up to a given ``length`` with possible ``replacements``."""

    SAMPLE_SIZE = 32  # Used to control the number of samples we generate internally at once

    def __init__(self, length: int, replacements: bool = False, num_samples: int | None = None):
        # Params
        self._length: Final[int] = length
        self._replacements: Final[bool] = replacements
        self._num_samples: Final[bool | None] = num_samples

    @property
    def num_samples(self) -> int:
        """Num samples."""
        if self._num_samples is None:
            return self._length

        return self._num_samples  # Fixed number of samples

    def __len__(self) -> int:
        """Len function."""
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        """Iter function."""
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


class BatchSampler(_types.Sampler[list[_IdxT]]):
    r"""Sample batches of indexes from a given sample."""

    def __init__(self, sampler: _types.Sampler[_IdxT], batch_size: int, drop_last: bool) -> None:
        # Params
        self._batch_size: Final[int] = batch_size
        self._drop_last: Final[bool] = drop_last

        # State
        self._sampler = sampler

    def __iter__(self) -> Iterator[list[_IdxT]]:
        """Iter function."""
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
        """Len function."""
        if self._drop_last:
            return len(self._sampler) // self._batch_size

        return (len(self._sampler) + self._batch_size - 1) // self._batch_size


class IterableSampler(_types.Sampler[list[None]]):
    def __iter__(self) -> Iterator[list[None]]:
        """Iter function."""
        yield from itertools.repeat([None])


class DistributedSampler(_types.Sampler[_IdxT]):
    """A sampler that distributes data across multiple processes, ensuring each process
    receives a unique and non-overlapping subset of the dataset.

    This sampler is designed for use with JAX's distributed training capabilities.
    It handles shuffling, dropping the last incomplete batch (if specified),
    and partitioning the dataset based on the process rank and number of replicas.
    """

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        dataset: "reax.data.Dataset",
        num_replicas: int | None = None,
        process_index: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        """Initializes the DistributedSampler.

        Args:
            dataset: The dataset to sample from.
            num_replicas: The total number of replicas (processes).
                Defaults to jax.process_count().
            process_index: The index of the current process. Defaults to jax.process_index().
            shuffle: Whether to shuffle the data.
            seed: The seed for shuffling.
            drop_last: Whether to drop the last incomplete batch.
        """
        if num_replicas == 0:
            raise ValueError("Number of replicas cannot be 0.")

        # Params
        self._num_replicas: Final[int] = self._init_num_replicas(num_replicas)
        self._process_index: Final[int] = self._init_process_index(process_index)
        if self._process_index >= self._num_replicas:
            raise ValueError(
                f"Process index ({self._process_index}) must be less than the number of replicas "
                f"({self._num_replicas})."
            )

        self._shuffle: Final[bool] = shuffle
        self._drop_last: Final[bool] = drop_last
        self._len_dataset = datasets.len_dataset(dataset)
        self._num_samples: Final[float] = self._init_num_samples(
            self._len_dataset, drop_last, self._num_replicas
        )
        self._total_size: Final[int] = self._num_samples * self._num_replicas
        self._seed: Final[int] = seed

        # State
        self._epoch: int = 0

    @staticmethod
    def _init_num_samples(len_dataset: int | float, drop_last: bool, num_replicas: int) -> float:
        """Calculates the number of samples each process should receive.

        If the dataset length is evenly divisible by the number of replicas,
        no data is dropped. Otherwise, the dataset is split to the nearest
        available length that is evenly divisible.

        Args:
            len_dataset: The total size of the dataset.
            drop_last: Whether to drop the last incomplete batch.
            num_replicas: The number of replicas.

        Returns:
            The number of samples each process should receive.
        """
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if drop_last and len_dataset % num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            return math.ceil((len_dataset - num_replicas) / num_replicas)

        return math.ceil(len_dataset / num_replicas)

    @staticmethod
    def _init_num_replicas(num_replicas: int | None) -> int:
        return num_replicas if num_replicas is not None else jax.process_count()

    @staticmethod
    def _init_process_index(process_index: int | None) -> int:
        return process_index if process_index is not None else jax.process_index()

    def __iter__(self) -> Iterator[_IdxT]:
        """Returns an iterator over the sampled indices.

        The indices are shuffled (if specified) and partitioned across processes.
        If `drop_last` is True, the last incomplete batch is removed.  Otherwise,
        the dataset is padded to ensure each process receives the same number of samples.

        Returns:
            An iterator over the sampled indices.
        """
        if self._shuffle:
            # deterministically shuffle based on epoch and seed
            key = jax.random.key(self._seed + self._epoch)
            indices = jax.random.permutation(key, self._len_dataset).tolist()
        else:
            indices = list(range(self._len_dataset))

        if self._drop_last:
            # remove the last set of indices
            indices = indices[: self._total_size]
        else:
            # pad up to evenly divisible
            padding_size = self._total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]

        assert len(indices) == self.total_size

        # subsample
        indices = indices[self._process_index : self._total_size : self._num_replicas]
        assert len(indices) == self._num_samples

        return iter(indices)

    def __len__(self) -> int:
        """Returns the number of samples in the sampler.

        Returns:
            The number of samples.
        """
        return self._num_samples

    @property
    def total_size(self) -> int:
        return self._total_size

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for shuffling.

        This allows for different shuffles in each epoch.

        Args:
            epoch: The epoch number.
        """
        self._epoch = epoch


def create_sampler(
    dataset: _types.Dataset[_T_co],
    batch_size: int | None = None,
    replacements: bool = False,
    shuffle: bool = False,
    sampler: "reax.data.Sampler[_T_co]" = None,
    drop_last: bool = False,
) -> "reax.data.Sampler[_T_co]":
    """Create sampler."""
    if sampler is None:
        # Need this special case because jax arrays are not really subclasses of jax.Array and won't
        # be matched by singledispatch
        if isinstance(dataset, jax.Array):
            sampler = create_sequence_sampler(dataset, replacements=replacements, shuffle=shuffle)
        else:
            sampler = _create_sampler(dataset, replacements=replacements, shuffle=shuffle)

    if batch_size is not None:
        sampler = BatchSampler(sampler, batch_size, drop_last)

    return sampler


@functools.singledispatch
def _create_sampler(
    dataset: _types.Dataset[_T_co], replacements: bool = False, shuffle: bool = False
) -> "reax.data.Sampler[_T_co]":
    """Create sampler."""
    raise TypeError(f"Unsupported type {type(dataset).__name__}")


with contextlib.suppress(ImportError):
    import torch.utils.data

    @_create_sampler.register(torch.utils.data.IterableDataset)
    def create_torch_iterable_dataset_sampler(
        dataset: torch.utils.data.IterableDataset[_T_co],
        replacements: bool = False,
        shuffle: bool = False,
    ):
        """Create torch iterable dataset sampler."""
        return create_iterable_sampler(dataset, replacements=replacements, shuffle=shuffle)

    @_create_sampler.register(torch.utils.data.Dataset)
    def create_torch_dataset_sampler(
        dataset: torch.utils.data.Dataset[_T_co], replacements: bool = False, shuffle: bool = False
    ):
        """Create torch dataset sampler."""
        return create_sequence_sampler(dataset, replacements=replacements, shuffle=shuffle)


@_create_sampler.register(Sequence)
@_create_sampler.register(np.ndarray)
def create_sequence_sampler(
    dataset, replacements: bool = False, shuffle: bool = False
) -> "reax.data.Sampler[_T_co]":
    """Create sequence sampler."""
    if shuffle:
        return RandomSampler(len(dataset), replacements=replacements)

    return SequentialSampler(len(dataset))


@_create_sampler.register(Iterable)
def create_iterable_sampler(
    dataset: Iterable[_T_co], replacements: bool = False, shuffle: bool = False
) -> _types.Sampler[None] | _types.Sampler[list[None]]:
    """Create iterable sampler."""
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

    return IterableSampler()


def create_batch_sampler(
    dataset: Sequence[_T_co], replacements: bool = False, shuffle: bool = False
) -> BatchSampler[int]:
    """Create batch sampler."""
    if shuffle:
        return RandomSampler(len(dataset), replacements=replacements)

    return SequentialSampler(len(dataset))
