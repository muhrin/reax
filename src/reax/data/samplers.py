from collections.abc import Iterable, Iterator, Sequence, Sized
import contextlib
import functools
import itertools
import math
from typing import TYPE_CHECKING, Any, Final, Generic, TypeVar

import beartype
import jax
import jaxtyping as jt
import numpy as np
from typing_extensions import Never

from . import _types

if TYPE_CHECKING:
    import reax
    import reax.data

__all__ = (
    "SequentialSampler",
    "RandomSampler",
    "ChunkedSampler",
    "IterableSampler",
    "DistributedSampler",
)

_T_co = TypeVar("_T_co", covariant=True)
_IdxT = TypeVar("_IdxT")
Empty = list[Never]


def _try_len(obj: object) -> int | None:
    """Return ``len(obj)``, or ``None`` if it has no (usable) length.

    Wrappers such as :class:`ChunkedSampler` always define ``__len__`` but raise ``TypeError`` when
    the wrapped sampler is unsized, so an ``isinstance(obj, Sized)`` check isn't enough.
    """
    if not isinstance(obj, Sized):
        return None
    try:
        return len(obj)
    except TypeError:
        return None


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
        self._num_samples: Final[int | None] = num_samples

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
                yield from map(int, np.random.randint(0, high=total, size=(self.SAMPLE_SIZE,)))

            yield from map(
                int, np.random.randint(0, high=total, size=(self.num_samples % self.SAMPLE_SIZE,))
            )
        else:
            for _ in range(self.num_samples // total):
                yield from map(int, np.random.permutation(total))

            yield from map(int, np.random.permutation(total)[: self.num_samples % total])


class ChunkedSampler(Generic[_IdxT], _types.Sampler[list[_IdxT]]):
    r"""Sample batches of indexes from a given sample."""

    def __init__(self, sampler: _types.Sampler[_IdxT], batch_size: int, drop_last: bool) -> None:
        # Params
        self._batch_size: Final[int] = batch_size
        self._drop_last: Final[bool] = drop_last

        # State
        self._sampler = sampler

    @property
    def sampler(self) -> _types.Sampler[_IdxT]:
        """Return the wrapped sampler."""
        return self._sampler

    @sampler.setter
    def sampler(self, value: _types.Sampler[_IdxT]) -> None:
        """Replace the wrapped sampler (e.g. with a sharded distributed sampler)."""
        self._sampler = value

    def __iter__(self) -> Iterator[list[_IdxT]]:
        """Iter function."""
        sampler_iter = iter(self._sampler)
        while True:
            batch = list(itertools.islice(sampler_iter, self._batch_size))
            if not batch:
                return
            if self._drop_last and len(batch) < self._batch_size:
                return
            yield batch

    def __len__(self) -> int:
        """Len function.

        Raises:
            TypeError: If the wrapped sampler is unsized.
        """
        if not isinstance(self._sampler, Sized):
            raise TypeError(f"Wrapped sampler {type(self._sampler).__name__} has no length.")
        if self._drop_last:
            return len(self._sampler) // self._batch_size

        return (len(self._sampler) + self._batch_size - 1) // self._batch_size


class IterableSampler(_types.Sampler[list[None]]):
    """Endless sampler for iterable (non random-access) datasets.

    Each item is a placeholder "index" ``[None]``: the iterable fetcher ignores the values and just
    pulls one element from the dataset per entry, stopping when the dataset is exhausted.
    """

    def __iter__(self) -> Iterator[list[None]]:
        """Iter function."""
        yield from itertools.repeat([None])


class DistributedSampler(_types.Sampler[_T_co | Empty]):
    """A sampler that distributes data across multiple processes.

    This sampler is a pure sharding wrapper around an inner sampler (e.g.,
    SequentialSampler, RandomSampler, or BatchSampler). It distributes the inner
    sampler's items across processes using a strided partition. It doesn't know
    about indices vs. batches—it treats the inner stream as an opaque sequence.
    """

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        inner: _types.Sampler[_T_co],
        num_replicas: int | None = None,
        process_index: int | None = None,
        drop_last: bool = False,
    ) -> None:
        """Initializes the DistributedSampler.

        Args:
            inner: The inner sampler to distribute.
            num_replicas: The total number of replicas (processes).
                Defaults to jax.process_count().
            process_index: The index of the current process. Defaults to jax.process_index().
            drop_last: Whether to drop the last incomplete step across ranks.
        """
        # Params
        self._inner = inner
        self._num_replicas: Final[int] = self._init_num_replicas(num_replicas)
        self._process_index: Final[int] = self._init_process_index(process_index)

        if self._num_replicas <= 0:
            raise ValueError(f"Number of replicas must be positive, got {self._num_replicas}.")
        if self._process_index < 0:
            raise ValueError(f"Process index must be non-negative, got {self._process_index}.")
        if self._process_index >= self._num_replicas:
            raise ValueError(
                f"Process index ({self._process_index}) must be less than the number of replicas "
                f"({self._num_replicas})."
            )

        self._drop_last: Final[bool] = drop_last

        # Sized inner: we know the global step count arithmetically.
        self._global_steps: Final[int | None] = _try_len(inner)

    @staticmethod
    def _init_num_replicas(num_replicas: int | None) -> int:
        return num_replicas if num_replicas is not None else jax.process_count()

    @staticmethod
    def _init_process_index(process_index: int | None) -> int:
        return process_index if process_index is not None else jax.process_index()

    @property
    def inner(self) -> _types.Sampler[_T_co]:
        """Return the wrapped inner sampler."""
        return self._inner

    def _step_count(self, global_steps: int) -> int:
        """Total number of steps this rank must emit (uniform across ranks)."""
        if self._drop_last:
            return global_steps // self._num_replicas
        return math.ceil(global_steps / self._num_replicas)

    def _real_count(self, global_steps: int) -> int:
        """Number of real (non-empty) items this rank will produce."""
        if self._drop_last:
            return global_steps // self._num_replicas
        return math.ceil((global_steps - self._process_index) / self._num_replicas)

    def __iter__(self) -> Iterator[_T_co | Empty]:
        """Returns an iterator over the distributed samples.

        Yields real items from the inner sampler for this rank. When ``drop_last``
        is False and the inner sampler is sized, short ranks are padded with empty
        ``[]`` entries so that every rank yields the same number of steps.
        """
        inner_iter = iter(self._inner)
        global_steps = self._global_steps

        if global_steps is None:
            yield from itertools.islice(inner_iter, self._process_index, None, self._num_replicas)
            return

        if self._drop_last:
            real = self._real_count(global_steps)
            # Truncate the inner stream to real * R before striding
            inner_iter = itertools.islice(inner_iter, real * self._num_replicas)
            yield from itertools.islice(inner_iter, self._process_index, None, self._num_replicas)
        else:
            real = self._real_count(global_steps)
            steps = self._step_count(global_steps)
            yield from itertools.islice(inner_iter, self._process_index, None, self._num_replicas)
            for _ in range(steps - real):
                yield []

    def __len__(self) -> int:
        """Returns the number of steps this sampler will yield."""
        if self._global_steps is None:
            raise TypeError("Inner sampler is unsized and has no length.")
        return self._step_count(self._global_steps)


def create_sampler(
    dataset: _types.Dataset[_T_co],
    batch_size: int | None = None,
    replacements: bool = False,
    shuffle: bool = False,
    sampler: "reax.data.Sampler[Any] | None" = None,
    drop_last: bool = False,
) -> "reax.data.Sampler[Any] | reax.data.BatchSampler[Any]":
    """Create a sampler for ``dataset``, batching it if ``batch_size`` is given.

    If ``sampler`` is passed it is used as-is (and batched if requested), otherwise a default
    sampler is created based on the dataset type.
    """
    base: reax.data.Sampler[Any]
    if sampler is not None:
        base = sampler
    elif isinstance(dataset, jax.Array):
        # Need this special case because jax arrays are not really subclasses of jax.Array and
        # won't be matched by singledispatch
        base = create_sequence_sampler(dataset, replacements=replacements, shuffle=shuffle)
    else:
        base = _create_sampler(dataset, replacements=replacements, shuffle=shuffle)

    if batch_size is None:
        return base

    return ChunkedSampler(base, batch_size, drop_last)


@functools.singledispatch
def _create_sampler(
    dataset: _types.Dataset[_T_co], replacements: bool = False, shuffle: bool = False
) -> "reax.data.Sampler[Any]":
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
) -> "reax.data.Sampler[int]":
    """Create sequence sampler."""
    if shuffle:
        return RandomSampler(len(dataset), replacements=replacements)

    return SequentialSampler(len(dataset))


@_create_sampler.register(Iterable)
def create_iterable_sampler(
    dataset: Iterable[Any], replacements: bool = False, shuffle: bool = False
) -> IterableSampler:
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
