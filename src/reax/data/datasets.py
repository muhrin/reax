import bisect
from collections.abc import Sequence
import itertools
import math
from typing import TYPE_CHECKING, Iterable, TypeVar, Union, cast
import warnings

import beartype
import jax.random
import jaxtyping as jt
import numpy as np

if TYPE_CHECKING:
    import reax

_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)


__all__ = ("ArrayDataset", "ConcatDataset", "Subset", "random_split")


class ConcatDataset(Sequence[_T_co]):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """

    datasets: list[Sequence[_T_co]]
    cumulative_sizes: list[int]

    @staticmethod
    def cumsum(sequence: Sequence):
        result, csum = [], 0
        for entry in sequence:
            length = len(entry)
            result.append(length + csum)
            csum += length
        return result

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(self, datasets: Iterable[Sequence[_T_co]]) -> None:
        super().__init__()
        self.datasets = list(datasets)
        if len(self.datasets) == 0:
            raise ValueError("datasets should not be an empty iterable")

        for dataset in self.datasets:
            if not isinstance(dataset, Sequence):
                raise TypeError(
                    f"Dataset must support random access (__getitem__), got "
                    f"{dataset.__class__.__name__}"
                )

        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]


class ArrayDataset(Sequence[tuple[jt.Array, ...]]):
    r"""Dataset wrapping arrays.

    Each sample will be retrieved by indexing arrays along the first dimension.

    :params *arrays: arrays that have the same size of the first dimension.
    """

    arrays: tuple[jt.Array, ...]

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(self, *arrays: Union[jax.Array, np.ndarray]) -> None:
        if not all(arrays[0].shape[0] == array.shape[0] for array in arrays):
            raise ValueError("Size mismatch between tensors")
        self.arrays = arrays

    def __getitem__(self, index: int) -> tuple[jt.Array, ...]:
        return tuple(array[index] for array in self.arrays)

    def __len__(self):
        return self.arrays[0].shape[0]


class Subset(Sequence[_T_co]):
    r"""
    Subset of a dataset at specified indices.
    """

    dataset: Sequence[_T_co]
    indices: Sequence[int]

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(self, dataset: Sequence[_T_co], indices: Sequence[int]) -> None:
        """
        :param dataset: The whole Dataset
        :param indices: Indices in the whole set selected for subset
        """
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx: Union[int, list[int]]) -> Union[_T_co, list[_T_co]]:
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]

        return self.dataset[self.indices[idx]]

    def __getitems__(self, indices: list[int]) -> list[_T_co]:
        if callable(getattr(self.dataset, "__getitems__", None)):
            return self.dataset.__getitems__([self.indices[idx] for idx in indices])

        return [self.dataset[self.indices[idx]] for idx in indices]

    def __len__(self) -> int:
        return len(self.indices)


def random_split(
    rng: "reax.Generator",
    dataset: Sequence[_T],
    lengths: Sequence[Union[int, float]],
) -> list[Subset[_T]]:
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    Example:
        >>> # xdoctest: +SKIP
        >>> key1 = torch.Generator().manual_seed(42)
        >>> key2 = torch.Generator().manual_seed(42)
        >>> random_split(key1, range(10), [3, 7])
        >>> random_split(key2, range(30), [0.3, 0.3, 0.4])

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: list[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(math.floor(len(dataset) * frac))  # type: ignore[arg-type]
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(
                    f"Length of split at index {i} is 0. " f"This might result in an empty dataset."
                )

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = jax.random.permutation(rng.make_key(), sum(lengths)).tolist()
    lengths = cast(Sequence[int], lengths)
    return [
        Subset(dataset, indices[offset - length : offset])
        for offset, length in zip(itertools.accumulate(lengths), lengths)
    ]
