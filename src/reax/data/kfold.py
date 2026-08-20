"""K-Fold cross-validation index partitioner for reax."""

from collections.abc import Iterator, Sequence
from itertools import islice
from typing import TYPE_CHECKING, TypeVar

import jax
import jax.numpy as jnp
import numpy as np

from . import datasets as datasets_

if TYPE_CHECKING:
    import reax

T = TypeVar("T")

__all__ = ("KFold",)


class KFold:
    """Distributed-aware K-Fold cross-validation index partitioner."""

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        seed: int | None = 42,
        engine: "reax.Engine | None" = None,
    ):
        if n_splits < 2:
            raise ValueError(f"K-fold cross-validation requires at least 2 splits, got {n_splits}.")
        self.n_splits = n_splits
        self.shuffle = shuffle

        # Synchronise seed across distributed ranks
        if engine is not None and shuffle:
            if seed is None:
                # Initialize matching array structures on all ranks
                raw = jnp.zeros((), dtype=jnp.uint32)

                if engine.is_global_zero:
                    # Extract a valid integer representation from the nnx PRNGKey
                    key = engine.rngs.default()
                    raw = jax.random.key_data(key)[0]

                # Broadcast the array and extract the integer scalar
                broadcasted_seed = engine.broadcast(raw, src=0)
                self.seed = int(broadcasted_seed)
            else:
                self.seed = seed
        else:
            self.seed = seed if seed is not None else 42

    # -- public API ----------------------------------------------------------

    def split(self, n_samples: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Lazily yield ``(train_indices, val_indices)`` for each fold."""
        if self.n_splits > n_samples:
            raise ValueError(f"Cannot have n_splits ({self.n_splits}) > n_samples ({n_samples}).")

        indices = np.arange(n_samples)

        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(indices)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            val_idx = indices[start:stop]
            train_idx = np.concatenate([indices[:start], indices[stop:]])
            yield train_idx, val_idx
            current = stop

    def get_fold(
        self,
        dataset: Sequence[T],
        fold: int,
    ) -> "tuple[datasets_.Subset[T], datasets_.Subset[T]]":
        """Return ``(train_subset, val_subset)`` for the given *fold* index."""
        if not 0 <= fold < self.n_splits:
            raise ValueError(f"fold index {fold} out of range [0, {self.n_splits}).")

        train_idx, val_idx = next(islice(self.split(len(dataset)), fold, None))

        return (
            # Pass arrays directly to avoid massive list materialization
            datasets_.Subset(dataset, train_idx),
            datasets_.Subset(dataset, val_idx),
        )
