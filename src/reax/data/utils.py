from collections.abc import Iterable, Mapping
import dataclasses
from typing import Any, Callable, Optional, TypeVar, Union, cast

import jax
import numpy as np

from reax.utils import containers

__all__ = "extract_batch_size", "sized_len", "get_registry"

Extractor = Callable[[Any], Iterable[int]]
T = TypeVar("T", bound=type)


class BatchSizer:
    """Tool for extracting batch sizes from dataset"""

    def __init__(self):
        self._registry = containers.TypeRegistry[Extractor]()

    def register(
        self,
        entry_type: Union[T, tuple[type, ...]],
        extractor: Callable[[Union[T, tuple[type, ...]]], Iterable[int]],
    ):
        self._registry.register(entry_type, extractor)

    def extract_batch_size(self, batch) -> Iterable[int]:
        """Try to extract batch size."""
        extractor = self._registry.find(batch)
        if extractor is not None:
            yield from extractor(batch)
        else:
            yield from self._fallback_extractor(batch)

    def _fallback_extractor(self, batch) -> Iterable[int]:
        if isinstance(batch, (Iterable, Mapping)):
            if isinstance(batch, Mapping):
                batch = batch.values()

            for entry in batch:
                yield from self.extract_batch_size(entry)

        elif dataclasses.is_dataclass(batch) and not isinstance(batch, type):
            batch = cast(dataclasses.dataclass, batch)
            for field in dataclasses.fields(batch):
                yield from self.extract_batch_size(getattr(batch, field.name))

        else:
            yield None


_registry = None  # pylint: disable=invalid-name


def _array_batch_size(batch: Union[jax.Array, np.ndarray]):
    if batch.ndim == 0:
        yield 1
    else:
        yield batch.shape[0]


try:
    import torch

    def _tensor_batch_size(batch: torch.Tensor):
        if batch.ndim == 0:
            yield 1
        else:
            yield batch.shape[0]

except ImportError:
    torch = None


def get_registry() -> BatchSizer:
    global _registry  # pylint: disable=global-statement
    if _registry is None:
        _registry = BatchSizer()
        _registry.register((jax.Array, np.ndarray), _array_batch_size)
        if torch is not None:
            _registry.register(torch.Tensor, _tensor_batch_size)

    return _registry


def extract_batch_size(batch) -> int:
    error_msg = (
        "Could not determine batch size automatically.  You can provide this manually using "
        "`self.log(..., batch_size=size)`"
    )
    batch_size = None
    try:
        for size in get_registry().extract_batch_size(batch):
            if batch_size is None:
                batch_size = size
            elif size is not None and size != batch_size:
                # TODO: Turn this into a warning
                print(
                    f"Could not determine batch size unambiguously, found {batch_size} and {size}"
                )
                break
    except RecursionError:
        raise RecursionError(error_msg) from None

    if batch_size is None:
        raise RuntimeError(error_msg)

    return batch_size


def sized_len(dataloader) -> Optional[int]:
    try:
        return len(dataloader)
    except (TypeError, NotImplementedError):
        return None
