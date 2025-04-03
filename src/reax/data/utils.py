from collections.abc import Iterable, Mapping
import dataclasses
import logging
from typing import Any, Callable, Optional, TypeVar, Union, cast

import jax
import numpy as np

from reax.utils import containers

from .. import plugins

_LOGGER = logging.getLogger(__name__)

__all__ = "extract_batch_size", "sized_len", "get_registry"

Extractor = Callable[[Any], Iterable[int]]
T = TypeVar("T", bound=type)


class BatchSizer:
    """Tool for extracting batch sizes from dataset."""

    def __init__(self):
        self._registry = containers.TypeRegistry[Extractor]()

    def register(
        self,
        entry_type: Union[T, tuple[type, ...]],
        extractor: Callable[[Union[T, tuple[type, ...]]], Iterable[int]],
    ):
        """Register function."""
        self._registry.register(entry_type, extractor)

    def extract_batch_size(self, batch) -> Iterable[int]:
        """Try to extract batch size."""
        extractor = self._registry.find(batch)
        if extractor is not None:
            yield from extractor(batch)
        else:
            yield from self._fallback_extractor(batch)

    def _fallback_extractor(self, batch) -> Iterable[int]:
        """Fallback extractor."""
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
    """Array batch size."""
    if batch.ndim == 0:
        yield 1
    else:
        yield batch.shape[0]


try:
    import torch

    def _tensor_batch_size(batch: "torch.Tensor"):
        """Tensor batch size."""
        if batch.ndim == 0:
            yield 1
        else:
            yield batch.shape[0]

except ImportError:
    torch = None


def get_registry() -> BatchSizer:
    """Get registry."""
    global _registry  # pylint: disable=global-statement
    if _registry is None:
        _registry = BatchSizer()
        _registry.register((jax.Array, np.ndarray), _array_batch_size)
        if torch is not None:
            _registry.register(torch.Tensor, _tensor_batch_size)

        for sizer in plugins.get_batch_sizers():
            _registry.register(*sizer)

    return _registry


def extract_batch_size(batch) -> int:
    """Extract batch size."""
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
                _LOGGER.warning(
                    "Could not determine batch size unambiguously, found %d and %d",
                    batch_size,
                    size,
                )
                break
    except RecursionError:
        raise RecursionError(error_msg) from None

    if batch_size is None:
        raise RuntimeError(error_msg)

    return batch_size


def sized_len(dataloader) -> Optional[int]:
    """Sized len."""
    try:
        return len(dataloader)
    except (TypeError, NotImplementedError):
        return None
