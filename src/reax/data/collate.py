import contextlib
import copy
from typing import (
    Any,
    Callable,
    Mapping,
    MutableMapping,
    MutableSequence,
    Optional,
    Sequence,
    TypeVar,
)

import jax
import numpy as np

from reax.utils import containers

T_co = TypeVar("T_co", covariant=True)
U = TypeVar("U")
CollateFn = Callable[[Sequence[T_co]], U]

DEFAULT_COLLATE_ERR_MSG_FORMAT = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found {}"
)


class Collator:
    def __init__(self):
        super().__init__()
        self._registry = containers.TypeRegistry[CollateFn]()

    def register(self, entry_type: T_co, collate_fn: Callable[[Sequence[T_co]], U]):
        self._registry.register(entry_type, collate_fn)

    def collate(self, batch: Any) -> Any:
        elem = batch[0]

        collate_fn = self._registry.find(elem)
        if collate_fn is not None:
            return collate_fn(batch)

        # Try the fallback options
        return self._fallback_collate(batch)

    def _fallback_collate(self, batch: Any) -> Any:
        """Called when the batch element doesn't match any type registered with the collator"""
        elem = batch[0]
        elem_type = type(elem)

        if isinstance(elem, Mapping):
            try:
                if isinstance(elem, MutableMapping):
                    # The mapping type may have extra properties, so we can't just
                    # use `type(data)(...)` to create the new mapping.
                    # Create a clone and update it if the mapping type is mutable.
                    clone = copy.copy(elem)
                    clone.update({key: self.collate([d[key] for d in batch]) for key in elem})
                    return clone
                # else:
                return elem_type({key: self.collate([d[key] for d in batch]) for key in elem})
            except TypeError:
                # The mapping type may not support `copy()` / `update(mapping)`
                # or `__init__(iterable)`.
                return {key: self.collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
            return elem_type(*(self.collate(samples) for samples in zip(*batch)))
        elif isinstance(elem, Sequence):
            # check to make sure that the elements in batch have consistent size
            it = iter(batch)
            elem_size = len(next(it))
            if not all(len(elem) == elem_size for elem in it):
                raise RuntimeError("each element in list of batch should be of equal size")
            # else:
            transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

            if isinstance(elem, tuple):
                return [self.collate(samples) for samples in transposed]  # Backwards compatibility.
            # else:
            try:
                if isinstance(elem, MutableSequence):
                    # The sequence type may have extra properties, so we can't just
                    # use `type(data)(...)` to create the new sequence.
                    # Create a clone and update it if the sequence type is mutable.
                    clone = copy.copy(elem)  # type: ignore[arg-type]
                    for i, samples in enumerate(transposed):
                        clone[i] = self.collate(samples)
                    return clone

                return elem_type([self.collate(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `copy()` / `__setitem__(index, item)`
                # or `__init__(iterable)` (e.g., `range`).
                return [self.collate(samples) for samples in transposed]

        raise TypeError(DEFAULT_COLLATE_ERR_MSG_FORMAT.format(elem_type))


def collate_jax_array_fn(batch: Sequence[jax.Array]):
    return np.stack(batch)


def collate_numpy_scalar_fn(batch):
    return np.asarray(batch)


def collate_int_fn(batch):
    return np.asarray(batch)


def collate_numpy_array_fn(batch: Sequence[np.ndarray]):
    return np.stack(batch)


_default_collator: Optional[Collator] = None


def get_default_collator() -> Collator:
    """Get or create the default collator"""
    global _default_collator  # pylint: disable=global-statement
    if _default_collator is not None:
        return _default_collator

    collator = Collator()
    collator.register(jax.Array, collate_jax_array_fn)

    with contextlib.suppress(ImportError):
        import torch
        import torch.utils.data._utils

        collator.register(
            torch.Tensor,
            torch.utils.data._utils.collate.collate_tensor_fn,  # pylint: disable=protected-access
        )

    collator.register(np.ndarray, collate_numpy_array_fn)

    # Use the same hierarchy as numpy: https://numpy.org/doc/stable/reference/arrays.scalars.html
    collator.register((np.bool_, np.number, np.object_), collate_numpy_scalar_fn)

    collator.register(int, collate_int_fn)

    _default_collator = collator  # Cache the collator
    return collator


def default_collate(batch) -> Any:
    return get_default_collator().collate(batch)
