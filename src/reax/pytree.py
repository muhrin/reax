"""Lightweight pytree base class used for reax metrics, optimizers and friends.

Subclasses of :class:`PyTree` are registered with JAX as pytrees and turned into dataclasses
automatically:

* Annotated fields describe the state that is stored.  Fields declared with
  ``field(static=True)`` become part of the tree structure rather than leaves, so they are
  hashable Python data under :func:`jax.jit`.
* If neither the class nor any of its parents writes an ``__init__``, one is generated from the
  fields.  Otherwise the hand-written ``__init__`` is used (and inherited), so its parameters can
  be anything and only what it assigns to ``self`` is stored.
* Instances are immutable once ``__init__`` returns, use :meth:`PyTree.replace` to get an updated
  copy.  Neither ``replace()`` nor JAX's unflattening call ``__init__`` again.
"""

from collections.abc import Callable
import dataclasses
from typing import Any, TypeVar

import jax
import numpy as np
from typing_extensions import dataclass_transform

from ._vendor.simple_pytree import Pytree as _Pytree

__all__ = ("PyTree", "field", "is_array", "partition", "combine")

T = TypeVar("T")


def field(*, static: bool = False, **kwargs) -> Any:
    """Declare a field of a :class:`PyTree`, taking the same arguments as
    :func:`dataclasses.field`.

    Args:
        static: if ``True`` the value is part of the tree structure rather than a leaf, i.e. it is
            treated as a compile-time constant by :func:`jax.jit` and must be hashable.
    """
    if static:
        kwargs["metadata"] = {**kwargs.get("metadata", {}), "static": True}
    return dataclasses.field(**kwargs)


@dataclass_transform(field_specifiers=(field, dataclasses.field))
class PyTree(_Pytree):
    """Base class for immutable objects that are also JAX pytrees."""

    _pytree__custom_init: bool = False

    def __init_subclass__(cls, **kwargs):
        # Let simple-pytree look at the class body first, it finds static fields from the
        # `dataclasses.Field` objects that the dataclass decorator then removes
        super().__init_subclass__(**kwargs)

        # Only generate an `__init__` if no one has written one, otherwise we would silently
        # replace a parent's hand-written `__init__` (the same rule as equinox)
        custom_init = "__init__" in cls.__dict__ or any(
            getattr(base, "_pytree__custom_init", False) for base in cls.__mro__[1:]
        )
        cls._pytree__custom_init = custom_init
        dataclasses.dataclass(cls, init=not custom_init, eq=False)


def is_array(element: Any) -> bool:
    """Returns ``True`` if the element is a JAX or NumPy array."""
    return isinstance(element, (np.ndarray, np.generic, jax.Array))


def partition(tree: T, filter_fn: Callable[[Any], bool]) -> tuple[T, T]:
    """Split a pytree into two with the same structure, the first contains the leaves that pass
    the filter and the second the rest, with ``None`` in place of the missing leaves.
    """
    passed = jax.tree.map(lambda leaf: leaf if filter_fn(leaf) else None, tree)
    rest = jax.tree.map(lambda leaf: None if filter_fn(leaf) else leaf, tree)
    return passed, rest


def combine(*trees: T) -> T:
    """The inverse of :func:`partition`, for each leaf this takes the first that isn't ``None``."""

    def _combine(*leaves):
        for leaf in leaves:
            if leaf is not None:
                return leaf
        return None

    return jax.tree.map(_combine, *trees, is_leaf=lambda leaf: leaf is None)
