# Vendored from simple-pytree 0.2.3 (https://github.com/cgarciae/simple-pytree,
# commit 182178837e4e8b75021c6e622ea249564703dae3), which is no longer developed.
#
# MIT License
#
# Copyright (c) 2023 Cristian Garcia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Changes from upstream:
#   * Dropped the fallback for jax < 0.4.7.
#   * A dataclass field counts as static if its metadata has ``pytree_node=False`` *or*
#     ``static=True`` (the latter is what ``equinox.field(static=True)`` sets).
#   * Static data is stored in the treedef as a sorted tuple of ``(name, value)`` pairs
#     instead of an (unhashable) ``MappingProxyType``.  Static fields that were never set on
#     the instance are skipped rather than raising a ``KeyError``.
#   * ``replace()`` always copies the instance and swaps the values, it never calls
#     ``__init__`` (``dataclasses.replace`` does, which breaks custom ``__init__`` methods).
# pylint: skip-file
from abc import ABCMeta
import copy as copy_
import dataclasses
from functools import partial
import importlib.util
import typing as tp

import jax

__all__ = ("Pytree", "PytreeMeta", "is_static_field")

P = tp.TypeVar("P", bound="Pytree")

_NODE_FIELDS = "_pytree__node_fields"


def is_static_field(field: dataclasses.Field) -> bool:
    """Whether the passed dataclass field is marked as static (i.e. not a pytree node)."""
    return not field.metadata.get("pytree_node", True) or bool(field.metadata.get("static", False))


class PytreeMeta(ABCMeta):
    def __call__(cls: type[P], *args: tp.Any, **kwargs: tp.Any) -> P:
        obj: P = cls.__new__(cls, *args, **kwargs)
        obj.__dict__["_pytree__initializing"] = True
        try:
            obj.__init__(*args, **kwargs)
        finally:
            del obj.__dict__["_pytree__initializing"]

        vars_dict = vars(obj)
        vars_dict[_NODE_FIELDS] = tuple(
            sorted(field for field in vars_dict if field not in cls._pytree__static_fields)
        )
        return obj


class Pytree(metaclass=PytreeMeta):
    _pytree__initializing: bool
    _pytree__class_is_mutable: bool
    _pytree__static_fields: tuple[str, ...]
    _pytree__node_fields: tuple[str, ...]
    _pytree__setter_descriptors: frozenset[str]

    def __init_subclass__(cls, mutable: bool = False, **kwargs):
        super().__init_subclass__(**kwargs)

        # gather class info
        class_vars = vars(cls)
        setter_descriptors = set()
        static_fields = _inherited_static_fields(cls)

        # add special static fields
        static_fields.add(_NODE_FIELDS)

        for field, value in class_vars.items():
            if isinstance(value, dataclasses.Field) and is_static_field(value):
                static_fields.add(field)

            # add setter descriptors
            if hasattr(value, "__set__"):
                setter_descriptors.add(field)

        static_fields = tuple(sorted(static_fields))

        # init class variables
        cls._pytree__initializing = False
        cls._pytree__class_is_mutable = mutable
        cls._pytree__static_fields = static_fields
        cls._pytree__setter_descriptors = frozenset(setter_descriptors)

        jax.tree_util.register_pytree_with_keys(
            cls,
            partial(cls._pytree__flatten, with_key_paths=True),
            cls._pytree__unflatten,  # ty: ignore[invalid-argument-type]
            flatten_func=partial(cls._pytree__flatten, with_key_paths=False),
        )

        # flax serialization support
        if importlib.util.find_spec("flax") is not None:
            from flax import serialization

            serialization.register_serialization_state(
                cls,
                partial(cls._to_flax_state_dict, cls._pytree__static_fields),
                partial(cls._from_flax_state_dict, cls._pytree__static_fields),
            )

    @classmethod
    def _pytree__flatten(
        cls,
        pytree: "Pytree",
        *,
        with_key_paths: bool,
    ) -> tuple[tuple[tp.Any, ...], tuple[tuple[str, tp.Any], ...]]:
        all_vars = vars(pytree).copy()
        static = tuple(
            (name, all_vars.pop(name)) for name in pytree._pytree__static_fields if name in all_vars
        )

        if with_key_paths:
            node_values = tuple(
                (jax.tree_util.GetAttrKey(field), all_vars.pop(field))
                for field in pytree._pytree__node_fields
            )
        else:
            node_values = tuple(all_vars.pop(field) for field in pytree._pytree__node_fields)

        if all_vars:
            raise ValueError(
                f"Unexpected fields in {cls.__name__}: {', '.join(all_vars.keys())}."
                "You cannot add new fields to a Pytree after it has been initialized."
            )

        return node_values, static

    @classmethod
    def _pytree__unflatten(
        cls: type[P],
        static_fields: tuple[tuple[str, tp.Any], ...],
        node_values: tuple[tp.Any, ...],
    ) -> P:
        pytree = object.__new__(cls)
        static = dict(static_fields)
        pytree.__dict__.update(zip(static[_NODE_FIELDS], node_values, strict=True))
        pytree.__dict__.update(static)
        return pytree

    @classmethod
    def _to_flax_state_dict(
        cls, static_field_names: tuple[str, ...], pytree: "Pytree"
    ) -> dict[str, tp.Any]:
        from flax import serialization

        state_dict = {
            name: serialization.to_state_dict(getattr(pytree, name))
            for name in pytree.__dict__
            if name not in static_field_names
        }
        return state_dict

    @classmethod
    def _from_flax_state_dict(
        cls,
        static_field_names: tuple[str, ...],
        pytree: P,
        state: dict[str, tp.Any],
    ) -> P:
        """Restore the state of a data class."""
        from flax import serialization

        state = state.copy()  # copy the state so we can pop the restored fields.
        updates = {}
        for name in pytree.__dict__:
            if name in static_field_names:
                continue
            if name not in state:
                raise ValueError(
                    f"Missing field {name} in state dict while restoring"
                    f" an instance of {type(pytree).__name__},"
                    f" at path {serialization.current_path()}"
                )
            value = getattr(pytree, name)
            value_state = state.pop(name)
            updates[name] = serialization.from_state_dict(value, value_state, name=name)
        if state:
            names = ",".join(state.keys())
            raise ValueError(
                f'Unknown field(s) "{names}" in state dict while'
                f" restoring an instance of {type(pytree).__name__}"
                f" at path {serialization.current_path()}"
            )
        return pytree.replace(**updates)

    def replace(self: P, **kwargs: tp.Any) -> P:
        """Return a copy of this object with the passed fields replaced by new values.

        This never calls ``__init__``, the stored values are swapped directly.
        """
        unknown_keys = set(kwargs) - set(vars(self))
        if unknown_keys:
            raise ValueError(
                f"Trying to replace unknown fields {unknown_keys} for '{type(self).__name__}'"
            )

        pytree = copy_.copy(self)
        pytree.__dict__.update(kwargs)
        return pytree

    if not tp.TYPE_CHECKING:

        def __setattr__(self: P, field: str, value: tp.Any):
            if self._pytree__initializing or field in self._pytree__setter_descriptors:
                pass
            elif not hasattr(self, field) and not self._pytree__initializing:
                raise AttributeError(f"Cannot add new fields to {type(self)} after initialization")
            elif not self._pytree__class_is_mutable:
                raise AttributeError(f"{type(self)} is immutable, trying to update field {field}")

            object.__setattr__(self, field, value)


def _inherited_static_fields(cls: type) -> set[str]:
    static_fields = set()
    for parent_class in cls.mro():
        if parent_class is not cls and parent_class is not Pytree:
            if issubclass(parent_class, Pytree):
                static_fields.update(parent_class._pytree__static_fields)
            elif dataclasses.is_dataclass(parent_class):
                for field in dataclasses.fields(parent_class):
                    if is_static_field(field):
                        static_fields.add(field.name)
    return static_fields
