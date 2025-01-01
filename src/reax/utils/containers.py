from collections.abc import Hashable, Mapping
from typing import Iterable, Optional, TypeVar, Union

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class BaseRegistry(Mapping[K, V]):
    """Registry: a homogenous mapping where the keys are all the same type and the values are
    (a possibly) different type
    """

    def __init__(self, init: dict[K, V] = None):
        self._registry = {}
        if init:
            self.register_many(init)

    def __len__(self) -> int:
        """Len function."""
        return len(self._registry)

    def __getitem__(self, key: K) -> V:
        """Getitem function."""
        return self._registry[key]

    def __iter__(self):
        """Iter function."""
        return iter(self._registry)

    def items(self) -> Iterable[tuple[K, V]]:
        """Items function."""
        return self._registry.items()

    def register(self, key: K, obj: V):
        """Register function."""
        self._registry[key] = obj

    def register_many(self, objects: dict[K, V]):
        """Register many."""
        for vals in objects.items():
            self.register(*vals)

    def unregister(self, key: K) -> V:
        """Unregister function."""
        return self._registry.pop(key)


class Registry(BaseRegistry[str, V]):
    """Simple registry of objects with unique names."""

    def find(self, starts_with: str) -> Iterable[tuple[str, V]]:
        """Find function."""
        for name, obj in self._registry.items():
            if name.startswith(starts_with):
                yield name, obj


class TypeRegistry(BaseRegistry[Union[type, tuple[type, ...]], V]):
    def find(self, obj: type) -> Optional[V]:
        """Find function."""
        obj_type = type(obj)
        # First, try to match the type directly in this registry
        if obj_type in self:
            return self[obj_type]

        for collate_type in self:
            if isinstance(obj, collate_type):
                return self[collate_type]

        return None
