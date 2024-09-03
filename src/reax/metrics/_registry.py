from typing import Union

from . import metric as metric_
from ..utils import containers

__all__ = ("Registry", "get_registry", "set_registry", "get")


class Registry(containers.Registry[metric_.Metric]):
    def register(self, key: str, obj: Union[metric_.Metric, type[metric_.Metric]]):
        try:
            if issubclass(obj, metric_.Metric):
                # Try to instantiate it
                obj = obj()
        except TypeError:
            pass

        if not isinstance(obj, metric_.Metric):
            raise ValueError(
                f"metric must be a subclass or instance of `reax.Metric`, got {type(obj).__name__}"
            )

        self._registry[key] = obj.empty()


# Helpers to make it easy to choose a metric using a string
_registry = Registry()


def get_registry() -> Registry:
    global _registry  # pylint: disable=global-variable-not-assigned
    return _registry


def set_registry(registry: Registry) -> None:
    global _registry  # pylint: disable=global-statement
    _registry = registry


def get(name: str) -> metric_.Metric:
    """Convenience method to get a metric using its name"""
    return get_registry()[name]
