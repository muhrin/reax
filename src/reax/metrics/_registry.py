from typing import Union

from . import _metric as metric_
from . import collections
from ..utils import containers

__all__ = ("Registry", "get_registry", "set_registry", "build_collection")


class Registry(containers.Registry[metric_.Metric]):
    def register(self, key: str, obj: Union[metric_.Metric, type[metric_.Metric]]):
        """Register function."""
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
    """Get registry."""
    global _registry  # pylint: disable=global-variable-not-assigned
    return _registry


def set_registry(registry: Registry) -> None:
    """Set registry."""
    global _registry  # pylint: disable=global-statement
    _registry = registry


def get(name: str) -> metric_.Metric:
    """Convenience method to get a metric using its name."""
    return get_registry()[name]


def build_collection(items: Union[str, dict, list]) -> collections.MetricCollection:
    return collections.MetricCollection(_get_metrics(items))


def _get_metrics(
    items: Union[metric_.Metric, str, dict, list]
) -> Union[metric_.Metric, list[metric_.Metric], dict[str, metric_.Metric]]:
    reg = get_registry()

    if isinstance(items, metric_.Metric):
        return items

    if isinstance(items, str):
        return reg[items]

    if isinstance(items, dict):
        return {name: _get_metrics(value) for name, value in items.items()}

    if isinstance(items, list):
        return [_get_metrics(entry) for entry in items]

    raise TypeError(f"Unknown metrics type: {type(items).__name__}")
