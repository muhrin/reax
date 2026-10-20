import logging
from typing import TYPE_CHECKING

from typing_extensions import override

from .. import plugins
from ..utils import containers
from . import _metric as metric_
from . import collections

if TYPE_CHECKING:
    import reax

__all__ = ("Registry", "get_registry", "set_registry", "build_collection")


_LOGGER = logging.getLogger(__name__)


class Registry(containers.Registry[metric_.Metric]):
    def __getitem__(self, key: str) -> metric_.Metric:
        try:
            return self._registry[key]
        except KeyError as exc:
            raise KeyError(f"Metric not found: {key}") from exc

    @override
    def register(  # pylint: disable=arguments-renamed
        self, key: str, obj: "metric_.Metric | type[metric_.Metric] | reax.types.MetricType"
    ):
        """Register a metric, either as an instance or a type that can be instantiated without
        arguments.

        Raises:
            ValueError: If the object is not a :class:`~reax.Metric` (or a type thereof that can be
                instantiated without arguments).
        """
        if isinstance(obj, type) and issubclass(obj, metric_.Metric):
            try:
                obj = obj()
            except TypeError:
                # Needs constructor arguments, this will be reported below
                pass

        if not isinstance(obj, metric_.Metric):
            raise ValueError(
                f"metric must be a subclass or instance of `reax.Metric`, got {type(obj).__name__}"
            )

        self._registry[key] = obj.empty()


# Helpers to make it easy to choose a metric using a string
_registry: Registry | None = None  # pylint: disable=invalid-name


def get_registry() -> Registry:
    """Get registry."""
    global _registry  # pylint: disable=global-statement,global-variable-not-assigned # noqa: F824
    if _registry is None:
        _registry = Registry()
        for name, metric in plugins.get_metrics().items():
            try:
                _registry.register(name, metric)
            except ValueError as exc:
                _LOGGER.warning("Failed to load metric %s: %s", name, exc)

    return _registry


def set_registry(registry: Registry) -> None:
    """Set registry."""
    global _registry  # pylint: disable=global-statement
    _registry = registry


def get(name: str) -> metric_.Metric:
    """Convenience method to get a metric using its name."""
    return get_registry()[name]


def build_collection(items: str | dict | list) -> collections.MetricCollection:
    return collections.MetricCollection(_get_metrics(items))


def _get_metrics(
    items: metric_.Metric | str | dict | list,
) -> metric_.Metric | list[metric_.Metric] | dict[str, metric_.Metric]:
    if isinstance(items, dict):
        return {name: _get_metric(value) for name, value in items.items()}

    if isinstance(items, list):
        return [_get_metric(entry) for entry in items]

    return _get_metric(items)


def _get_metric(item: metric_.Metric | str) -> metric_.Metric:
    if isinstance(item, metric_.Metric):
        return item

    if isinstance(item, str):
        return get_registry()[item]

    raise TypeError(f"Unknown metrics type: {type(item).__name__}")
