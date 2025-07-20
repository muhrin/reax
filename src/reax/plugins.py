import logging
from typing import TYPE_CHECKING

import stevedore
import stevedore.extension

if TYPE_CHECKING:
    import reax

_LOGGER = logging.getLogger(__name__)


def load_failed(_manager, entrypoint, exception):
    _LOGGER.warning("Error loading REAX plugin from entrypoint '%s':\n%s", entrypoint, exception)


def get_batch_sizers() -> list:
    """Get all REAX types and type helper instances registered as extensions"""
    mgr = stevedore.extension.ExtensionManager(
        namespace="reax.plugins.batch_sizers",
        invoke_on_load=False,
        on_load_failure_callback=load_failed,
    )

    sizers: list = []

    def get_type(extension: stevedore.extension.Extension):
        try:
            sizers.extend(extension.plugin())
        except Exception:  # pylint: disable=broad-except
            _LOGGER.exception("Failed to get types plugin from %s", extension.name)

    try:
        mgr.map(get_type)
    except stevedore.extension.NoMatches:
        pass

    return sizers


def get_metrics() -> "dict[str, reax.Metric]":
    """Get registered metrics from plugins"""
    mgr = stevedore.extension.ExtensionManager(
        namespace="reax.plugins.metrics",
        invoke_on_load=False,
        on_load_failure_callback=load_failed,
    )

    metrics: "dict[str, reax.Metric]" = {}

    def get_type(extension: stevedore.extension.Extension):
        try:
            metrics[extension.name] = extension.plugin
        except Exception:  # pylint: disable=broad-except
            _LOGGER.exception("Failed to get types plugin from %s", extension.name)

    try:
        mgr.map(get_type)
    except stevedore.extension.NoMatches:
        pass

    return metrics
