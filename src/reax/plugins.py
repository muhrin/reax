import logging

import stevedore
import stevedore.extension

logger = logging.getLogger(__name__)


def load_failed(_manager, entrypoint, exception):
    logger.warning("Error loading REAX plugin from entrypoing '%s':\n%s", entrypoint, exception)


def get_batch_sizers() -> list:
    """Get all mincepy types and type helper instances registered as extensions"""
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
            logger.exception("Failed to get types plugin from %s", extension.name)

    try:
        mgr.map(get_type)
    except stevedore.extension.NoMatches:
        pass

    return sizers
