from typing import Any

import jax

from . import _strategies

__all__ = ("SingleDevice",)


class SingleDevice(_strategies.Strategy):
    """Strategy for a single device e.g. a single GPU"""

    def __init__(self, device: jax.Device):
        self._device = device

    def to_device(self, value: Any) -> Any:
        return jax.device_put(value, self._device)

    def from_device(self, value: Any) -> Any:
        return jax.device_get(value)
