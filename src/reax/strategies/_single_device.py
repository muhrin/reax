from typing import Any

import jax
from typing_extensions import override

from . import _strategies

__all__ = ("SingleDevice",)


class SingleDevice(_strategies.Strategy):
    """Strategy for a single device e.g. a single GPU."""

    def __init__(self, device: jax.Device):
        self._device = device

    @override
    def to_device(self, value: Any) -> Any:
        """To device."""
        return jax.device_put(value, self._device)

    @override
    def from_device(self, value: Any) -> Any:
        """From device."""
        return jax.device_get(value)

    @property
    @override
    def is_global_zero(self) -> bool:
        """Is global zero."""
        return True

    @override
    def broadcast(self, obj: _strategies.BroadcastT, src: int = 0) -> _strategies.BroadcastT:
        """Broadcast function."""
        return obj
