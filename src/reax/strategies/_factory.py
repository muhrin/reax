from typing import TYPE_CHECKING, Optional

from . import _jax, _single_device, _utils

if TYPE_CHECKING:
    import reax

__all__ = ("create",)


def create(name: str, platform: Optional[str], **kwargs) -> "reax.Strategy":
    if name == "auto":
        if "devices" not in kwargs or kwargs["devices"] == "auto":
            num_devices = _utils.probe_local_device_count(platform)
        else:
            num_devices = kwargs["devices"]

        if num_devices > 1:
            name = "ddp"
        else:
            name = "single"

    if name == "single":
        return _single_device.SingleDevice(platform)

    if name == "ddp":
        return _jax.JaxDdpStrategy(platform, **kwargs)

    raise ValueError(f"Unknown strategy: {name}")
