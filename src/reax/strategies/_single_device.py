from typing import TYPE_CHECKING, Any, TypeVar

import jax
import jaxtyping as jt
from typing_extensions import override

from . import _strategies

if TYPE_CHECKING:
    import reax

__all__ = ("SingleDevice",)

_OutT = TypeVar("_OutT")


class SingleDevice(_strategies.Strategy):
    """Strategy for a single device e.g. a single GPU."""

    def __init__(self, platform: str):
        self._device = jax.Device = (
            jax.devices()[0] if platform == "auto" else jax.devices(platform)[0]
        )

    @property
    def device(self) -> jax.Device:
        return self._device

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
    def broadcast(self, obj: jt.PyTreeDef, src: int = 0) -> Any:
        """Broadcast function."""
        return obj

    @override
    def all_gather(self, obj: jt.PyTreeDef) -> Any:
        return obj

    @override
    def all_reduce(self, obj: jt.PyTree, reduce_op: str = "mean") -> jt.PyTree:
        """Reduces a tensor from several distributed processes to one aggregated tensor.

        Args:
            obj: the pytree to sync and reduce
            reduce_op: the reduction operation. Defaults to 'mean'/'avg'.
                Can also be a string 'sum' to calculate the sum during reduction.

        Return:
            reduced value
        """
        return obj

    @override
    def barrier(self, name: str | None = None) -> None:
        """Synchronizes all processes which blocks processes until the whole group enters this
        function.

        Args:
            name: an optional name to pass into barrier.
        """
        return None  # Nothing to do in a single device

    @override
    def compute(self, metric: "reax.Metric[_OutT]") -> _OutT:
        return metric.compute()
