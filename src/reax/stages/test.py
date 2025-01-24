"""Test loop."""

import logging
from typing import TYPE_CHECKING, Optional, Union
import weakref

import beartype
import jaxtyping as jt
from typing_extensions import override

from . import stages
from .. import keys

if TYPE_CHECKING:
    import reax

__all__ = ("Test",)

_LOGGER = logging.getLogger(__name__)


class Test(stages.EpochStage):
    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        module: "reax.Module",
        dataloader,
        strategy: "reax.Strategy",
        *,
        max_batches: Union[int, float] = keys.NO_LIMIT,
        parent: Optional["reax.Stage"] = None,
    ):
        """Init function."""
        super().__init__(
            "test", module, dataloader, strategy, max_batches=max_batches, parent=parent
        )

    @override
    def _on_started(self):
        """On started."""
        super()._on_started()
        self._module.on_test_epoch_start(weakref.proxy(self))

    @override
    def _on_stopped(self):
        """On stopped."""
        super()._on_stopped()
        self._module.on_test_epoch_end(weakref.proxy(self))

    @override
    def _step(self) -> "reax.stages.MetricResults":
        """Step function."""
        return self._module.predict_step(self.batch, self._iter)
