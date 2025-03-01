"""Test loop."""

import logging
from typing import TYPE_CHECKING, Optional
import weakref

import beartype
import jaxtyping as jt
from typing_extensions import override

from . import common, stages

if TYPE_CHECKING:
    import reax

__all__ = ("Test",)

_LOGGER = logging.getLogger(__name__)


class Test(stages.EpochStage):
    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        module: "reax.Module",
        strategy: "reax.Strategy",
        *,
        dataloader: "Optional[reax.DataLoader]" = None,
        datamodule: "Optional[reax.DataModule]" = None,
        max_batches: Optional[int] = None,
        parent: Optional["reax.Stage"] = None,
    ):
        """Init function."""
        if dataloader is None:
            datamanager = common.get_datasource(datamodule, module)
            dataloader = datamanager.get_loader_proxy("val_dataloader")
        else:
            datamanager = None

        super().__init__(
            "test",
            module,
            dataloader,
            strategy,
            max_batches=max_batches,
            parent=parent,
            datamanager=datamanager,
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
