"""Test loop."""

import logging
from typing import TYPE_CHECKING, Optional, Union
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
        rng: "Optional[reax.Generator]",
        *,
        dataloader: "Optional[reax.DataLoader]" = None,
        datamodule: "Optional[reax.DataModule]" = None,
        fast_dev_run: Union[bool, int] = False,
        limit_batches: Optional[Union[int, float]] = None,
        parent: Optional["reax.Stage"] = None,
    ):
        """Init function."""
        if dataloader is None:
            datamanager = common.get_datasource(datamodule, module)
            dataloader = datamanager.get_loader_proxy("test_dataloader")
        else:
            datamanager = None

        super().__init__(
            "test",
            module,
            dataloader,
            strategy,
            rng,
            fast_dev_run=fast_dev_run,
            limit_batches=limit_batches,
            parent=parent,
            datamanager=datamanager,
        )

    @override
    def _on_starting(self):
        super()._on_starting()
        self._module.on_test_start(weakref.proxy(self))

    @override
    def _on_epoch_start(self):
        super()._on_epoch_start()
        self._module.on_test_epoch_start(weakref.proxy(self))

    @override
    def _step(self) -> "reax.stages.MetricResults":
        return self._module.test_step(self.batch, self._iter)

    @override
    def _on_epoch_end(self):
        super()._on_epoch_end()
        self._module.on_test_epoch_end(weakref.proxy(self))

    @override
    def _on_stopping(self) -> None:
        super()._on_stopping()
        self._module.on_test_end(weakref.proxy(self))
