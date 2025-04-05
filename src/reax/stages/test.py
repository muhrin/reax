"""Test loop."""

import logging
from typing import TYPE_CHECKING, Optional, Union
import weakref

import beartype
import jaxtyping as jt
from typing_extensions import override

from . import stages

if TYPE_CHECKING:
    import reax

__all__ = ("Test",)

_LOGGER = logging.getLogger(__name__)


class Test(stages.EpochStage):
    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        module: "reax.Module",
        datamanager: "reax.data.DataSourceManager",
        strategy: "reax.Strategy",
        rng: "Optional[reax.Generator]",
        *,
        fast_dev_run: Union[bool, int] = False,
        limit_batches: Optional[Union[int, float]] = None,
    ):
        """Init function."""
        super().__init__(
            "test",
            module,
            datamanager,
            strategy,
            rng,
            fast_dev_run=fast_dev_run,
            limit_batches=limit_batches,
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
