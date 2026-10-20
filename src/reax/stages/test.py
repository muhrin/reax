"""Test loop."""

import logging
from typing import TYPE_CHECKING, Any
import weakref

import beartype
import jaxtyping as jt
from typing_extensions import override

from . import _loops, stages

if TYPE_CHECKING:
    import reax

__all__ = ("Test", "TestEpoch")

_LOGGER = logging.getLogger(__name__)


class Test(stages.ModuleStage):
    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        module: "reax.Module",
        datamanager: "reax.data.DataSourceManager",
        engine: "reax.Engine",
        *,
        fast_dev_run: bool | int = False,
        limit_batches: int | float | None = None,
    ):
        """Init function."""
        super().__init__(
            "test",
            module,
            datamanager,
            engine,
            enable_checkpointing=False,
            fast_dev_run=fast_dev_run,
        )
        # Params
        self._loop = TestEpoch(self, fast_dev_run=fast_dev_run, limit_batches=limit_batches)

    @property
    def batch_idx(self) -> int:
        return self._loop.batch_idx

    @property
    def dataloader(self) -> "reax.DataLoader":
        return self._loop.dataloader

    @property
    def dataloaders(self) -> "reax.DataLoader":
        return self._loop.dataloaders

    @property
    def max_batches(self) -> int | float | None:
        return self._loop.max_batches

    @property
    def limit_batches(self) -> int | float | None:
        return self._loop.limit_batches

    @property
    def num_batches(self) -> int | float | None:
        return self._loop.num_batches

    @override
    def _on_started(self):
        super()._on_started()
        self._module.on_test_start(weakref.proxy(self))

    @override
    def _run(self) -> "reax.stages.MetricResults":
        return self.run_loop(self._loop)

    @override
    def _on_stopping(self) -> None:
        super()._on_stopping()
        self._module.on_test_end(weakref.proxy(self))


class TestEpoch(_loops.LoopOverBatches[Test]):
    def __init__(
        self,
        stage: "Test",
        *,
        fast_dev_run: bool | int = False,
        limit_batches: int | float | None = None,
    ):
        super().__init__("test", stage, fast_dev_run=fast_dev_run, limit_batches=limit_batches)

    @override
    def _on_starting(self):
        super()._on_starting()
        self._stage.configure_module(self._next_batch)

    @override
    def _on_started(self):
        super()._on_started()
        self._stage.module.on_test_epoch_start(weakref.proxy(self._stage), weakref.proxy(self))

    @override
    def _on_iteration_starting(self):
        super()._on_iteration_starting()
        self._stage.module.on_test_batch_start(
            weakref.proxy(self._stage), weakref.proxy(self), self.batch
        )

    @override
    def _step(self) -> Any:
        return self._stage.module.test_step(self.batch, self._iter)

    @override
    def _on_iteration_finishing(self, outputs: Any, /):
        """On iteration finishing."""
        super()._on_iteration_finishing(outputs)
        self._stage.module.on_test_batch_end(
            weakref.proxy(self._stage), weakref.proxy(self), outputs
        )

    @override
    def _on_epoch_end(self) -> None:
        self._stage.module.on_test_epoch_end(weakref.proxy(self._stage), weakref.proxy(self))
