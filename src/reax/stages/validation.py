from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
import weakref

import beartype
import jaxtyping as jt
from typing_extensions import override

from . import _loops, stages

if TYPE_CHECKING:
    import reax


__all__ = "Validate", "ValidationEpoch"


@runtime_checkable
class ValidationStage(_loops.ModuleBatchHost, Protocol):
    """Structural interface a stage must provide to run a :class:`ValidationEpoch`.

    ``ValidationEpoch`` is generic over this protocol rather than a concrete stage, so
    validation can be driven by custom stages that expose the module-batch interface.
    """


class Validate(stages.ModuleStage):
    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        module: "reax.Module",
        datamanager: "reax.data.DataSourceManager",
        engine: "reax.Engine",
        *,
        name: str = "validate",
        fast_dev_run: bool | int = False,
        limit_batches: int | float | None = None,
        enable_checkpointing: bool = True,
    ):
        """Init function."""
        super().__init__(
            name,
            module,
            datamanager,
            engine,
            enable_checkpointing=enable_checkpointing,
            fast_dev_run=fast_dev_run,
        )

        if module.validation_step is None:
            raise RuntimeError(
                f"Cannot perform validation as the module '{type(module).__name__}' does not "
                f"define validation_step()"
            )

        # Params
        self._loop = ValidationEpoch(self, fast_dev_run=fast_dev_run, limit_batches=limit_batches)

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
        self._module.on_validation_start(weakref.proxy(self))

    @override
    def _run(self) -> "reax.stages.MetricResults":
        return self.run_loop(self._loop)

    @override
    def _on_stopping(self) -> None:
        super()._on_stopping()
        self._module.on_validation_end(weakref.proxy(self))


class ValidationEpoch(_loops.LoopOverBatches[ValidationStage]):
    def __init__(
        self,
        stage: ValidationStage,
        *,
        name: str = "validate",
        fast_dev_run: bool | int = False,
        limit_batches: int | float | None = None,
        dataloader_name: str | None = "val",
    ):
        if stage.module.validation_step is None:
            raise RuntimeError(
                f"Cannot perform validation as the module '{type(stage.module).__name__}' does not "
                f"define validation_step()"
            )
        super().__init__(
            name,
            stage,
            fast_dev_run=fast_dev_run,
            limit_batches=limit_batches,
            dataloader_name=dataloader_name,
        )

    @override
    def _on_starting(self):
        super()._on_starting()
        self._stage.configure_module(self._next_batch)

    @override
    def _on_started(self):
        super()._on_started()
        self._stage.module.on_validation_epoch_start(weakref.proxy(self._stage))

    @override
    def _on_iteration_starting(self):
        super()._on_iteration_starting()
        self._stage.module.on_validation_batch_start(
            weakref.proxy(self._stage), self.batch, self.batch_idx
        )

    @override
    def _step(self) -> Any:
        return self._stage.module.validation_step(self.batch, self._iter)

    @override
    def _on_iteration_finishing(self, outputs: Any, /):
        """On iteration finishing."""
        super()._on_iteration_finishing(outputs)
        self._stage.module.on_validation_batch_end(
            weakref.proxy(self._stage), outputs, self.batch, self.batch_idx
        )

    @override
    def _on_epoch_end(self) -> None:
        self._stage.module.on_validation_epoch_end(weakref.proxy(self._stage))
