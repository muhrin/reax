from typing import TYPE_CHECKING, Any
import weakref

import beartype
import jaxtyping as jt
from typing_extensions import override

from . import stages

if TYPE_CHECKING:
    import reax


__all__ = ("Validate",)


class Validate(stages.EpochStage):
    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        module: "reax.Module",
        datamanager: "reax.data.DataSourceManager",
        engine: "reax.Engine",
        *,
        fast_dev_run: bool | int = False,
        limit_batches: int | float | None = None,
        name: str = "validate",
        enable_checkpointing: bool = True,
    ):
        """Init function."""
        if getattr(module, "validation_step") is None:
            raise RuntimeError(
                f"Cannot perform validation as the module '{type(module).__name__}' does not "
                f"define validation_step()"
            )

        super().__init__(
            name,
            module,
            datamanager,
            engine,
            rngs=None,
            dataloader_name="val",
            fast_dev_run=fast_dev_run,
            limit_batches=limit_batches,
            enable_checkpointing=enable_checkpointing,
        )

        # Params
        self._mod: "reax.Module" = module

    @property
    def epoch(self) -> int:
        """Get the current epoch."""
        return self._run_count if self.parent is None else self.parent.epoch

    @override
    def _on_starting(self):
        super()._on_starting()
        self._mod.on_validation_start(weakref.proxy(self))

    @override
    def _on_epoch_start(self):
        super()._on_epoch_start()
        self._mod.on_validation_epoch_start(weakref.proxy(self))

    @override
    def _on_iteration_starting(self):
        super()._on_iteration_starting()
        self._mod.on_validation_batch_start(self, self.batch, self.batch_idx)

    @override
    def _step(self) -> "reax.stages.MetricResults":
        return self._mod.validation_step(self.batch, self._iter)

    @override
    def _on_iteration_finishing(self, outputs: Any, /):
        """On iteration finishing."""
        super()._on_iteration_finishing(outputs)
        self._mod.on_validation_batch_end(self, outputs, self.batch, self.batch_idx)

    @override
    def _on_epoch_end(self) -> None:
        super()._on_epoch_end()
        self._mod.on_validation_epoch_end(weakref.proxy(self))

    @override
    def _on_stopping(self) -> None:
        super()._on_stopping()
        self._mod.on_validation_end(weakref.proxy(self))
