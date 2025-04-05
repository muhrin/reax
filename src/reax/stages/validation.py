from typing import TYPE_CHECKING, Any, Optional, Union
import weakref

from typing_extensions import override

from . import stages

if TYPE_CHECKING:
    import reax


__all__ = ("Validate",)


class Validate(stages.EpochStage):
    def __init__(
        self,
        module: "reax.Module",
        datamanager: "reax.data.DataSourceManager",
        strategy: "reax.Strategy",
        *,
        fast_dev_run: Union[bool, int] = False,
        limit_batches: Optional[Union[int, float]] = None,
        name: str = "validate",
        enable_checkpointing: bool = True,
    ):
        """Init function."""
        super().__init__(
            name,
            module,
            datamanager,
            strategy,
            rng=None,
            dataloader_name="val",
            fast_dev_run=fast_dev_run,
            limit_batches=limit_batches,
            enable_checkpointing=enable_checkpointing,
        )

    @override
    def _on_starting(self):
        super()._on_starting()
        self._module.on_validation_start(weakref.proxy(self))

    @override
    def _on_epoch_start(self):
        super()._on_epoch_start()
        self._module.on_validation_epoch_start(weakref.proxy(self))

    @override
    def _on_iteration_starting(self):
        super()._on_iteration_starting()
        self._module.on_validation_batch_start(self, self.batch, self.batch_idx)

    @override
    def _step(self) -> "reax.stages.MetricResults":
        return self._module.validation_step(self.batch, self._iter)

    @override
    def _on_iteration_finishing(self, outputs: Any, /):
        """On iteration finishing."""
        super()._on_iteration_finishing(outputs)
        self._module.on_validation_batch_end(self, outputs, self.batch, self.batch_idx)

    @override
    def _on_epoch_end(self) -> None:
        super()._on_epoch_end()
        self._module.on_validation_epoch_end(weakref.proxy(self))

    @override
    def _on_stopping(self) -> None:
        super()._on_stopping()
        self._module.on_validation_end(weakref.proxy(self))
