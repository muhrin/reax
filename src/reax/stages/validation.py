from typing import TYPE_CHECKING, Optional, Union
import weakref

from typing_extensions import override

from . import common, stages

if TYPE_CHECKING:
    import reax


__all__ = ("Validate",)


class Validate(stages.EpochStage):
    def __init__(
        self,
        module: "reax.Module",
        strategy: "reax.Strategy",
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
            dataloader = datamanager.get_loader_proxy("val_dataloader")
        else:
            datamanager = None

        super().__init__(
            "validate",
            module,
            dataloader,
            strategy,
            None,
            fast_dev_run=fast_dev_run,
            limit_batches=limit_batches,
            parent=parent,
            datamanager=datamanager,
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
    def _step(self) -> "reax.stages.MetricResults":
        return self._module.validation_step(self.batch, self._iter)

    @override
    def _on_epoch_end(self) -> None:
        super()._on_epoch_end()
        self._module.on_validation_epoch_end(weakref.proxy(self))

    @override
    def _on_stopping(self) -> None:
        super()._on_stopping()
        self._module.on_validation_end(weakref.proxy(self))
