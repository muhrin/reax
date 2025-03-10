from typing import TYPE_CHECKING, Any, Optional, Union
import weakref

import beartype
import jaxtyping as jt
from typing_extensions import override

from . import common, stages

if TYPE_CHECKING:
    import reax

__all__ = ("Predict",)


class Predict(stages.EpochStage):
    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        module: "reax.Module",
        strategy: "reax.Strategy",
        *,
        dataloader: "Optional[reax.DataLoader]" = None,
        datamodule: "Optional[reax.DataModule]" = None,
        fast_dev_run: Union[bool, int] = False,
        limit_batches: Optional[int] = None,
        keep_predictions=True,
        parent: Optional["reax.Stage"] = None,
    ):
        """Init function."""
        if dataloader is None:
            datamanager = common.get_datasource(datamodule, module)
            dataloader = datamanager.get_loader_proxy("predict_dataloader")
        else:
            datamanager = None

        super().__init__(
            "predict",
            module,
            dataloader,
            strategy,
            None,
            fast_dev_run=fast_dev_run,
            limit_batches=limit_batches,
            parent=parent,
            datamanager=datamanager,
        )
        self._keep_predictions = keep_predictions
        self._all_outputs = []

    @property
    def all_outputs(self) -> list[Any]:
        return self._all_outputs

    @override
    def _on_starting(self):
        super()._on_starting()
        self._module.on_predict_start(weakref.proxy(self))

    @override
    def _on_epoch_start(self):
        super()._on_epoch_start()
        self._module.on_predict_epoch_start(weakref.proxy(self))

    @override
    def _on_iteration_starting(self):
        super()._on_iteration_starting()
        self._module.on_predict_batch_start(self, self.batch, self.batch_idx)

    @override
    def _step(self) -> "reax.stages.MetricResults":
        """Step function."""
        return self._module.predict_step(self.batch, self._iter)

    @override
    def _on_iteration_finishing(self, outputs: Any, /):
        """On iteration finishing."""
        super()._on_iteration_finishing(outputs)
        if self._keep_predictions:
            self._all_outputs.append(outputs)
        self._module.on_predict_batch_end(self, outputs, self.batch, self.batch_idx)

    @override
    def _on_epoch_end(self) -> None:
        super()._on_epoch_end()
        self._module.on_predict_epoch_end(weakref.proxy(self))

    @override
    def _on_stopped(self) -> None:
        super()._on_stopped()
        self._module.on_predict_end(weakref.proxy(self))
