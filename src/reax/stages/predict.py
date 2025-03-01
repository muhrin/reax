from typing import TYPE_CHECKING, Any, Optional

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
        max_batches: Optional[int] = None,
        keep_predictions=True,
        parent: Optional["reax.Stage"] = None,
    ):
        """Init function."""
        if dataloader is None:
            datamanager = common.get_datasource(datamodule, module)
            dataloader = datamanager.get_loader_proxy("val_dataloader")
        else:
            datamanager = None

        super().__init__(
            "predict",
            module,
            dataloader,
            strategy,
            None,
            max_batches=max_batches,
            parent=parent,
            datamanager=datamanager,
        )
        self._keep_predictions = keep_predictions
        self._all_outputs = []

    @property
    def all_outputs(self) -> list[Any]:
        return self._all_outputs

    @override
    def _step(self) -> "reax.stages.MetricResults":
        """Step function."""
        return self._module.predict_step(self.batch, self._iter)

    @override
    def _on_iteration_finishing(self, outputs: Any, /):
        """On iteration finishing."""
        if self._keep_predictions:
            self._all_outputs.append(outputs)
