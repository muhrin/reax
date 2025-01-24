from typing import TYPE_CHECKING, Optional, Union
import weakref

from typing_extensions import override

from . import stages
from .. import keys

if TYPE_CHECKING:
    import reax


__all__ = ("Validate",)


class Validate(stages.EpochStage):
    def __init__(
        self,
        module: "reax.Module",
        dataloader: "reax.DataLoader",
        strategy: "reax.Strategy",
        *,
        max_batches: Union[int, float] = keys.NO_LIMIT,
        parent: Optional["reax.Stage"] = None,
    ):
        """Init function."""
        super().__init__(
            "validate", module, dataloader, strategy, max_batches=max_batches, parent=parent
        )

    @override
    def _on_starting(self):
        """On starting."""
        super()._on_starting()
        self._module.on_validation_start(weakref.proxy(self))

    @override
    def _on_started(self):
        """On started."""
        super()._on_started()
        self._module.on_validation_epoch_start(weakref.proxy(self))

    @override
    def _step(self) -> "reax.stages.MetricResults":
        """Step function."""
        return self._module.validation_step(self.batch, self._iter)

    @override
    def _on_stopping(self) -> None:
        """On stopping."""
        self._module.on_validation_epoch_end(weakref.proxy(self))
        super()._on_stopping()

    @override
    def _on_stopped(self):
        """On stopped."""
        super()._on_stopped()
        self._module.on_validation_end(weakref.proxy(self))
