"""Stage for evaluating dataset statistics"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional, TypeVar, Union

import beartype
import jaxtyping as jt
from lightning_utilities.core import overrides
from typing_extensions import override

from reax import data, metrics, modules

from . import stages

if TYPE_CHECKING:
    import reax


__all__ = ("EvaluateStats",)

OutT = TypeVar("OutT")


class EvaluateStats(stages.EpochStage):
    """
    A stage that can be used to evaluate statistics (in the form of metrics) on a dataset
    """

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        stats: Union["reax.Metric", Sequence["reax.Metric"], dict[str, "reax.Metric"]],
        strategy: Optional["reax.Strategy"],
        rng: "reax.Generator",
        *,
        dataloader: "Optional[reax.DataLoader]" = None,
        datamodule: "Optional[reax.DataModule]" = None,
        dataset_name: str = "train",
        fast_dev_run: Union[bool, int] = False,
        limit_batches: Optional[Union[int, float]] = None,
    ):
        """Init function."""
        super().__init__(
            "stats",
            None,
            strategy,
            rng,
            dataloader=dataloader,
            datamodule=datamodule,
            limit_batches=limit_batches,
            fast_dev_run=fast_dev_run,
        )

        # Params
        self._stats = metrics.MetricCollection(stats)
        self._dataset_name = dataset_name

    @property
    def dataloader(self) -> "Optional[reax.DataLoader]":
        """Dataloader function."""
        if self._dataloader is None:
            if self._datamodule is not None and overrides.is_overridden(
                "val_dataloader", self._datamodule, data.DataModule
            ):
                self._dataloader = self._datamodule.val_dataloader()
            elif self._module is not None and overrides.is_overridden(
                "val_dataloader", self._module, modules.Module
            ):
                self._dataloader = self._module.val_dataloader()

        return self._dataloader

    @override
    def _step(self) -> None:
        """Step function."""
        # Calculate and log all the stats
        for name, stat in self._stats.items():
            if isinstance(self.batch, tuple):
                self.log(name, stat.create(*self.batch), on_step=False, on_epoch=True, logger=True)
            else:
                self.log(name, stat.create(self.batch), on_step=False, on_epoch=True, logger=True)
