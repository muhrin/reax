from typing import TYPE_CHECKING, Any
import weakref

import beartype
import jax
import jaxtyping as jt
from typing_extensions import override

from . import _loops, stages

if TYPE_CHECKING:
    import reax


__all__ = "Predict", "PredictionEpoch"


class Predict(stages.ModuleStage):
    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        module: "reax.Module",
        datamanager: "reax.data.DataSourceManager",
        engine: "reax.Engine",
        *,
        fast_dev_run: bool | int = False,
        limit_batches: int | float | None = None,
        keep_predictions=True,
    ):
        """Init function."""
        super().__init__("predict", module, datamanager, engine, fast_dev_run=fast_dev_run)

        # Params
        self._loop = PredictionEpoch(
            self,
            fast_dev_run=fast_dev_run,
            limit_batches=limit_batches,
            keep_predictions=keep_predictions,
        )

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
    def predictions(self) -> list[Any] | list[list[Any]]:
        return self._loop.all_outputs

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
        self._module.on_predict_start(weakref.proxy(self))

    @override
    def run(self) -> Any:
        return super().run()

    @override
    def _run(self) -> Any:
        self.run_loop(self._loop)

    @override
    def log(self, *args, **kwargs):
        raise RuntimeError(
            "Log is not supported by the predict stage, use `predict_step` to compute the "
            "predictions"
        )

    @override
    def _on_stopping(self) -> None:
        super()._on_stopping()
        self._module.on_predict_end(weakref.proxy(self))


class PredictionEpoch(_loops.LoopOverBatches[Predict]):
    def __init__(
        self,
        stage: "reax.stages.Predict",
        *,
        fast_dev_run: bool | int = False,
        limit_batches: int | float | None = None,
        keep_predictions=True,
    ):
        super().__init__("predict", stage, fast_dev_run=fast_dev_run, limit_batches=limit_batches)

        self._keep_predictions = keep_predictions
        self._all_outputs: list[Any] | list[list[Any]] = []

    @property
    def all_outputs(self) -> list[Any] | list[list[Any]]:
        return self._all_outputs

    @override
    def _on_starting(self):
        super()._on_starting()
        self._all_outputs = []
        self._stage.configure_module(self._next_batch)

    @override
    def _on_started(self):
        super()._on_started()
        self._stage.module.on_predict_epoch_start(weakref.proxy(self._stage))

    @override
    def _on_iteration_starting(self):
        super()._on_iteration_starting()
        self._stage.module.on_predict_batch_start(
            weakref.proxy(self._stage), self.batch, self.batch_idx
        )

    @override
    def _step(self) -> Any:
        return self._stage.module.predict_step(self.batch, self._iter)

    @override
    def _on_iteration_finishing(self, outputs: Any, /):
        """On iteration finishing."""
        super()._on_iteration_finishing(outputs)
        if self._keep_predictions:
            cpu = jax.devices("cpu")[0]
            self._all_outputs.append(jax.device_put(outputs, cpu))
        self._stage.module.on_predict_batch_end(
            weakref.proxy(self._stage), outputs, self.batch, self.batch_idx
        )

    @override
    def _on_epoch_end(self) -> None:
        self._stage.module.on_predict_epoch_end(weakref.proxy(self._stage))
