from typing import TYPE_CHECKING, Any, Optional, TypeVar, Union
import weakref

import beartype
import jaxtyping as jt
import optax
from typing_extensions import override

from . import common, stages
from .. import exceptions
from .. import optimizers as optimizers_
from ..lightning import rank_zero

if TYPE_CHECKING:
    import reax


__all__ = ("Train",)


_T_co = TypeVar("_T_co", covariant=True)


class Train(stages.EpochStage):
    """One training epoch."""

    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(
        self,
        module: "reax.Module",
        strategy: "reax.Strategy",
        optimizers: "list[reax.Optimizer]",
        *,
        dataloader: "Optional[reax.DataLoader]" = None,
        datamodule: "Optional[reax.DataModule]" = None,
        min_updates: int = 0,
        max_updates: Optional[Union[int, float]] = None,
        max_batches: Optional[Union[int, float]] = None,
        accumulate_grad_batches: int = 1,
        parent: Optional["reax.Stage"] = None,
        stopper: Optional[common.Stopper] = None,
        datamanager: "Optional[common.DataSourceManager]" = None,
    ):
        if datamanager is None:
            if dataloader is None:
                datamanager = common.get_datasource(datamodule, module)
                dataloader = datamanager.get_loader_proxy("val_dataloader")
            else:
                datamanager = None
        else:
            assert dataloader is not None

        super().__init__(
            "fit",
            module,
            dataloader,
            strategy,
            max_batches=max_batches,
            parent=parent,
            datamanager=datamanager,
        )
        # Params
        self._min_updates = min_updates
        self._max_updates = max_updates
        self._accumulate_grad_batches = accumulate_grad_batches

        # State
        self._optimizers = optimizers
        self._stopper = stopper
        self._stopper.add_condition(lambda: self.updates >= self._min_updates)

    @property
    def updates(self) -> int:
        """Get the number of gradient updates that have been applied."""
        return sum(opt.update_count for opt in self._optimizers)

    @property
    def optimizers(self) -> Optional[list["reax.Optimizer"]]:
        """Optimizers function."""
        return self._optimizers

    @override
    def run(self) -> list["reax.Optimizer"]:
        """Run function."""
        super().run()
        return self._optimizers

    @override
    def _on_starting(self):
        """On starting."""
        super()._on_starting()

        if not self._optimizers:
            opts = self._module.configure_optimizers()
            if opts is None:
                rank_zero.rank_zero_warn(
                    "`LightningModule.configure_optimizers` returned `None`, this fit will run "
                    "with no optimizer"
                )
                opt = optimizers_.mock_optimizer
                opts = opt, opt.init(self._module.parameters())

            if not isinstance(opts, list):
                opts = [opts]

            optimizers: list[optimizers_.Optimizer] = []
            for opt, state in opts:
                # Move optimizer parameters to device
                state = self._strategy.to_device(state)
                if self._accumulate_grad_batches > 1:
                    stepper = optax.MultiSteps(opt, every_k_schedule=self._accumulate_grad_batches)
                    state = stepper.init(self._module.parameters())
                    opt = stepper.gradient_transformation()

                optimizers.append(optimizers_.Optimizer(opt, state))

            # Create the `Optimizer` instances
            self._optimizers = optimizers

        self._module.on_train_start(weakref.proxy(self))

    @override
    def _on_started(self):
        """On started."""
        super()._on_started()
        self._module.on_train_epoch_start(weakref.proxy(self))

    @override
    def _step(self) -> Any:
        """Step function."""
        if self._module.parameters() is None:
            raise exceptions.MisconfigurationException(
                "Module does not have any parameters set, this should have been done by now."
            )

        res = self._module.training_step(self.batch, self._iter)
        if self._module.automatic_optimization:
            if isinstance(res, dict):
                grad = res["grad"]
            else:
                _loss, grad = res
            opt = self._optimizers[0]
            self._module.on_before_optimizer_step(opt, grad)
            opt = opt.update_module(self._module, grad)
            self._optimizers = [opt]

        if (self._min_updates is None or self.updates >= self._min_updates) and (
            self._max_updates is not None and self.updates >= self._max_updates
        ):
            self.stop("Max updates reached")

        return res

    @override
    def _on_stopping(self) -> None:
        """On stopping."""
        self._module.on_train_epoch_end(weakref.proxy(self))
        super()._on_stopping()

    @override
    def _done(self) -> bool:
        """Done function."""
        if self.max_batches is not None and self.batch_idx >= self.max_batches:
            rank_zero.rank_zero_debug(
                f"`{type(self).__name__}` done: max_batches.{self.max_batches!r}` reached."
            )
            return True

        if self._max_updates is not None and self.updates >= self._max_updates:
            rank_zero.rank_zero_debug(
                f"`{type(self).__name__}` done: `max_updates={self._max_updates!r}` reached."
            )
            return True

        if self._stopper.stop_requested:
            if self._stopper.can_stop:
                rank_zero.rank_zero_debug(
                    f"`{type(self).__name__}` stopped: `{type(self).__name__}.should_stop` was set."
                )
            else:
                self._warning_cache.info(
                    f"Trainer was signaled to stop but the required "
                    f"`min_epochs={self.parent.min_iters!r}` or"
                    f" `min_steps={self._min_updates!r}` has not been met. "
                    f"Training will continue..."
                )
            return self._stopper.can_stop

        return False
