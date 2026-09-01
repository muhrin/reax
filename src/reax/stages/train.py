from typing import TYPE_CHECKING, Any, Final, TypeVar
import weakref

import beartype
from flax import nnx
import jaxtyping as jt
import optax
from typing_extensions import override

from reax import exceptions
from reax import optimizers as optimizers_
from reax.lightning import rank_zero

from . import common, stages

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
        datamanager: "reax.data.DataSourceManager",
        engine: "reax.Engine",
        optimizers: "list[reax.Optimizer]",
        *,
        rngs: nnx.Rngs | None = None,
        fast_dev_run: bool | int = False,
        min_updates: int = 0,
        max_updates: int | float | None = None,
        limit_batches: int | float | None = None,
        accumulate_grad_batches: int = 1,
        stopper: common.Stopper | None = None,
    ):
        super().__init__(
            "fit",
            module,
            datamanager,
            engine,
            rngs=rngs,
            dataloader_name="train",
            fast_dev_run=fast_dev_run,
            limit_batches=limit_batches,
            enable_checkpointing=True,
        )
        # Params
        self._mod: "Final[reax.Module]" = module
        self._min_updates: Final[int] = min_updates
        self._max_updates: Final[int | float | None] = max_updates
        self._accumulate_grad_batches: Final[int] = accumulate_grad_batches

        # State
        self._optimizers = optimizers
        self._stopper = stopper if stopper is not None else common.Stopper()
        self._stopper.add_condition(lambda: self.updates >= self._min_updates)

    @property
    def num_training_batches(self) -> int | float | None:
        return self.max_batches

    @property
    def updates(self) -> int:
        """Get the number of gradient updates that have been applied."""
        return sum(opt.update_count for opt in self._optimizers)

    @property
    def optimizers(self) -> "list[reax.Optimizer] | None":
        """Optimizers function."""
        return self._optimizers

    @override
    def run(self) -> "list[reax.Optimizer]":
        """Run function."""
        super().run()
        return self._optimizers

    @override
    def _on_starting(self):
        """On starting."""
        super()._on_starting()

        if not self._optimizers:
            opts = self._mod.configure_optimizers()
            if opts is None:
                rank_zero.rank_zero_warn(
                    "`reax.Module.configure_optimizers` returned `None`, this fit will run "
                    "with no optimizer"
                )
                opt = optimizers_.mock_optimizer
                opts = opt, opt.init(self._mod.parameters())

            if not isinstance(opts, list):
                opts = [opts]

            optimizers: list[optimizers_.Optimizer] = []
            for opt, state in opts:
                # Move optimizer parameters to device
                state = self._engine.to_device(state)
                if self._accumulate_grad_batches > 1:
                    stepper = optax.MultiSteps(opt, every_k_schedule=self._accumulate_grad_batches)
                    state = stepper.init(self._mod.parameters())
                    opt = stepper.gradient_transformation()

                optimizers.append(optimizers_.Optimizer(opt, state))

            # Create the `Optimizer` instances
            self._optimizers = optimizers

        self._mod.on_train_start(weakref.proxy(self))

    @override
    def _on_iteration_starting(self):
        super()._on_iteration_starting()
        self._mod.on_train_batch_start(weakref.proxy(self), self.batch, self.batch_idx)

    @override
    def _on_epoch_start(self):
        """On started."""
        super()._on_epoch_start()
        self._mod.on_train_epoch_start(weakref.proxy(self))

    @override
    def _step(self) -> Any:
        """Step function."""
        if self._mod.parameters() is None:
            raise exceptions.MisconfigurationException(
                "Module does not have any parameters set, this should have been done in "
                ".configure_model()."
            )

        res = self._mod.training_step(self.batch, self._iter)
        if self._mod.automatic_optimization:
            if isinstance(res, dict):
                grad = res["grad"]
                loss = res.get("loss", None)
            else:
                loss, grad = res
            opt = self._optimizers[0]
            self._mod.on_before_optimizer_step(opt, grad)
            opt = opt.update_module(self._mod, grad, value=loss)
            self._optimizers = [opt]

        if (self._min_updates is None or self.updates >= self._min_updates) and (
            self._max_updates is not None and self.updates >= self._max_updates
        ):
            self.stop("Max updates reached")

        return res

    @override
    def _on_iteration_finishing(self, outputs: Any, /):
        super()._on_iteration_finishing(outputs)
        self._mod.on_train_batch_end(weakref.proxy(self), outputs, self.batch, self.batch_idx)

    @override
    def _on_epoch_end(self) -> None:
        super()._on_epoch_end()
        self._mod.on_train_epoch_end(weakref.proxy(self))

    @override
    def _on_stopping(self) -> None:
        super()._on_stopping()
        self._mod.on_train_end(weakref.proxy(self))

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
                lines = ["Trainer was signaled to stop but the required "]
                if self.parent is not None:
                    lines.append(f"`min_epochs={self.parent.min_iters!r}` or")
                lines.append(
                    f" `min_steps={self._min_updates!r}` has not been met. "
                    f"Training will continue..."
                )
                self._warning_cache.info("".join(lines))
            return self._stopper.can_stop

        return False
