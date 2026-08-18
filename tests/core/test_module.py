import jax.numpy as jnp
import numpy as np
import optax.losses
import pytest

import reax
from reax import demos


@pytest.mark.parametrize("reduce_fn", ["mean", "sum"])
def test_module_log(test_trainer, reduce_fn):
    class TestModule(demos.BoringModel):
        def __init__(self, reduce_fn):
            super().__init__()
            self.reduce_fn = reduce_fn
            self.losses = []

        def on_train_epoch_start(self, stage: "reax.stages.Train", /) -> None:
            super().on_train_epoch_start(stage)
            self.losses = []

        def training_step(self, batch, batch_idx: int, /):
            res = super().training_step(batch, batch_idx)
            loss = res["loss"]
            self.log("loss", loss, on_epoch=True, reduce_fx=self.reduce_fn)
            self.losses.append(loss.item())
            return res

        @staticmethod
        def loss(preds, labels):
            """Loss function."""
            if labels is None:
                labels = jnp.ones_like(preds)
            # An arbitrary loss to have a loss that updates the model weights during `Trainer.fit` calls
            return optax.losses.squared_error(preds, labels).mean() ** 0.5

    module = TestModule(reduce_fn)
    stage = test_trainer.fit(module, max_epochs=1)

    reduce = np.mean if reduce_fn == "mean" else np.sum
    assert np.isclose(reduce(module.losses), stage._fit_epoch.callback_metrics["loss"])
