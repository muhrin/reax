import jax.numpy as jnp
import numpy as np
import optax.losses
import pytest

import reax
from reax import demos


@pytest.mark.parametrize("reduce_fn", ["mean", "sum"])
@pytest.mark.parametrize("batch_size_mode", ["supplied", "none"])
def test_module_log(test_trainer, reduce_fn, batch_size_mode):
    class TestModule(demos.BoringModel):
        def __init__(self, reduce_fn, batch_size_mode):
            super().__init__()
            self.reduce_fn = reduce_fn
            self.batch_size_mode = batch_size_mode
            self.losses = []
            self.batch_sizes = []

        def on_train_epoch_start(self, stage: "reax.stages.Train", /) -> None:
            super().on_train_epoch_start(stage)
            self.losses = []
            self.batch_sizes = []

        def training_step(self, batch, batch_idx: int, /):
            res = super().training_step(batch, batch_idx)
            loss = res["loss"]

            # Varying batch size per step (8, 16, 24...) or None
            if self.batch_size_mode == "supplied":
                batch_size = (batch_idx + 1) * 8
            else:
                batch_size = None

            self.log("loss", loss, on_epoch=True, batch_size=batch_size, reduce_fn=self.reduce_fn)

            self.losses.append(loss.item())
            if batch_size is not None:
                self.batch_sizes.append(batch_size)
            return res

        @staticmethod
        def loss(preds, labels):
            """Loss function."""
            if labels is None:
                labels = jnp.ones_like(preds)
            return optax.losses.squared_error(preds, labels).mean() ** 0.5

    module = TestModule(reduce_fn, batch_size_mode)
    stage = test_trainer.fit(module, max_epochs=1)

    # Compute expected metric based on combination
    if reduce_fn == "sum":
        expected_metric = np.sum(module.losses)
    elif batch_size_mode == "supplied":
        expected_metric = np.average(module.losses, weights=module.batch_sizes)
    else:  # reduce_fn == "mean" and batch_size_mode == "none"
        expected_metric = np.mean(module.losses)

    assert np.isclose(expected_metric, stage._fit_epoch.callback_metrics["loss"])
