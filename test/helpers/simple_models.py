# Copyright (C) 2024  Martin Uhrin
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Most of this file is covered by the following license.  To find what has been modified you
# can perform a diff with the file at:
# https://github.com/Lightning-AI/pytorch-lightning/blob/9177ec09caadcf88859e1f1e3e10a18e8832069a/tests/tests_pytorch/helpers/simple_models.py
#
# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import functools
from typing import Any, Callable, Optional

from flax import linen
import jax.typing
import jaxtyping as jt
import optax
from optax import losses

import reax


class ClassificationModel(reax.Module):
    def __init__(self, num_features=32, num_classes=3, lr=0.01):
        super().__init__()

        self.lr = lr
        self.num_classes = num_classes

        layers = []
        for i in range(3):
            layers.append(linen.Dense(num_features))
            layers.append(linen.activation.relu)
        layers.append(linen.Dense(num_classes))
        self._model = linen.Sequential(layers)

        # Define our settings for measuring accuracy
        self.acc = reax.metrics.Accuracy(mode="multiclass", num_classes=num_classes)

    @staticmethod
    def loss_fn(logits, labels):
        return losses.softmax_cross_entropy(logits, labels).mean()

    def configure_optimizers(self):
        assert self.parameters() is not None  # nosec B101
        optimiser = optax.adam(learning_rate=self.lr)
        state = optimiser.init(self.parameters())
        return optimiser, state

    def setup(self, stage: reax.Stage):
        if isinstance(stage, reax.stages.Train) and self.parameters() is None:
            batch = next(iter(stage.dataloader))
            inputs = batch[0]

            params = self._model.init(self.trainer.rng_key(), inputs)
            self.set_parameters(params)

    def forward(self, x):
        return jax.jit(self._model.apply)(self.parameters(), x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_one_hot = jax.nn.one_hot(y, self.num_classes, dtype=y.dtype)

        (loss, logits), grads = jax.value_and_grad(self.step, argnums=0, has_aux=True)(
            self.parameters(), x, y_one_hot, self._model.apply, self.loss_fn
        )

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.acc.create(logits, y), prog_bar=True)
        return {"grad": grads, "loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        y_one_hot = jax.nn.one_hot(y, self.num_classes, dtype=y.dtype)
        self.log("val_loss", losses.softmax_cross_entropy(logits, y_one_hot), prog_bar=False)
        self.log("val_acc", self.acc.create(logits, y), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        y_one_hot = jax.nn.one_hot(y, self.num_classes, dtype=y.dtype)
        self.log("test_loss", losses.softmax_cross_entropy(logits, y_one_hot), prog_bar=False)
        self.log("test_acc", self.acc.create(logits, y), prog_bar=True)

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        return self.forward(x)

    @staticmethod
    @functools.partial(jax.jit, static_argnums=[3, 4], donate_argnums=[0, 1, 2])
    def step(
        params: jt.PyTree,
        inputs: Any,
        targets: Any,
        model: Callable[[jt.PyTree, Any], Any],
        loss_fn: Callable,
    ) -> tuple[jax.Array, Optional[reax.metrics.MetricCollection]]:
        """Calculate loss and, optionally metrics"""
        predictions = model(params, inputs)
        return loss_fn(predictions, targets), predictions
