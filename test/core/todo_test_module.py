# Copyright (C) 2025  Martin Uhrin
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
# https://github.com/Lightning-AI/pytorch-lightning/blob/9177ec09caadcf88859e1f1e3e10a18e8832069a/tests/tests_pytorch/core/test_module.py
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
import sys
from unittest.mock import Mock
import weakref

from flax import linen as nn
import optax
import pytest

import reax
from reax.demos import boring_classes

# from tests_pytorch.helpers.runif import RunIf


def test_lightning_module_not_abstract():
    """Test that the reax.Module can be instantiated (it is not an abstract class)."""
    _ = reax.Module()


def test_property_current_epoch():
    """Test that the current_epoch in LightningModule is accessible via the Trainer."""
    model = boring_classes.BoringModel()
    assert model.current_epoch == 0

    trainer = Mock(current_epoch=123)
    model.trainer = trainer
    assert model.current_epoch == 123


def test_property_global_update():
    """Test that the global_step in LightningModule is accessible via the Trainer."""
    model = boring_classes.BoringModel()
    assert model.global_updates == 0

    trainer = Mock(global_step=123)
    model.trainer = trainer
    assert model.global_updates == 123


@pytest.mark.skip(reason="Not supported yet")
def test_property_process_index():
    """Test that the global rank in ReaxModule is accessible via the Trainer."""
    model = boring_classes.BoringModel()
    assert model.process_index == 0

    trainer = Mock(process_index=123)
    model.trainer = trainer
    assert model.process_index == 123


@pytest.mark.skip(reason="Not supported yet")
def test_property_local_process_index():
    """Test that the local rank in LightningModule is accessible via the Trainer."""
    model = boring_classes.BoringModel()
    assert model.local_process_index == 0

    trainer = Mock(local_rank=123)
    model.trainer = trainer
    assert model.local_process_index == 123


def test_property_logger(tmp_path):
    """Test that the logger in LightningModule is accessible via the Trainer."""
    model = boring_classes.BoringModel()
    assert model.logger is None

    logger = reax.loggers.TensorBoardLogger(tmp_path)
    trainer = reax.Trainer(logger=logger)
    model.trainer = trainer
    assert model.logger == logger


def test_property_loggers(tmp_path):
    """Test that loggers in LightningModule is accessible via the Trainer."""
    model = boring_classes.BoringModel()
    assert model.loggers == []

    logger = reax.loggers.TensorBoardLogger(tmp_path)
    trainer = reax.Trainer(logger=logger)
    model.trainer = trainer
    assert model.loggers == [logger]

    logger0 = reax.loggers.TensorBoardLogger(tmp_path)
    logger1 = reax.loggers.TensorBoardLogger(tmp_path)
    trainer = reax.Trainer(logger=[logger0, logger1])
    model.trainer = trainer
    assert model.loggers == [logger0, logger1]


def test_1_optimizer_toggle_model():
    """Test toggle_model runs when only one optimizer is used."""
    model = boring_classes.BoringModel()
    trainer = Mock()
    model.trainer = trainer
    params = model.parameters()
    optimizer = torch.optim.SGD(params, lr=0.1)
    trainer.optimizers = [optimizer]

    assert not model._param_requires_grad_state
    # toggle optimizer was failing with a single optimizer
    model.toggle_optimizer(optimizer)
    assert model._param_requires_grad_state
    model.untoggle_optimizer(optimizer)
    assert not model._param_requires_grad_state


def test_toggle_untoggle_2_optimizers_no_shared_parameters(tmp_path):
    class TestModel(boring_classes.BoringModel):
        def __init__(self):
            super().__init__()
            self.automatic_optimization = False
            self.layer_1 = nn.Sequential(
                [nn.Dense(32), nn.relu, nn.Dense(32), nn.relu, nn.Dense(32)]
            )

            self.layer_2 = nn.Sequential(
                [nn.relu, nn.Dense(32), nn.relu, nn.Dense(32), nn.relu, nn.Dense(2)]
            )

            # set some weights to require no gradient to check that toggle/untoggle works as expected.
            self.layer_1[2].weight.requires_grad = False
            self.layer_1[4].weight.requires_grad = False

            self.layer_2[1].weight.requires_grad = False
            self.layer_2[3].weight.requires_grad = False

        def training_step(self, batch, batch_idx):
            opt1, opt2 = self.optimizers()

            # Use the first optimizer, toggle it
            self.toggle_optimizer(opt1)
            loss = self.step(batch)
            opt1.zero_grad()
            self.manual_backward(loss)
            assert self.layer_1[0].weight.requires_grad is True
            assert self.layer_1[2].weight.requires_grad is False
            assert self.layer_1[4].weight.requires_grad is False

            assert self.layer_2[1].weight.requires_grad is False
            assert self.layer_2[3].weight.requires_grad is False
            assert self.layer_2[5].weight.requires_grad is False
            opt1.step()
            self.untoggle_optimizer(opt1)

            # Use the second optimizer, toggle it
            self.toggle_optimizer(opt2)
            loss = self.step(batch)
            opt2.zero_grad()
            self.manual_backward(loss)
            assert self.layer_1[0].weight.requires_grad is False
            assert self.layer_1[2].weight.requires_grad is False
            assert self.layer_1[4].weight.requires_grad is False

            assert self.layer_2[1].weight.requires_grad is False
            assert self.layer_2[3].weight.requires_grad is False
            assert self.layer_2[5].weight.requires_grad is True
            opt2.step()
            self.untoggle_optimizer(opt2)

        def configure_optimizers(self):
            optimizer_1 = optax.sgd(learning_rate=0.1)
            optimizer_2 = optax.adam(learning_rate=0.1)

            opt_1_data = [optimizer_1, optimizer_1.init(self.parameters()["layer_1"])]
            opt_2_data = [optimizer_2, optimizer_2.init(self.parameters()["layer_2"])]
            return [opt_1_data, opt_2_data]

    model = TestModel()

    trainer = reax.Trainer(default_root_dir=tmp_path)
    trainer.fit(model, max_epochs=1, limit_train_batches=8, limit_val_batches=0)


def test_toggle_untoggle_3_optimizers_shared_parameters(tmp_path):
    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.automatic_optimization = False
            self.layer_1 = nn.Sequential(
                nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 32)
            )

            self.layer_2 = nn.Sequential(
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 2),
            )

            self.layer_3 = nn.Sequential(
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 2),
            )

            # set some weights to require no gradient to check that toggle/untoggle works as expected.
            self.layer_1[2].weight.requires_grad = False
            self.layer_1[4].weight.requires_grad = False

            self.layer_2[1].weight.requires_grad = False
            self.layer_2[3].weight.requires_grad = False

            self.layer_3[1].weight.requires_grad = False
            self.layer_3[5].weight.requires_grad = False

        def training_step(self, batch, batch_idx):
            opt1, opt2, opt3 = self.optimizers()

            # Use the first optimizer, toggle it
            self.toggle_optimizer(opt1)
            loss = self.step(batch)
            opt1.zero_grad()
            self.manual_backward(loss)
            assert self.layer_1[0].weight.requires_grad is True
            assert self.layer_1[2].weight.requires_grad is False
            assert self.layer_1[4].weight.requires_grad is False

            assert self.layer_2[1].weight.requires_grad is False
            assert self.layer_2[3].weight.requires_grad is False
            assert self.layer_2[5].weight.requires_grad is True

            assert self.layer_3[1].weight.requires_grad is False
            assert self.layer_3[3].weight.requires_grad is False
            assert self.layer_3[5].weight.requires_grad is False
            opt1.step()
            self.untoggle_optimizer(opt1)

            # Use the second optimizer, toggle it
            self.toggle_optimizer(opt2)
            loss = self.step(batch)
            opt2.zero_grad()
            self.manual_backward(loss)
            assert self.layer_1[0].weight.requires_grad is False
            assert self.layer_1[2].weight.requires_grad is False
            assert self.layer_1[4].weight.requires_grad is False

            assert self.layer_2[1].weight.requires_grad is False
            assert self.layer_2[3].weight.requires_grad is False
            assert self.layer_2[5].weight.requires_grad is True

            assert self.layer_3[1].weight.requires_grad is False
            assert self.layer_3[3].weight.requires_grad is True
            assert self.layer_3[5].weight.requires_grad is False
            opt2.step()
            self.untoggle_optimizer(opt2)

            # Use the third optimizer, toggle it
            self.toggle_optimizer(opt3)
            loss = self.step(batch)
            opt3.zero_grad()
            self.manual_backward(loss)
            assert self.layer_1[0].weight.requires_grad is True
            assert self.layer_1[2].weight.requires_grad is False
            assert self.layer_1[4].weight.requires_grad is False

            assert self.layer_2[1].weight.requires_grad is False
            assert self.layer_2[3].weight.requires_grad is False
            assert self.layer_2[5].weight.requires_grad is False

            assert self.layer_3[1].weight.requires_grad is False
            assert self.layer_3[3].weight.requires_grad is True
            assert self.layer_3[5].weight.requires_grad is False
            opt3.step()
            self.untoggle_optimizer(opt3)

        @staticmethod
        def combine_generators(gen_1, gen_2):
            yield from gen_1
            yield from gen_2

        def configure_optimizers(self):
            optimizer_1 = SGD(
                self.combine_generators(self.layer_1.parameters(), self.layer_2.parameters()),
                lr=0.1,
            )
            optimizer_2 = Adam(
                self.combine_generators(self.layer_2.parameters(), self.layer_3.parameters()),
                lr=0.1,
            )
            optimizer_3 = SGD(
                self.combine_generators(self.layer_3.parameters(), self.layer_1.parameters()),
                lr=0.1,
            )
            return [optimizer_1, optimizer_2, optimizer_3]

    model = TestModel()
    trainer = reax.Trainer(default_root_dir=tmp_path)
    trainer.fit(model, max_epochs=1, limit_train_batches=8)


# @pytest.mark.parametrize(
#     ("accelerator", "device"),
#     [
#         pytest.param("gpu", "cuda:0", marks=RunIf(min_cuda_gpus=1)),
#         pytest.param("mps", "mps:0", marks=RunIf(mps=True)),
#     ],
# )
# def test_device_placement(tmp_path, accelerator, device):
#     model = BoringModel()
#     trainer = Trainer(
#         default_root_dir=tmp_path, fast_dev_run=True, accelerator=accelerator, devices=1
#     )
#     trainer.fit(model)
#
#     def assert_device(device: torch.device) -> None:
#         assert model.device == device
#         for p in model.parameters():
#             assert p.device == device
#
#     assert_device(torch.device("cpu"))
#     model.to(torch.device(device))
#     assert_device(torch.device(device))
#     trainer.test(model)
#     assert_device(torch.device("cpu"))
#     trainer.predict(model, dataloaders=model.train_dataloader())
#     assert_device(torch.device("cpu"))
#
#
# @RunIf(skip_windows=True, max_torch="2.1.0")
# def test_sharded_tensor_state_dict(single_process_pg):
#     from torch.distributed._shard.sharded_tensor import empty as sharded_tensor_empty
#     from torch.distributed._sharding_spec import ChunkShardingSpec
#
#     class BoringModelWithShardedTensor(BoringModel):
#         def __init__(self, spec):
#             super().__init__()
#             self.sharded_tensor = sharded_tensor_empty(spec, 10, 20)
#             self.sharded_tensor.local_shards()[0].tensor.fill_(0)
#
#     spec = ChunkShardingSpec(
#         dim=0,
#         placements=[
#             "rank:0/cpu",
#         ],
#     )
#
#     m_0 = BoringModelWithShardedTensor(spec)
#     m_0.sharded_tensor.local_shards()[0].tensor.fill_(1)
#     assert (
#         "sharded_tensor" in m_0.state_dict()
#     ), 'Expect "sharded_tensor" to appear in the state dict'
#
#     m_1 = BoringModelWithShardedTensor(spec)
#     assert not torch.allclose(
#         m_1.sharded_tensor.local_shards()[0].tensor, m_0.sharded_tensor.local_shards()[0].tensor
#     ), "Expect the shards to be different before `m_1` loading `m_0`'s state dict"
#
#     m_1.load_state_dict(m_0.state_dict(), strict=False)
#     assert torch.allclose(
#         m_1.sharded_tensor.local_shards()[0].tensor, m_0.sharded_tensor.local_shards()[0].tensor
#     ), "Expect the shards to be same after `m_1` loading `m_0`'s state dict"
#


def test_lightning_module_configure_gradient_clipping(tmp_path):
    """Test custom gradient clipping inside `configure_gradient_clipping` hook."""

    class TestModel(boring_classes.BoringModel):
        has_validated_gradients = False
        custom_gradient_clip_val = 1e-2

        def configure_gradient_clipping(
            self, optimizer, gradient_clip_val, gradient_clip_algorithm
        ):
            assert gradient_clip_val == self.trainer.gradient_clip_val
            assert gradient_clip_algorithm == self.trainer.gradient_clip_algorithm

            for pg in optimizer.param_groups:
                for p in pg["params"]:
                    p.grad.clamp_(min=0, max=self.custom_gradient_clip_val)

    model = TestModel()
    trainer = reax.Trainer(default_root_dir=tmp_path, gradient_clip_val=1e-4)
    trainer.fit(model, max_epochs=1, limit_train_batches=1, limit_val_batches=0)

    optimizer = model.optimizers()
    for pg in optimizer.param_groups:
        for p in pg["params"]:
            if p.grad is not None:
                assert p.grad.min() >= 0
                assert p.grad.max() <= model.custom_gradient_clip_val


def test_lightning_module_configure_gradient_clipping_different_argument_values(tmp_path):
    """Test that setting gradient clipping arguments in `Trainer` and cusotmizing gradient clipping inside
    `configure_gradient_clipping` with different values raises an exception."""

    class TestModel(boring_classes.BoringModel):
        custom_gradient_clip_val = 1e-2

        def configure_gradient_clipping(
            self, optimizer, gradient_clip_val, gradient_clip_algorithm
        ):
            self.clip_gradients(optimizer, gradient_clip_val=self.custom_gradient_clip_val)

    model = TestModel()
    trainer = reax.Trainer(default_root_dir=tmp_path, gradient_clip_val=1e-4)
    with pytest.raises(
        reax.exceptions.MisconfigurationException,
        match=r"gradient_clip_val=0.0001\)` and have passed `clip_gradients\(gradient_clip_val=0.01",
    ):
        trainer.fit(model, max_epochs=1, limit_train_batches=2, limit_val_batches=0)

    class TestModel(boring_classes.BoringModel):
        custom_gradient_clip_algorithm = "foo"

        def configure_gradient_clipping(
            self, optimizer, gradient_clip_val, gradient_clip_algorithm
        ):
            self.clip_gradients(
                optimizer, gradient_clip_algorithm=self.custom_gradient_clip_algorithm
            )

    model = TestModel()
    trainer = reax.Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=0,
        gradient_clip_algorithm="norm",
    )
    with pytest.raises(
        reax.exceptions.MisconfigurationException,
        match=r"gradient_clip_algorithm='norm'\)` and have passed `clip_gradients\(gradient_clip_algorithm='foo'",
    ):
        trainer.fit(model)


def test_proper_refcount():
    torch_module = nn.Module()
    lightning_module = LightningModule()

    assert sys.getrefcount(torch_module) == sys.getrefcount(lightning_module)


def test_lightning_module_scriptable():
    """Test that the LightningModule is `torch.jit.script`-able.

    Regression test for #15917.

    """
    model = boring_classes.BoringModel()
    trainer = reax.Trainer()
    model.trainer = trainer
    torch.jit.script(model)


def test_trainer_reference_recursively():
    ensemble = reax.Module()
    inner = reax.Module()
    ensemble.inner = inner

    assert inner._trainer is None
    with pytest.raises(RuntimeError, match="attached to a `Trainer"):
        _ = ensemble.trainer

    trainer = Mock()
    ensemble.trainer = trainer
    # references match
    assert ensemble.trainer is inner.trainer
