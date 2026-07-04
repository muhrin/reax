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
# https://github.com/Lightning-AI/pytorch-lightning/blob/f9babd1def4c703e639dfc34fd1877ac4e7b9435/tests/tests_pytorch/callbacks/test_callback_hook_outputs.py
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
import pytest

import reax
from reax import demos


@pytest.mark.parametrize("single_cb", [False, True])
def test_train_step_no_return(tmp_path, single_cb: bool):
    """Tests that only training_step can be used."""

    class Listener(reax.TrainerListener):
        def on_train_batch_end(self, trainer, stage, outputs, *_):
            assert "loss" in outputs

        def on_validation_batch_end(self, trainer, stage, outputs, *_):
            assert "x" in outputs

        def on_test_batch_end(self, trainer, stage, outputs, *_):
            assert "x" in outputs

    class TestModel(demos.boring_classes.BoringModel):
        def on_train_batch_end(self, stage, outputs, *_):
            assert "loss" in outputs

        def on_validation_batch_end(self, stage, outputs, *_):
            assert "x" in outputs

        def on_test_batch_end(self, stage, outputs, *_):
            assert "x" in outputs

    model = TestModel()

    trainer = reax.Trainer(
        listeners=Listener() if single_cb else [Listener()],
        default_root_dir=tmp_path,
        enable_model_summary=False,
        log_every_n_steps=1,
    )

    assert any(isinstance(c, Listener) for c in trainer.listeners)

    trainer.fit(
        model,
        limit_train_batches=2,
        limit_val_batches=2,
        max_epochs=1,
    )
