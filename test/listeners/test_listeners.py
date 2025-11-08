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
# https://github.com/Lightning-AI/pytorch-lightning/blob/f9babd1def4c703e639dfc34fd1877ac4e7b9435/tests/tests_pytorch/listeners/test_listeners.py
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
from pathlib import Path
from re import escape
from unittest.mock import Mock

import pytest

import reax
from reax import demos


def test_listeners_configured_in_model(tmp_path):
    """Test the listener system with listeners added through the model hook.

    ..warning::
        REAX behaviour deviates from that of lightning here.  Model listeners are only attached
        to the trainer during the stage.  Once the stage is complete, the listeners are back to
        how they were beforehand.
    """
    model_listener_mock = Mock(spec=reax.TrainerListener, model=reax.TrainerListener())
    trainer_listener_mock = Mock(spec=reax.TrainerListener, model=reax.TrainerListener())

    class TestModel(demos.BoringModel):
        def configure_listeners(self):
            return [model_listener_mock]

    model = TestModel()
    trainer_options = {
        "default_root_dir": tmp_path,
        "enable_checkpointing": False,
        "enable_progress_bar": False,
    }

    def assert_expected_calls(_trainer, model_listener, trainer_listener):
        # assert that the rest of calls are the same as for trainer listeners
        expected_calls = [m for m in trainer_listener.method_calls if m]
        assert expected_calls
        assert model_listener.method_calls == expected_calls

    # .fit()
    trainer_options.update(listeners=[trainer_listener_mock])
    trainer = reax.Trainer(**trainer_options)

    assert trainer_listener_mock in trainer.listeners
    assert model_listener_mock not in trainer.listeners
    trainer.fit(model, fast_dev_run=True)

    assert model_listener_mock not in trainer.listeners
    # assert trainer.listeners[-1] == model_listener_mock <- not REAX behaviour
    assert_expected_calls(trainer, model_listener_mock, trainer_listener_mock)

    # .test()
    for fn in ("test", "validate"):
        model_listener_mock.reset_mock()
        trainer_listener_mock.reset_mock()

        trainer_options.update(listeners=[trainer_listener_mock])
        trainer = reax.Trainer(**trainer_options)

        trainer_fn = getattr(trainer, fn)
        trainer_fn(model)

        assert model_listener_mock not in trainer.listeners
        # assert trainer.listeners[-1] == model_listener_mock <- not REAX behaviour
        assert_expected_calls(trainer, model_listener_mock, trainer_listener_mock)


def test_configure_listeners_hook_multiple_calls(tmp_path):
    """Test that subsequent calls to `configure_listeners` do not change the listeners list."""
    model_listener_mock = Mock(spec=reax.TrainerListener, model=reax.TrainerListener())

    class TestModel(demos.BoringModel):
        def configure_listeners(self):
            return model_listener_mock

    model = TestModel()
    trainer = reax.Trainer(default_root_dir=tmp_path, enable_checkpointing=False)

    listeners_before_fit = trainer.listeners.copy()
    assert listeners_before_fit

    trainer.fit(model, fast_dev_run=True)
    listeners_after_fit = trainer.listeners.copy()
    assert (
        listeners_after_fit == listeners_before_fit
    )  # + [model_listener_mock] <- not REAX behaviour

    for fn in ("test", "validate"):
        trainer_fn = getattr(trainer, fn)
        trainer_fn(model)

        listeners_after = trainer.listeners.copy()
        assert listeners_after == listeners_after_fit

        trainer_fn(model)
        listeners_after = trainer.listeners.copy()
        assert listeners_after == listeners_after_fit


class OldStatefulCallback(reax.TrainerListener):
    def __init__(self, state):
        self.state = state

    @property
    def state_key(self):
        return type(self)

    def state_dict(self):
        return {"state": self.state}

    def load_state_dict(self, state_dict) -> None:
        self.state = state_dict["state"]


@pytest.mark.skip(reason="not yet supported")
def test_resume_listener_state_saved_by_type_stateful(tmp_path):
    """Test that a legacy checkpoint that didn't use a state key before can still be loaded, using
    state_dict/load_state_dict.
    """
    model = demos.BoringModel()
    listener = OldStatefulCallback(state=111)
    trainer = reax.Trainer(default_root_dir=tmp_path, listeners=[listener])
    trainer.fit(model, max_updates=1)
    ckpt_path = Path(trainer.checkpoint_listener.best_model_path)
    assert ckpt_path.exists()

    listener = OldStatefulCallback(state=222)
    trainer = reax.Trainer(default_root_dir=tmp_path, listeners=[listener])
    trainer.fit(model, max_updates=2, ckpt_path=ckpt_path)
    assert listener.state == 111


@pytest.mark.skip(reason="not yet supported")
def test_resume_incomplete_listeners_list_warning(tmp_path):
    model = demos.BoringModel()
    listener0 = reax.listeners.ModelCheckpoint(monitor="epoch")
    listener1 = reax.listeners.ModelCheckpoint(monitor="global_step")
    trainer = reax.Trainer(
        default_root_dir=tmp_path,
        listeners=[listener0, listener1],
    )
    trainer.fit(model, max_updates=1)
    ckpt_path = trainer.checkpoint_listener.best_model_path

    trainer = reax.Trainer(
        default_root_dir=tmp_path,
        listeners=[listener1],  # one listener is missing!
    )
    with pytest.warns(
        UserWarning,
        match=escape(f"Please add the following listeners: [{repr(listener0.state_key)}]"),
    ):
        trainer.fit(model, ckpt_path=ckpt_path, max_steps=1)

    trainer = reax.Trainer(
        default_root_dir=tmp_path,
        max_steps=1,
        listeners=[listener1, listener0],  # all listeners here, order switched
    )
    with no_warning_call(UserWarning, match="Please add the following listeners:"):
        trainer.fit(model, ckpt_path=ckpt_path)
