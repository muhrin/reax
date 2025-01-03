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
# https://github.com/Lightning-AI/pytorch-lightning/blob/9177ec09caadcf88859e1f1e3e10a18e8832069a/tests/tests_pytorch/trainer/test_trainer.py
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


import logging
import math

from flax import linen
import jaxtyping as jt
import pytest

import reax
from reax import demos


def test_trainer_error_when_input_not_reax_module():
    """Test that a useful error gets raised when the Trainer methods receive something other than a LightningModule."""
    with pytest.raises(jt.TypeCheckError, match="Expected type: <class 'reax.modules.Module'>."):
        reax.Trainer(linen.Dense(2))


def test_trainer_max_steps_and_epochs(tmp_path):
    """Verify model trains according to specified max steps."""
    mod = demos.BoringModel()
    num_train_samples = math.floor(len(mod.train_dataloader()) * 0.5)

    # define less train steps than epochs
    kwargs = dict(
        module=mod,
        default_root_dir=tmp_path,
        logger=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )

    trainer = reax.Trainer(**kwargs)
    max_updates = num_train_samples + 10
    trainer.fit(max_epochs=3, max_updates=max_updates)

    # todo: assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert trainer.global_updates == max_updates, "Model did not stop at max_steps"
    trainer.finalize()

    # define less train epochs than steps
    trainer = reax.Trainer(**kwargs)
    max_epochs = 2
    trainer.fit(
        max_epochs=max_epochs,
        max_updates=3 * 2 * num_train_samples,
        limit_train_batches=0.5,
    )

    # todo: assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert trainer.global_updates == num_train_samples * max_epochs
    assert trainer.current_epoch == max_epochs, "Model did not stop at max_epochs"
    trainer.finalize()

    trainer = reax.Trainer(**kwargs)
    # if max_steps is positive and max_epochs is infinity, use max_steps
    trainer.fit(max_epochs=float("inf"), max_updates=3)

    # todo: assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert trainer.global_updates == 3


@pytest.mark.parametrize(
    ("max_epochs", "max_updates", "incorrect_variable"),
    [
        (-100, -1, "max_epochs"),
        (1, -2, "max_updates"),
    ],
)
def test_trainer_max_steps_and_epochs_validation(max_epochs, max_updates, incorrect_variable):
    """Don't allow `max_epochs` or `max_updates` to be less than -1 or a float."""
    with pytest.raises(
        ValueError,
        match=f"`{incorrect_variable}` must be a non-negative integer or -1",
    ):
        reax.Trainer(demos.BoringModel()).fit(max_epochs=max_epochs, max_updates=max_updates)


def test_trainer_min_steps_and_epochs(tmp_path):
    """Verify model trains according to specified min steps."""
    num_train_samples = math.floor(len(demos.BoringModel().train_dataloader()) * 0.5)

    class CustomModel(demos.BoringModel):
        def training_step(self, *args, **kwargs):
            # try to force stop right after first step
            if self.global_updates > 0:
                self.trainer.should_step = True

            return super().training_step(*args, **kwargs)

    model = CustomModel()

    trainer_kwargs = {
        "default_root_dir": tmp_path,
        # "val_check_interval": 2, # todo: not supported for now
        "logger": False,
        # "enable_model_summary": False,
        "enable_progress_bar": False,
    }
    fit_kwargs = {
        "limit_train_batches": 0.5,
        "min_epochs": 1,
        "max_epochs": 7,
        # define less min steps than 1 epoch
        "min_updates": num_train_samples // 2,
    }
    trainer = reax.Trainer(model, **trainer_kwargs)
    trainer.fit(**fit_kwargs)

    # assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert trainer.current_epoch > 0
    assert (
        trainer.global_updates >= num_train_samples
    ), "Model did not train for at least min_epochs"
    trainer.finalize()

    # define less epochs than min_steps
    fit_kwargs["min_updates"] = math.floor(num_train_samples * 1.5)
    trainer = reax.Trainer(model, **trainer_kwargs)
    trainer.fit(**fit_kwargs)

    # assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert trainer.current_epoch > 0
    assert trainer.global_updates >= math.floor(
        num_train_samples * 1.5
    ), "Model did not train for at least min_steps"


def test_trainer_min_steps_and_min_epochs_not_reached(tmp_path, caplog):
    """Test that min_epochs/min_steps in Trainer are enforced even if EarlyStopping is triggered."""

    class TestModel(demos.BoringModel):
        training_step_invoked = 0

        def training_step(self, batch, batch_idx):
            output = super().training_step(batch, batch_idx)
            output["loss"] = output["loss"] * 0.0  # force minimal loss to trigger early stopping
            self.log("loss", output["loss"])
            self.training_step_invoked += 1
            if self.current_epoch < 2:
                assert not self.trainer.should_stop
            else:
                assert self.trainer.should_stop
            return output

    model = TestModel()
    early_stop = reax.listeners.EarlyStopping(
        monitor="loss", patience=0, check_on_train_epoch_end=True
    )
    min_epochs = 5
    trainer = reax.Trainer(
        model,
        default_root_dir=tmp_path,
        enable_progress_bar=False,
        listeners=[early_stop],
    )
    with caplog.at_level(logging.INFO, logger="reax"):
        trainer.fit(
            min_epochs=min_epochs,
            limit_val_batches=0,
            limit_train_batches=2,
        )

    message = (
        f"Trainer was signaled to stop but the required "
        f"`min_epochs={min_epochs!r}` or `min_steps=-1` has not been met. "
        f"Training will continue..."
    )
    num_messages = sum(1 for record in caplog.records if message in record.message)
    assert num_messages == 1
    assert model.training_step_invoked == min_epochs * 2


def test_trainer_max_updates_accumulate_batches(tmp_path):
    """Verify model trains according to specified max steps with grad accumulated batches."""
    model = demos.BoringModel()
    num_train_samples = math.floor(len(model.train_dataloader()) * 0.5)

    # define less train steps than epochs
    max_updates = num_train_samples + 10
    trainer = reax.Trainer(
        model,
        default_root_dir=tmp_path,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(
        limit_train_batches=0.5,
        max_updates=max_updates,
        accumulate_grad_batches=10,
    )

    # TODO: assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert trainer.global_updates == max_updates, "Model did not stop at max_updates"


def test_disabled_validation(tmp_path):
    """Verify that `limit_val_batches=0` disables the validation loop unless `fast_dev_run=True`."""

    class CurrentModel(demos.BoringModel):
        validation_step_invoked = False

        def validation_step(self, *args, **kwargs):
            self.validation_step_invoked = True
            return super().validation_step(*args, **kwargs)

    model = CurrentModel()

    trainer_options = {
        "default_root_dir": tmp_path,
        "enable_progress_bar": False,
        "fast_dev_run": False,
    }
    trainer = reax.Trainer(model, **trainer_options)
    trainer.fit(max_epochs=2, limit_train_batches=0.4, limit_val_batches=0.0)

    # check that limit_val_batches=0 turns off validation
    # assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert trainer.current_epoch == 2
    assert (
        not model.validation_step_invoked
    ), "`validation_step` should not run when `limit_val_batches=0`"

    # check that limit_val_batches has no influence when fast_dev_run is turned on
    model = CurrentModel()
    trainer_options.update(fast_dev_run=True)
    trainer = reax.Trainer(model, **trainer_options)
    trainer.fit(max_epochs=2, limit_train_batches=0.4, limit_val_batches=0.0)

    # TODO: assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert trainer.current_epoch == 1
    assert model.validation_step_invoked, "did not run `validation_step` with `fast_dev_run=True`"


@pytest.mark.parametrize("stage", ["fit", "validate", "test"])
def test_trainer_setup_call(tmp_path, stage):
    """Test setup call gets the correct stage."""

    class CurrentModel(demos.BoringModel):
        def setup(self, stage, batch):
            super().setup(stage, batch)
            self.stage = stage

    class CurrentListener(reax.TrainerListener):
        def setup(self, trainer, stage):
            assert stage.module is not None
            self.stage = stage

    model = CurrentModel()
    listener = CurrentListener()
    trainer = reax.Trainer(
        model, default_root_dir=tmp_path, enable_checkpointing=False, listeners=[listener]
    )

    if stage == "fit":
        trainer.fit(max_epochs=1)
    elif stage == "validate":
        trainer.validate()
    else:
        trainer.test()

    assert str(listener.stage) == stage
    assert str(model.stage) == stage


@pytest.mark.parametrize("return_predictions", [None, False, True])
def test_predict_return_predictions_cpu(return_predictions, tmp_path):
    reax.seed_everything(42)
    model = reax.demos.BoringModel()

    trainer = reax.Trainer(model, fast_dev_run=True, default_root_dir=tmp_path)
    preds = trainer.predict(
        dataloaders=model.train_dataloader(), return_predictions=return_predictions
    )
    if return_predictions or return_predictions is None:
        assert len(preds) == 1
        assert preds[0].shape == (1, 2)


def test_trainer_access_in_configure_optimizers(tmp_path):
    """Verify that the configure optimizer function can reference the trainer."""

    class TestModel(demos.BoringModel):
        def configure_optimizers(self):
            assert (
                self.trainer is not None
            ), "Expect to have access to the trainer within `configure_optimizers`"

    train_data = reax.data.ReaxDataLoader(demos.RandomDataset(32, 64))

    model = TestModel()
    trainer = reax.Trainer(model, default_root_dir=tmp_path, fast_dev_run=True)
    trainer.fit(train_data)
