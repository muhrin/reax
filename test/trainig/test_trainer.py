from argparse import Namespace
from contextlib import nullcontext, suppress
from copy import deepcopy
import gc
import logging
import math
import os
from pathlib import Path
import pickle
from unittest import mock
from unittest.mock import ANY, Mock, call, patch

import jax.numpy as jnp
import pytest

import reax
from reax import demos


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
