import jax
import jax.numpy as jnp
import pytest

import reax
from reax import demos, testing


@pytest.mark.multiproc
def test_trainer_fit_multiprocess(tmp_path):
    testing.in_subprocess(_run_trainer_fit)(str(tmp_path))


def _run_trainer_fit(tmp_path):
    trainer = reax.Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        devices=2,
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
    )
    model = demos.BoringModel()
    fit = trainer.fit(model, max_epochs=1, limit_train_batches=0.5, limit_val_batches=0.0)
    assert fit.state.finished
    assert trainer.current_epoch == 1
    assert trainer.global_updates > 0
    trainer.finalize()


@pytest.mark.multiproc
def test_trainer_validate_multiprocess(tmp_path):
    testing.in_subprocess(_run_trainer_validate)(str(tmp_path))


def _run_trainer_validate(tmp_path):
    trainer = reax.Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        devices=2,
        logger=False,
        enable_progress_bar=False,
    )
    model = demos.BoringModel()
    result = trainer.validate(model, limit_batches=2)
    trainer.finalize()
    assert result is not None
