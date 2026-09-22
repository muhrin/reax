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
import os
import time
from unittest.mock import Mock

import jax.numpy as jnp
import pytest

from reax import exceptions, stages
from reax.listeners import ModelCheckpoint


def _identity(x):
    return x


def _make_trainer(tmp_path, *, is_global_zero=True, loggers=()):
    """Build a fake trainer whose ``save_checkpoint`` really writes a file."""

    def _save(filepath, weights_only):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "wb") as fh:
            fh.write(b"ckpt")

    trainer = Mock()
    trainer.is_global_zero = is_global_zero
    trainer.default_root_dir = str(tmp_path)
    trainer.global_updates = 0
    trainer.current_epoch = 0
    trainer.loggers = list(loggers)
    trainer.save_checkpoint.side_effect = _save
    trainer.strategy.barrier = Mock()
    return trainer


class TestInit:
    def test_default_triggers(self):
        ckpt = ModelCheckpoint()
        assert ckpt._every_n_epochs == 1
        assert ckpt._every_n_train_steps == 0
        assert ckpt._train_time_interval is None

    def test_explicit_triggers(self):
        ckpt = ModelCheckpoint(every_n_train_steps=5)
        assert ckpt._every_n_train_steps == 5
        assert ckpt._every_n_epochs == 0

        ckpt = ModelCheckpoint(every_n_epochs=3)
        assert ckpt._every_n_epochs == 3
        assert ckpt._every_n_train_steps == 0

    def test_invalid_mode(self):
        with pytest.raises(exceptions.MisconfigurationException, match=r"`mode` can be"):
            ModelCheckpoint(mode="banana")

    def test_monitor_mode_starts(self):
        assert ModelCheckpoint(mode="min")._kth_value == float("inf")
        assert ModelCheckpoint(mode="max")._kth_value == float("inf")  # -(-inf)

    def test_properties(self, tmp_path):
        dirpath = str(tmp_path / "ckpt")
        ckpt = ModelCheckpoint(dirpath=dirpath, mode="max", save_top_k=3)
        assert ckpt.dirpath is not None
        assert "ckpt" in ckpt.dirpath
        assert ckpt.save_top_k == 3
        assert ckpt.best_model_path == ""
        assert ckpt.best_model_score is None

    def test_dirpath_expanduser(self, tmp_path):
        # dirpath passed through os.path.realpath/expanduser
        ckpt = ModelCheckpoint(dirpath=str(tmp_path / "sub"))
        assert ckpt.dirpath == os.path.realpath(os.path.expanduser(str(tmp_path / "sub")))


def test_format_checkpoint_name(tmp_path):
    ckpt = ModelCheckpoint(dirpath=str(tmp_path))

    # default name uses epoch/step
    assert ckpt.format_checkpoint_name(
        {"epoch": jnp.array(1), "step": jnp.array(2)}
    ) == os.path.join(ckpt.dirpath, "epoch=1-step=2.ckpt")

    # custom filename with named metrics
    name = ckpt.format_checkpoint_name(
        {"val_loss": jnp.array(0.5), "epoch": jnp.array(3)}, filename="{epoch:02d}-{val_loss:.2f}"
    )
    assert name == os.path.join(ckpt.dirpath, "epoch=03-val_loss=0.50.ckpt")

    # version suffix
    name = ckpt.format_checkpoint_name(
        {"epoch": jnp.array(0), "step": jnp.array(0)}, filename="{epoch}-{step}", version=1
    )
    assert name.endswith("epoch=0-step=0-v1.ckpt")

    # no dirpath -> bare filename (with default auto-insert)
    ckpt2 = ModelCheckpoint()
    assert (
        ckpt2.format_checkpoint_name({"epoch": jnp.array(4), "step": jnp.array(9)})
        == "epoch=4-step=9.ckpt"
    )


def test_format_checkpoint_name_prefix():
    ckpt = ModelCheckpoint(enable_version_counter=False)
    # unknown placeholder defaults to 0 in the rendered name
    name = ckpt._format_checkpoint_name("{missing_metric}", {"missing_metric": jnp.array(0)})
    assert "missing_metric=0" in name
    # prefix is joined with CHECKPOINT_JOIN_CHAR
    name = ckpt._format_checkpoint_name("{epoch}", {"epoch": jnp.array(0)}, prefix="best")
    assert name == "best-epoch=0"


def test_check_monitor_top_k():
    ckpt = ModelCheckpoint(monitor="val_loss", save_top_k=2, mode="min")

    # current is None -> False
    assert ckpt.check_monitor_top_k(None) is False

    # fewer than k models stored -> True
    ckpt._best_k_models = {}
    assert bool(ckpt.check_monitor_top_k(jnp.array(5.0))) is True

    # ordering branch: not better than the kth (worst) -> False
    ckpt._best_k_models = {"a": jnp.array(1.0), "b": jnp.array(2.0)}
    ckpt._kth_best_model_path = "b"
    assert bool(ckpt.check_monitor_top_k(jnp.array(5.0))) is False

    # ordering branch: better than kth -> True
    ckpt._best_k_models = {"a": jnp.array(5.0), "b": jnp.array(9.0)}
    ckpt._kth_best_model_path = "a"
    assert bool(ckpt.check_monitor_top_k(jnp.array(0.5))) is True

    # save_top_k == -1 -> always True once current is not None
    ckpt = ModelCheckpoint(monitor="val_loss", save_top_k=-1)
    assert ckpt.check_monitor_top_k(jnp.array(1.0)) is True


def test_check_monitor_top_k_max_mode():
    ckpt = ModelCheckpoint(monitor="acc", save_top_k=1, mode="max")
    ckpt._best_k_models = {"a": jnp.array(0.5)}
    ckpt._kth_best_model_path = "a"
    assert bool(ckpt.check_monitor_top_k(jnp.array(0.9))) is True
    assert bool(ckpt.check_monitor_top_k(jnp.array(0.1))) is False


def test_should_save_on_train_epoch_end():
    # explicit user preference is honoured
    ckpt = ModelCheckpoint(save_on_train_epoch_end=True)
    trainer = Mock()
    assert ckpt._should_save_on_train_epoch_end(trainer, Mock(spec=stages.Train)) is True

    ckpt = ModelCheckpoint(save_on_train_epoch_end=False)
    assert ckpt._should_save_on_train_epoch_end(trainer, Mock(spec=stages.Train)) is False

    # non-fit stage: defer unless we are on a Train stage
    ckpt = ModelCheckpoint()
    trainer = Mock()
    trainer.stage = "not-a-fit"
    assert ckpt._should_save_on_train_epoch_end(trainer, Mock(spec=stages.Train)) is True
    assert ckpt._should_save_on_train_epoch_end(trainer, Mock(spec=stages.Validate)) is False


def test_should_remove_checkpoint(tmp_path):
    ckpt = ModelCheckpoint(dirpath=str(tmp_path))
    trainer = _make_trainer(tmp_path)

    # same path -> never remove
    assert ckpt._should_remove_checkpoint(trainer, "/a/b", "/a/b") is False
    # previous lives inside the checkpoint dir -> removable
    assert ckpt._should_remove_checkpoint(trainer, str(tmp_path / "x.ckpt"), "current") is True
    # previous outside the dir -> keep it
    assert ckpt._should_remove_checkpoint(trainer, "/elsewhere/x.ckpt", "current") is False


def test_get_dirpath(tmp_path):
    # explicit dirpath wins
    ckpt = ModelCheckpoint(dirpath=str(tmp_path))
    trainer = _make_trainer(tmp_path)
    assert ckpt._get_dirpath(trainer) == ckpt.dirpath

    # falls back to default_root_dir
    ckpt = ModelCheckpoint()
    assert ckpt._get_dirpath(trainer) == str(tmp_path)

    # and finally to empty string
    trainer = _make_trainer(tmp_path)
    trainer.default_root_dir = None
    assert ckpt._get_dirpath(trainer) == ""


def test_get_metric_interpolated_filepath_name_version(tmp_path):
    ckpt = ModelCheckpoint(dirpath=str(tmp_path), enable_version_counter=True)
    trainer = _make_trainer(tmp_path)
    existing = os.path.join(ckpt.dirpath, "epoch=0-step=0.ckpt")
    os.makedirs(ckpt.dirpath, exist_ok=True)
    with open(existing, "wb") as fh:
        fh.write(b"x")

    candidates = {"epoch": jnp.array(0), "step": jnp.array(0)}
    result = ckpt._get_metric_interpolated_filepath_name(trainer, candidates)
    assert result.endswith("epoch=0-step=0-v1.ckpt")

    # with version counter disabled, no bumping
    ckpt = ModelCheckpoint(dirpath=str(tmp_path), enable_version_counter=False)
    result = ckpt._get_metric_interpolated_filepath_name(trainer, candidates)
    assert result == existing


def test_save_none_monitor_checkpoint(tmp_path):
    ckpt = ModelCheckpoint(dirpath=str(tmp_path), enable_version_counter=False, save_top_k=1)
    trainer = _make_trainer(tmp_path)
    candidates = {"epoch": jnp.array(0), "step": jnp.array(0)}
    ckpt._save_topk_checkpoint(trainer, candidates)

    assert ckpt.best_model_path
    assert os.path.exists(ckpt.best_model_path)

    # a second, worse/identical save replaces the previous (save_top_k == 1)
    previous = ckpt.best_model_path
    candidates = {"epoch": jnp.array(1), "step": jnp.array(1)}
    ckpt._save_topk_checkpoint(trainer, candidates)
    assert ckpt.best_model_path != previous
    assert not os.path.exists(previous)
    assert os.path.exists(ckpt.best_model_path)


def test_save_topk_zero_saves_nothing(tmp_path):
    ckpt = ModelCheckpoint(dirpath=str(tmp_path), save_top_k=0)
    trainer = _make_trainer(tmp_path)
    os.makedirs(ckpt.dirpath, exist_ok=True)
    candidates = {"epoch": jnp.array(0), "step": jnp.array(0)}
    ckpt._save_topk_checkpoint(trainer, candidates)
    assert ckpt.best_model_path == ""
    assert os.listdir(ckpt.dirpath) == []


def test_save_monitor_checkpoint_verbose_not_topk(tmp_path):
    ckpt = ModelCheckpoint(dirpath=str(tmp_path), monitor="val_loss", save_top_k=1, verbose=True)
    trainer = _make_trainer(tmp_path)
    os.makedirs(ckpt.dirpath, exist_ok=True)

    # seed the store so the current score is not better than the kth
    ckpt._best_k_models = {"best": jnp.array(0.1)}
    ckpt._kth_best_model_path = "best"
    candidates = {"val_loss": jnp.array(0.9), "epoch": jnp.array(0), "step": jnp.array(0)}
    # no new file written because not in the top-k
    before = set(os.listdir(ckpt.dirpath))
    ckpt._save_monitor_checkpoint(trainer, candidates)
    assert set(os.listdir(ckpt.dirpath)) == before

    # a genuinely better score IS saved
    candidates = {"val_loss": jnp.array(0.05), "epoch": jnp.array(0), "step": jnp.array(0)}
    ckpt._save_monitor_checkpoint(trainer, candidates)
    assert jnp.isclose(ckpt.best_model_score, 0.05)
    assert os.path.exists(ckpt.best_model_path)


def test_update_best_and_save_evicts_worst(tmp_path):
    ckpt = ModelCheckpoint(dirpath=str(tmp_path), monitor="val_loss", save_top_k=1, mode="min")
    trainer = _make_trainer(tmp_path)
    os.makedirs(ckpt.dirpath, exist_ok=True)

    def _candidates(v):
        return {"val_loss": jnp.array(v), "epoch": jnp.array(0), "step": jnp.array(0)}

    ckpt._save_monitor_checkpoint(trainer, _candidates(5.0))
    first_path = ckpt.best_model_path

    # worse than what we saw (save_top_k=1) -> not saved
    ckpt._save_monitor_checkpoint(trainer, _candidates(6.0))
    assert ckpt.best_model_path == first_path
    assert len(ckpt._best_k_models) == 1

    # better than what we saw -> store updated, best score improved
    ckpt._save_monitor_checkpoint(trainer, _candidates(1.0))
    assert jnp.isclose(ckpt.best_model_score, 1.0)
    assert os.path.exists(ckpt.best_model_path)
    assert len(ckpt._best_k_models) == 1


def test_save_last_checkpoint_replaces_previous(tmp_path):
    ckpt = ModelCheckpoint(dirpath=str(tmp_path), save_last=True, enable_version_counter=False)
    trainer = _make_trainer(tmp_path)
    os.makedirs(ckpt.dirpath, exist_ok=True)

    candidates = {"epoch": jnp.array(0), "step": jnp.array(0)}
    ckpt._save_last_checkpoint(trainer, candidates)
    first = ckpt.last_model_path
    assert first.endswith("last.ckpt")
    assert os.path.exists(first)

    # replace the previous "last" (same versionless name)
    ckpt._save_last_checkpoint(trainer, candidates)
    assert ckpt.last_model_path == first
    assert not os.path.exists(first + "-copy") if os.path.exists(first + "-copy") else True
    assert os.path.exists(first)


def test_save_last_link(tmp_path):
    ckpt = ModelCheckpoint(dirpath=str(tmp_path), save_last="link", save_top_k=1)
    trainer = _make_trainer(tmp_path, is_global_zero=True)
    os.makedirs(ckpt.dirpath, exist_ok=True)

    candidates = {"epoch": jnp.array(0), "step": jnp.array(0)}
    # first save establishes _last_checkpoint_saved (a real file)
    ckpt._save_none_monitor_checkpoint(trainer, candidates)
    ckpt._save_last_checkpoint(trainer, candidates)

    # "last" should now be a symlink pointing at the real saved checkpoint
    saved = next(p for p in os.listdir(ckpt.dirpath) if p.endswith(".ckpt") and p != "last.ckpt")
    assert os.path.islink(ckpt.last_model_path)
    assert os.path.realpath(ckpt.last_model_path) == os.path.realpath(
        os.path.join(ckpt.dirpath, saved)
    )


def test_do_save_epochs_gating(tmp_path):
    ckpt = ModelCheckpoint(
        dirpath=str(tmp_path), monitor=None, every_n_epochs=2, enable_version_counter=False
    )
    trainer = _make_trainer(tmp_path)
    os.makedirs(ckpt.dirpath, exist_ok=True)
    candidates = {"epoch": jnp.array(0), "step": jnp.array(0)}

    # epoch 0 (current_epoch 0) -> (0+1) % 2 != 0 -> no topk save, but last is saved
    trainer.current_epoch = 0
    ckpt._do_save(trainer, candidates)
    assert ckpt.best_model_path == ""

    # epoch 1 -> (1+1) % 2 == 0 -> topk save happens
    trainer.current_epoch = 1
    candidates = {"epoch": jnp.array(1), "step": jnp.array(0)}
    ckpt._do_save(trainer, candidates)
    assert ckpt.best_model_path != ""


def test_monitor_candidates_fallback(tmp_path):
    ckpt = ModelCheckpoint()
    stage = Mock(spec=stages.EpochStage)
    stage.listener_metrics = {}
    trainer = Mock(current_epoch=7, global_updates=42)
    out = ckpt._monitor_candidates(stage, trainer)
    assert int(out["epoch"]) == 7
    assert int(out["step"]) == 42


def test_on_fit_start_sets_time():
    ckpt = ModelCheckpoint()
    ckpt._last_time_checked = None
    before = time.monotonic()
    ckpt.on_fit_start(Mock(), Mock(spec=stages.Fit))
    assert ckpt._last_time_checked is not None
    assert ckpt._last_time_checked >= before


def test_resolve_ckpt_dir(tmp_path):
    # 1. explicit dirpath short-circuits
    ckpt = ModelCheckpoint(dirpath=str(tmp_path / "mine"))
    trainer = _make_trainer(tmp_path)
    assert ckpt._ModelCheckpoint__resolve_ckpt_dir(trainer) == ckpt.dirpath

    # 2. use logger save_dir + name + version
    logger = Mock(save_dir=str(tmp_path / "logs"), name="testlogger", version=3)
    ckpt = ModelCheckpoint()
    trainer = _make_trainer(tmp_path, loggers=[logger])
    path = ckpt._ModelCheckpoint__resolve_ckpt_dir(trainer)
    assert os.path.basename(path) == "checkpoints"
    assert "testlogger" in path
    assert "version_3" in path

    # 3. logger with no save_dir uses default_root_dir, version passed through as-is
    logger = Mock(save_dir=None, name="testlogger", version="abc")
    ckpt = ModelCheckpoint()
    trainer = _make_trainer(tmp_path, loggers=[logger])
    path = ckpt._ModelCheckpoint__resolve_ckpt_dir(trainer)
    assert os.path.basename(path) == "checkpoints"
    assert "testlogger" in path and "abc" in path

    # 4. no loggers -> default_root_dir/checkpoints
    ckpt = ModelCheckpoint()
    trainer = _make_trainer(tmp_path)
    path = ckpt._ModelCheckpoint__resolve_ckpt_dir(trainer)
    assert os.path.basename(path) == "checkpoints"


def test_warn_if_dir_not_empty(tmp_path):
    # empty dir -> no warning
    d = tmp_path / "empty"
    d.mkdir()
    ckpt = ModelCheckpoint(dirpath=str(d))
    ckpt._ModelCheckpoint__warn_if_dir_not_empty(str(d))

    # non-empty dir -> warning is emitted (rank_zero_warn)
    d2 = tmp_path / "nonempty"
    d2.mkdir()
    (d2 / "old.ckpt").write_text("old")
    ckpt = ModelCheckpoint(dirpath=str(d2))
    with pytest.warns(UserWarning, match="is not empty"):
        ckpt._ModelCheckpoint__warn_if_dir_not_empty(str(d2))


def _train_stage(fast_dev_run=False, enable_checkpointing=True, listener_metrics=None):
    stage = Mock(spec=stages.EpochStage)
    stage.fast_dev_run = fast_dev_run
    stage.enable_checkpointing = enable_checkpointing
    stage.listener_metrics = listener_metrics or {}
    return stage


def test_should_skip_saving_checkpoint(tmp_path):
    ckpt = ModelCheckpoint(dirpath=str(tmp_path))
    trainer = _make_trainer(tmp_path)
    trainer.sanity_checking = False
    ckpt._last_global_step_saved = -1
    stage = _train_stage()

    assert bool(ckpt._should_skip_saving_checkpoint(trainer, stage)) is False

    assert (
        bool(ckpt._should_skip_saving_checkpoint(trainer, _train_stage(fast_dev_run=True))) is True
    )

    assert (
        bool(ckpt._should_skip_saving_checkpoint(trainer, _train_stage(enable_checkpointing=False)))
        is True
    )

    trainer.sanity_checking = True
    assert bool(ckpt._should_skip_saving_checkpoint(trainer, stage)) is True
    trainer.sanity_checking = False

    ckpt._last_global_step_saved = 5
    trainer.global_updates = 5
    assert bool(ckpt._should_skip_saving_checkpoint(trainer, stage)) is True
    trainer.global_updates = 6
    assert bool(ckpt._should_skip_saving_checkpoint(trainer, stage)) is False


def test_on_train_epoch_end_saves_when_validated(tmp_path):
    ckpt = ModelCheckpoint(dirpath=str(tmp_path), save_on_train_epoch_end=True)
    os.makedirs(ckpt.dirpath, exist_ok=True)
    trainer = _make_trainer(tmp_path)
    trainer.sanity_checking = False
    ckpt._last_global_step_saved = -1
    trainer.global_updates = 1
    trainer.current_epoch = 0

    # explicit True short-circuits the stage-type validation-mode branch;
    # stage must be a real Train for the isinstance check in _do_save
    stage = Mock(spec=stages.Train)
    stage.fast_dev_run = False
    stage.listener_metrics = {"val_loss": jnp.array(0.5)}
    ckpt.on_train_epoch_end(trainer, stage)

    assert any(p.endswith(".ckpt") for p in os.listdir(ckpt.dirpath))


def test_on_train_batch_end_every_n_train_steps(tmp_path):
    ckpt = ModelCheckpoint(dirpath=str(tmp_path), every_n_train_steps=2)
    os.makedirs(ckpt.dirpath, exist_ok=True)
    trainer = _make_trainer(tmp_path)
    trainer.sanity_checking = False
    ckpt._last_global_step_saved = -1
    ckpt._last_time_checked = None

    stage = _train_stage()
    trainer.global_updates = 1
    ckpt.on_train_batch_end(trainer, stage)
    assert not [p for p in os.listdir(ckpt.dirpath) if p.endswith(".ckpt")]

    trainer.global_updates = 2
    ckpt.on_train_batch_end(trainer, stage)
    assert [p for p in os.listdir(ckpt.dirpath) if p.endswith(".ckpt")]


def test_on_train_batch_end_time_interval(tmp_path):
    import datetime

    ckpt = ModelCheckpoint(
        dirpath=str(tmp_path),
        every_n_train_steps=1,
        train_time_interval=datetime.timedelta(1),
    )
    os.makedirs(ckpt.dirpath, exist_ok=True)
    trainer = _make_trainer(tmp_path)
    trainer.sanity_checking = False
    ckpt._last_global_step_saved = -1
    ckpt._last_time_checked = time.monotonic() - 10
    trainer.engine.broadcast.side_effect = _identity

    stage = _train_stage()
    trainer.global_updates = 1
    ckpt.on_train_batch_end(trainer, stage)
    assert any(p.endswith(".ckpt") for p in os.listdir(ckpt.dirpath))
