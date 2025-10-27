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
# https://github.com/Lightning-AI/pytorch-lightning/blob/9177ec09caadcf88859e1f1e3e10a18e8832069a/tests/tests_pytorch/trainer/test_dataloaders.py
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
import os
from unittest.mock import Mock, call, patch

from lightning_utilities.test import warning
import numpy
import pytest

import reax
from reax.demos import boring_classes

# from tests_pytorch.helpers.dataloaders import (
#     CustomInfDataloader,
#     CustomNotImplementedErrorDataloader,
# )
# from tests_pytorch.helpers.runif import RunIf


class MultiValDataLoaderBoringModel(boring_classes.BoringModel):
    def val_dataloader(self):
        # return [
        #     reax.ReaxDataLoader(boring_classes.RandomDataset(32, 64)),
        #     reax.ReaxDataLoader(boring_classes.RandomDataset(32, 64), batch_size=8),
        # ]
        return reax.ReaxDataLoader(boring_classes.RandomDataset(32, 64))

    # def validation_step(self, batch, batch_idx, dataloader_idx):
    #     return super().validation_step(batch, batch_idx)
    def validation_step(self, batch, batch_idx):
        return super().validation_step(batch, batch_idx)


class MultiTestDataLoaderBoringModel(boring_classes.BoringModel):
    def test_dataloader(self):
        # return [
        #     reax.ReaxDataLoader(boring_classes.RandomDataset(32, 64)),
        #     reax.ReaxDataLoader(boring_classes.RandomDataset(32, 64), batch_size=8),
        # ]
        return reax.ReaxDataLoader(boring_classes.RandomDataset(32, 64))

    # def test_step(self, batch, batch_idx, dataloader_idx):
    #     return super().test_step(batch, batch_idx)
    def test_step(self, batch, batch_idx):
        return super().test_step(batch, batch_idx)


class MultiEvalDataLoaderModel(MultiValDataLoaderBoringModel, MultiTestDataLoaderBoringModel):
    pass


def test_fit_train_loader_only(tmp_path):
    model = boring_classes.BoringModel()
    train_dataloader = model.train_dataloader()

    model.train_dataloader = None
    model.val_dataloader = None
    model.test_dataloader = None

    model.validation_step = None
    model.test_step = None

    trainer = reax.Trainer(default_root_dir=tmp_path)
    trainer.fit(model, train_dataloaders=train_dataloader, fast_dev_run=True)


def test_fit_val_loader_only(tmp_path):
    model = boring_classes.BoringModel()
    train_dataloader = model.train_dataloader()
    val_dataloader = model.val_dataloader()

    model.train_dataloader = None
    model.val_dataloader = None
    model.test_dataloader = None

    model.test_step = None

    trainer = reax.Trainer(default_root_dir=tmp_path)
    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, fast_dev_run=True
    )


@pytest.mark.parametrize("dataloader_options", [{"val_check_interval": 10000}])
def test_dataloader_config_errors_runtime(tmp_path, dataloader_options):
    model = boring_classes.BoringModel()
    trainer = reax.Trainer(default_root_dir=tmp_path)
    with pytest.raises(
        ValueError, match="less than or equal to the number of the training batches"
    ):
        trainer.fit(model, max_epochs=1, **dataloader_options)


#
# @pytest.mark.parametrize(
#     "dataloader_options",
#     [
#         {"limit_train_batches": -0.1},
#         {"limit_train_batches": 1.2},
#         {"limit_val_batches": -0.1},
#         {"limit_val_batches": 1.2},
#         {"limit_test_batches": -0.1},
#         {"limit_test_batches": 1.2},
#         {"val_check_interval": -0.1},
#         {"val_check_interval": 1.2},
#         {"overfit_batches": -0.1},
#         {"overfit_batches": 1.2},
#     ],
# )
# def test_dataloader_config_errors_init(tmp_path, dataloader_options):
#     with pytest.raises(reax.exceptions.MisconfigurationException, match="passed invalid value"):
#         reax.Trainer(default_root_dir=tmp_path, max_epochs=1, **dataloader_options)


# def test_multiple_val_dataloader(tmp_path):
#     """Verify multiple val_dataloader."""
#     model = MultiValDataLoaderBoringModel()
#     trainer = reax.Trainer(default_root_dir=tmp_path)
#     trainer.fit(model, max_epochs=1, limit_val_batches=0.3, limit_train_batches=1.0)
#
#     # verify there are 2 val loaders
#     assert len(trainer.val_dataloaders) == 2, "Multiple val_dataloaders not initiated properly"
#

# @pytest.mark.parametrize("ckpt_path", [None, "best", "specific"])
# def test_multiple_eval_dataloader(tmp_path, ckpt_path):
#     """Verify multiple evaluation dataloaders."""
#     model = MultiEvalDataLoaderModel()
#     trainer = Trainer(
#         default_root_dir=tmp_path, max_epochs=1, limit_val_batches=10, limit_train_batches=100
#     )
#     trainer.fit(model)
#     ckpt_path = (
#         trainer.checkpoint_callback.best_model_path if ckpt_path == "specific" else ckpt_path
#     )
#
#     trainer.validate(ckpt_path=ckpt_path, verbose=False)
#     # verify there are 2 loaders
#     assert len(trainer.val_dataloaders) == 2
#
#     trainer.test(ckpt_path=ckpt_path, verbose=False)
#     assert len(trainer.test_dataloaders) == 2


def test_train_dataloader_passed_to_fit(tmp_path):
    """Verify that train dataloader can be passed to fit."""
    # only train passed to fit
    model = boring_classes.BoringModel()
    train_loader = model.train_dataloader()
    trainer = reax.Trainer(default_root_dir=tmp_path)
    fit_options = {"train_dataloaders": train_loader}
    stage = trainer.fit(model, fast_dev_run=2, **fit_options)
    assert stage.num_training_batches == 2
    assert isinstance(stage.train_dataloader, reax.data.DeviceDataLoader)
    assert stage.train_dataloader.parent == train_loader


@pytest.mark.parametrize(
    "ckpt_path",
    [
        None,
        # TODO: not supported yet "best",
        "specific",
    ],
)
@pytest.mark.parametrize(
    "n",
    [
        1,
        # TODO: Don't support multiple dataloaders yet 2
    ],
)
def test_dataloaders_passed_to_fn(tmp_path, ckpt_path, n):
    """Verify that dataloaders can be passed."""
    train_dataloaders = reax.ReaxDataLoader(boring_classes.RandomDataset(32, 64))
    if n == 1:
        model = boring_classes.BoringModel()
        eval_dataloaders = reax.ReaxDataLoader(boring_classes.RandomDataset(32, 64))
    else:
        model = MultiEvalDataLoaderModel()
        eval_dataloaders = [
            reax.ReaxDataLoader(boring_classes.RandomDataset(32, 64)),
            reax.ReaxDataLoader(boring_classes.RandomDataset(32, 64)),
        ]

    # multiple val dataloaders passed to fit
    trainer = reax.Trainer(default_root_dir=tmp_path)

    fit = trainer.fit(
        model,
        train_dataloaders=train_dataloaders,
        val_dataloaders=eval_dataloaders,
        max_epochs=1,
        limit_val_batches=0.1,
        limit_train_batches=0.2,
    )
    if n > 1:
        assert len(fit.val_dataloaders) == n
    else:
        assert isinstance(fit.val_dataloaders, reax.data.DeviceDataLoader)
        assert isinstance(fit.val_dataloaders.parent, reax.ReaxDataLoader)

    if ckpt_path == "specific":
        ckpt_path = trainer.checkpoint_listener.best_model_path

    test = trainer.test(model, dataloaders=eval_dataloaders, ckpt_path=ckpt_path)
    if n > 1:
        assert len(test.dataloaders) == n
    else:
        assert isinstance(test.dataloader, reax.data.DeviceDataLoader)
        assert isinstance(test.dataloader.parent, reax.ReaxDataLoader)

    validate = trainer.validate(model, dataloaders=eval_dataloaders, ckpt_path=ckpt_path)
    if n > 1:
        assert len(validate.dataloaders) == n
    else:
        assert isinstance(validate.dataloaders, reax.data.DeviceDataLoader)
        assert isinstance(validate.dataloaders.parent, reax.ReaxDataLoader)


class DummyModel(boring_classes.BoringModel):
    def training_step(self, batch, batch_idx):
        self.log("loss", self.global_updates)
        return super().training_step(batch, batch_idx)

    def on_validation_epoch_end(self, *_):
        self.log("val_log", self.current_epoch)


class Counter(reax.TrainerListener):
    def __init__(self):
        super().__init__()
        self.train_epoch_count = 0
        self.val_epoch_count = 0
        self.test_epoch_count = 0
        self.train_batches_seen = 0
        self.val_batches_seen = 0
        self.test_batches_seen = 0

    def on_train_batch_start(self, *_):
        self.train_batches_seen += 1

    def on_train_epoch_start(self, *_):
        self.train_epoch_count += 1

    def on_validation_batch_start(self, *_):
        self.val_batches_seen += 1

    def on_test_batch_start(self, *_):
        self.test_batches_seen += 1

    def on_validation_epoch_start(self, *_):
        self.val_epoch_count += 1

    def on_test_epoch_start(self, *_):
        self.test_epoch_count += 1


#
#
# def test_inf_dataloaders_with_limit_percent_batches(tmp_path):
#     """Verify inf train, val & test dataloaders (e.g. IterableDataset) passed with batch limit in percent."""
#     epoch_cb = Counter()
#     trainer = Trainer(
#         default_root_dir=tmp_path,
#         num_sanity_val_steps=0,
#         max_epochs=1,
#         callbacks=[epoch_cb],
#         limit_train_batches=1.0,
#         limit_val_batches=1.0,
#         limit_test_batches=1.0,
#     )
#     model = DummyModel()
#
#     batch_size = 8
#     train_dl = DataLoader(dataset=RandomIterableDataset(32, 128), batch_size=batch_size)
#     val_dl = DataLoader(dataset=RandomIterableDataset(32, 128), batch_size=batch_size)
#     test_dl = DataLoader(dataset=RandomIterableDataset(32, 128), batch_size=batch_size)
#
#     num_batches = 128 / batch_size
#     for dl in (train_dl, val_dl, test_dl):
#         if has_len_all_ranks(dl, trainer.strategy):
#             assert len(dl) == num_batches
#         else:
#             assert sum(1 for _ in dl) == num_batches
#
#     trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
#
#     assert trainer.num_training_batches == float("inf")
#     assert epoch_cb.train_epoch_count == 1
#
#     assert trainer.num_val_batches[0] == float("inf")
#     assert epoch_cb.val_epoch_count == 1
#
#     trainer.test(model, dataloaders=test_dl)
#     assert trainer.num_test_batches[0] == float("inf")
#     assert epoch_cb.test_epoch_count == 1
#


@pytest.mark.skip(reason="Don't support anticipating end of dataloader yet")
@pytest.mark.parametrize(
    ("dataset", "limit_train_batches"),
    [
        (boring_classes.RandomDataset(32, 128), 10),
        (boring_classes.RandomIterableDataset(32, 128), 10),
        # TODO: (boring_classes.RandomIterableDatasetWithLen(32, 128), 10),
    ],
)
def test_dataloaders_with_limit_train_batches(tmp_path, dataset, limit_train_batches):
    """Verify inf train, val & test dataloaders (e.g. IterableDataset) passed with batch limit as number."""
    epoch_listener = Counter()
    max_epochs = 2
    trainer = reax.Trainer(default_root_dir=tmp_path, listeners=[epoch_listener])
    model = DummyModel()

    batch_size = 8
    train_dl = reax.ReaxDataLoader(dataset=dataset, batch_size=batch_size)
    val_dl = reax.ReaxDataLoader(dataset=dataset, batch_size=batch_size)

    fit = trainer.fit(
        model,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl,
        num_sanity_val_steps=0,
        max_epochs=max_epochs,
        limit_train_batches=limit_train_batches,
    )

    assert fit.num_training_batches == limit_train_batches
    assert epoch_listener.train_epoch_count == max_epochs
    assert epoch_listener.train_batches_seen == limit_train_batches * max_epochs


@pytest.mark.skip(reason="Don't support anticipating end of dataloader yet")
@pytest.mark.parametrize(
    "dataset",
    [
        boring_classes.RandomDataset(32, 128),
        boring_classes.RandomIterableDataset(32, 128),
        # boring_classes.RandomIterableDatasetWithLen(32, 128),
    ],
)
def test_dataloaders_with_limit_val_batches(tmp_path, dataset):
    """Verify inf train, val & test dataloaders (e.g. IterableDataset) passed with batch limit as number."""
    epoch_listener = Counter()
    listeners = [epoch_listener]
    enable_checkpointing = False

    max_epochs = 2
    limit_val_batches = 10
    trainer = reax.Trainer(
        default_root_dir=tmp_path, listeners=listeners, enable_checkpointing=enable_checkpointing
    )
    model = DummyModel()

    batch_size = 8
    train_dl = reax.ReaxDataLoader(dataset=dataset, batch_size=batch_size)
    val_dl = reax.ReaxDataLoader(dataset=dataset, batch_size=batch_size)

    fit = trainer.fit(
        model,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl,
        num_sanity_val_steps=0,
        max_epochs=max_epochs,
        limit_val_batches=limit_val_batches,
    )
    assert fit.limit_val_batches == limit_val_batches
    assert epoch_listener.val_epoch_count == max_epochs
    assert epoch_listener.val_batches_seen == limit_val_batches * max_epochs


@pytest.mark.skip(reason="Not fully ported yet")
@pytest.mark.parametrize(
    "dataset",
    [
        boring_classes.RandomDataset(32, 128),
        boring_classes.RandomIterableDataset(32, 128),
        # TODO: boring_classes.RandomIterableDatasetWithLen(32, 128),
    ],
)
def test_datasets_dataloaders_with_limit_num_batches(tmp_path, dataset):
    """Verify inf train, val & test dataloaders (e.g. IterableDataset) passed with batch limit as number."""
    epoch_cb = Counter()
    max_epochs = 2
    limit_batches = 10
    trainer = reax.Trainer(default_root_dir=tmp_path, listeners=[epoch_cb])
    model = DummyModel()

    batch_size = 8
    train_dl = reax.data.ReaxDataLoader(dataset=dataset, batch_size=batch_size)
    val_dl = reax.data.ReaxDataLoader(dataset=dataset, batch_size=batch_size)
    test_dl = reax.data.ReaxDataLoader(dataset=dataset, batch_size=batch_size)

    fit = trainer.fit(
        model,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl,
        max_epochs=max_epochs,
        limit_train_batches=limit_batches,
        limit_val_batches=limit_batches,
        num_sanity_val_steps=0,
    )

    assert fit.num_training_batches == limit_batches
    assert fit.num_val_batches[0] == limit_batches
    assert epoch_cb.train_epoch_count == max_epochs
    assert epoch_cb.train_batches_seen == limit_batches * max_epochs
    assert epoch_cb.val_epoch_count == max_epochs
    assert epoch_cb.val_batches_seen == limit_batches * max_epochs

    test = trainer.test(model, dataloaders=test_dl, limit_batches=limit_batches)
    assert test.num_batches[0] == limit_batches
    assert epoch_cb.test_epoch_count == 1


@pytest.mark.skip(reason="Not fully ported yet")
@pytest.mark.parametrize(
    ("limit_train_batches", "limit_val_batches", "limit_test_batches"),
    [(1.0, 1.0, 1.0), (0.2, 0.4, 0.4)],
)
def test_dataloaders_with_limit_percent_batches(
    tmp_path, limit_train_batches, limit_val_batches, limit_test_batches
):
    """Verify num_batches for train, val & test dataloaders passed with batch limit in percent."""
    model = MultiEvalDataLoaderModel()
    # train, multiple val and multiple test passed with percent_check
    trainer = reax.Trainer(default_root_dir=tmp_path)
    fit = trainer.fit(
        model,
        max_epochs=1,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
    )
    expected_train_batches = int(len(fit.train_dataloader) * limit_train_batches)
    expected_val_batches = [
        int(len(dataloader) * limit_val_batches) for dataloader in fit.val_dataloaders
    ]
    assert fit.num_training_batches == expected_train_batches
    assert fit.num_val_batches == expected_val_batches

    test = trainer.test(model, limit_batches=limit_test_batches)
    expected_test_batches = [
        int(len(dataloader) * limit_test_batches) for dataloader in test.dataloaders
    ]
    assert test.num_batches == expected_test_batches


@pytest.mark.skip(reason="Not updated yet")
@pytest.mark.parametrize(
    ("limit_train_batches", "limit_val_batches", "limit_test_batches"), [(1, 2, 3), (1, 2, 1e50)]
)
def test_dataloaders_with_limit_num_batches(
    tmp_path, limit_train_batches, limit_val_batches, limit_test_batches
):
    """Verify num_batches for train, val & test dataloaders passed with batch limit as number."""
    model = MultiEvalDataLoaderModel()

    # train, multiple val and multiple test passed with percent_check
    trainer = reax.Trainer(default_root_dir=tmp_path)

    with patch.object(
        trainer.fit_loop.epoch_loop.val_loop,
        "_evaluation_step",
        wraps=trainer.fit_loop.epoch_loop.val_loop._evaluation_step,
    ) as mocked:
        fit = trainer.fit(
            model,
            max_epochs=1,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            num_sanity_val_steps=0,
        )
        assert fit.num_training_batches == limit_train_batches
        assert fit.num_val_batches == [limit_val_batches] * len(fit.val_dataloaders)
        assert mocked.call_count == limit_val_batches * len(fit.val_dataloaders)

    with patch.object(
        trainer.test_loop,
        "_evaluation_step",
        wraps=trainer.test_loop._evaluation_step,
    ) as mocked:
        test = trainer.test(model, limit_batches=limit_test_batches)
        test_dataloader_lengths = [len(x) for x in model.test_dataloader()]
        if limit_test_batches > 1e10:
            # when the limit is greater than the number of test batches it should be the num in loaders
            assert test.num_batches == test_dataloader_lengths
            assert mocked.call_count == sum(test_dataloader_lengths)
        else:
            assert test.num_batches == [limit_test_batches] * len(test.dataloaders)
            assert mocked.call_count == limit_test_batches * len(test.dataloaders)


@pytest.mark.parametrize("fast_dev_run", [True, 1, 3, -1])
def test_dataloaders_with_fast_dev_run(tmp_path, fast_dev_run):
    """Verify num_batches for train, val & test dataloaders passed with fast_dev_run."""
    model = MultiEvalDataLoaderModel()
    trainer = reax.Trainer(default_root_dir=tmp_path)
    fit_options = {"max_epochs": 2, "fast_dev_run": fast_dev_run}

    if fast_dev_run == -1:
        with pytest.raises(reax.exceptions.MisconfigurationException, match="should be >= 0"):
            trainer.fit(model, **fit_options)
    else:
        fit = trainer.fit(model, **fit_options)

        # fast_dev_run is set to True when it is 1
        if fast_dev_run == 1:
            fast_dev_run = True

        assert fit.fast_dev_run is fast_dev_run

        if fast_dev_run is True:
            fast_dev_run = 1

        assert fit.limit_train_batches == fast_dev_run
        assert fit.limit_val_batches == fast_dev_run

        assert fit.num_sanity_val_steps == 0
        assert fit.max_epochs == 1

        assert fit.enable_validation
        assert fit.num_training_batches == fast_dev_run
        # TODO: assert fit.num_val_batches == [fast_dev_run] * len(fit.val_dataloaders)
        assert fit.num_val_batches == fast_dev_run

        test = trainer.test(model, fast_dev_run=fast_dev_run)
        assert test.limit_batches == fast_dev_run
        # TODO: assert test.num_batches == [fast_dev_run] * len(test.dataloaders)
        assert test.num_batches == fast_dev_run


@pytest.mark.parametrize(
    "ckpt_path",
    [
        None,
        # TODO: not supported yet "best",
        "specific",
    ],
)
def test_mixing_of_dataloader_options(tmp_path, ckpt_path):
    """Verify that dataloaders can be passed to fit."""
    model = boring_classes.BoringModel()
    eval_dataloader = reax.ReaxDataLoader(boring_classes.RandomDataset(32, 64))

    # fit model
    trainer = reax.Trainer(default_root_dir=tmp_path)
    fit = trainer.fit(
        model,
        val_dataloaders=eval_dataloader,
        max_epochs=1,
        limit_val_batches=0.1,
        limit_train_batches=0.2,
    )
    assert (
        isinstance(fit.val_dataloaders, reax.data.DeviceDataLoader)
        and fit.val_dataloaders.parent == eval_dataloader
    )

    ckpt_path = (
        trainer.checkpoint_listener.best_model_path if ckpt_path == "specific" else ckpt_path
    )
    test = trainer.test(model, dataloaders=eval_dataloader, ckpt_path=ckpt_path)
    assert isinstance(test.dataloaders, reax.data.DeviceDataLoader)
    assert test.dataloaders.parent == eval_dataloader


# def test_warning_on_zero_len_dataloader():
#     """Test that a warning is raised if a zero-length dataloader is defined."""
#     model = boring_classes.BoringModel()
#     trainer = reax.Trainer()
#     trainer.strategy.connect(model)
#     train_dataloader = reax.ReaxDataLoader(boring_classes.RandomDataset(32, 0))
#     val_dataloader = reax.ReaxDataLoader(boring_classes.RandomDataset(32, 0))
#     trainer._data_connector.attach_data(
#         model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
#     )
#
#     with pytest.warns(UserWarning, match="Total length of `CombinedLoader` across ranks is zero"):
#         trainer.fit_loop.setup_data()
#     assert trainer.num_training_batches == 0
#
#     trainer.state.fn = "validate"
#     with pytest.warns(UserWarning, match="Total length of `DataLoader` across ranks is zero"):
#         trainer.validate_loop.setup_data()
#     assert trainer.num_val_batches == [0]


# @RunIf(skip_windows=True)
# @pytest.mark.parametrize("ckpt_path", [None, "best", "specific"])
# @pytest.mark.parametrize("stage", ["train", "test", "val"])
# @patch("lightning.fabric.utilities.data._num_cpus_available", return_value=4)
# def test_warning_with_few_workers(_, tmp_path, ckpt_path, stage):
#     """Test that error is raised if dataloader with only a few workers is used."""
#     model = BoringModel()
#
#     train_dl = model.train_dataloader()
#     train_dl.num_workers = 0
#
#     val_dl = model.val_dataloader()
#     val_dl.num_workers = 0
#
#     trainer = Trainer(
#         default_root_dir=tmp_path, max_epochs=1, limit_val_batches=0.1, limit_train_batches=0.2
#     )
#
#     with pytest.warns(UserWarning, match=f"The '{stage}_dataloader' does not have many workers"):
#         if stage == "test":
#             if ckpt_path in ("specific", "best"):
#                 trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
#             ckpt_path = (
#                 trainer.checkpoint_callback.best_model_path
#                 if ckpt_path == "specific"
#                 else ckpt_path
#             )
#             trainer.test(model, dataloaders=train_dl, ckpt_path=ckpt_path)
#         else:
#             trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
#
#
# @RunIf(skip_windows=True)
# @pytest.mark.parametrize("ckpt_path", [None, "best", "specific"])
# @pytest.mark.parametrize("stage", ["train", "test", "val"])
# @patch("lightning.fabric.utilities.data._num_cpus_available", return_value=4)
# def test_warning_with_few_workers_multi_loader(_, tmp_path, ckpt_path, stage):
#     """Test that a warning is emitted if the dataloader only has a few workers."""
#
#     class CustomModel(MultiEvalDataLoaderModel):
#         def training_step(self, batch, batch_idx):
#             return super().training_step(batch["a_b"][0], batch_idx)
#
#     model = CustomModel()
#     val_dl = DataLoader(RandomDataset(32, 64))
#     val_dl.num_workers = 0
#
#     train_dl = DataLoader(RandomDataset(32, 64))
#     train_dl.num_workers = 0
#
#     train_multi_dl = {"a_b": [train_dl, train_dl], "c_d_e": [train_dl, train_dl, train_dl]}
#     val_multi_dl = [val_dl, val_dl]
#     test_multi_dl = [train_dl, train_dl]
#
#     trainer = Trainer(
#         default_root_dir=tmp_path, max_epochs=1, limit_val_batches=0.1, limit_train_batches=0.2
#     )
#
#     with pytest.warns(
#         UserWarning,
#         match=f"The '{stage}_dataloader' does not have many workers",
#     ):
#         if stage == "test":
#             if ckpt_path in ("specific", "best"):
#                 trainer.fit(model, train_dataloaders=train_multi_dl, val_dataloaders=val_multi_dl)
#             ckpt_path = (
#                 trainer.checkpoint_callback.best_model_path
#                 if ckpt_path == "specific"
#                 else ckpt_path
#             )
#             trainer.test(model, dataloaders=test_multi_dl, ckpt_path=ckpt_path)
#         else:
#             trainer.fit(model, train_dataloaders=train_multi_dl, val_dataloaders=val_multi_dl)
#
#
# class NumpyRandomDataset(Dataset):
#     # this dataset uses numpy instead of torch to produce random numbers
#     size = 16
#
#     def __getitem__(self, index):
#         return numpy.random.randint(0, 100, 3)
#
#     def __len__(self):
#         return self.size
#
#
# def _user_worker_init_fn(_):
#     pass
#
#
# def test_auto_add_worker_init_fn():
#     """Test Trainer adds a default worker_init_fn to the dataloader when seed_everything() is used."""
#     dataset = Mock()
#     dataloader = DataLoader(dataset)
#
#     # without pl.seed_everything()
#     _auto_add_worker_init_fn(dataloader, 0)
#     assert dataloader.worker_init_fn is None
#
#     # with forcefully avoiding it
#     seed_everything(0, workers=False)
#     _auto_add_worker_init_fn(dataloader, 0)
#     assert dataloader.worker_init_fn is None
#
#     # when user already has a worker_init_fn
#     user_function = _user_worker_init_fn
#     dataloader.worker_init_fn = user_function
#     _auto_add_worker_init_fn(dataloader, 0)
#     assert dataloader.worker_init_fn is user_function
#     dataloader.worker_init_fn = None
#
#     # main use case
#     seed_everything(0, workers=True)
#     _auto_add_worker_init_fn(dataloader, 0)
#     assert dataloader.worker_init_fn is not None
#
#
# class MultiProcessModel(BoringModel):
#     def __init__(self):
#         super().__init__()
#         self.batches_seen = []
#
#     def training_step(self, batch, batch_idx):
#         self.batches_seen.append(batch)
#         # the actual training step is not needed for the assertions below
#         return super().training_step(torch.rand(1, 32, device=self.device), batch_idx)
#
#     def on_train_epoch_end(self):
#         world_size = 2
#         num_samples = NumpyRandomDataset.size
#         all_batches = torch.cat(self.batches_seen)
#         all_batches = self.all_gather(all_batches)
#         assert all_batches.shape[0] == world_size
#         all_batches = all_batches.view(-1, 3)
#         assert len(torch.unique(all_batches, dim=0)) == num_samples
#
#
# @RunIf(min_cuda_gpus=2)
# def test_auto_add_worker_init_fn_distributed(tmp_path, monkeypatch):
#     """Test that the lightning worker_init_fn takes care of dataloaders in multi-gpu/multi-node training."""
#     dataset = NumpyRandomDataset()
#     num_workers = 2
#     batch_size = 2
#
#     dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
#     seed_everything(0, workers=True)
#     trainer = Trainer(
#         default_root_dir=tmp_path, max_epochs=1, accelerator="gpu", devices=2, strategy="ddp_spawn"
#     )
#     model = MultiProcessModel()
#     model.val_dataloader = None
#     trainer.fit(model, train_dataloaders=dataloader)
#


# def test_warning_with_small_dataloader_and_logging_interval(tmp_path):
#     """Test that a warning message is shown if the dataloader length is too short for the chosen logging interval."""
#     model = boring_classes.BoringModel()
#     dataloader = reax.ReaxDataLoader(boring_classes.RandomDataset(32, length=10))
#     model.train_dataloader = lambda: dataloader
#
#     with pytest.warns(
#         UserWarning,
#         match=r"The number of training batches \(10\) is smaller than the logging interval",
#     ):
#         trainer = reax.Trainer(
#             default_root_dir=tmp_path,
#             log_every_n_steps=11,
#             logger=reax.loggers.CsvLogger(tmp_path),
#         )
#         trainer.fit(model, max_epochs=1)
#
#     with pytest.warns(
#         UserWarning,
#         match=r"The number of training batches \(1\) is smaller than the logging interval",
#     ):
#         trainer = reax.Trainer(
#             default_root_dir=tmp_path,
#             log_every_n_steps=2,
#             logger=reax.loggers.CsvLogger(tmp_path),
#         )
#         trainer.fit(model, max_epochs=1, limit_train_batches=1)
#
#     with warning.no_warning_call(UserWarning, match="The number of training batches"):
#         trainer = reax.Trainer(default_root_dir=tmp_path, fast_dev_run=True, log_every_n_steps=2)
#         trainer.fit(model)


# def test_warning_with_iterable_dataset_and_len(tmp_path):
#     """Tests that a warning message is shown when an IterableDataset defines `__len__`."""
#     model = boring_classes.BoringModel()
#     original_dataset = model.train_dataloader().dataset
#
#     class IterableWithoutLen(IterableDataset):
#         def __iter__(self):
#             return iter(original_dataset)
#
#     class IterableWithLen(IterableWithoutLen):
#         def __len__(self):
#             return len(original_dataset)
#
#     # with __len__ defined
#     trainer = reax.Trainer(default_root_dir=tmp_path, max_steps=3)
#     dataloader = reax.ReaxDataLoader(IterableWithLen(), batch_size=16)
#     assert has_len_all_ranks(dataloader, trainer.strategy)
#     assert has_iterable_dataset(dataloader)
#     with pytest.warns(UserWarning, match="Your `IterableDataset` has `__len__` defined."):
#         trainer.validate(model, dataloaders=[dataloader])
#     with pytest.warns(UserWarning, match="Your `IterableDataset` has `__len__` defined."):
#         trainer.fit(model, train_dataloaders=dataloader, val_dataloaders=[dataloader])
#     with pytest.warns(UserWarning, match="Your `IterableDataset` has `__len__` defined."):
#         trainer.test(model, dataloaders=[dataloader])
#     with pytest.warns(UserWarning, match="Your `IterableDataset` has `__len__` defined."):
#         trainer.predict(model, dataloaders=[dataloader])
#
#     # without __len__ defined
#     trainer = reax.Trainer(default_root_dir=tmp_path, max_steps=3)
#     dataloader = reax.ReaxDataLoader(IterableWithoutLen(), batch_size=16)
#     assert not has_len_all_ranks(dataloader, trainer.strategy)
#     assert has_iterable_dataset(dataloader)
#     trainer.validate(model, dataloaders=dataloader)
#     trainer.fit(model, train_dataloaders=dataloader, val_dataloaders=[dataloader])
#     trainer.test(model, dataloaders=dataloader)
#     trainer.predict(model, dataloaders=dataloader)


# @pytest.mark.parametrize("yield_at_all", [False, True])
# def test_iterable_dataset_stop_iteration_at_epoch_beginning(yield_at_all, tmp_path):
#     """Test that the training loop skips execution if the iterator is empty from the start."""
#
#     class TestDataset(reax.data.IterableDataset):
#         def __init__(self, gen):
#             self.gen = gen
#
#         def __iter__(self):
#             return iter(self.gen())
#
#     class TestModel(boring_classes.BoringModel):
#         def gen(self):
#             # produce data in epoch 0, no data otherwise
#             if yield_at_all and self.current_epoch == 0:
#                 yield torch.rand(32)
#                 yield torch.rand(32)
#                 yield torch.rand(32)
#
#     model = TestModel()
#     train_dataloader = reax.ReaxDataLoader(TestDataset(model.gen), batch_size=2)
#     trainer = reax.Trainer(
#         default_root_dir=tmp_path,
#         logger=False,
#         enable_model_summary=False,
#     )
#     trainer.fit(model, train_dataloaders=train_dataloader, max_epochs=2)
#     assert trainer.global_updates == 2 * yield_at_all
#     # even though the generator might not yield any data, the fit_loop still advances so the
#     # current epoch gets increased
#     assert trainer.current_epoch == 2


# class DistribSamplerCallback(Callback):
#     def __init__(self, expected_seeds=(0, 0, 0)):
#         self.expected_seed = expected_seeds
#
#     def on_train_start(self, trainer, pl_module):
#         train_sampler = trainer.train_dataloader.sampler
#         assert isinstance(train_sampler, DistributedSampler)
#         assert train_sampler.shuffle
#         assert train_sampler.seed == self.expected_seed[0]
#
#     def on_validation_start(self, trainer, pl_module):
#         val_sampler = trainer.val_dataloaders.sampler
#         assert isinstance(val_sampler, DistributedSampler)
#         assert not val_sampler.shuffle
#         assert val_sampler.seed == self.expected_seed[1]
#
#     def on_test_start(self, trainer, pl_module):
#         test_sampler = trainer.test_dataloaders.sampler
#         assert isinstance(test_sampler, DistributedSampler)
#         assert not test_sampler.shuffle
#         assert test_sampler.seed == self.expected_seed[2]
#
#
# @RunIf(min_cuda_gpus=2, skip_windows=True)
# def test_dataloader_distributed_sampler(tmp_path):
#     """Test DistributedSampler and it's arguments for DDP backend."""
#     seed_everything(123)
#     model = BoringModel()
#     trainer = Trainer(
#         accelerator="gpu",
#         devices=[0, 1],
#         num_nodes=1,
#         strategy="ddp_spawn",
#         default_root_dir=tmp_path,
#         max_steps=1,
#         callbacks=[DistribSamplerCallback(expected_seeds=(123, 123, 123))],
#     )
#     trainer.fit(model)
#     trainer.test(model)
#
#
# class TestModelUniqueDDPSampling(BoringModel):
#     def __init__(self):
#         super().__init__()
#         self.seen_samples = []
#
#     def training_step(self, batch, batch_idx):
#         self.seen_samples.extend(batch.tolist())
#         # the actual training step is not needed for the test
#         return super().training_step(torch.rand(1, 32, device=self.device), batch_idx)
#
#     def on_train_end(self):
#         seen_samples = self.all_gather(self.seen_samples)
#         # The samples should be unique across all processes
#         assert set(torch.cat(seen_samples).view(-1).tolist()) == set(range(32))
#
#
# @RunIf(standalone=True)
# def test_distributed_sampler_without_global_seed(tmp_path):
#     """Test that the samples are non-overlapping in DDP when shuffling is enabled and no global seed is set."""
#     # This test must run without a global seed set (e.g. through `seed_everything`), to ensure that each process
#     # starts with a different initial state.
#     assert "PL_GLOBAL_SEED" not in os.environ
#     train_dataloader = DataLoader(range(32), shuffle=True, batch_size=4)
#     trainer = Trainer(
#         default_root_dir=tmp_path,
#         num_sanity_val_steps=False,
#         logger=False,
#         enable_progress_bar=False,
#         accelerator="cpu",
#         devices=2,
#         strategy="ddp",
#         max_epochs=1,
#     )
#     trainer.fit(TestModelUniqueDDPSampling(), train_dataloader)
#
#
# class ModelWithDataLoaderDistributedSampler(BoringModel):
#     def train_dataloader(self):
#         dataloader = super().train_dataloader()
#         dist_sampler = DistributedSampler(dataloader.dataset, shuffle=True, seed=11)
#         return DataLoader(
#             dataloader.dataset, batch_size=32, drop_last=False, sampler=dist_sampler, shuffle=False
#         )
#
#
# @RunIf(min_cuda_gpus=2, skip_windows=True)
# def test_dataloader_distributed_sampler_already_attached(tmp_path):
#     """Test DistributedSampler and it's arguments for DDP backend when DistSampler already included on dataloader."""
#     seed_everything(123)
#     model = ModelWithDataLoaderDistributedSampler()
#     trainer = Trainer(
#         accelerator="gpu",
#         devices=[0, 1],
#         num_nodes=1,
#         strategy="ddp_spawn",
#         default_root_dir=tmp_path,
#         max_steps=100,
#         callbacks=[DistribSamplerCallback(expected_seeds=(11, 123, 0))],
#         use_distributed_sampler=True,
#     )
#     trainer.fit(model)
#     assert trainer.state.finished, "DDP Training failed"
#
#
# @pytest.mark.parametrize(
#     ("mode", "num_training_batches"),
#     [("min_size", 16), ("max_size_cycle", 64), ("max_size", 64), ("sequential", 64 + 16 * 4)],
# )
# def test_fit_multiple_train_loaders(tmp_path, mode, num_training_batches):
#     """Integration test for multiple train iterables."""
#
#     class CustomBoringModel(BoringModel):
#         def train_dataloader(self):
#             loaders_a_b = [DataLoader(RandomDataset(32, 64)), DataLoader(RandomDataset(32, 16))]
#             loaders_c_d_e = [
#                 DataLoader(RandomDataset(32, 16)),
#                 DataLoader(RandomDataset(32, 16)),
#                 DataLoader(RandomDataset(32, 16)),
#             ]
#             # Dict[str, List[DataLoader]]
#             loaders = {"a_b": loaders_a_b, "c_d_e": loaders_c_d_e}
#             return CombinedLoader(loaders, mode)
#
#         def training_step(self, batch, batch_idx):
#             assert len(batch) == 2
#             assert len(batch["a_b"]) == 2
#             assert len(batch["c_d_e"]) == 3
#             return super().training_step(batch["a_b"][0], batch_idx)
#
#     model = CustomBoringModel()
#     trainer = Trainer(max_epochs=1, default_root_dir=tmp_path)
#
#     if mode == "sequential":
#         with pytest.raises(ValueError, match="FitLoop` does not support"):
#             trainer.fit(model)
#     else:
#         trainer.fit(model)
#
#     # verify the num_training_batches according to the mode
#     assert num_training_batches == trainer.num_training_batches
#


# @pytest.mark.parametrize("check_interval", [50, 1.0])
# @pytest.mark.parametrize(
#     "dataloader_wrapper", [CustomNotImplementedErrorDataloader, CustomInfDataloader]
# )
# def test_train_dataloader_not_implemented_error(tmp_path, check_interval, dataloader_wrapper):
#     """Test not_implemented_error train data loader (e.g. IterableDataset)"""
#
#     class CustomBoringModel(boring_classes.BoringModel):
#         def train_dataloader(self):
#             return dataloader_wrapper(reax.ReaxDataLoader(boring_classes.RandomDataset(32, 64)))
#
#         def val_dataloader(self):
#             return dataloader_wrapper(reax.ReaxDataLoader(boring_classes.RandomDataset(32, 64)))
#
#     model = CustomBoringModel()
#     trainer = reax.Trainer(default_root_dir=tmp_path)
#     trainer.fit(model, max_updates=5, max_epochs=1, val_check_interval=check_interval)
#     # verify training completed


# @pytest.mark.parametrize(
#     "stage",
#     [RunningStage.TRAINING, RunningStage.VALIDATING, RunningStage.TESTING, RunningStage.PREDICTING],
# )
# @pytest.mark.parametrize(
#     "dataloader_wrapper", [CustomNotImplementedErrorDataloader, CustomInfDataloader]
# )
# def test_inf_dataloader_raise_error_with_partial_batch_limits(tmp_path, stage, dataloader_wrapper):
#     """Test limit_batch error with inf dataloader (e.g. IterableDataset)"""
#     model = BoringModel()
#     setattr(
#         model,
#         f"{stage.dataloader_prefix}_dataloader",
#         lambda: dataloader_wrapper(DataLoader(RandomDataset(32, 64))),
#     )
#     trainer_kwargs = {
#         "default_root_dir": tmp_path,
#         "max_epochs": 1,
#         f"limit_{stage.dataloader_prefix}_batches": 0.5,
#     }
#     trainer = Trainer(**trainer_kwargs)
#     trainer_fn = "fit" if stage == RunningStage.TRAINING else stage.value
#
#     with pytest.raises(
#         MisconfigurationException, match=r"IterableDataset`.*limit_.*_batches\)`.*`1.0` or an int"
#     ):
#         getattr(trainer, trainer_fn)(model)
#


def test_dataloaders_load_only_once(tmp_path):
    model = boring_classes.BoringModel()
    tracker = Mock()

    model.train_dataloader = Mock(wraps=model.train_dataloader)
    model.val_dataloader = Mock(wraps=model.val_dataloader)
    model.test_dataloader = Mock(wraps=model.test_dataloader)

    tracker.attach_mock(model.train_dataloader, "train_dataloader")
    tracker.attach_mock(model.val_dataloader, "val_dataloader")
    tracker.attach_mock(model.test_dataloader, "test_dataloader")

    trainer = reax.Trainer(default_root_dir=tmp_path)
    trainer.fit(model, limit_train_batches=0.3, limit_val_batches=0.3, max_epochs=3)

    model.train_dataloader.assert_called_once()
    model.val_dataloader.assert_called_once()
    model.test_dataloader.assert_not_called()

    assert tracker.mock_calls == [call.val_dataloader(), call.train_dataloader()]


def test_dataloaders_load_only_once_no_sanity_check(tmp_path):
    model = boring_classes.BoringModel()

    # logger file to get meta
    trainer = reax.Trainer(default_root_dir=tmp_path)

    tracker = Mock()
    model.train_dataloader = Mock(wraps=model.train_dataloader)
    model.val_dataloader = Mock(wraps=model.val_dataloader)
    model.test_dataloader = Mock(wraps=model.test_dataloader)

    tracker.attach_mock(model.train_dataloader, "train_dataloader")
    tracker.attach_mock(model.val_dataloader, "val_dataloader")
    tracker.attach_mock(model.test_dataloader, "test_dataloader")

    trainer.fit(
        model,
        limit_train_batches=0.3,
        limit_val_batches=0.3,
        num_sanity_val_steps=0,
        max_epochs=3,
    )

    # verify the sequence
    expected_sequence = [call.train_dataloader(), call.val_dataloader()]
    assert tracker.mock_calls == expected_sequence


@pytest.mark.skip(reason="Not fully ported yet")
@pytest.mark.parametrize(
    (
        "num_sanity_val_steps",
        "check_val_every_n_epoch",
        "reload_dataloaders_every_n_epochs",
        "train_reload_epochs_expect",
        "val_reload_epochs_expect",
        "val_step_epochs_expect",
    ),
    [
        # general case where sanity check reloads the dataloaders for validation on current_epoch=0
        (
            0,
            1,
            1,
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        ),
        (
            1,
            1,
            1,
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        ),
        # case where check_val_every_n_epoch < reload_dataloaders_every_n_epochs so expected val_reload_epoch
        # and val_step_epoch will be different
        (0, 1, 2, [0, 2, 4, 6, 8], [0, 2, 4, 6, 8], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        (1, 1, 2, [0, 2, 4, 6, 8], [2, 4, 6, 8], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        (0, 3, 4, [0, 4, 8], [2, 8], [2, 5, 8]),
        (1, 3, 4, [0, 4, 8], [2, 8], [2, 5, 8]),
        # case where check_val_every_n_epoch > reload_dataloaders_every_n_epochs so expected val_reload_epoch
        # and val_step_epoch will be same
        (0, 2, 1, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 3, 5, 7, 9], [1, 3, 5, 7, 9]),
        (1, 2, 1, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 3, 5, 7, 9], [1, 3, 5, 7, 9]),
        (0, 3, 2, [0, 2, 4, 6, 8], [2, 5, 8], [2, 5, 8]),
        (1, 3, 2, [0, 2, 4, 6, 8], [2, 5, 8], [2, 5, 8]),
        (0, 5, 2, [0, 2, 4, 6, 8], [4, 9], [4, 9]),
        (1, 5, 2, [0, 2, 4, 6, 8], [4, 9], [4, 9]),
        # case where check_val_every_n_epoch = reload_dataloaders_every_n_epochs so expected val_reload_epoch
        # and val_step_epoch will be same
        (0, 2, 2, [0, 2, 4, 6, 8], [1, 3, 5, 7, 9], [1, 3, 5, 7, 9]),
        (1, 2, 2, [0, 2, 4, 6, 8], [1, 3, 5, 7, 9], [1, 3, 5, 7, 9]),
    ],
)
def test_dataloaders_load_every_n_epochs_infrequent_val(
    tmp_path,
    num_sanity_val_steps,
    check_val_every_n_epoch,
    reload_dataloaders_every_n_epochs,
    train_reload_epochs_expect,
    val_reload_epochs_expect,
    val_step_epochs_expect,
):
    """Test dataloader reload behavior when infrequently checking validation set (via check_val_every_n_epoch)"""
    sanity_val_check_epochs, train_reload_epochs, val_reload_epochs = [], [], []
    sanity_val_step_epochs, val_step_epochs = [], []

    class TestModel(boring_classes.BoringModel):
        def train_dataloader(self):
            train_reload_epochs.append(self.current_epoch)
            return super().train_dataloader()

        def val_dataloader(self):
            if self.trainer.sanity_checking:
                sanity_val_check_epochs.append(self.current_epoch)
            else:
                val_reload_epochs.append(self.current_epoch)
            return super().val_dataloader()

        def validation_step(self, *args, **kwargs):
            if self.trainer.sanity_checking:
                sanity_val_step_epochs.append(self.current_epoch)
            else:
                val_step_epochs.append(self.current_epoch)

            return super().validation_step(*args, **kwargs)

    model = TestModel()

    trainer = reax.Trainer(default_root_dir=tmp_path)
    trainer.fit(
        model,
        limit_train_batches=1,
        limit_val_batches=1,
        check_val_every_n_epoch=check_val_every_n_epoch,
        reload_dataloaders_every_n_epochs=reload_dataloaders_every_n_epochs,
        max_epochs=10,
        num_sanity_val_steps=num_sanity_val_steps,
    )

    # Verify epoch of reloads
    sanity_val_check_epochs_expect = [0] if num_sanity_val_steps else []
    assert sanity_val_check_epochs == sanity_val_step_epochs == sanity_val_check_epochs_expect
    assert train_reload_epochs == train_reload_epochs_expect
    assert val_reload_epochs == val_reload_epochs_expect
    assert val_step_epochs == val_step_epochs_expect


def test_dataloaders_load_every_n_epochs_frequent_val(tmp_path):
    """Test dataloader reload behavior when frequently checking validation set (via val_check_interval)"""
    train_reload_epochs, val_reload_epochs, val_check_epochs = [], [], []

    class TestModel(boring_classes.BoringModel):
        def train_dataloader(self):
            train_reload_epochs.append(self.current_epoch)
            return super().train_dataloader()

        def val_dataloader(self):
            val_reload_epochs.append(self.current_epoch)
            return super().val_dataloader()

        def on_validation_epoch_end(self, stage: "reax.stages.Validate", /):
            val_check_epochs.append(self.current_epoch)

    model = TestModel()

    trainer = reax.Trainer(default_root_dir=tmp_path)

    model.test_dataloader = Mock(wraps=model.test_dataloader)

    trainer.fit(
        model,
        limit_train_batches=0.3,
        limit_val_batches=0.3,
        val_check_interval=0.3,
        reload_dataloaders_every_n_epochs=1,
        max_epochs=3,
    )
    trainer.test(model)

    # Verify epoch of reloads
    assert train_reload_epochs == [0, 1, 2]
    assert val_reload_epochs == [0, 1, 2]
    model.test_dataloader.assert_called_once()

    # Verify validation happens 3 times per epoch + 1 for sanity check
    assert val_check_epochs == [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]


@pytest.mark.parametrize("n", ["test", -1])
def test_dataloaders_load_every_n_epochs_exception(tmp_path, n):
    with pytest.raises(reax.exceptions.MisconfigurationException, match="should be an int >"):
        trainer = reax.Trainer(default_root_dir=tmp_path)
        trainer.fit(boring_classes.BoringModel(), reload_dataloaders_every_n_epochs=n)


def test_dataloaders_load_every_epoch_no_sanity_check(tmp_path):
    class TestModel(boring_classes.BoringModel):
        def validation_step(self, batch, batch_idx):
            self.log("dummy_val", 5.0)
            return super().validation_step(batch, batch_idx)

    model = TestModel()

    # This callback tests that the evaluation metrics are available by the time we run checkpointing
    checkpoint_listener = reax.listeners.ModelCheckpoint(monitor="dummy_val", save_top_k=1)

    # logger file to get meta
    trainer = reax.Trainer(default_root_dir=tmp_path, listeners=[checkpoint_listener])

    tracker = Mock()
    model.train_dataloader = Mock(wraps=model.train_dataloader)
    model.val_dataloader = Mock(wraps=model.val_dataloader)
    model.test_dataloader = Mock(wraps=model.test_dataloader)

    tracker.attach_mock(model.train_dataloader, "train_dataloader")
    tracker.attach_mock(model.val_dataloader, "val_dataloader")
    tracker.attach_mock(model.test_dataloader, "test_dataloader")

    trainer.fit(
        model,
        limit_train_batches=0.3,
        limit_val_batches=0.3,
        reload_dataloaders_every_n_epochs=1,
        max_epochs=3,
        num_sanity_val_steps=0,
    )
    trainer.test(model)

    expected_calls = [
        call.train_dataloader(),
        call.val_dataloader(),
        call.train_dataloader(),
        call.val_dataloader(),
        call.train_dataloader(),
        call.val_dataloader(),
        call.test_dataloader(),
    ]
    assert tracker.mock_calls == expected_calls


#
# @pytest.mark.parametrize("sanity_check", [False, True])
# def test_dataloaders_load_only_once_passed_loaders(tmp_path, monkeypatch, sanity_check):
#     model = boring_classes.BoringModel()
#     train_dataloader = model.train_dataloader()
#     val_dataloader = model.val_dataloader()
#     test_dataloader = model.test_dataloader()
#
#     # delete dataloader methods on the model
#     model.train_dataloader = None
#     model.val_dataloader = None
#     model.test_dataloader = None
#
#     stages: list[str] = []
#
#     trainer = reax.Trainer(default_root_dir=tmp_path)
#
#     original_request_dataloader = reax.data.DataSourceManager.get_dataloader
#
#     def side_effect_request_dataloader(self, name):
#         stages.append(str(trainer.stage))
#         return original_request_dataloader(self, name)
#
#     # patch.object(reax.data.DataSourceManager, "_request_dataloader", side_effect_request_dataloader)
#
#     # request_dataloader_mock = Mock(wraps=side_effect_request_dataloader)
#     monkeypatch.setattr(
#         reax.data.DataSourceManager, "get_dataloader", side_effect_request_dataloader
#     )
#     # monkeypatch.setattr(
#     #     lightning.pytorch.loops.evaluation_loop, "_request_dataloader", request_dataloader_mock
#     # )
#
#     trainer.fit(
#         model,
#         train_dataloader,
#         val_dataloader,
#         limit_train_batches=0.3,
#         limit_val_batches=0.3,
#         max_epochs=3,
#         num_sanity_val_steps=1 if sanity_check else 0,
#     )
#     # assert request_dataloader_mock.call_count == 2
#     assert len(stages) == 2
#
#     request_dataloader_mock.reset_mock()
#     trainer.test(model, dataloaders=test_dataloader)
#     assert request_dataloader_mock.call_count == 1
#
#     expected = ["sanity_check", "train", "test"] if sanity_check else ["train", "validate", "test"]
#     assert stages == expected


def test_dataloaders_reset_and_attach(tmp_path):
    """Test that repeated calls to Trainer.{fit,validate,test,predict} properly reset dataloaders
    before attaching the new one."""
    # the assertions compare the datasets and not dataloaders since we patch and replace the samplers
    dataloader_0 = reax.ReaxDataLoader(dataset=boring_classes.RandomDataset(32, 64))
    dataloader_1 = reax.ReaxDataLoader(dataset=boring_classes.RandomDataset(32, 64))
    dataloader_2 = reax.ReaxDataLoader(dataset=boring_classes.RandomDataset(32, 64))
    dataloader_3 = reax.ReaxDataLoader(dataset=boring_classes.RandomDataset(32, 64))
    model = boring_classes.BoringModel()
    trainer = reax.Trainer(default_root_dir=tmp_path)
    fit_kwargs = dict(max_updates=1, limit_val_batches=1)

    # 1st fit
    fit = trainer.fit(
        model, train_dataloaders=dataloader_0, val_dataloaders=dataloader_1, **fit_kwargs
    )
    assert fit.train_dataloader.dataset is dataloader_0.dataset
    assert fit.val_dataloaders.dataset is dataloader_1.dataset
    # 2nd fit
    fit = trainer.fit(
        model, train_dataloaders=dataloader_2, val_dataloaders=dataloader_3, **fit_kwargs
    )
    assert fit.train_dataloader.dataset is dataloader_2.dataset
    assert fit.val_dataloaders.dataset is dataloader_3.dataset

    # 1st validate
    val = trainer.validate(model, dataloaders=dataloader_0, limit_batches=1)
    assert val.dataloaders.dataset is dataloader_0.dataset
    # 2nd validate
    val = trainer.validate(model, dataloaders=dataloader_1, limit_batches=1)
    assert val.dataloaders.dataset is dataloader_1.dataset

    # 1st test
    test = trainer.test(model, dataloaders=dataloader_0, limit_batches=1)
    assert test.dataloaders.dataset is dataloader_0.dataset
    # 2nd test
    test = trainer.test(model, dataloaders=dataloader_1, limit_batches=1)
    assert test.dataloaders.dataset is dataloader_1.dataset

    # 1st predict
    predict = trainer.predict(model, dataloaders=dataloader_0, limit_batches=1)
    assert predict.dataloaders.dataset is dataloader_0.dataset
    # 2nd predict
    predict = trainer.predict(model, dataloaders=dataloader_1, limit_batches=1)
    assert predict.dataloaders.dataset is dataloader_1.dataset


#
# @pytest.mark.parametrize("mode", ["min_size", "max_size_cycle"])
# def test_correct_dataloader_idx_in_hooks(tmp_path, mode):
#     """Check the correct dataloader_idx inside hooks."""
#
#     class CustomBoringModel(BoringModel):
#         def __init__(self):
#             super().__init__()
#             self.val_call_count = 0
#             self.test_call_count = 0
#
#         def assert_dataloader_idx_hook(self, dataloader_idx):
#             if self.trainer.training:
#                 assert dataloader_idx == 0
#             elif self.trainer.validating:
#                 assert dataloader_idx == (0 if self.val_call_count <= 5 else 1)
#             elif self.trainer.testing:
#                 assert dataloader_idx == (0 if self.test_call_count <= 5 else 1)
#
#         def transfer_batch_to_device(self, batch, device, dataloader_idx):
#             self.assert_dataloader_idx_hook(dataloader_idx)
#             return super().transfer_batch_to_device(batch, device, dataloader_idx)
#
#         def on_before_batch_transfer(self, batch, dataloader_idx):
#             # incrementing here since this is the first hook called at each step
#             if self.trainer.validating:
#                 self.val_call_count += 1
#             elif self.trainer.testing:
#                 self.test_call_count += 1
#
#             self.assert_dataloader_idx_hook(dataloader_idx)
#             return super().on_before_batch_transfer(batch, dataloader_idx)
#
#         def on_after_batch_transfer(self, batch, dataloader_idx):
#             self.assert_dataloader_idx_hook(dataloader_idx)
#             return super().on_after_batch_transfer(batch, dataloader_idx)
#
#         def training_step(self, batch, batch_idx):
#             return super().training_step(batch["a"], batch_idx)
#
#         def validation_step(self, batch, batch_idx, dataloader_idx):
#             self.assert_dataloader_idx_hook(dataloader_idx)
#             out = super().validation_step(batch, batch_idx)
#             loss = out.pop("x")
#             out[f"val_loss_{dataloader_idx}"] = loss
#             return out
#
#         def test_step(self, batch, batch_idx, dataloader_idx):
#             self.assert_dataloader_idx_hook(dataloader_idx)
#             out = super().test_step(batch, batch_idx)
#             loss = out.pop("y")
#             out[f"test_loss_{dataloader_idx}"] = loss
#             return out
#
#         def predict(self, batch, batch_idx, dataloader_idx):
#             self.assert_dataloader_idx_hook(dataloader_idx)
#             return super().predict(batch, batch_idx, dataloader_idx)
#
#         def train_dataloader(self):
#             return CombinedLoader(
#                 {"a": DataLoader(RandomDataset(32, 64)), "b": DataLoader(RandomDataset(32, 64))},
#                 mode=mode,
#             )
#
#         def val_dataloader(self):
#             return [DataLoader(RandomDataset(32, 64)), DataLoader(RandomDataset(32, 64))]
#
#         def test_dataloader(self):
#             return [DataLoader(RandomDataset(32, 64)), DataLoader(RandomDataset(32, 64))]
#
#         def predict_dataloader(self):
#             return [DataLoader(RandomDataset(32, 64)), DataLoader(RandomDataset(32, 64))]
#
#     model = CustomBoringModel()
#     trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=5)
#
#     trainer.fit(model)
#     trainer.test(model)
#     preds = trainer.predict(model)
#
#     assert len(preds) == 2
#     assert all(len(x) == 5 for x in preds)
#
#
def test_request_dataloader(tmp_path):
    """This test asserts dataloader can be wrapped."""

    class DataLoaderWrapper(reax.data.DataLoader):
        def __init__(self, loader):
            self.loader = loader
            self._iter = iter(self.loader)

        def __iter__(self):
            self._iter = iter(self.loader)
            return self._iter

        def __next__(self):
            return next(self._iter)

        @property
        def dataset(self):
            return self.loader.dataset

        @property
        def sampler(self):
            return self.loader.sampler

        def with_new_sampler(self, sampler):
            return DataLoaderWrapper(self.loader.with_new_sampler(sampler))

    class TestModel(boring_classes.BoringModel):
        def __init__(self):
            super().__init__()
            self.on_train_batch_start_called = False
            self.on_val_batch_start_called = False

        def train_dataloader(self):
            loader = super().train_dataloader()
            return DataLoaderWrapper(loader)

        def on_train_batch_start(self, *_) -> None:
            assert isinstance(
                self.trainer.train_dataloader, reax.data.DeviceDataLoader
            ) and isinstance(self.trainer.train_dataloader.parent, DataLoaderWrapper)
            self.on_train_batch_start_called = True

        def val_dataloader(self):
            loader = super().val_dataloader()
            return DataLoaderWrapper(loader)

        def on_validation_batch_start(self, *_):
            assert isinstance(
                self.trainer.val_dataloaders, reax.data.DeviceDataLoader
            ) and isinstance(self.trainer.val_dataloaders.parent, DataLoaderWrapper)
            self.on_val_batch_start_called = True

    trainer = reax.Trainer(default_root_dir=tmp_path)
    model = TestModel()
    trainer.fit(model, limit_train_batches=2, limit_val_batches=2, max_epochs=1)
    trainer.test(model, limit_batches=2)
    assert model.on_train_batch_start_called
    assert model.on_val_batch_start_called


#
# @pytest.mark.parametrize("num_loaders", [1, 2])
# def test_multiple_dataloaders_with_random_sampler_overfit_batches(num_loaders, tmp_path):
#     class TestModel(BoringModel):
#         def training_step(self, batch, batch_idx):
#             assert all(
#                 isinstance(dl.sampler, SequentialSampler) for dl in self.trainer.train_dataloader
#             )
#             return super().training_step(batch[0], batch_idx)
#
#         def _create_dataloader(self):
#             ds = RandomDataset(32, 64)
#             return DataLoader(ds, sampler=RandomSampler(ds))
#
#         def train_dataloader(self):
#             return [self._create_dataloader() for _ in range(num_loaders)]
#
#         validation_step = None
#
#     trainer = Trainer(default_root_dir=tmp_path, overfit_batches=1.0, max_epochs=1)
#     trainer.fit(TestModel())
