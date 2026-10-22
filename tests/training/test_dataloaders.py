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
from unittest.mock import Mock, call, patch

import jaxtyping as jt
import numpy as np
import pytest

import reax
from reax.demos import boring_classes


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


def _iterable_dataloader(batch_size: int = 8) -> reax.ReaxDataLoader:
    """A dataloader over an iterable dataset, so it has no length."""
    return reax.ReaxDataLoader(boring_classes.RandomIterableDataset(32, 128), batch_size=batch_size)


def test_inf_dataloaders_with_limit_percent_batches(tmp_path):
    """Verify inf train, val & test dataloaders (e.g. IterableDataset) passed with batch limit
    in percent."""
    epoch_cb = Counter()
    trainer = reax.Trainer(default_root_dir=tmp_path, listeners=[epoch_cb])
    model = DummyModel()

    batch_size = 8
    train_dl = _iterable_dataloader(batch_size=batch_size)
    val_dl = _iterable_dataloader(batch_size=batch_size)
    test_dl = _iterable_dataloader(batch_size=batch_size)

    num_batches = 128 // batch_size
    for dl in (train_dl, val_dl, test_dl):
        assert reax.data.utils.sized_len(dl) is None
        assert sum(1 for _ in dl) == num_batches

    trainer.fit(
        model,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl,
        max_epochs=1,
        num_sanity_val_steps=0,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
    )

    assert epoch_cb.train_epoch_count == 1
    assert epoch_cb.train_batches_seen == num_batches

    assert epoch_cb.val_epoch_count == 1
    assert epoch_cb.val_batches_seen == num_batches

    trainer.test(model, dataloaders=test_dl, limit_batches=1.0)
    assert epoch_cb.test_epoch_count == 1
    assert epoch_cb.test_batches_seen == num_batches


@pytest.mark.parametrize(
    ("dataset", "limit_train_batches"),
    [
        (boring_classes.RandomDataset(32, 128), 10),
        (boring_classes.RandomIterableDataset(32, 128), 10),
        # TODO: (boring_classes.RandomIterableDatasetWithLen(32, 128), 10),
    ],
)
def test_dataloaders_with_limit_train_batches(tmp_path, dataset, limit_train_batches):
    """Verify inf train, val & test dataloaders (e.g. IterableDataset) passed with batch limit as
    number."""
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


@pytest.mark.parametrize(
    "dataset",
    [
        boring_classes.RandomDataset(32, 128),
        boring_classes.RandomIterableDataset(32, 128),
        # boring_classes.RandomIterableDatasetWithLen(32, 128),
    ],
)
def test_dataloaders_with_limit_val_batches(tmp_path, dataset):
    """Verify inf train, val & test dataloaders (e.g. IterableDataset) passed with batch limit as
    number."""
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
    """Verify inf train, val & test dataloaders (e.g. IterableDataset) passed with batch limit as
    number."""
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
            # when the limit is greater than the number of test batches
            # it should be the num in loaders
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


# def test_warning_with_small_dataloader_and_logging_interval(tmp_path):
#     """Test that a warning message is shown if the dataloader length is too short for the chosen
#     logging interval."""
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


@pytest.mark.parametrize("yield_at_all", [False, True])
def test_iterable_dataset_stop_iteration_at_epoch_beginning(yield_at_all, tmp_path):
    """Test that the training loop skips execution if the iterator is empty from the start."""

    class TestDataset:
        def __init__(self, gen):
            self.gen = gen

        def __iter__(self):
            return iter(self.gen())

    class TestModel(boring_classes.BoringModel):
        def gen(self):
            # produce data in epoch 0, no data otherwise
            if yield_at_all and self.trainer.current_epoch == 0:
                for _ in range(3):
                    yield np.random.rand(32).astype(np.float32)

    model = TestModel()
    train_dataloader = reax.ReaxDataLoader(TestDataset(model.gen), batch_size=2)
    trainer = reax.Trainer(default_root_dir=tmp_path)
    trainer.fit(model, train_dataloaders=train_dataloader, max_epochs=2)
    # 3 samples with a batch size of 2 gives 2 batches (and updates) in the first epoch
    assert trainer.global_updates == 2 * yield_at_all
    # even though the generator might not yield any data, the fit loop still advances so the
    # number of completed epochs is increased
    assert trainer.epochs_completed == 2


@pytest.mark.parametrize("check_interval", [50, 1.0])
def test_train_dataloader_with_no_length(tmp_path, check_interval):
    """Test training with dataloaders that have no length (e.g. IterableDataset)"""

    class CustomBoringModel(boring_classes.BoringModel):
        def train_dataloader(self):
            return _iterable_dataloader()

        def val_dataloader(self):
            return _iterable_dataloader()

    model = CustomBoringModel()
    trainer = reax.Trainer(default_root_dir=tmp_path)
    trainer.fit(model, max_updates=5, max_epochs=1, val_check_interval=check_interval)
    # verify training completed
    assert trainer.global_updates == 5


@pytest.mark.parametrize("stage", ["train", "val", "test", "predict"])
def test_inf_dataloader_raise_error_with_partial_batch_limits(tmp_path, stage):
    """Test limit_batch error with inf dataloader (e.g. IterableDataset)"""
    model = boring_classes.BoringModel()
    setattr(model, f"{stage}_dataloader", _iterable_dataloader)
    trainer = reax.Trainer(default_root_dir=tmp_path)

    with pytest.raises(ValueError, match="Cannot use a fractional batch limit"):
        if stage == "train":
            trainer.fit(model, max_epochs=1, limit_train_batches=0.5)
        elif stage == "val":
            trainer.validate(model, limit_batches=0.5)
        elif stage == "test":
            trainer.test(model, limit_batches=0.5)
        else:
            trainer.predict(model, limit_batches=0.5)


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
        # case where check_val_every_n_epoch < reload_dataloaders_every_n_epochs so expected
        # val_reload_epoch and val_step_epoch will be different
        (0, 1, 2, [0, 2, 4, 6, 8], [0, 2, 4, 6, 8], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        (1, 1, 2, [0, 2, 4, 6, 8], [2, 4, 6, 8], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        (0, 3, 4, [0, 4, 8], [2, 8], [2, 5, 8]),
        (1, 3, 4, [0, 4, 8], [2, 8], [2, 5, 8]),
        # case where check_val_every_n_epoch > reload_dataloaders_every_n_epochs so expected
        # val_reload_epoch and val_step_epoch will be same
        (0, 2, 1, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 3, 5, 7, 9], [1, 3, 5, 7, 9]),
        (1, 2, 1, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 3, 5, 7, 9], [1, 3, 5, 7, 9]),
        (0, 3, 2, [0, 2, 4, 6, 8], [2, 5, 8], [2, 5, 8]),
        (1, 3, 2, [0, 2, 4, 6, 8], [2, 5, 8], [2, 5, 8]),
        (0, 5, 2, [0, 2, 4, 6, 8], [4, 9], [4, 9]),
        (1, 5, 2, [0, 2, 4, 6, 8], [4, 9], [4, 9]),
        # case where check_val_every_n_epoch = reload_dataloaders_every_n_epochs so expected
        # val_reload_epoch and val_step_epoch will be same
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
    """Test dataloader reload behavior when infrequently checking validation set (via
    check_val_every_n_epoch)"""
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
    """Test dataloader reload behavior when frequently checking validation set (via
    val_check_interval)"""
    train_reload_epochs, val_reload_epochs, val_check_epochs = [], [], []

    class TestModel(boring_classes.BoringModel):
        def train_dataloader(self):
            train_reload_epochs.append(self.current_epoch)
            return super().train_dataloader()

        def val_dataloader(self):
            val_reload_epochs.append(self.current_epoch)
            return super().val_dataloader()

        def on_validation_epoch_end(
            self, stage: "reax.Stage", loop: "reax.stages.ValidationEpoch", /
        ):
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
    assert val_reload_epochs == [-1, 0, 1, 2]
    model.test_dataloader.assert_called_once()

    # Verify validation happens 1 for sanity check + 3 times per epoch
    assert val_check_epochs == [-1, 0, 0, 0, 1, 1, 1, 2, 2, 2]


@pytest.mark.parametrize("n", ["test", -1])
def test_dataloaders_load_every_n_epochs_exception(tmp_path, n):
    with pytest.raises((reax.exceptions.MisconfigurationException, jt.TypeCheckError)):
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


def test_dataloaders_reset_and_attach(tmp_path):
    """Test that repeated calls to Trainer.{fit,validate,test,predict} properly reset dataloaders
    before attaching the new one.
    """
    # the assertions compare the datasets and not dataloaders since we patch and replace the
    # samplers
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
