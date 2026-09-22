from unittest.mock import MagicMock

from flax import nnx
import jax.numpy as jnp
import numpy as np
import pytest

import reax.data
from reax.data import _datasource_manager, _loaders, datamodules


class CountingDataModule(datamodules.DataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.cleared = 0
        self.events = []
        self.dataset = kwargs.get("dataset") or [_loaders_test_data()]

    def prepare_data(self):
        self.events.append("prepare_data")

    def setup(self, stage):
        self.events.append(f"setup:{stage}")

    def teardown(self, stage):
        self.events.append(f"teardown:{stage}")

    def on_exception(self, exception):
        self.events.append(f"exception:{type(exception).__name__}")

    def train_dataloader(self):
        return _loaders.ReaxDataLoader(self.dataset, batch_size=2)


def _loaders_test_data():
    return (jnp.ones((4, 2)),)


def test_datamodule_rngs_getter_setter():
    source = datamodules.DataModule()
    assert isinstance(source.rngs, nnx.Rngs)
    new = nnx.Rngs(7)
    source.rngs = new
    assert source.rngs is new


def test_datamodule_from_datasets_returns_from_datasets():
    source = datamodules.DataModule.from_datasets(train_dataset=[1, 2, 3])
    assert isinstance(source, datamodules.FromDatasets)
    assert source._train_dataset == [1, 2, 3]
    assert source._val_dataset is None
    assert source._test_dataset is None
    assert source._predict_dataset is None
    assert source._batch_size == 1


def test_from_datasets_dataloaders():
    source = datamodules.FromDatasets(
        train_dataset=[1, 2, 3, 4],
        val_dataset=[1, 2],
        test_dataset=[1, 2, 3, 4],
        predict_dataset=[1, 2, 3, 4],
        batch_size=2,
    )
    train = list(source.train_dataloader())
    assert [b.tolist() for b in train] == [[1, 2], [3, 4]]

    val = list(source.val_dataloader())
    assert [b.tolist() for b in val] == [[1, 2]]

    test = list(source.test_dataloader())
    assert [b.tolist() for b in test] == [[1, 2], [3, 4]]

    predict = list(source.predict_dataloader())
    assert [b.tolist() for b in predict] == [[1, 2], [3, 4]]


def test_from_datasets_defaults_to_none():
    source = datamodules.FromDatasets()
    assert source._train_dataset is None
    assert source._val_dataset is None
    assert source._test_dataset is None
    assert source._predict_dataset is None
    assert source._batch_size == 1


def test_manager_create_with_datamodule():
    source = CountingDataModule()
    manager = _datasource_manager.create_manager(datamodule=source)
    assert manager.source is source
    assert manager.has_dataloader("train")
    assert not manager.has_dataloader("bogus_name")


def test_manager_create_with_module():
    from reax import modules

    class MyModule(modules.Module):
        def train_dataloader(self):
            return _loaders.ReaxDataLoader([1, 2], batch_size=2)

    module = MyModule()
    manager = _datasource_manager.create_manager(module=module)
    assert manager.source is module
    assert manager.has_dataloader("train")


def test_manager_source_base_type_raises_for_invalid_source():
    class InvalidSource:
        pass

    manager = _datasource_manager.DataSourceManager(InvalidSource())
    with pytest.raises(RuntimeError, match="expected datasource"):
        manager._source_base_type


def test_manager_get_dataloader():
    source = CountingDataModule()
    manager = _datasource_manager.create_manager(datamodule=source)
    loader = manager.get_dataloader("train")
    assert isinstance(loader, _loaders.ReaxDataLoader)

    # The loader is now cached:
    assert manager.get_dataloader("train") is loader


def test_manager_has_dataloader_from_pre_supplied_loaders():
    source = CountingDataModule()
    loader = _loaders.ReaxDataLoader([1, 2], batch_size=2)
    manager = _datasource_manager.create_manager(source, train=loader)
    # Pre-supplied loaders are found in `_loaders`:
    assert manager._loaders["train"] is loader
    assert manager.get_dataloader("train") is loader
    assert manager.has_dataloader("train")


def test_manager_create_filters_none_loaders():
    source = CountingDataModule()
    manager = _datasource_manager.create_manager(source, val=None, train=[1, 2])
    assert "val" not in manager._loaders
    assert "train" in manager._loaders


def test_manager_create_prefers_datamodule_over_module():
    from reax import modules

    class MyModule(modules.Module):
        def train_dataloader(self):
            return _loaders.ReaxDataLoader([1, 2], batch_size=2)

    datamodule = CountingDataModule()
    module = MyModule()
    manager = _datasource_manager.create_manager(module=module, datamodule=datamodule)
    assert manager.source is datamodule


def test_manager_events():
    source = CountingDataModule()
    manager = _datasource_manager.create_manager(datamodule=source)

    manager.prepare_data()
    manager.setup("fit")
    manager.teardown("fit")
    manager.on_exception(ValueError("ouch"))

    assert source.events == [
        "prepare_data",
        "setup:fit",
        "teardown:fit",
        "exception:ValueError",
    ]


def test_manager_events_no_source():
    manager = _datasource_manager.DataSourceManager(None)
    manager.prepare_data()
    manager.setup("fit")
    manager.teardown("fit")
    manager.on_exception(Exception("boom"))
    assert manager.source is None


def test_manager_reset_clears_cache():
    source = CountingDataModule()
    manager = _datasource_manager.create_manager(datamodule=source)
    manager.get_dataloader("train")
    assert len(manager._from_datasource) == 1

    manager.reset()
    assert manager._from_datasource == {}


def test_manager_setup_dataloader_with_engine():
    source = CountingDataModule()
    engine = MagicMock()
    wrapped = MagicMock()
    engine.setup_dataloaders.return_value = wrapped
    manager = _datasource_manager.create_manager(datamodule=source, engine=engine)

    loader = manager.get_dataloader("train")
    assert loader is wrapped
    engine.setup_dataloaders.assert_called_once()


def test_manager_setup_dataloader_without_engine_passthrough():
    source = CountingDataModule()
    manager = _datasource_manager.create_manager(datamodule=source)
    loader = manager.get_dataloader("train")
    assert manager.source is source
    assert loader is not None
