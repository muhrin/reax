import pytest

from reax import exceptions
from reax.data import _datasources


def test_data_source_default_attributes():
    source = _datasources.DataSource()
    assert source.prepare_data_per_node is True
    assert source.allow_zero_length_dataloader_with_multiple_devices is False


def test_data_source_dataloader_stubs_raise():
    source = _datasources.DataSource()
    with pytest.raises(exceptions.MisconfigurationException, match="must be implemented"):
        source.train_dataloader()
    with pytest.raises(exceptions.MisconfigurationException, match="must be implemented"):
        source.test_dataloader()
    with pytest.raises(exceptions.MisconfigurationException, match="must be implemented"):
        source.predict_dataloader()
