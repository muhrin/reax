import dataclasses
import logging

import jax.numpy as jnp
import numpy as np
import pytest

from reax.data import utils


def test_extract_batch_size_numpy_nd():
    assert utils.extract_batch_size(np.ones((4, 3))) == 4


def test_extract_batch_size_numpy_0d():
    assert utils.extract_batch_size(np.array(5)) == 1


def test_extract_batch_size_jax_array():
    assert utils.extract_batch_size(jnp.ones((4, 3))) == 4


def test_extract_batch_size_nested_dict():
    batch = {"x": np.ones((7, 2)), "y": np.ones((7, 1))}
    assert utils.extract_batch_size(batch) == 7


def test_extract_batch_size_tuple():
    batch = (np.ones((5, 2)), np.ones((5, 3)))
    assert utils.extract_batch_size(batch) == 5


def test_extract_batch_size_mismatch(caplog):
    with caplog.at_level(logging.WARNING, logger="reax.data.utils"):
        result = utils.extract_batch_size((np.ones((5, 2)), np.ones((3, 2))))
    assert result == 5
    assert any("Could not determine batch size unambiguously" in m for m in caplog.messages)


def test_extract_batch_size_unresolvable_raises():
    class NoBatchSize:
        pass

    with pytest.raises(RuntimeError, match="Could not determine batch size"):
        utils.extract_batch_size(NoBatchSize())


def test_extract_batch_size_recursion_raises():
    container = []
    container.append(container)
    with pytest.raises(RecursionError, match="Could not determine batch size"):
        utils.extract_batch_size(container)


@dataclasses.dataclass
class Sample:
    x: np.ndarray
    y: np.ndarray


def test_extract_batch_size_dataclass_fields():
    batch = Sample(x=np.ones((2, 1)), y=np.ones((2, 1)))
    assert utils.extract_batch_size(batch) == 2


def test_sized_len():
    assert utils.sized_len([1, 2, 3]) == 3


def test_sized_len_none():
    class NoLen:
        def __len__(self):
            raise NotImplementedError

    assert utils.sized_len(NoLen()) is None


def test_get_registry_caches():
    first = utils.get_registry()
    second = utils.get_registry()
    assert first is second
    assert isinstance(first, utils.BatchSizer)


def test_batch_sizer_register_and_find():
    sizer = utils.BatchSizer()
    sizer.register(str, lambda batch: [len(batch)])
    assert list(sizer.extract_batch_size("hello")) == [5]


def test_batch_sizer_fallback_unknown_yields_none():
    sizer = utils.BatchSizer()
    assert list(sizer.extract_batch_size(object())) == [None]


def test_batch_sizer_fallback_iterable():
    sizer = utils.BatchSizer()
    sizer.register(np.ndarray, utils._array_batch_size)
    batch = [np.ones((4, 3)), np.ones((4, 3))]
    assert 4 in list(sizer.extract_batch_size(batch))


def test_batch_sizer_fallback_mapping():
    sizer = utils.BatchSizer()
    sizer.register(np.ndarray, utils._array_batch_size)
    batch = {"a": np.ones((3, 1))}
    assert 3 in list(sizer.extract_batch_size(batch))


def test_batch_sizer_fallback_dataclass():
    sizer = utils.BatchSizer()
    sizer.register(np.ndarray, utils._array_batch_size)
    batch = Sample(x=np.ones((6, 1)), y=np.ones((6, 1)))
    assert 6 in list(sizer.extract_batch_size(batch))


def test_array_batch_size_0d_and_nd():
    assert list(utils._array_batch_size(np.array(1))) == [1]
    assert list(utils._array_batch_size(np.ones((8, 2)))) == [8]
