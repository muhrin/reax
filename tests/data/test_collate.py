from collections import OrderedDict, namedtuple
from collections.abc import Mapping
import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from reax.data import collate


@pytest.fixture
def collator():
    return collate.Collator()


def test_collate_empty_batch_raises(collator):
    with pytest.raises(ValueError, match="non-empty batch"):
        collator.collate([])


def test_collate_registered_type(collator):
    collator.register(str, lambda batch: [item.upper() for item in batch])
    assert collator.collate(["a", "b"]) == ["A", "B"]


def test_collate_jax_array():
    batch = [jnp.array([1, 2]), jnp.array([3, 4])]
    result = collate.default_collate(batch)
    assert isinstance(result, np.ndarray)
    assert result.tolist() == [[1, 2], [3, 4]]


def test_collate_numpy_array():
    batch = [np.array([1, 2]), np.array([3, 4])]
    result = collate.default_collate(batch)
    assert result.tolist() == [[1, 2], [3, 4]]


def test_collate_int():
    assert collate.default_collate([1, 2, 3]).tolist() == [1, 2, 3]
    assert collate.collate_int_fn([7]).tolist() == [7]


def test_collate_numpy_scalar():
    assert collate.collate_numpy_scalar_fn([np.int64(1), np.int64(2)]).tolist() == [1, 2]


def test_get_default_collator_caches():
    first = collate.get_default_collator()
    second = collate.get_default_collator()
    assert first is second


def test_default_collate_dict():
    # Uses the default collator (which registers `int`)
    batch = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    result = collate.default_collate(batch)
    assert isinstance(result, dict)
    assert result["a"].tolist() == [1, 3]
    assert result["b"].tolist() == [2, 4]


def test_fallback_collate_with_registered_scalars(collator):
    # Register scalar types so the nested values can be collated
    collator.register(int, lambda batch: np.asarray(batch))

    def _check(fn):
        batch = fn([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
        assert batch["a"].tolist() == [1, 3]
        assert batch["b"].tolist() == [2, 4]

    _check(collator.collate)


def test_fallback_collate_ordered_dict(collator):
    collator.register(int, lambda batch: np.asarray(batch))
    batch = [OrderedDict([("a", 1)]), OrderedDict([("a", 2)])]
    result = collator.collate(batch)
    assert isinstance(result, OrderedDict)
    assert result["a"].tolist() == [1, 2]


def test_fallback_collate_mapping_init_error(collator):
    # Mapping that doesn't support `type(elem)(iterable)` -> TypeError path
    class WeirdMapping(Mapping):
        def __init__(self, mapping):
            self._map = dict(mapping)

        def __getitem__(self, key):
            return self._map[key]

        def __iter__(self):
            return iter(self._map)

        def __len__(self):
            return len(self._map)

    collator.register(int, lambda batch: np.asarray(batch))
    batch = [WeirdMapping({"a": 1}), WeirdMapping({"a": 2})]
    result = collator.collate(batch)
    assert result["a"].tolist() == [1, 2]


def test_fallback_collate_mapping_constructor_rejects_dict(collator):
    # A read-only Mapping whose constructor does not accept the iterable of
    # key-value pairs that `type(elem)(iterable)` would pass -> raises TypeError,
    # so the plain-dict fallback path is taken.
    class NoDictCtor(Mapping):
        def __getitem__(self, key):
            return self._map[key]

        def __iter__(self):
            return iter(self._map)

        def __len__(self):
            return len(self._map)

        def __init__(self, *args, **kwargs):
            if args or kwargs:
                raise TypeError("no callable constructor with arguments")

    collator.register(int, lambda batch: np.asarray(batch))
    m1, m2 = NoDictCtor(), NoDictCtor()
    m1._map, m2._map = {"a": 1}, {"a": 2}
    # `elem_type({...})` raises -> falls back to a plain dict.
    result = collator.collate([m1, m2])
    assert result["a"].tolist() == [1, 2]


def test_fallback_collate_namedtuple(collator):
    collator.register(int, lambda batch: np.asarray(batch))
    Point = namedtuple("Point", ["x", "y"])
    batch = [Point(1, 2), Point(3, 4)]
    result = collator.collate(batch)
    assert isinstance(result, Point)
    assert result.x.tolist() == [1, 3]
    assert result.y.tolist() == [2, 4]


def test_fallback_collate_tuple(collator):
    collator.register(int, lambda batch: np.asarray(batch))
    batch = [(1, 2), (3, 4)]
    result = collator.collate(batch)
    assert isinstance(result, list)
    assert result[0].tolist() == [1, 3]
    assert result[1].tolist() == [2, 4]


def test_fallback_collate_list(collator):
    collator.register(int, lambda batch: np.asarray(batch))
    batch = [[1, 2], [3, 4]]
    result = collator.collate(batch)
    assert isinstance(result, list)
    assert result[0].tolist() == [1, 3]


def test_fallback_collate_list_size_mismatch_raises(collator):
    with pytest.raises(RuntimeError, match="equal size"):
        collator.collate([[1, 2], [3]])


def test_fallback_collate_range_fallback(collator):
    # range supports Sequence but not `type(elem)(iterable)` -> hits the except TypeError branch
    collator.register(int, lambda batch: np.asarray(batch))
    batch = [range(3), range(3)]
    result = collator.collate(batch)
    # Returns a plain list (not a range), one entry per transposed column
    assert isinstance(result, list)
    assert [item.tolist() for item in result] == [[0, 0], [1, 1], [2, 2]]


def test_fallback_collate_unsupported_raises(collator):
    class NotCollatable:
        pass

    with pytest.raises(TypeError, match="default_collate"):
        collator.collate([NotCollatable(), NotCollatable()])


@dataclasses.dataclass
class Point:
    x: int
    y: int


def test_fallback_collate_dataclass_raises(collator):
    # Dataclasses aren't Mapping/Sequence/namedtuple -> TypeError
    batch = [Point(1, 2), Point(3, 4)]
    with pytest.raises(TypeError, match="default_collate"):
        collator._fallback_collate(batch)


def test_custom_register(collator):
    collator.register(complex, lambda batch: np.asarray(batch))
    assert collator.collate([1 + 2j, 3 + 4j]).tolist() == [(1 + 2j), (3 + 4j)]
