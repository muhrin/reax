import numpy as np
import pytest

from reax.data import fetchers


def fake_collate(items):
    return [i for i in items]


class GetitemsDataset:
    def __getitems__(self, indices):
        return [f"data-{i}" for i in indices]


class GetitemDataset:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]


def test_map_fetcher_uses_getitems_when_present():
    fetcher = fetchers._MapFetcher(GetitemsDataset(), fake_collate)
    assert fetcher.fetch([0, 1, 2]) == ["data-0", "data-1", "data-2"]


def test_map_fetcher_falls_back_to_getitem():
    dataset = GetitemDataset(list(range(5)))
    fetcher = fetchers._MapFetcher(dataset, fake_collate)
    assert fetcher.fetch([0, 2, 4]) == [0, 2, 4]


def test_map_fetcher_empty_index():
    dataset = GetitemDataset(list(range(5)))
    fetcher = fetchers._MapFetcher(dataset, fake_collate)
    assert fetcher.fetch([]) == []


def test_iterable_fetcher_fetches_in_order():
    fetcher = fetchers._IterableFetcher(iter([1, 2, 3, 4]), fake_collate)
    assert fetcher.fetch([None, None]) == [1, 2]
    # State is preserved across fetch calls:
    assert fetcher.fetch([None]) == [3]


def test_iterable_fetcher_raises_stop_iteration_on_empty():
    fetcher = fetchers._IterableFetcher(iter([]), fake_collate)
    with pytest.raises(StopIteration):
        fetcher.fetch([None])


def test_iterable_fetcher_exhausted_mid_batch_sets_flag():
    fetcher = fetchers._IterableFetcher(iter([1, 2]), fake_collate)
    # Asked for 3 but only 2 available: fetches both, marks as ended
    assert fetcher.fetch([None] * 3) == [1, 2]
    assert fetcher._ended

    # Subsequent fetches raise StopIteration
    with pytest.raises(StopIteration):
        fetcher.fetch([None])


def test_create_fetcher_for_list_is_map_fetcher():
    fetcher = fetchers.create_fetcher([1, 2, 3], fake_collate)
    assert isinstance(fetcher, fetchers._MapFetcher)
    assert fetcher.fetch([0, 2]) == [1, 3]


def test_create_fetcher_for_generator_is_iterable_fetcher():
    fetcher = fetchers.create_fetcher((x for x in [1, 2, 3]), fake_collate)
    assert isinstance(fetcher, fetchers._IterableFetcher)
    assert fetcher.fetch([None, None]) == [1, 2]


def test_create_fetcher_for_jax_array_is_iterable_fetcher():
    import jax.numpy as jnp

    fetcher = fetchers.create_fetcher(jnp.arange(3), fake_collate)
    assert isinstance(fetcher, fetchers._IterableFetcher)


def test_create_fetcher_unsupported_type_raises():
    with pytest.raises(TypeError, match="Unsupported type"):
        fetchers.create_fetcher(object(), fake_collate)
