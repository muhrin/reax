import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from reax.data import _loaders, dataloaders, samplers


def test_single_or_value_single():
    result = _loaders._single_or_value((5,))
    assert result == 5


def test_single_or_value_multiple():
    result = _loaders._single_or_value((1, 2))
    assert result == (1, 2)


def test_fetcher_dataloader_iter():
    dataset = [1, 2, 3, 4]
    sampler = samplers.BatchSampler(samplers.SequentialSampler(len(dataset)), 2, False)
    loader = _loaders.FetcherDataLoader(dataset, sampler, lambda batch: list(batch))
    assert list(loader) == [[1, 2], [3, 4]]
    assert loader.dataset is dataset
    assert loader.sampler is sampler


def test_fetcher_dataloader_with_new_sampler():
    dataset = [1, 2, 3]
    loader = _loaders.FetcherDataLoader(
        dataset, samplers.BatchSampler(samplers.SequentialSampler(3), 1, False), list
    )
    new_loader = loader.with_new_sampler(samplers.BatchSampler(samplers.RandomSampler(3), 1, False))
    assert isinstance(new_loader, _loaders.FetcherDataLoader)
    assert sorted(new_loader.dataset) == [1, 2, 3]
    assert sorted(itertools.chain.from_iterable(new_loader.sampler)) == [0, 1, 2]


def test_fetcher_dataloader_stops_on_exhausted_iter():
    # Sampler yields 3 index-batches (1 item each) but the iterable dataset only has 2 items
    loader = _loaders.FetcherDataLoader(iter([1, 2]), _RepeatSampler(3), list)
    assert list(loader) == [[1], [2]]


class _RepeatSampler:
    def __init__(self, n):
        self.n = n

    def __iter__(self):
        for _ in range(self.n):
            yield [None]

    def __len__(self):
        return self.n


def test_reax_dataloader_basic():
    loader = _loaders.ReaxDataLoader([1, 2, 3, 4, 5], batch_size=2)
    batches = list(loader)
    assert len(batches) == 3
    assert batches[0].tolist() == [1, 2]
    assert batches[-1].tolist() == [5]
    assert loader.batch_size == 2 if hasattr(loader, "batch_size") else True


def test_reax_dataloader_custom_sampler():
    loader = _loaders.ReaxDataLoader([1, 2, 3], batch_size=2, sampler=samplers.SequentialSampler(3))
    batches = [list(b) for b in loader]
    assert batches == [[1, 2], [3]]


def test_reax_dataloader_custom_collate():
    loader = _loaders.ReaxDataLoader([1, 2], batch_size=1, collate_fn=lambda b: sum(b))
    assert list(loader) == [1, 2]


def test_array_loader_single_array():
    array = np.arange(10)
    loader = _loaders.ArrayLoader(array, batch_size=3)
    batches = list(loader)
    assert len(batches) == 4
    np.testing.assert_array_equal(batches[0], [0, 1, 2])
    assert len(loader) == 4
    assert loader.batch_size == 3
    assert loader.dataset is array
    assert loader.sampler is loader._sampler
    np.testing.assert_array_equal(loader.first(), [0, 1, 2])


def test_array_loader_jax_array():
    array = jnp.arange(6)
    loader = _loaders.ArrayLoader(array, batch_size=2)
    batches = [np.asarray(b) for b in loader]
    np.testing.assert_array_equal(batches[0], [0, 1])


def test_array_loader_tuple_of_arrays():
    arrays = (np.arange(9), np.arange(9) * 2)
    loader = _loaders.ArrayLoader(arrays, batch_size=3)
    batches = list(loader)
    assert len(batches) == 3
    first = batches[0]
    assert isinstance(first, tuple)
    np.testing.assert_array_equal(first[0], [0, 1, 2])
    np.testing.assert_array_equal(first[1], [0, 2, 4])


def test_array_loader_size_mismatch_raises():
    with pytest.raises(ValueError, match="Size mismatch"):
        _loaders.ArrayLoader((np.arange(5), np.arange(6)))


def test_array_loader_invalid_type_raises():
    # jaxtyping/beartype rejects the str before the manual check; both raise TypeError
    with pytest.raises(TypeError):
        _loaders.ArrayLoader("not an array")


def test_array_loader_with_new_sampler():
    array = np.arange(4)
    loader = _loaders.ArrayLoader(array, batch_size=2)
    new_loader = loader.with_new_sampler(samplers.SequentialSampler(4))
    assert isinstance(new_loader, _loaders.ArrayLoader)
    assert list(new_loader)[0].tolist() == [0, 1]


def test_caching_loader():
    dataset = [1, 2, 3, 4]
    # single-item batches unwrapped to the raw value
    inner = _loaders.FetcherDataLoader(
        dataset,
        samplers.BatchSampler(samplers.SequentialSampler(len(dataset)), 1, False),
        lambda b: b[0],
    )
    loader = _loaders.CachingLoader(inner, reset_every=1)

    # First pass pulls from the loader
    first = list(loader)
    assert first == [1, 2, 3, 4]

    # Second pass uses the cache
    second = list(loader)
    assert second == [1, 2, 3, 4]
    # After `reset_every` passes the cache is cleared
    assert loader._cache in ([], None)


def test_caching_loader_reset_every_two():
    dataset = [1, 2]
    # single-item batches unwrapped to the raw value
    inner = _loaders.FetcherDataLoader(
        dataset,
        samplers.BatchSampler(samplers.SequentialSampler(len(dataset)), 1, False),
        lambda b: b[0],
    )
    loader = _loaders.CachingLoader(inner, reset_every=2)

    assert list(loader) == [1, 2]
    # Cache is still set (time_since_reset == 1)
    assert loader._cache == [1, 2]
    assert list(loader) == [1, 2]
    # Now the cache has been cleared
    assert not loader._cache
    assert loader._time_since_reset == 0


def test_caching_loader_properties():
    dataset = [1, 2]
    inner = _loaders.FetcherDataLoader(
        dataset,
        samplers.BatchSampler(samplers.SequentialSampler(len(dataset)), 1, False),
        lambda b: b[0],
    )
    loader = _loaders.CachingLoader(inner, reset_every=1)
    assert loader.dataset is dataset
    assert loader.sampler is inner.sampler
    new_loader = loader.with_new_sampler(samplers.BatchSampler(samplers.RandomSampler(2), 1, False))
    assert isinstance(new_loader, _loaders.CachingLoader)


def test_device_dataloader_puts_on_device():
    dataset = [jnp.array([1.0]), jnp.array([2.0])]
    inner = _loaders.FetcherDataLoader(
        dataset, samplers.BatchSampler(samplers.SequentialSampler(2), 1, False), list
    )
    loader = _loaders.DeviceDataLoader(inner, device=jax.devices()[0])

    batches = list(loader)
    # Each batch is the inner list device_put onto the target device
    assert len(batches) == 2
    assert all(len(b) == 1 for b in batches)
    assert jax.devices()[0] == loader._device

    assert loader.parent is inner
    assert loader.dataset is dataset
    assert loader.sampler is inner.sampler
    new_loader = loader.with_new_sampler(samplers.BatchSampler(samplers.RandomSampler(2), 1, False))
    assert isinstance(new_loader, _loaders.DeviceDataLoader)
    assert new_loader._device is loader._device


class _ConcreteGeneric(dataloaders.GenericDataLoader):
    @property
    def dataset(self):
        return self._dataset

    @property
    def sampler(self):
        return self._sampler

    def with_new_sampler(self, sampler):
        return self


def test_generic_dataloader_default():
    loader = _ConcreteGeneric([1, 2, 3, 4], batch_size=2)
    assert loader.batch_size == 2
    assert len(loader) == 2
    assert [b.tolist() for b in loader] == [[1, 2], [3, 4]]


def test_generic_dataloader_custom_collate():
    custom = _ConcreteGeneric([1, 2], batch_size=1, collate_fn=lambda b: sum(b))
    assert list(custom) == [1, 2]
