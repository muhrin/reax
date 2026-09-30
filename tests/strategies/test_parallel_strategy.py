import math

import jax
import pytest

import reax
from reax.data import samplers
from reax.strategies import _parallel


class FakeParallelStrategy(_parallel.ParallelStrategy):
    """A ParallelStrategy whose device ops are no-ops, letting us test the
    dataloader setup logic without starting the JAX distributed runtime."""

    def __init__(self, process_index: int = 0, process_count: int = 2):
        self._process_index = process_index
        self._process_count = process_count
        self._device = jax.local_devices()[0]

    @property
    def process_index(self) -> int:
        return self._process_index

    @property
    def process_count(self) -> int:
        return self._process_count

    @property
    def device(self):
        return self._device

    @property
    def is_global_zero(self) -> bool:
        return self._process_index == 0

    def to_device(self, value):
        return value

    def from_device(self, value):
        return value

    def broadcast(self, obj, src: int = 0):
        return obj

    def all_gather(self, obj):
        return obj

    def all_reduce(self, obj, reduce_op: str = "mean"):
        return obj

    def barrier(self, name: str | None = None) -> None:
        return None

    def compute(self, metric):
        return metric.compute()


def _make_loader(n: int):
    return reax.data.ReaxDataLoader(list(range(n)))


def _inner_indices(loader):
    """Extract the per-rank index list from the setup-produced DataLoader."""
    # loader.sampler is DistributedSampler, loader.sampler.inner is BatchSampler
    return [int(i) for batch in loader.sampler for i in batch]


def test_base_strategy_dataloader_passthrough():
    strategy = FakeParallelStrategy()
    loader = _make_loader(10)
    res = strategy.setup_dataloader(loader)
    assert res is not loader
    assert isinstance(res.sampler, samplers.DistributedSampler)


def test_base_strategy_wraps_raw_dataset():
    strategy = FakeParallelStrategy(process_index=0, process_count=2)
    wrapped = strategy.setup_dataloader([1, 2, 3])
    assert isinstance(wrapped, reax.data.DataLoader)
    # A raw sequence is wrapped in a BatchSampler, then DistributedSampler.
    assert isinstance(wrapped.sampler, samplers.DistributedSampler)
    assert isinstance(wrapped.sampler.inner, samplers.BatchSampler)


def test_parallel_setup_dataloader_wraps_batch_sampler():
    strategy = FakeParallelStrategy(process_index=0, process_count=2)
    loader = _make_loader(10)
    assert isinstance(loader.sampler, samplers.BatchSampler)
    inner_before = loader.sampler

    result = strategy.setup_dataloader(loader)

    # ReaxDataLoader.with_new_sampler returns a new loader. Wait, no, the old test did:
    # assert result is loader
    # because they did `data.sampler.sampler = sampler`.
    # Now we do `data = data.with_new_sampler(sampler)`.
    # So result is not loader.
    assert result is not loader
    outer = result.sampler
    assert isinstance(outer, samplers.DistributedSampler)
    assert outer.inner is inner_before
    assert outer._num_replicas == 2
    assert outer._process_index == 0


def test_parallel_setup_dataloader_shards_indices_across_replicas():
    n = 10
    c = 2
    r0 = FakeParallelStrategy(process_index=0, process_count=c).setup_dataloader(_make_loader(n))
    r1 = FakeParallelStrategy(process_index=1, process_count=c).setup_dataloader(_make_loader(n))

    idx0 = _inner_indices(r0)
    idx1 = _inner_indices(r1)

    # Both ranks yield a uniform step count of ceil(n / c).
    assert len(idx0) == math.ceil(n / c)
    assert len(list(r1.sampler)) == math.ceil(n / c)
    assert len(idx1) == math.ceil(n / c)
    # Union covers every sample exactly once (no loss when evenly divisible).
    assert sorted(idx0 + idx1) == list(range(n))
    assert set(idx0).isdisjoint(set(idx1))


def test_parallel_setup_dataloader_is_deterministic():
    n = 10
    strategy = FakeParallelStrategy(process_index=0, process_count=2)

    a = _inner_indices(strategy.setup_dataloader(_make_loader(n)))
    b = _inner_indices(strategy.setup_dataloader(_make_loader(n)))

    assert a == b
    assert len(a) == math.ceil(n / 2)


def test_parallel_setup_dataloader_padded_split_covers_dataset():
    # 7 samples, 2 replicas -> ceil(7/2) == 4 steps per rank (uniform). Rank 1 is
    # padded with an empty batch; real indices still cover every sample exactly once.
    n = 7
    c = 2
    r0 = FakeParallelStrategy(process_index=0, process_count=c).setup_dataloader(_make_loader(n))
    r1 = FakeParallelStrategy(process_index=1, process_count=c).setup_dataloader(_make_loader(n))

    idx0 = _inner_indices(r0)
    idx1 = _inner_indices(r1)

    # Both ranks yield the same step count.
    assert len(idx0) == math.ceil(n / c)
    assert len(list(r1.sampler)) == math.ceil(n / c)
    # Rank 0: 4 real indices, Rank 1: 3 real + 1 padding empty.
    assert len(idx0) == math.ceil(n / c)
    assert len(idx1) == math.ceil((n - 1) / c)
    assert min(idx0 + idx1) >= 0
    assert max(idx0 + idx1) <= n - 1
    assert set(idx0 + idx1) == set(range(n))
    # Disjoint real indices (no overlap).
    assert set(idx0).isdisjoint(set(idx1))


def test_parallel_setup_dataloader_default_process_count_is_replicas():
    # When process_index/process_count come from jax defaults (single process),
    # a single replica sees the full dataset.
    loader = reax.data.ReaxDataLoader(list(range(5)))
    sampler = samplers.DistributedSampler(
        samplers.RandomSampler(len(loader.dataset)), num_replicas=1, process_index=0
    )
    assert sorted(int(i) for i in sampler) == list(range(5))
    assert len(sampler) == 5


def test_parallel_strategy_is_abc():
    with pytest.raises(TypeError):
        _parallel.ParallelStrategy()
