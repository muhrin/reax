import jax
import jax.numpy as jnp
import pytest

import reax
from reax import testing
from reax.demos import RandomDataset
from reax.metrics import Sum
from reax.strategies import JaxDdpStrategy


def _make_engine() -> reax.Engine:
    strategy = JaxDdpStrategy(platform="cpu", devices=2)
    return reax.Engine(strategy=strategy, logger=False)


@pytest.mark.multiproc
def test_engine_broadcast():
    testing.in_subprocess(_run_engine_broadcast)()


def _run_engine_broadcast():
    engine = _make_engine()
    payload = jnp.full((2,), float(engine.strategy.process_index))
    broadcasted = engine.broadcast(payload, src=0)
    assert jnp.all(broadcasted == 0.0)


@pytest.mark.multiproc
def test_engine_all_reduce():
    testing.in_subprocess(_run_engine_all_reduce)()


def _run_engine_all_reduce():
    engine = _make_engine()
    value = jnp.array([float(engine.strategy.process_index)])
    total = engine.all_reduce(value, reduce_op="sum")
    assert total == 1.0
    average = engine.all_reduce(value, reduce_op="mean")
    assert average == 0.5


@pytest.mark.multiproc
def test_engine_barrier():
    testing.in_subprocess(_run_engine_barrier)()


def _run_engine_barrier():
    engine = _make_engine()
    # Engine-level delegation with the default (name=None) argument must work.
    engine.barrier()
    assert engine.strategy.process_count == 2


@pytest.mark.multiproc
def test_engine_to_device():
    testing.in_subprocess(_run_engine_to_device)()


def _run_engine_to_device():
    engine = _make_engine()
    value = jnp.arange(4)
    moved = engine.to_device(value)
    device = engine.device
    moved = jax.device_get(moved)
    assert isinstance(device, jax.Device)
    assert jnp.allclose(moved, jnp.arange(4))


@pytest.mark.multiproc
def test_engine_compute_metric():
    testing.in_subprocess(_run_engine_compute_metric)()


def _run_engine_compute_metric():
    engine = _make_engine()
    rank = engine.strategy.process_index
    values = jnp.array([rank * 3 + i + 1 for i in range(3)], dtype=jnp.float64)
    result = engine.compute(Sum().update(values))
    # (1+2+3) + (4+5+6) = 21
    assert result == 21.0


@pytest.mark.multiproc
def test_engine_setup_dataloaders_sharded():
    testing.in_subprocess(_run_engine_setup_dataloaders)()


def _run_engine_setup_dataloaders():
    engine = _make_engine()
    ds = RandomDataset(size=4, length=10)
    loader = engine.setup_dataloaders(reax.ReaxDataLoader(ds, batch_size=2))
    # With 2 ranks each shard must cover half the (padded) dataset: 5 samples => batches of 2 => 3 batches
    batches = list(loader)
    assert len(batches) == 3
    # The two shards must be disjoint over the dataset items (each rank gets distinct half)
    # Batch shapes are (2, 4)
    assert batches[0].shape == (2, 4)
