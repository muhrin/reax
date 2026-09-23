import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from reax import testing
from reax.metrics import Average, Sum
from reax.strategies import JaxDdpStrategy


@pytest.mark.multiproc
def test_ddp_broadcast_pytree():
    testing.in_subprocess(_run_ddp_broadcast)()


def _run_ddp_broadcast():
    strategy = JaxDdpStrategy(platform="cpu", devices=2)
    # Each rank holds a rank-specific value; after broadcast everyone has rank-0's value.
    payload = {
        "x": jnp.full((3,), float(strategy.process_index)),
        "y": jnp.arange(4) + strategy.process_index,
    }
    broadcasted = strategy.broadcast(payload, src=0)
    assert jnp.all(broadcasted["x"] == 0.0)
    assert jnp.allclose(broadcasted["y"], jnp.arange(4))


@pytest.mark.multiproc
def test_ddp_broadcast_string():
    testing.in_subprocess(_run_ddp_broadcast_string)()


def _run_ddp_broadcast_string():
    strategy = JaxDdpStrategy(platform="cpu", devices=2)
    # `broadcast_one_to_all` requires identical leaf shapes across ranks, so every
    # rank must hold the same-length string; after broadcast everyone has src rank's.
    received = strategy.broadcast("rf", src=0)
    assert received == "rf"


@pytest.mark.multiproc
def test_ddp_broadcast_from_rank1():
    testing.in_subprocess(_run_ddp_broadcast_rank1)()


def _run_ddp_broadcast_rank1():
    strategy = JaxDdpStrategy(platform="cpu", devices=2)
    payload = jnp.full((2,), float(strategy.process_index))
    broadcasted = strategy.broadcast(payload, src=1)
    # Regardless of rank, everyone should now hold rank-1's value (1.0)
    assert jnp.all(broadcasted == 1.0)


@pytest.mark.multiproc
def test_ddp_all_reduce_mean():
    testing.in_subprocess(_run_ddp_all_reduce_mean)()


def _run_ddp_all_reduce_mean():
    strategy = JaxDdpStrategy(platform="cpu", devices=2)
    # Rank 0 has [0.0], rank 1 has [1.0] -> gathered (2,1); mean over it is 0.5
    value = jnp.array([float(strategy.process_index)])
    result = strategy.all_reduce(value, reduce_op="mean")
    assert result == 0.5


@pytest.mark.multiproc
def test_ddp_all_reduce_sum():
    testing.in_subprocess(_run_ddp_all_reduce_sum)()


def _run_ddp_all_reduce_sum():
    strategy = JaxDdpStrategy(platform="cpu", devices=2)
    # Rank 0 has [0.0], rank 1 has [1.0] -> gathered (2,1); sum over it is 1.0
    value = jnp.array([float(strategy.process_index)])
    result = strategy.all_reduce(value, reduce_op="sum")
    assert result == 1.0


@pytest.mark.multiproc
def test_ddp_all_gather():
    testing.in_subprocess(_run_ddp_all_gather)()


def _run_ddp_all_gather():
    strategy = JaxDdpStrategy(platform="cpu", devices=2)
    value = jnp.array([float(strategy.process_index)])
    gathered = strategy.all_gather(value)
    # process_allgather prepends a leading axis of length process_count
    assert jnp.asarray(gathered).shape[0] == 2
    flat = sorted(jnp.asarray(gathered).ravel().tolist())
    assert flat == [0.0, 1.0]


@pytest.mark.multiproc
def test_ddp_barrier():
    testing.in_subprocess(_run_ddp_barrier)()


def _run_ddp_barrier():
    strategy = JaxDdpStrategy(platform="cpu", devices=2)
    # Both the default (name=None) and a named barrier must not hang or crash.
    strategy.barrier()
    strategy.barrier(name="custom_barrier")
    assert strategy.process_count == 2


@pytest.mark.multiproc
def test_ddp_compute_metric():
    testing.in_subprocess(_run_ddp_compute_metric)()


def _run_ddp_compute_metric():
    strategy = JaxDdpStrategy(platform="cpu", devices=2)

    # Each rank builds a metric on different data, but identical values per rank
    # so the merged result is deterministic and checkable.
    # Rank 0 sees [1, 2, 3], rank 1 sees [4, 5, 6].
    rank = strategy.process_index
    values = jnp.array([rank * 3 + i + 1 for i in range(3)], dtype=jnp.float64)
    metric = Sum().update(values)

    # strategy.compute() gathers dynamic leaves over the process axis, unbatching back
    # to per-rank metrics, then merges them.
    result = strategy.compute(metric)
    # Sum across all ranks: (1+2+3) + (4+5+6) = 21
    assert result == 21.0


@pytest.mark.multiproc
def test_ddp_compute_mean_metric():
    testing.in_subprocess(_run_ddp_compute_mean)()


def _run_ddp_compute_mean():
    strategy = JaxDdpStrategy(platform="cpu", devices=2)
    rank = strategy.process_index
    values = jnp.array([rank * 3 + i + 1 for i in range(3)], dtype=jnp.float64)
    metric = Average().update(values)

    result = strategy.compute(metric)
    # Mean over all 6 values = (1+2+3+4+5+6) / 6 = 21/6 = 3.5
    assert result == 3.5
