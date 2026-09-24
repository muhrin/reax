import socket

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from reax import strategies
from reax.metrics import Average, Sum
from reax.strategies._jax import JaxDdpStrategy
from reax.strategies._single_device import SingleDevice


def _make_metric(values: jnp.ndarray):
    return Sum().update(values)


def test_single_device_creation():
    strategy = SingleDevice("cpu")
    assert strategy.device.platform == "cpu"

    auto_strategy = SingleDevice("auto")
    assert auto_strategy.device is not None
    assert isinstance(auto_strategy.device.platform, str)


def test_single_device_properties():
    strategy = SingleDevice("cpu")
    assert strategy.is_global_zero
    assert strategy.device is strategy._device


def test_single_device_to_from_device():
    strategy = SingleDevice("cpu")
    value = jnp.arange(6).reshape(2, 3)
    on_device = strategy.to_device(value)
    assert on_device.device == strategy.device
    back = strategy.from_device(on_device)
    assert isinstance(back, (jnp.ndarray, jax.Array, np.ndarray))
    assert np.allclose(np.asarray(back), value)


def test_single_device_to_device_pytree():
    strategy = SingleDevice("cpu")
    pytree = {"a": jnp.ones(3), "b": [jnp.zeros(2), jnp.array(1.0)]}
    on_device = strategy.to_device(pytree)
    restored = strategy.from_device(on_device)
    assert isinstance(restored["a"], (jnp.ndarray, jax.Array, np.ndarray))
    assert jnp.allclose(restored["a"], jnp.ones(3))
    assert jnp.allclose(restored["b"][0], jnp.zeros(2))
    assert restored["b"][1] == 1.0


def test_single_device_broadcast_is_identity():
    strategy = SingleDevice("cpu")
    value = jnp.arange(3)
    assert strategy.broadcast(value) is value
    assert strategy.broadcast(value, src=1) is value


def test_single_device_all_gather_is_identity():
    strategy = SingleDevice("cpu")
    value = {"x": jnp.arange(4)}
    assert strategy.all_gather(value) is value


@pytest.mark.parametrize("reduce_op", ("mean", "sum", "max"))
def test_single_device_all_reduce_is_identity(reduce_op):
    strategy = SingleDevice("cpu")
    value = jnp.array([2.0, 3.0])
    assert strategy.all_reduce(value, reduce_op=reduce_op) is value


def test_single_device_barrier_is_noop():
    strategy = SingleDevice("cpu")
    assert strategy.barrier() is None
    assert strategy.barrier(name="name") is None


def test_single_device_compute_calls_metric_compute():
    strategy = SingleDevice("cpu")
    metric = _make_metric(jnp.array([1.0, 2.0, 3.0]))
    assert strategy.compute(metric) == 6.0


def test_single_device_compute_average_metric():
    strategy = SingleDevice("cpu")
    metric = Average().update(jnp.array([2.0, 4.0, 6.0]))
    assert strategy.compute(metric) == 4.0


def test_single_device_teardown_is_noop():
    strategy = SingleDevice("cpu")
    assert strategy.teardown() is None


def test_single_device_abc_conformance():
    # A fully-implemented strategy must be instantiable (no missing abstract methods)
    assert isinstance(SingleDevice("cpu"), strategies.Strategy)


# ---------------------------------------------------------------------------
# JaxDdpStrategy helpers that do not require starting the distributed runtime
# ---------------------------------------------------------------------------


def test_jax_ddp_probe_local_device_count():
    assert JaxDdpStrategy.probe_local_device_count() == jax.local_device_count()


@pytest.mark.parametrize("min_port, max_port", ((50000, 51000), (49152, 65535)))
def test_jax_ddp_get_available_port(min_port, max_port):
    port = JaxDdpStrategy.get_available_port(min_port=min_port, max_port=max_port)
    assert min_port <= port <= max_port


def test_jax_ddp_get_available_port_exhaustion_raises():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("localhost", 50001))
    try:
        with pytest.raises(OSError, match="Could not find an available port"):
            JaxDdpStrategy.get_available_port(min_port=50001, max_port=50001, max_attempts=3)
    finally:
        s.close()


def test_jax_ddp_get_available_port_retries_past_in_use_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("localhost", 50010))
    try:
        port = JaxDdpStrategy.get_available_port(
            min_port=50000,
            max_port=50020,
            max_attempts=50,
        )
        assert 50000 <= port <= 50020
    finally:
        s.close()
