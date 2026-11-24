import jax
import pytest

from reax import strategies, testing


def jax_strategy():
    strategy = strategies.JaxDdpStrategy(platform="cpu", devices=1)
    assert strategy.process_count == 1
    assert strategy.process_index == 0

    # Check that the JAX global methods agree
    assert jax.process_count() == 1
    assert jax.process_index() == 0


test_jax_strategy = pytest.mark.multiproc(testing.in_subprocess(jax_strategy))


def test_auto_platform_is_left_to_jax(monkeypatch):
    # "auto" isn't a JAX platform, passing it on makes JAX fail to initialise any backend
    updates = []
    monkeypatch.setattr(jax.config, "update", lambda name, value: updates.append((name, value)))
    monkeypatch.setattr(strategies.JaxDdpStrategy, "_distributed_init", lambda self, **kw: (0, 1))
    monkeypatch.setenv("REAX_PROCESS_ID", "0")
    monkeypatch.setenv("REAX_COORDINATOR_ADDRESS", "localhost:1234")
    monkeypatch.setenv("REAX_NUM_PROCESSES", "1")

    strategies.JaxDdpStrategy(platform="auto")
    assert ("jax_platforms", "auto") not in updates

    strategies.JaxDdpStrategy(platform="cpu")
    assert ("jax_platforms", "cpu") in updates


def test_auto_platform_not_passed_on(monkeypatch):
    # The platform given to `_init` is what's used for the config and the children's environment
    received = []
    monkeypatch.setattr(
        strategies.JaxDdpStrategy, "_init", lambda self, p, d: received.append(p) or (0, 1, [])
    )
    strategies.JaxDdpStrategy(platform="auto")
    strategies.JaxDdpStrategy(platform="cpu")
    assert received == [None, "cpu"]
