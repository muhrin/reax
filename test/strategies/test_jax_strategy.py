import jax

from reax import strategies, testing


def jax_strategy():

    strategy = strategies.JaxDdpStrategy(platform="cpu", devices=1)
    assert strategy.process_count == 1
    assert strategy.process_index == 0

    # Check that the JAX global methods agree
    assert jax.process_count() == 1
    assert jax.process_index() == 0


test_jax_strategy = testing.in_subprocess(jax_strategy)
