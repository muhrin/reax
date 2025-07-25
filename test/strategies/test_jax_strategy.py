import functools
import multiprocessing

from reax import strategies


def in_subprocess(fn):
    """Run a test within a subprocess, necessary for certain strategies."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with multiprocessing.Pool(processes=1) as pool:
            return pool.apply(fn, *args, **kwargs)

    return wrapper


def jax_strategy():
    import jax

    strategy = strategies.JaxDdpStrategy(platform="cpu", devices=1)
    assert strategy.process_count == 1
    assert strategy.process_index == 0

    # Check that the JAX global methods agree
    assert jax.process_count() == 1
    assert jax.process_index() == 0


test_jax_strategy = in_subprocess(jax_strategy)
