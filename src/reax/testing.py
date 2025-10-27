import functools
import multiprocessing as mp


def in_subprocess(fn):
    """Run a test within a subprocess, necessary for certain strategies."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # We need to use the 'spawn' method as otherwise the initialised JAX distribution from this
        # process will still be active
        spawn_ctx = mp.get_context("spawn")
        with spawn_ctx.Pool(processes=1) as pool:
            return pool.apply(fn, *args, **kwargs)

    return wrapper
