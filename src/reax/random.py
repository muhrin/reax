"""Random number generation utilities.

REAX roots all randomness in a single integer seed held by the :class:`reax.Engine`.  From it an
:class:`flax.nnx.Rngs` is built with a set of *independent* named streams:

* ``params`` - model parameter initialisation (identical on every process),
* ``data`` - dataset splitting, shuffling, k-fold partitioning (identical on every process),
* ``dropout`` - stochastic layers (different on every process),
* ``default`` - anything else (identical on every process).

Because each stream is derived from the root key independently, drawing from one never shifts
the sequence of another; e.g. adding a dropout layer won't change how the data is split.

Model-side randomness (parameter initialisation, dropout, anything inside ``jit``) uses JAX.
Data-side randomness (splitting, shuffling, k-fold partitioning) is drawn from the JAX keys but
done with NumPy on the host, see :func:`numpy_rng`.  New data utilities should follow this rule.
"""

import os
import random
import secrets
import sys
from typing import Any, Final
import zlib

from flax import nnx
import jax
import numpy

# Only `seed_everything` is re-exported at the top level; use the rest as `reax.random.<name>`
__all__ = ("seed_everything",)

#: Streams that are identical across all processes.
SHARED_STREAMS: Final[tuple[str, ...]] = ("default", "params", "data")
#: Streams that are folded with the process index, so they differ between processes.
PER_PROCESS_STREAMS: Final[tuple[str, ...]] = ("dropout",)
#: All the named streams created by :func:`create_rngs`.
STREAMS: Final[tuple[str, ...]] = SHARED_STREAMS + PER_PROCESS_STREAMS

_MAX_SEED: Final[int] = 2**31 - 1


def seed_everything(seed: int | None, workers: bool = False):
    """Seed the python, numpy and (if installed) torch global random number generators.

    Note:
        JAX randomness in REAX is controlled by the ``seed`` passed to :class:`reax.Engine` (or
        :class:`reax.Trainer`), which also seeds the python and numpy global generators, so you
        only need to call this yourself if you aren't using an engine.
    """
    numpy.random.seed(seed)
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass

    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"

    return seed


def seed_globals(seed: int) -> None:
    """Seed the global python and numpy generators (and torch, but only if already imported).

    This keeps code that still uses the global generators in step with the engine seed, without
    paying for a torch import.
    """
    numpy.random.seed(seed)
    random.seed(seed)
    if (torch := sys.modules.get("torch")) is not None:
        torch.manual_seed(seed)


def draw_seed() -> int:
    """Draw a fresh seed from OS entropy."""
    return secrets.randbelow(_MAX_SEED + 1)


def _stream_id(name: str) -> int:
    """A stable (across processes and python sessions) integer id for a stream name."""
    return zlib.crc32(name.encode("utf-8"))


def create_rngs(seed: int, process_index: int = 0) -> nnx.Rngs:
    """Create the :class:`flax.nnx.Rngs` with REAX's independent named streams.

    Args:
        seed: The root seed.
        process_index: The global rank of this process, folded into the per-process streams
            (see :data:`PER_PROCESS_STREAMS`).

    Returns:
        The RNGs, with one stream per entry of :data:`STREAMS`.
    """
    root = jax.random.key(seed)
    keys = {}
    for name in STREAMS:
        key = jax.random.fold_in(root, _stream_id(name))
        if name in PER_PROCESS_STREAMS:
            key = jax.random.fold_in(key, process_index)
        keys[name] = key

    return nnx.Rngs(**keys)


def to_key(rng: "nnx.Rngs | nnx.RngStream | jax.Array | int", stream: str = "default") -> jax.Array:
    """Get a single JAX key from any of the commonly used RNG representations.

    Args:
        rng: An :class:`flax.nnx.Rngs` (a key is drawn from ``stream``), an
            :class:`flax.nnx.RngStream` (a key is drawn from it), a JAX key, or an integer seed.
        stream: The stream to draw from if ``rng`` is an :class:`flax.nnx.Rngs`.

    Returns:
        A JAX key.
    """
    if isinstance(rng, nnx.Rngs):
        return getattr(rng, stream)()
    if isinstance(rng, nnx.RngStream):
        return rng()
    if isinstance(rng, int):
        return jax.random.key(rng)

    return rng


def _key_words(key: jax.Array) -> list[int]:
    """Get all the words making up a JAX key as python integers (on the host)."""
    if jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key):
        key = jax.random.key_data(key)
    return [int(word) for word in numpy.asarray(key).ravel()]


def numpy_rng(
    rng: "nnx.Rngs | nnx.RngStream | jax.Array | int", *extra: int, stream: str = "data"
) -> numpy.random.Generator:
    """Get a NumPy generator, seeded from a JAX key, for host-side (data) randomness.

    REAX does all data-side randomness (splitting, shuffling, k-fold partitioning) with NumPy on
    the host.  The work happens where the data lives, without device round trips, XLA compilation
    or accelerator-backend quirks, while the seed still comes from the engine's RNGs.

    Args:
        rng: An :class:`flax.nnx.Rngs` (a key is drawn from ``stream``), an
            :class:`flax.nnx.RngStream` (a key is drawn from it), a JAX key, or an integer seed.
        *extra: Additional integers mixed into the seed, e.g. an epoch number.
        stream: The stream to draw from if ``rng`` is an :class:`flax.nnx.Rngs`.

    Returns:
        A NumPy generator.

    Note:
        Prefer using the generator directly.  If something really needs a plain integer seed
        (e.g. a sampler that stores it), draw one from the generator:
        ``numpy_rng(rngs).integers(2**31)``.
    """
    return numpy.random.default_rng([*_key_words(to_key(rng, stream=stream)), *extra])


def rngs_state_dict(rngs: nnx.Rngs) -> dict[str, dict[str, Any]]:
    """Get a serialisable (e.g. via msgpack) representation of the RNGs state."""
    state = {}
    for name, stream in vars(rngs).items():
        if isinstance(stream, nnx.RngStream):
            state[name] = {
                "key": numpy.asarray(jax.random.key_data(stream.key.get_value())),
                "count": numpy.asarray(stream.count.get_value()),
            }
    return state


def load_rngs_state_dict(rngs: nnx.Rngs, state: dict[str, dict[str, Any]]) -> None:
    """Restore the RNGs state from a dictionary created by :func:`rngs_state_dict`."""
    for name, stream_state in state.items():
        stream = getattr(rngs, name, None)
        if not isinstance(stream, nnx.RngStream):
            continue
        impl = jax.random.key_impl(stream.key.get_value())
        stream.key.set_value(
            jax.random.wrap_key_data(jax.numpy.asarray(stream_state["key"]), impl=impl)
        )
        dtype = stream.count.get_value().dtype
        stream.count.set_value(jax.numpy.asarray(stream_state["count"], dtype=dtype))
