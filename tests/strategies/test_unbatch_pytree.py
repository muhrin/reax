import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from reax.metrics import Sum
from reax.strategies._jax import unbatch_pytree


def test_unbatch_scalar_leaf():
    """A scalar (0-d) array leaf of shape (B,) is split into B scalars."""
    original = jnp.array(5.0)
    batched = jnp.array([1.0, 2.0, 3.0])

    result = unbatch_pytree(batched, original)

    assert len(result) == 3
    for entry, expected in zip(result, (1.0, 2.0, 3.0)):
        assert entry.shape == ()
        assert entry == expected


def test_unbatch_1d_vector_leaf():
    """A 1D array of shape (B, D) is split into B vectors of shape (D,)."""
    original = jnp.zeros(2)
    batched = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    result = unbatch_pytree(batched, original)

    assert len(result) == 3
    for k, expected in enumerate(
        (jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]), jnp.array([5.0, 6.0]))
    ):
        assert result[k].shape == (2,)
        assert jnp.allclose(result[k], expected)


def test_unbatch_2d_batch():
    """A 2D array of shape (B, H, W) is split into B arrays of shape (H, W)."""
    original = jnp.zeros((3, 4))
    batched = jnp.arange(4 * 3 * 4, dtype=jnp.float64).reshape(4, 3, 4)

    result = unbatch_pytree(batched, original)

    assert len(result) == 4
    for entry in result:
        assert entry.shape == (3, 4)
    # Verify the data is preserved, row by row
    assert jnp.allclose(result[0], batched[0])
    assert jnp.allclose(result[3], batched[3])


def test_unbatch_nested_dict():
    """A nested pytree (dict with a list) is split at the batch axis on every leaf."""
    original = {"a": jnp.zeros(2), "b": [jnp.zeros(3), jnp.zeros(1)]}
    batched = {
        "a": jnp.arange(5 * 2, dtype=jnp.float64).reshape(5, 2),
        "b": [
            jnp.arange(5 * 3, dtype=jnp.float64).reshape(5, 3),
            jnp.arange(5 * 1, dtype=jnp.float64).reshape(5, 1),
        ],
    }

    result = unbatch_pytree(batched, original)

    assert len(result) == 5
    for k, entry in enumerate(result):
        assert set(entry.keys()) == {"a", "b"}
        assert entry["a"].shape == (2,)
        assert entry["b"][0].shape == (3,)
        assert entry["b"][1].shape == (1,)
        assert jnp.allclose(entry["a"], batched["a"][k])
        assert jnp.allclose(entry["b"][0], batched["b"][0][k])
        assert jnp.allclose(entry["b"][1], batched["b"][1][k])


def test_unbatch_roundtrip_with_metric():
    """Round trip: partition a metric, batch over a fake process axis, then
    unbatch back to per-process metrics and merge them."""
    metric = Sum().update(jnp.array([1.0, 2.0, 3.0]))
    dynamic, static = eqx.partition(metric, eqx.is_array)

    # Simulate an all_gather that prepends a leading process axis (B=2)
    batched = jax.tree.map(lambda x: jnp.stack([x, x]), dynamic)
    unbatched = unbatch_pytree(batched, dynamic)
    unbatched = [eqx.combine(entry, static) for entry in unbatched]

    assert len(unbatched) == 2
    merged = unbatched[0].merge(unbatched[1])
    # Original accumulator is 6.0. Stacking it twice and merging sums to 12.0
    assert merged.compute() == 12.0
    # Each entry matches the original metric's accumulator
    for entry in unbatched:
        assert entry.compute() == 6.0


def test_unbatch_structure_mismatch_raises():
    """When the structure of `batched` and `original` differ, a ValueError is raised."""
    original = {"a": jnp.zeros(2), "b": [jnp.zeros(3)]}
    batched = {"a": jnp.zeros((5, 2)), "other": jnp.zeros((5, 3))}

    with pytest.raises(ValueError, match="Structure"):
        unbatch_pytree(batched, original)


def test_unbatch_mismatched_batch_size_raises():
    """Leaves with different leading dims raise a ValueError."""
    original = {"a": jnp.zeros(2), "b": jnp.zeros(3)}
    # batched['a'] has leading dim 5, batched['b'] has leading dim 4
    batched = {"a": jnp.zeros((5, 2)), "b": jnp.zeros((4, 3))}

    with pytest.raises(ValueError, match="same leading batch dimension"):
        unbatch_pytree(batched, original)


def test_unbatch_empty_batch_returns_empty_list():
    """A batch of size 0 returns an empty list (no unbatched entries)."""
    original = jnp.zeros(2)
    batched = jnp.zeros((0, 2))

    result = unbatch_pytree(batched, original)

    assert result == []


def test_unbatch_empty_pytree_returns_self():
    """A pytree with no leaves (empty dict) returns a single-element list with itself."""
    original = {}
    batched = {}

    result = unbatch_pytree(batched, original)

    assert result == [batched]


def test_unbatch_scalar_jax_array_returns_self():
    """A scalar jax array (0-d) batched leaf cannot be split, so it's returned as-is."""
    original = jnp.array(5.0)
    batched = jnp.array(7.0)  # 0-d -> shape[0] raises IndexError

    result = unbatch_pytree(batched, original)

    assert result == [batched]
    assert result[0].shape == ()


def test_unbatch_non_jax_original_leaf():
    """When an `original` leaf is not an array (e.g. a python scalar), target shape is
    None and the first element of each split is taken."""
    # Use a nested None in original to hit the None branch
    original = {"a": jnp.zeros(2), "flag": None}
    batched = {"a": jnp.arange(4 * 2, dtype=jnp.float64).reshape(4, 2), "flag": None}

    result = unbatch_pytree(batched, original)

    assert len(result) == 4
    for k, entry in enumerate(result):
        assert entry["a"].shape == (2,)
        assert entry["flag"] is None
        assert jnp.allclose(entry["a"], batched["a"][k])
