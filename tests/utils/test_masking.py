import jax.numpy as jnp
import pytest

from reax.utils import masking


def test_prepare_mask():
    vals = jnp.array([[1.0, 2.0], [3.0, 4.0]])

    # A 1D entity mask is reshaped to broadcast against the values
    mask = masking.prepare_mask(vals, jnp.array([True, False]))
    assert mask.shape == (2, 1)

    # A full-shape mask is kept as is
    mask = masking.prepare_mask(vals, jnp.array([[True, False], [True, True]]))
    assert mask.shape == vals.shape

    with pytest.raises(ValueError):
        masking.prepare_mask(vals, jnp.array([True]))
    with pytest.raises(ValueError):
        masking.prepare_mask(vals, jnp.zeros((2, 3), dtype=bool))


def test_count_valid():
    vals = jnp.array([[1.0, 2.0], [3.0, 4.0]])

    assert masking.count_valid(vals, None) == vals.size
    # Entity mask: each selected entity counts once per feature
    assert masking.count_valid(vals, masking.prepare_mask(vals, jnp.array([True, False]))) == 2
    mask = masking.prepare_mask(vals, jnp.array([[True, False], [True, True]]))
    assert masking.count_valid(vals, mask) == 3
