import jax.numpy as jnp
import numpy as np
import pytest

from reax.utils import arrays


def test_to_base_passthrough_scalar():
    assert arrays.to_base(3) == 3
    assert arrays.to_base(3.5) == 3.5


def test_to_base_passthrough_list():
    lst = [1, 2, 3]
    assert arrays.to_base(lst) is lst


def test_to_base_jnp_scalar():
    assert arrays.to_base(jnp.array(5.0)) == 5.0


def test_to_base_jnp_array():
    arr = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    assert arrays.to_base(arr) == [[1.0, 2.0], [3.0, 4.0]]


def test_to_scalar_numpy():
    assert arrays.to_scalar(np.array(7)) == 7


def test_to_scalar_jax():
    assert arrays.to_scalar(jnp.array(7.0)) == 7.0


def test_to_scalar_passthrough():
    assert arrays.to_scalar(7) == 7


def test_infer_backend_numpy():
    assert arrays.infer_backend({"a": np.ones(3)}) is np


def test_infer_backend_jax():
    assert arrays.infer_backend({"a": jnp.ones(3)}) is jnp


def test_infer_backend_default():
    assert arrays.infer_backend({"a": 1}) is jnp


def test_infer_backend_mixed_raises():
    with pytest.raises(ValueError, match="Cannot mix"):
        arrays.infer_backend({"a": np.ones(3), "b": jnp.ones(3)})
