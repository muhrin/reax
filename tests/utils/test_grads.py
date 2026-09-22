import jax.numpy as jnp
import numpy as np
import pytest

from reax.utils import grads


def test_grad_norm_computes_per_param_and_total():
    g = {"w1": jnp.array([3.0, 4.0]), "w2": jnp.array([5.0, 12.0])}
    norms = grads.grad_norm(g, 2.0)
    # per-param 2-norms
    assert norms["grad_2.0_norm/w1"] == pytest.approx(5.0)
    assert norms["grad_2.0_norm/w2"] == pytest.approx(13.0)
    # overall: sqrt(5^2 + 13^2)
    assert norms["grad_2.0_norm_total"] == pytest.approx((25.0 + 169.0) ** 0.5)


def test_grad_norm_inf():
    g = {"a": jnp.array([1.0, -7.0, 2.0])}
    norms = grads.grad_norm(g, "inf")
    assert norms["grad_inf_norm/a"] == pytest.approx(7.0)
    assert norms["grad_inf_norm_total"] == pytest.approx(7.0)


def test_grad_norm_nested_keys_with_separator():
    g = {"layer": {"dense": jnp.array([1.0, 2.0])}}
    norms = grads.grad_norm(g, 2.0, group_separator=".")
    assert "grad_2.0_norm.layer.dense" in norms
    assert norms["grad_2.0_norm_total"] == pytest.approx(np.linalg.norm([1.0, 2.0]))


def test_grad_norm_none_values_skipped():
    g = {"a": jnp.array([1.0, 2.0]), "b": None}
    norms = grads.grad_norm(g, 2.0)
    assert "grad_2.0_norm/a" in norms
    assert not any("b" in k for k in norms)


def test_grad_norm_invalid_norm_type():
    with pytest.raises(ValueError, match=r"norm_type.*positive"):
        grads.grad_norm({"a": jnp.array([1.0])}, 0.0)
