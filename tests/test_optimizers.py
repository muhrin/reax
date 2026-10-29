import jax
import jax.numpy as jnp
import optax

import reax


def test_optimizer_update_forwards_extra_args():
    """`value` should reach the wrapped transform's `update` as a keyword argument, not nested
    under a literal `extra_args` kwarg.
    """
    received_keys = set()

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None, **extra_args):
        del params
        # `update` is jitted so only record the (static) names, and use the value in the result
        received_keys.update(extra_args)
        return jax.tree.map(lambda u: u * extra_args["value"], updates), state

    transform = optax.GradientTransformationExtraArgs(init_fn, update_fn)

    params = {"w": jnp.array(1.0)}
    optimizer = reax.optimizers.Optimizer.from_params(transform, params)
    grad = {"w": jnp.array(0.5)}

    new_params, _ = optimizer.update(params, grad, value=3.0)

    assert received_keys == {"value"}
    assert jnp.allclose(new_params["w"], 1.0 + 0.5 * 3.0)
