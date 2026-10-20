import jax.numpy as jnp
import optax

import reax


def test_optimizer_update_forwards_extra_args():
    """`value` should reach the wrapped transform's `update` as a keyword argument, not nested
    under a literal `extra_args` kwarg.
    """
    received_kwargs = {}

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None, **extra_args):
        del params
        received_kwargs.update(extra_args)
        return updates, state

    transform = optax.GradientTransformationExtraArgs(init_fn, update_fn)

    params = {"w": jnp.array(1.0)}
    optimizer = reax.optimizers.Optimizer.from_params(transform, params)
    grad = {"w": jnp.array(0.5)}

    optimizer.update(params, grad, value=3.0)

    assert received_kwargs == {"value": 3.0}
