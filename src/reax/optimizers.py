import functools
from typing import TYPE_CHECKING, Any, Union

import equinox
import jax
import jax.numpy as jnp
import jaxtyping as jt
import optax

if TYPE_CHECKING:
    import reax

__all__ = ("Optimizer",)


class Optimizer(equinox.Module):
    optimizer: optax.GradientTransformation
    state: optax.OptState
    update_count: Union[
        int, jt.Int[jax.Array, ""]
    ]  # Total number of times this optimizer was used to parameters

    @classmethod
    def from_params(cls, opt: optax.GradientTransformation, params: optax.Params) -> "Optimizer":
        """From params."""
        return cls(opt, opt.init(params))

    def __init__(
        self,
        opt: optax.GradientTransformation,
        state: optax.OptState,
        count: Union[int, jt.Int[jax.Array, ""]] = 0,
    ):
        """Init function."""
        self.optimizer = opt
        self.state = state
        self.update_count = count

    @equinox.filter_jit
    def update(self, params: optax.Params, grad: jt.PyTree) -> tuple[Any, "Optimizer"]:
        """Return an updated version of the passed parameters using the passed gradients.

        :return: A tuple of the updated parameters and an instance of this optimizer updated with
            the new state.
        :rtype: tuple[Any, "Optimizer"]
        """
        updates, new_state = self.optimizer.update(grad, self.state, params=params)
        params = optax.apply_updates(params, updates)

        count = getattr(new_state, "gradient_step", self.update_count + 1)
        return params, type(self)(self.optimizer, new_state, count=count)

    def update_module(self, module: "reax.Module", grad: jt.PyTree) -> "Optimizer":
        """Perform an inplace update of the module parameters given the passed gradients.

        :return: An instance of this optimizer updated with the new state.
        :rtype: "Optimizer"
        """
        new_params, new_state = _update(self.optimizer, self.state, grad, module.parameters())
        module.set_parameters(new_params)
        count = getattr(new_state, "gradient_step", self.update_count + 1)
        return type(self)(self.optimizer, new_state, count=count)


@functools.partial(jax.jit, static_argnames=("optimizer",), donate_argnames="params")
def _update(optimizer: optax.GradientTransformation, state, grad: dict, params):
    """Jax jitted function that performs an optimizer update based on the passed gradients and
    parameters."""
    updates, new_state = optimizer.update(grad, state, params=params)
    params = optax.apply_updates(params, updates)

    return params, new_state


mock_optimizer = optax.GradientTransformation(
    init=lambda params: {},
    update=lambda grad, opt_state, *_, **__: (jax.tree.map(jnp.zeros_like, grad), opt_state),
)
