from typing import TYPE_CHECKING, Any, Union

import beartype
import equinox
import jax
import jaxtyping as jt
import optax

if TYPE_CHECKING:
    import reax

__all__ = ("Optimizer",)


@jt.jaxtyped(typechecker=beartype.beartype)
class Optimizer(equinox.Module):
    optimizer: optax.GradientTransformation
    state: optax.OptState
    update_count: Union[
        int, jt.Int[jax.Array, ""]
    ]  # Total number of times this optimizer was used to parameters

    @classmethod
    def from_params(cls, opt: optax.GradientTransformation, params: optax.Params) -> "Optimizer":
        return cls(opt, opt.init(params))

    def __init__(self, opt: optax.GradientTransformation, state: optax.OptState, count: int = 0):
        self.optimizer = opt
        self.state = state
        self.update_count = count

    # @equinox.filter_jit
    def update(self, params: optax.Params, grad: jt.PyTree) -> tuple[Any, "Optimizer"]:
        """
        Return an updated version of the passed parameters using the passed gradients

        :return: a tuple of the updated parameters and an instance of this optimizer updated with
            the new state
        """
        updates, new_state = self.optimizer.update(grad, self.state, params=params)
        params = optax.apply_updates(params, updates)

        count = getattr(new_state, "gradient_step", self.update_count + 1)
        return params, type(self)(self.optimizer, new_state, count=count)

    def update_module(self, module: "reax.Module", grad: jt.PyTree) -> "Optimizer":
        """Perform an inplace update of the module parameters given the passed gradients.

        :return: an instance of this optimizer updated with the new state
        """
        params, updated_opt = self.update(module.parameters(), grad)
        module.set_parameters(params)
        return updated_opt
