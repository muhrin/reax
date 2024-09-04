from typing import Any

import beartype
import equinox
import jaxtyping as jt
import optax

__all__ = ("Optimizer",)


@jt.jaxtyped(typechecker=beartype.beartype)
class Optimizer(equinox.Module):
    optimizer: optax.GradientTransformation
    state: optax.OptState

    @classmethod
    def from_params(cls, opt: optax.GradientTransformation, params: optax.Params) -> "Optimizer":
        return cls(opt, opt.init(params))

    def __init__(self, opt: optax.GradientTransformation, state: optax.OptState):
        self.optimizer = opt
        self.state = state

    @equinox.filter_jit
    def update(self, params: optax.Params, grads) -> tuple[Any, "Optimizer"]:
        updates, new_state = self.optimizer.update(grads, self.state, params=params)
        params = optax.apply_updates(params, updates)
        return params, type(self)(self.optimizer, new_state)
