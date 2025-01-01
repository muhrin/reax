from reax import hooks

__all__ = ("Checkpointer",)


class Checkpointer(hooks.TrainerListener):
    r"""Interface for trainer checkpointing."""
