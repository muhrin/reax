from collections.abc import Iterable
import contextlib
import functools
import logging
import os
from typing import TYPE_CHECKING, Any, Optional, TypeVar, Union
import weakref

import beartype
from flax import nnx
import fsspec.utils
import jax
import jaxtyping as jt

from . import data as data_
from . import hooks
from . import loggers as loggers_
from . import modules, optimizers, strategies, typing
from .training import _logger_connector
from .utils import events

if TYPE_CHECKING:
    import reax

__all__ = ("Engine",)

_OutT = TypeVar("_OutT")
_LOGGER = logging.getLogger(__name__)


class Engine:
    def __init__(
        self,
        accelerator: str = "auto",
        strategy: "Union[str, reax.Strategy]" = "auto",
        devices: list[int] | str | int = "auto",
        precision=None,
        logger: Union["reax.Logger", Iterable["reax.Logger"], bool] | None = True,
        listeners: "Optional[list[reax.TrainerListener], reax.TrainerListener]" = None,
        deterministic: bool = False,
        rngs: nnx.Rngs = None,
        default_root_dir: typing.Path | None = None,
    ):
        if deterministic:
            _LOGGER.warning("`deterministic=True` is not supported yet, ignoring.")
        if precision is not None:
            _LOGGER.warning(
                "`precision` other than None is not supported yet, ignoring.  "
                "There is still ongoing discussion on how to support this in JAX, "
                "see e.g.: https://github.com/jax-ml/jax/issues/22688"
            )
        # Params

        # State
        if isinstance(strategy, strategies.Strategy):
            self._strategy = strategy
        else:
            self._strategy = strategies.create(strategy, accelerator, devices=devices)
        self._default_root_dir = (
            os.fspath(default_root_dir) if default_root_dir is not None else os.getcwd()
        )
        self._events = events.EventGenerator[hooks.TrainerListener](
            default_args=(weakref.proxy(self),)
        )
        self._rngs = rngs if rngs is not None else nnx.Rngs(0)
        self._loggers: list[loggers_.Logger] = _init_loggers(logger, self.default_root_dir)
        self._logging = _logger_connector.TrainerLogging()

        if isinstance(listeners, hooks.TrainerListener):
            listeners = [listeners]
        if listeners:
            for listener in listeners:
                self._events.add_listener(listener)

    def finalize(self):
        """Clean up the trainer.

        After this called this object should no longer be interacted with.
        """
        self._events = None
        self._loggers = None

    @property
    def default_root_dir(self) -> str:
        """Get the fallback directory used for loggers and other components when not explicitly
        specified."""
        if _is_local_file_protocol(self._default_root_dir):
            return os.path.normpath(os.path.expanduser(self._default_root_dir))

        return self._default_root_dir

    @property
    def logger(self) -> Optional["reax.Logger"]:
        r"""Get the first (and main) logger."""
        if not self._loggers:
            return None

        return self._loggers[0]

    @property
    def loggers(self) -> list["reax.Logger"]:
        """Get all the loggers."""
        return self._loggers

    @loggers.setter
    def loggers(self, loggers: list["reax.Logger"] | None) -> None:
        """Loggers function."""
        self._loggers = loggers if loggers else []

    @property
    def rngs(self) -> nnx.Rngs:
        return self._rngs

    @property
    def strategy(self) -> strategies.Strategy:
        """Strategy function."""
        return self._strategy

    @property
    def is_global_zero(self) -> bool:
        """Whether this rank is rank zero."""
        return self._strategy.is_global_zero

    @property
    def local_process_index(self) -> int:
        """Rank of the process on the current host (0 → local device count − 1)."""
        return getattr(self.strategy, "local_rank", 0)

    @property
    def node_rank(self) -> int:
        return getattr(self.strategy, "node_rank", 0)

    @property
    def process_index(self) -> int:
        """Global rank of the current process across all process count (0 → process count − 1)."""
        return getattr(self.strategy, "process_index", 0)

    @property
    def process_count(self) -> int:
        """Total number of processes across all hosts."""
        return getattr(self.strategy, "process_count", 1)

    @property
    def device(self) -> jax.Device:
        """The current device in use"""
        return self._strategy.device

    @contextlib.contextmanager
    def default_device(self):
        """Context manager that explicitly sets the strategy's device as the default for the
        duration of the context"""
        with jax.default_device(self.device):
            yield

    def setup(self, *args) -> Any:
        setup_fn = functools.partial(_setup, engine=self)
        res = tuple(map(setup_fn, args))
        if len(args) == 1:
            return res[0]

        return res

    def setup_dataloaders(
        self, *args
    ) -> "Union[reax.data.DeviceDataLoader, list[reax.data.DeviceDataLoader]]":
        loaders = list(map(self.strategy.setup_dataloader, args))
        # Wrap in DeviceLoader so all data is already on the selected device
        loaders = [data_.DeviceDataLoader(loader, self.device) for loader in loaders]
        if len(args) == 1:
            return loaders[0]

        return loaders

    def to_device(self, data: jt.PyTree) -> jt.PyTree:
        return self._strategy.to_device(data)

    def barrier(self, name: str | None = None) -> None:
        """Wait for all processes to enter this call.

        Use this to synchronize all parallel processes, but only if necessary, otherwise the
        overhead of synchronization will cause your program to slow down.
        This method needs to be called on all processes. Failing to do so will cause your program
        to stall forever.

        """
        self._strategy.barrier(name=name)

    def broadcast(self, obj: jt.PyTree, src: int = 0) -> jt.PyTree:
        r"""Send a tensor from one process to all others.

        This method needs to be called on all processes. Failing to do so will cause your program
        to stall forever.

        Args:
            obj: The object to broadcast to all other members. Any pytree object is supported.
            src: The (global) rank of the process that should send the data to all others.

        Return:
            The transferred data, the same value on every rank.
        """
        return self._strategy.broadcast(obj, src=src)

    def all_reduce(self, obj: jt.PyTree, reduce_op: str = "mean") -> jt.PyTree:
        """Reduces a tensor from several distributed processes to one aggregated tensor.

        Args:
            obj: the pytree to sync and reduce
            reduce_op: the reduction operation. Defaults to 'mean'/'avg'.
                Can also be a string 'sum' to calculate the sum during reduction.

        Return:
            reduced value
        """
        return self._strategy.all_reduce(obj, reduce_op=reduce_op)

    def call(self, name: str, *args, **kwargs):
        """Call an even hook any necessary arguments"""
        self._events.fire_event(getattr(hooks.TrainerListener, name), *args, **kwargs)

    def compute(self, metric: "reax.Metric[_OutT]") -> _OutT:
        """Compute the value of a metric, unlike metric.compute(), in a parallel setting this method
        will compute the value across all processes."""
        return self._strategy.compute(metric)


@functools.singledispatch
def _setup(arg, engine: Engine) -> Any:
    raise TypeError(f"Unknown argument type: {arg.__class__.__name__}")


@_setup.register
def _(arg: optimizers.Optimizer, engine: Engine) -> optimizers.Optimizer:
    return optimizers.DistributedOptimizer(
        opt=arg.optimizer, state=arg.state, engine=engine, count=arg.update_count
    )


@_setup.register
def _(arg: modules.Module, engine: Engine) -> modules.Module:
    return arg.set_parameters(engine.to_device(arg.parameters()))


@_setup.register
def _(arg: data_.DataLoader, engine: Engine) -> data_.DataLoader:
    return engine.strategy.setup_dataloader(arg)


@jt.jaxtyped(typechecker=beartype.beartype)
def _init_loggers(
    logger: Union["reax.Logger", Iterable["reax.Logger"], bool] | None,
    default_root_dir: str,
) -> list["reax.Logger"]:
    """Init loggers."""
    if isinstance(logger, loggers_.Logger):
        return [logger]

    if isinstance(logger, (bool, type(None))):
        if logger:
            return [loggers_.CsvLogger(default_root_dir)]

        return []

    return list(logger)


def _is_local_file_protocol(path: typing.Path) -> bool:
    """Is local file protocol."""
    return fsspec.utils.get_protocol(str(path)) == "file"
