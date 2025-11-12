import contextlib
import functools
import logging
import os
from typing import TYPE_CHECKING, Any, TypeVar
import weakref

import beartype
from flax import nnx
import fsspec.utils
import jax
import jaxtyping as jt

from . import data as data_
from . import hooks
from . import loggers as loggers_
from . import modules, optimizers, strategies
from .data import DeviceDataLoader
from .training import _logger_connector
from .utils import events

if TYPE_CHECKING:
    import reax

__all__ = ("Engine",)

_OutT = TypeVar("_OutT")
_LOGGER = logging.getLogger(__name__)


class Engine:
    """
    The central execution and orchestration component for distributed training.

    The Engine serves as the high-level interface that binds a specific
    Strategy (e.g., local, DDP, sharded) to the execution flow,
    providing standardized methods necessary to build training loops or a full
    Trainer.

    It encapsulates the environment setup defined by the chosen strategy,
    managing details like accelerator configuration, process launching (e.g.,
    via MPI or Python's multiprocessing), and backend initialization.

    Attributes:
        strategy (object): The execution strategy defining the environment and distributed
            communication.
    """

    def __init__(
        self,
        accelerator: str = "auto",
        strategy: "str | reax.Strategy" = "auto",
        devices: list[int] | str | int = "auto",
        precision=None,
        logger: "reax.Logger | Iterable[reax.Logger] | bool | None" = True,
        listeners: "list[reax.TrainerListener] | reax.TrainerListener | None" = None,
        deterministic: bool = False,
        rngs: nnx.Rngs = None,
        default_root_dir: "reax.types.Path | None" = None,
    ):
        """
        Initializes the Engine with execution parameters and components.

        The Engine sets up the chosen execution strategy and initializes core
        components like loggers, event listeners, and the random number generator
        state.

        Args:
            accelerator (str): The type of hardware accelerator to use, e.g., ``'gpu'``, ``'tpu'``,
                or ``'cpu'``. Defaults to ``'auto'``.
            strategy (str | reax.Strategy): The distributed training strategy to use, e.g.,
                ``'ddp'``, ``'fsdp'``, or a :class:`reax.Strategy` instance. Defaults to ``'auto'``.
            devices (list[int] | str | int): The specific devices to target. Can be a list of
                indices, ``'auto'``, or an integer count.
            precision: Currently not supported in JAX/nnx, and a warning is issued if set.
            logger (reax.Logger | Iterable[reax.Logger] | bool | None): One or more loggers, or
                ``True`` (for default CSVLogger) or ``False`` (to disable logging).
            listeners (list[reax.TrainerListener] | reax.TrainerListener | None): A single or list
                of :class:`reax.TrainerListener` objects for hooking into events.
            deterministic (bool): Currently not supported, a warning is issued if set.
            rngs (nnx.Rngs): The initial random number generator state used for all training steps.
                Defaults to a new :class:`nnx.Rngs(0)`.
            default_root_dir (typing.Path | None): The default directory path for saving logs and
                checkpoints if not specified elsewhere. Defaults to the current working directory.
        """
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

        After this is called, this object should no longer be interacted with,
        as internal components (like event listeners and loggers) are cleared.
        """
        self._events = None
        self._loggers = None

    @property
    def default_root_dir(self) -> str:
        """
        Get the fallback directory used for loggers and other components when not explicitly
        specified, resolving path variables (e.g., ``~``) if necessary.

        Returns:
            str: The resolved default root directory path.
        """
        if _is_local_file_protocol(self._default_root_dir):
            return os.path.normpath(os.path.expanduser(self._default_root_dir))

        return self._default_root_dir

    @property
    def logger(self) -> "reax.Logger | None":
        r"""
        Get the first (and main) logger configured for the Engine.

        Returns:
            reax.Logger | None: The main logger, or None if no loggers are configured.
        """
        if not self._loggers:
            return None

        return self._loggers[0]

    @property
    def loggers(self) -> list["reax.Logger"]:
        """
        Get all the loggers configured for the Engine.

        Returns:
            list[reax.Logger]: A list of all configured loggers.
        """
        return self._loggers

    @loggers.setter
    def loggers(self, loggers: list["reax.Logger"] | None) -> None:
        """
        Set the list of loggers.

        Args:
            loggers (list[reax.Logger] | None): The new list of loggers.
        """
        self._loggers = loggers if loggers else []

    @property
    def rngs(self) -> nnx.Rngs:
        """
        The current random number generator state used by the Engine.

        Returns:
            nnx.Rngs: The RNG state object.
        """
        return self._rngs

    @property
    def strategy(self) -> strategies.Strategy:
        """
        The execution strategy currently employed by the Engine.

        Returns:
            strategies.Strategy: The underlying strategy object.
        """
        return self._strategy

    @property
    def is_global_zero(self) -> bool:
        """
        Whether this process is the global rank zero process.

        Returns:
            bool: True if the current process has a global rank of zero.
        """
        return self._strategy.is_global_zero

    @property
    def local_process_index(self) -> int:
        """
        Rank of the process on the current host (0 → local device count − 1).

        Returns:
            int: The local rank.
        """
        return getattr(self.strategy, "local_rank", 0)

    @property
    def node_rank(self) -> int:
        """
        The rank of the current node (host) among all nodes.

        Returns:
            int: The node rank.
        """
        return getattr(self.strategy, "node_rank", 0)

    @property
    def process_index(self) -> int:
        """
        Global rank of the current process across all hosts (0 → total process count − 1).

        Returns:
            int: The global rank.
        """
        return getattr(self.strategy, "process_index", 0)

    @property
    def process_count(self) -> int:
        """
        Total number of processes across all hosts participating in the distributed task.

        Returns:
            int: The total number of processes.
        """
        return getattr(self.strategy, "process_count", 1)

    @property
    def device(self) -> jax.Device:
        """
        The JAX device instance currently assigned to this process.

        Returns:
            jax.Device: The assigned JAX device.
        """
        return self._strategy.device

    @contextlib.contextmanager
    def default_device(self):
        """
        Context manager that explicitly sets the strategy's device as the default for the
        duration of the context, ensuring subsequent operations target this device.

        Yields:
            None: The context manager yields nothing.
        """
        with jax.default_device(self.device):
            yield

    def setup(self, *args) -> Any:
        """
        Prepares and transforms objects (Modules, Optimizers, DataLoaders) for distributed use
        according to the current strategy.

        For example, this method wraps Optimizers in a
        :class:`~reax.optimizers.DistributedOptimizer` and moves Module parameters to the
        correct device.

        Args:
            *args: One or more objects (:class:`~reax.modules.Module`,
                :class:`~reax.optimizers.Optimizer`, or :class:`~reax.data.DataLoader`) to set up.

        Returns:
            Any: The set up object(s), with distributed wrappers applied. If a single object was
            passed, a single object is returned; otherwise, a tuple is returned.
        """
        setup_fn = functools.partial(_setup, engine=self)
        res = tuple(map(setup_fn, args))
        if len(args) == 1:
            return res[0]

        return res

    def setup_dataloaders(
        self, *args
    ) -> "reax.data.DeviceDataLoader | list[reax.data.DeviceDataLoader]":
        """
        Prepares and wraps dataloaders for the distributed environment.

        This delegates to the strategy to potentially shard the data and wraps
        the result in a :class:`~reax.data.DeviceDataLoader` to ensure data is
        moved to the correct device automatically.

        Args:
            *args: One or more :class:`reax.DataLoader` instances.

        Returns:
            reax.data.DeviceDataLoader | list[reax.data.DeviceDataLoader]: The prepared
            dataloader(s), wrapped as a :class:`~reax.data.DeviceDataLoader`.
        """
        loaders = list(map(self.strategy.setup_dataloader, args))
        # Wrap in DeviceLoader so all data is already on the selected device
        loaders = list(map(self._wrap_loader, loaders))
        if len(args) == 1:
            return loaders[0]

        return loaders

    def to_device(self, data: jt.PyTree) -> jt.PyTree:
        """
        Moves the provided PyTree data structure to the Engine's assigned device.

        Args:
            data (jaxtyping.PyTree): The PyTree object (e.g., tensors, arrays) to move.

        Returns:
            jaxtyping.PyTree: The data moved to the device.
        """
        return self._strategy.to_device(data)

    def barrier(self, name: str | None = None) -> None:
        """
        Wait for all processes to enter this call.

        Use this to synchronize all parallel processes, but only if necessary, otherwise the
        overhead of synchronization will cause your program to slow down.
        **This method needs to be called on all processes.** Failing to do so will cause your
        program to stall forever.

        Args:
            name (str | None): An optional name for the barrier for debugging purposes.
        """
        self._strategy.barrier(name=name)

    def broadcast(self, obj: jt.PyTree, src: int = 0) -> jt.PyTree:
        r"""
        Send a tensor from one process to all others.

        **This method needs to be called on all processes.** Failing to do so will cause your
        program to stall forever.

        Args:
            obj (jaxtyping.PyTree): The object to broadcast to all other members.
                Any pytree object is supported.
            src (int): The (global) rank of the process that should send the data to all others.
                Defaults to 0.

        Returns:
            jaxtyping.PyTree: The transferred data, which is the same value on every rank.
        """
        return self._strategy.broadcast(obj, src=src)

    def all_reduce(self, obj: jt.PyTree, reduce_op: str = "mean") -> jt.PyTree:
        """
        Reduces a tensor from several distributed processes to one aggregated tensor (AllReduce).

        The result is the same on all processes.

        Args:
            obj (jaxtyping.PyTree): The pytree to synchronize and reduce.
            reduce_op (str): The reduction operation. Defaults to ``'mean'``/``'avg'``.
            Can also be ``'sum'``.

        Returns:
            jaxtyping.PyTree: The reduced value.
        """
        return self._strategy.all_reduce(obj, reduce_op=reduce_op)

    def call(self, name: str, *args, **kwargs):
        """
        Triggers a named event hook for all registered listeners.

        The event hook is expected to be a method on :class:`reax.hooks.TrainerListener`.

        Args:
            name (str): The name of the event hook method to call (e.g., ``'on_train_epoch_end'``).
            *args: Positional arguments to pass to the hook.
            **kwargs: Keyword arguments to pass to the hook.
        """
        self._events.fire_event(getattr(hooks.TrainerListener, name), *args, **kwargs)

    def compute(self, metric: "reax.typing.MetricInstance[_OutT]") -> _OutT:
        """
        Computes the final value of a metric across all distributed processes.

        Unlike calling ``metric.compute()`` directly, this method ensures the
        value is aggregated correctly across all ranks (e.g., summing up counts)
        before the final calculation.

        Args:
            metric (reax.types.MetricInstance): The metric object to compute the final value for.

        Returns:
            _OutT: The computed metric value.
        """
        return self._strategy.compute(metric)

    def _wrap_loader(self, loader: "reax.DataLoader") -> "reax.data.DeviceDataLoader":
        """
        Internal utility to wrap a standard DataLoader into a DeviceDataLoader.
        """
        if isinstance(loader, DeviceDataLoader):
            loader = loader.parent

        return data_.DeviceDataLoader(loader, self.device)


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
    logger: "reax.Logger | Iterable[reax.Logger] | bool | None",
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


def _is_local_file_protocol(path: "reax.types.Path") -> bool:
    """Is local file protocol."""
    return fsspec.utils.get_protocol(str(path)) == "file"
