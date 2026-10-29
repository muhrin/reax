import abc
from collections.abc import Callable
import functools
import types
from typing import Any, ClassVar, Generic, TypeVar, cast

import jax
import jax.numpy as jnp
from typing_extensions import Self, override

from .. import pytree

__all__ = ("Metric", "FromFun")

_OutT = TypeVar("_OutT")
_R = TypeVar("_R")


class hybrid_method(Generic[_R]):  # pylint: disable=invalid-name
    """A decorator that binds the function to the instance if called on an instance, or the class
    if called on the class."""

    def __init__(self, func: Callable[..., _R]):
        self.__func__ = func
        # Copy the function's metadata (name, docstring, etc.).  The descriptor itself isn't
        # callable, hence the cast.
        functools.update_wrapper(cast(Callable, self), func)

    def __get__(self, obj: Any, objtype: type | None = None) -> Callable[..., _R]:
        return types.MethodType(self.__func__, objtype if obj is None else obj)


class Metric(pytree.PyTree, Generic[_OutT], metaclass=abc.ABCMeta):
    """The base class for all metrics.

    To be compatible with JAX Metrics are designed to be immutable meaning that when we 'update'
    them, the update will return a new instance with the results of the update.

    To create a new instance, use the ``.create()`` method with the data the metric will be
    calculated on.

    Updating can be done either with the ``.update()`` call when updating from raw data, or the
    ``.merge()`` call when merging an existing metric instance.
    """

    @classmethod
    def from_fun(cls, function: Callable, name: str | None = None) -> type["FromFun[_OutT]"]:
        """Create a new metric from this one where a function is called before passing it on to this
        metric.

        Args:
            cls
            function (Callable): The function to call.

        Returns:
            type["FromFun[OutT]"]: The new metric type.
        """

        # Construct the desired new name (e.g., 'AccuracyMetricFromFun')
        new_name = name or f"{cls.__name__}FromFun"

        class FromFunction(FromFun):
            metric = cls

            @classmethod
            def func(cls, *args, **kwargs) -> "Metric[_OutT]":
                """Fun function."""
                return function(*args, **kwargs)

        # Set the __name__ and __qualname__ attributes on the new class
        FromFunction.__name__ = new_name

        # __qualname__ often includes the module and containing class,
        # so we derive it from the base class's qualname
        FromFunction.__qualname__ = f"{cls.__qualname__}.{new_name}"

        return FromFunction

    @hybrid_method
    def empty(self) -> "Metric[_OutT]":  # pylint: disable=no-self-argument
        """Create a new empty instance.

        This can be called on the class or an instance. By default, this will call the constructor
        with no arguments, if needed, subclasses can overwrite this with custom behaviour.
        """
        cls = self if isinstance(self, type) else type(self)
        return cls()

    @hybrid_method
    @abc.abstractmethod
    def create(self, *args, **kwargs) -> Self:
        """Create a new metric instance from data.

        This can be called on the class or an instance, subclasses can implement it as either a
        ``classmethod`` or a regular method (e.g. if the instance holds configuration that is
        needed to create the new instance).
        """

    def update(self, *args, **kwargs) -> Self:
        """Update the metric from new data and return a new instance."""
        return self.merge(self.create(*args, **kwargs))

    @abc.abstractmethod
    def merge(self, other: Self) -> Self:
        """Merge the metric with data from another metric instance of the same type."""

    @abc.abstractmethod
    def compute(self) -> _OutT:
        """Compute the metric."""

    def reduce(self, axis: int = 0) -> "Metric[_OutT]":
        """Collapses a vectorized (vmapped) metric into a single instance.

        The default implementation assumes the metric state is additive.
        Subclasses should override this if they require non-additive reduction
        (e.g., taking the max, or concatenating lists).
        """
        reduce_fn = type(self).reduce_fn

        def _apply(leaf):
            # If it's a JAX array and has the axis we want to reduce
            if isinstance(leaf, jax.Array) and leaf.ndim > axis:
                return reduce_fn(leaf, axis=axis)
            # Handle strings, None, or un-vmapped scalars
            return leaf

        return jax.tree_util.tree_map(_apply, self)

    @staticmethod
    def reduce_fn(x: jax.Array, axis: int = 0) -> jax.Array:
        return jnp.sum(x, axis=axis)


class FromFun(Metric[_OutT]):
    """Helper class apply a function before passing the result to an existing metric."""

    metric: type[Metric[_OutT]]
    func: ClassVar[Callable]
    _state: Metric[_OutT]

    def __init__(self, *args, state: Metric[_OutT] | None = None, **kwargs):
        super().__init__()
        if self.metric is None:
            raise RuntimeError(
                "The metric used by from_fun has not been set. "
                "This should be set as a class variable"
            )
        if state is None:
            state = self.metric(*args, **kwargs)
        self._state = state

    @property
    def is_empty(self) -> bool:
        return self._state is None

    @override
    @hybrid_method
    def empty(self) -> "FromFun[_OutT]":
        """Empty function."""
        if isinstance(self, type):
            cls = cast("type[FromFun[_OutT]]", self)
            return cls(state=cls.metric.empty())  # pylint: disable=not-callable

        instance = cast(FromFun, self)
        return type(instance)(state=self._state.empty())  # pylint: disable=protected-access

    @override
    @hybrid_method
    def create(self, *args, **kwargs) -> "FromFun":
        """Create function."""
        val = self._call_fn(*args, **kwargs)

        if isinstance(self, type):
            cls = self
            metric = cls.metric
        else:
            cls = type(self)
            metric = self._state

        cls = cast("type[FromFun]", cls)
        return cls(state=metric.create(*val))  # pylint: disable=not-callable

    @override
    def merge(self, other: "FromFun") -> "FromFun":
        """Merge function."""
        if self.is_empty:
            return other
        if other.is_empty:
            return self

        return type(self)(state=self._state.merge(other._state))  # pylint: disable=protected-access

    @override
    def update(self, *args, **kwargs) -> "FromFun":
        """Update function."""
        if self.is_empty:
            return self.create(*args, **kwargs)

        val = self._call_fn(*args, **kwargs)  # pylint: disable=protected-access
        return type(self)(state=self._state.update(*val))

    @override
    def reduce(self, *args, **kwargs) -> "FromFun":
        """Reduce function."""
        if self.is_empty:
            raise RuntimeError("Metric is empty!")

        return type(self)(state=self._state.reduce(*args, **kwargs))

    @override
    def compute(self) -> _OutT:
        """Compute function."""
        if self._state is None:
            raise RuntimeError("Nothing to compute, metric is empty!")
        return self._state.compute()

    @classmethod
    def _call_fn(cls, *args, **kwargs) -> tuple:
        """Call fn."""
        val = cls.func(*args, **kwargs)

        # Automatically unroll a tuple of return values
        if not isinstance(val, tuple):
            val = (val,)

        return val
