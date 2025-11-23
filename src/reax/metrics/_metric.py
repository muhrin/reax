import abc
from collections.abc import Callable
import functools
from typing import ClassVar, Generic, TypeVar, cast

import equinox
from typing_extensions import override

__all__ = ("Metric", "FromFun")

_OutT = TypeVar("_OutT")


class Metric(equinox.Module, Generic[_OutT], metaclass=abc.ABCMeta):
    """The base class for all metrics.

    To be compatible with JAX Metrics are designed to be immutable meaning that when we 'update'
    them, the update will return a new instance with the results of the update.

    To create a new instance, use the ``.create()`` method with the data the metric will be
    calculated on.

    Updating can be done either with the ``.update()`` call when updating from raw data, or the
    ``.merge()`` call when merging an existing metric instance.
    """

    @classmethod
    def from_fun(cls, function: Callable, name: str = None) -> type["FromFun[_OutT]"]:
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

    def empty(self) -> "Metric[_OutT]":
        """Create a new empty instance.

        By default, this will call the constructor with no arguments, if needed, subclasses can
        overwrite this with custom behaviour.
        """
        return type(self)()

    @abc.abstractmethod
    def create(self, *args, **kwargs) -> "Metric[_OutT]":
        """Create a new metric instance from data."""

    def update(self, *args, **kwargs) -> "Metric[_OutT]":
        """Update the metric from new data and return a new instance."""
        return self.merge(self.create(*args, **kwargs))

    @abc.abstractmethod
    def merge(self, other: "Metric") -> "Metric[_OutT]":
        """Merge the metric with data from another metric instance of the same type."""

    @abc.abstractmethod
    def compute(self) -> _OutT:
        """Compute the metric."""


def hybrid_method(func: Callable):
    """
    A decorator that binds the function to the instance if called on an instance,
    or the class if called on the class.

    Implemented as a function returning a descriptor object.
    """

    class HybridDescriptor:
        def __get__(self, obj, objtype=None):
            if obj is None:
                # Called on the class -> bind to class
                return func.__get__(objtype, objtype)
            # Called on an instance -> bind to instance
            return func.__get__(obj, objtype)

    # Create the descriptor instance
    descriptor = HybridDescriptor()

    # Copy metadata (name, docstring) from the original function to the descriptor
    functools.update_wrapper(descriptor, func)

    return descriptor


class FromFun(Metric[_OutT]):
    """Helper class apply a function before passing the result to an existing metric."""

    metric: ClassVar[type[Metric[_OutT]]]
    func: ClassVar[Callable]
    _state: Metric[_OutT]

    def __init__(self, *args, state: Metric[_OutT] = None, **kwargs):
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
    def empty(  # pylint: disable=no-self-argument, arguments-renamed
        self_or_cls,
    ) -> "FromFun[_OutT]":
        """Empty function."""
        if isinstance(self_or_cls, type):
            cls: "type[FromFun[_OutT]]" = cast(type(FromFun), self_or_cls)
            return cls(state=cls.metric.empty())  # pylint: disable=not-callable

        self = cast(FromFun, self_or_cls)
        return type(self)(state=self._state.empty())  # pylint: disable=protected-access

    @override
    @hybrid_method
    def create(  # pylint: disable=no-self-argument, arguments-differ
        self_or_cls, *args, **kwargs
    ) -> "FromFun":
        """Create function."""
        val = self_or_cls._call_fn(*args, **kwargs)

        if isinstance(self_or_cls, type):
            cls = self_or_cls
            metric = cls.metric
        else:
            cls = type(self_or_cls)
            metric = self_or_cls._state

        cls = cast(type(FromFun), cls)
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
