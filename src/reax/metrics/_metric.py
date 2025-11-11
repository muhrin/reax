import abc
from collections.abc import Callable
from typing import ClassVar, Generic, TypeVar

import equinox

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
    def from_fun(cls, function: Callable) -> type["FromFun[_OutT]"]:
        """Create a new metric from this one where a function is called before passing it on to this
        metric.

        Args:
            cls
            function (Callable): The function to call.

        Returns:
            type["FromFun[OutT]"]: The new metric type.
        """

        class FromFunction(FromFun):
            metric = cls()  # pylint: disable=abstract-class-instantiated

            @classmethod
            def func(cls, *args, **kwargs) -> "Metric[_OutT]":
                """Fun function."""
                return function(*args, **kwargs)

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


class FromFun(Metric[_OutT]):
    """Helper class apply a function before passing the result to an existing metric."""

    metric: ClassVar[type[Metric[_OutT]] | Metric[_OutT]]
    func: ClassVar[Callable]
    _state: Metric[_OutT]

    def __init__(self, state: Metric[_OutT] = None):
        super().__init__()
        if self.metric is None:
            raise RuntimeError(
                "The metric used by from_fun has not been set. "
                "This should be set as a class variable"
            )
        self._state = state

    @property
    def is_empty(self) -> bool:
        return self._state is None

    def empty(self) -> "FromFun[_OutT]":
        """Empty function."""
        if self.is_empty:
            return self

        return type(self)()

    @classmethod
    def create(cls, *args, **kwargs) -> "FromFun":
        """Create function."""
        val = cls._call_fn(*args, **kwargs)
        return cls(state=cls.metric.create(*val))

    def merge(self, other: "FromFun") -> "FromFun":
        """Merge function."""
        if self.is_empty:
            return other
        if other.is_empty:
            return self

        return type(self)(state=self._state.merge(other._state))  # pylint: disable=protected-access

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
