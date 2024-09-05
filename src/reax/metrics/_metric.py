import abc
from typing import Callable, ClassVar, Generic, Optional, TypeVar

import equinox

__all__ = ("Metric",)

OutT = TypeVar("OutT")


class Metric(equinox.Module, Generic[OutT], metaclass=abc.ABCMeta):

    @classmethod
    def from_fun(cls, function: Callable) -> type["FromFun[OutT]"]:
        """
        Create a new metric from this one where a function is called before passing it on to this
        metric.
        :param function: the function to call
        :return: the new metric type
        """

        class FromFunction(FromFun):
            metric = cls()  # pylint: disable=abstract-class-instantiated

            def fun(self, *args, **kwargs) -> "Metric[OutT]":
                return function(*args, **kwargs)

        return FromFunction

    def empty(self) -> "Metric":
        """Create a new empty instance.

        By default, this will call the constructor with no arguments, if needed, subclasses can
        overwrite this with custom behaviour.
        """
        return type(self)()

    @abc.abstractmethod
    def create(self, *args, **kwargs) -> "Metric":
        """Create a new metric instance from data"""

    def update(self, *args, **kwargs) -> "Metric":
        """Update the metric from new data and return a new instance"""
        return self.merge(self.create(*args, **kwargs))

    @abc.abstractmethod
    def merge(self, other: "Metric") -> "Metric":
        """Merge the metric with data from another metric instance of the same type"""

    @abc.abstractmethod
    def compute(self) -> OutT:
        """Compute the metric"""


ParentMetric = TypeVar("ParentMetric", bound=Metric)


class FromFun(Metric[OutT]):
    """
    Helper class apply a function before passing the result to an existing metric.
    """

    fun: ClassVar[Callable]
    metric: ClassVar[Metric]
    _state: Metric[OutT]

    def __init__(self, state: Optional[Metric[OutT]] = None):
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

    def empty(self) -> "FromFun[OutT]":
        if self.is_empty:
            return self

        return type(self)()

    def create(self, *args, **kwargs) -> "FromFun":
        val = self._call_fn(*args, **kwargs)
        return type(self)(state=self.metric.create(*val))

    def merge(self, other: "FromFun") -> "FromFun":
        if self.is_empty:
            return other
        if other.is_empty:
            return self

        return type(self)(state=self._state.merge(other._state))  # pylint: disable=protected-access

    def update(self, *args, **kwargs) -> "FromFun":
        if self.is_empty:
            return self.create(*args, **kwargs)

        val = self._call_fn(*args, **kwargs)  # pylint: disable=protected-access
        return type(self)(state=self._state.update(*val))

    def compute(self) -> OutT:
        if self._state is None:
            raise RuntimeError("Nothing to compute, metric is empty!")
        return self._state.compute()

    def _call_fn(self, *args, **kwargs) -> tuple:
        val = self.fun(*args, **kwargs)

        # Automatically unroll a tuple of return values
        if not isinstance(val, tuple):
            val = (val,)

        return val
