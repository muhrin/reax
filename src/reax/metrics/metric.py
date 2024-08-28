import abc
from typing import Callable, ClassVar, Generic, Optional, TypeVar

import beartype
import equinox
import jaxtyping as jt

__all__ = ("Metric",)

OutT = TypeVar("OutT")


class Metric(equinox.Module, Generic[OutT], metaclass=abc.ABCMeta):
    Self = TypeVar("Self", bound="Metric")

    @classmethod
    def from_fun(cls, function: Callable) -> type["FromFun[OutT]"]:
        """
        Create a new metric from this one where a function is called before passing it on to this
        metric.
        :param function: the function to call
        :return: the new metric type
        """

        class FromFunction(FromFun):
            parent = cls
            fun = function

        return FromFunction

    @classmethod
    def empty(cls) -> Self:
        """Create a new empty instance.

        By default, this will call the constructor with no arguments, if needed, subclasses can
        overwrite this with custom behaviour.
        """
        return cls()

    @classmethod
    @abc.abstractmethod
    def create(cls, *args, **kwargs) -> Self:
        """Create the metric from data"""

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


class FromFun(Metric):
    """
    Helper class apply a function before aggregating
    """

    parent: ClassVar[type[Metric[OutT]]]
    fun: ClassVar[Callable]
    metric: Optional[Metric[OutT]]

    def __init__(self, metric: Optional[Metric[OutT]] = None):
        super().__init__()
        self.metric = metric

    @classmethod
    def create(cls, *args, **kwargs) -> "FromFun":
        val = cls._call_fn(*args, **kwargs)
        return cls(metric=cls.parent.create(*val))

    def merge(self, other: "FromFun") -> "FromFun":
        if other.metric is None:
            return self
        if self.metric is None:
            return other

        return type(self)(metric=self.metric.merge(other.metric))

    @jt.jaxtyped(typechecker=beartype.beartype)
    def update(self, *args, **kwargs) -> "FromFun":
        cls = type(self)
        if self.metric is None:
            return cls.create(*args, **kwargs)

        val = cls._call_fn(*args, **kwargs)  # pylint: disable=protected-access
        return cls(metric=self.metric.update(*val))

    def compute(self) -> OutT:
        if self.metric is None:
            raise RuntimeError("Nothing to compute, metric is empty!")
        return self.metric.compute()

    @classmethod
    def _call_fn(cls, *args, **kwargs) -> tuple:
        val = cls.fun(*args, **kwargs)

        # Automatically unroll a tuple of return values
        if not isinstance(val, tuple):
            val = (val,)

        return val
