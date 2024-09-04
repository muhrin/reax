import abc
from typing import Any

__all__ = ("Strategy",)


class Strategy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def to_device(self, value: Any) -> Any:
        """Move the value to the device and return it"""

    @abc.abstractmethod
    def from_device(self, value: Any) -> Any:
        """Get a value from the device and return it"""
