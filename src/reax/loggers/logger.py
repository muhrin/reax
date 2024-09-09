import abc
import argparse
from typing import Any, Callable, Generic, Optional, TypeVar, Union

from reax.lightning import rank_zero

__all__ = ("Logger",)

Exp = TypeVar("Exp")  # Experiment type


class Logger(metaclass=abc.ABCMeta):
    """Base class for REAX loggers."""

    @property
    @abc.abstractmethod
    def name(self) -> Optional[str]:
        """Return the name of the experiment."""

    @property
    @abc.abstractmethod
    def version(self) -> Optional[Union[int, str]]:
        """Return the experiment version"""

    @property
    def root_dir(self) -> Optional[str]:
        """
        Returns the root directory where all experimental versions are saved.
        Returns `None` if the logger does not save locally.
        """
        return None

    @property
    def save_dir(self) -> Optional[str]:
        """
        Get the directory where logs are saved.
        Returns `None` if logger does not save locally.
        """
        return None

    @property
    def log_dir(self) -> Optional[str]:
        """
        Returns the directory used to store logs for the current experiment
        Returns `None` if logger does not save locally.
        """
        return None

    @property
    def group_separator(self) -> str:
        """The separator used to store to group data into subfolders"""
        return "/"

    @abc.abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        """Log the passed metrics.

        :param metrics: dictionary of metrics to log
        :param step: the current step
        """

    @abc.abstractmethod
    def log_hyperparams(
        self, params: Union[dict[str, Any], argparse.Namespace], *args: Any, **kwargs: Any
    ) -> None:
        """Log the passed hyperparameters.

        :param params: a `dict` or :class:`~argparse.Namespace` containing hyperparameters
        :param args: additional optional args
        :param kwargs: additional optional kwargs
        """

    def log_graph(self, model: Callable, *args, **kwargs) -> None:
        """Log the model graph.

        :param model: the model function
        :param args: the args to pass to the model
        :param kwargs: the kwargs to pass to the model
        """

    def save(self) -> None:
        """Save the log data"""

    def finalize(self, status: str) -> None:  # pylint: disable=unused-argument
        """Do postprocessing to be done at the end of an experiment

        :param status: a status string indicating the outcome of the experiment
        """
        self.save()


class WithDdp(Generic[Exp], metaclass=abc.ABCMeta):
    """Mixins that allows a logger to be compatible with DPP strategy."""

    @property
    def experiment(self) -> Union[Exp, "_DummyExperiment"]:
        """Returns the actual experiment if on rank 0 and otherwise the _DummyExperiment."""
        if rank_zero.rank_zero_only.rank > 0:
            return _DummyExperiment()

        return self._experiment

    @rank_zero.rank_zero_only
    def log_hyperparams(
        self, params: Union[dict[str, Any], argparse.Namespace], *args: Any, **kwargs: Any
    ) -> None:
        return self._log_hyperparams(params, *args, **kwargs)

    @rank_zero.rank_zero_only
    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        return self._log_metrics(metrics, step)

    @rank_zero.rank_zero_only
    def log_graph(self, model: Callable, *args, **kwargs) -> None:
        return self._log_graph(model, *args, **kwargs)

    @rank_zero.rank_zero_only
    def save(self) -> None:
        return self._save()

    @rank_zero.rank_zero_only
    def finalize(self, status: str) -> None:
        return self._finalize(status)

    @property
    def _experiment(self) -> Optional[Exp]:
        return None

    @abc.abstractmethod
    def _log_hyperparams(
        self, params: Union[dict[str, Any], argparse.Namespace], *args: Any, **kwargs: Any
    ) -> None:
        """Log hyperparams implementation"""

    @abc.abstractmethod
    def _log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics implementation"""

    def _log_graph(self, model: Callable, *args, **kwargs) -> None:
        """Log graph implementation"""

    def _save(self) -> None:
        """Save implementation"""

    def _finalize(self, status: str) -> None:  # pylint: disable=unused-argument
        """Finalize implementation"""
        self._save()


class _DummyExperiment:
    """Dummy experiment."""

    def nop(self, *args: Any, **kw: Any) -> None:
        pass

    def __getattr__(self, _: Any) -> Callable:
        return self.nop

    def __getitem__(self, idx: int) -> "_DummyExperiment":
        # enables self.logger.experiment[0].add_image(...)
        return self

    def __setitem__(self, *args: Any, **kwargs: Any) -> None:
        pass
