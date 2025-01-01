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
        """Return the experiment version."""

    @property
    def root_dir(self) -> Optional[str]:
        """Returns the root directory where all experimental versions are saved.

        Returns `None` if the logger does not save locally.
        """
        return None

    @property
    def save_dir(self) -> Optional[str]:
        """Get the directory where logs are saved.

        Returns `None` if logger does not save locally.
        """
        return None

    @property
    def log_dir(self) -> Optional[str]:
        """Returns the directory used to store logs for the current experiment
        Returns `None` if logger does not save locally.
        """
        return None

    @property
    def group_separator(self) -> str:
        """The separator used to store to group data into subfolders."""
        return "/"

    @abc.abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        """Log the passed metrics.
        :param metrics: Dictionary of metrics to log.
        :type metrics: dict[str, float]
        :param step: The current step, defaults to None.
        :type step: Optional[int], optional
        """

    @abc.abstractmethod
    def log_hyperparams(
        self, params: Union[dict[str, Any], argparse.Namespace], *args: Any, **kwargs: Any
    ) -> None:
        """Log the passed hyperparameters.
        :param params: A `dict` or :class:`~argparse.Namespace` containing hyperparameters.
        :type params: Union[dict[str, Any], argparse.Namespace]
        :param *args: Additional optional args.
        :type *args: Any
        :param **kwargs: Additional optional kwargs.
        :type **kwargs: Any
        """

    def log_graph(self, model: Callable, *args, **kwargs) -> None:
        """Log the model graph.
        :param model: The model function.
        :type model: Callable
        :param *args: The args to pass to the model.
        :param **kwargs: The kwargs to pass to the model.
        """

    def save(self) -> None:
        """Save the log data."""

    def finalize(self, status: str) -> None:  # pylint: disable=unused-argument
        """Do postprocessing to be done at the end of an experiment.
        :param status: A status string indicating the outcome of the experiment.
        :type status: str
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
        """Log hyperparams."""
        return self._log_hyperparams(params, *args, **kwargs)

    @rank_zero.rank_zero_only
    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics."""
        return self._log_metrics(metrics, step)

    @rank_zero.rank_zero_only
    def log_graph(self, model: Callable, *args, **kwargs) -> None:
        """Log graph."""
        return self._log_graph(model, *args, **kwargs)

    @rank_zero.rank_zero_only
    def save(self) -> None:
        """Save function."""
        return self._save()

    @rank_zero.rank_zero_only
    def finalize(self, status: str) -> None:
        """Finalize function."""
        return self._finalize(status)

    @property
    def _experiment(self) -> Optional[Exp]:
        """Experiment function."""
        return None

    @abc.abstractmethod
    def _log_hyperparams(
        self, params: Union[dict[str, Any], argparse.Namespace], *args: Any, **kwargs: Any
    ) -> None:
        """Log hyperparams implementation."""

    @abc.abstractmethod
    def _log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics implementation."""

    def _log_graph(self, model: Callable, *args, **kwargs) -> None:
        """Log graph implementation."""

    def _save(self) -> None:
        """Save implementation."""

    def _finalize(self, status: str) -> None:  # pylint: disable=unused-argument
        """Finalize implementation."""
        self._save()


class _DummyExperiment:
    """Dummy experiment."""

    def nop(self, *args: Any, **kw: Any) -> None:
        """Nop function."""
        pass

    def __getattr__(self, _: Any) -> Callable:
        """Getattr function."""
        return self.nop

    def __getitem__(self, idx: int) -> "_DummyExperiment":
        """Getitem function."""
        # enables self.logger.experiment[0].add_image(...)
        return self

    def __setitem__(self, *args: Any, **kwargs: Any) -> None:
        """Setitem function."""
        pass
