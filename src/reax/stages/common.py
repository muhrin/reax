from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Iterable,
    Optional,
    TypedDict,
    TypeVar,
    Union,
    cast,
)

import beartype
import jax
import jaxtyping as jt

from .. import data
from ..utils import events

if TYPE_CHECKING:
    import reax

__all__ = ("MetricResults", "StageListener")

_T_co = TypeVar("_T_co", covariant=True)
StageEvents = events.EventGenerator["StageListener"]


class StageListener:
    def on_stage_start(self, stage: "reax.Stage", /):
        """The stage is starting."""

    def on_stage_started(self, stage: "reax.Stage", /):
        """The stage is starting."""

    def on_stage_iter_starting(self, stage: "reax.Stage", step: int, /):
        """The stage is about to start an iteration."""

    def on_stage_iter_ending(self, stage: "reax.Stage", step: int, outputs: Any, /):
        """The stage just finished processing an iteration."""

    def on_stage_iter_ended(self, stage: "reax.Stage", step: int, outputs: Any, /):
        """The stage just finished processing an iteration."""

    def on_stage_end(self, stage: "reax.Stage", /):
        """The stage is ending."""

    def on_stage_ended(self, stage: "reax.Stage", /):
        """The stage has ended."""


class MetricResults(TypedDict):
    callback: dict[str, jax.Array]
    log: dict[str, jax.Array]
    pbar: dict[str, float]


class Stopper:
    """Class used by loops to manage their stop conditions."""

    def __init__(self):
        """Init function."""
        self._stop_requested = False
        self._conditions: list[Callable[[], bool]] = []

    @property
    def stop_requested(self) -> bool:
        """Stop requested."""
        return self._stop_requested

    @property
    def can_stop(self):
        """Can stop."""
        return all(condition() for condition in self._conditions)

    def do_stop(self) -> bool:
        """Do stop."""
        return self._stop_requested and self.can_stop

    def set(self):
        """Set function."""
        self._stop_requested = True

    def get(self) -> bool:
        """Get function."""
        return self._stop_requested

    def add_condition(self, condition: Callable[[], bool]) -> None:
        """Add condition."""
        self._conditions.append(condition)


@jt.jaxtyped(typechecker=beartype.beartype)
def batches_limit(
    batch_limit: Optional[Union[int, float]], dataloader: "reax.DataLoader"
) -> Optional[Union[int, float]]:
    """Return a maximum number of batches given a dataloader and an optional batches limit.

    If the dataloader has fewer entries than the batch limit, then this will be used, otherwise
    the batches limit.

    .. note:: Will return `None` if there is no limit.

    :param batch_limit: The batches limit.
    :type batch_limit: Union[int, float]
    :param dataloader: The dataloader.
    :type dataloader: "reax.DataLoader"
    :return: The maximum number of batches.
    """
    dataloader_size = data.sized_len(dataloader)
    if isinstance(batch_limit, int):
        if dataloader_size is None:
            return batch_limit

        return min(batch_limit, dataloader_size)

    if isinstance(batch_limit, float):
        if batch_limit in (None, 1.0):
            if dataloader_size is not None:
                return dataloader_size
            return None

        if dataloader_size is not None:
            # batch_limit is a finite float and we have a dataloader size
            batch_limit = cast(float, batch_limit)
            return int(round(batch_limit * dataloader_size))

        raise ValueError(
            f"Cannot determine number of batches from dataloader and batch_limit is "
            f"{batch_limit}"
        )

    # We can't say anything other than just 'go to the end'
    return float("inf")


class DataSourceManager(Generic[_T_co]):
    class LoaderProxy(data.DataLoader[_T_co]):
        def __init__(self, manager: "DataSourceManager[_T_co]", method_name: str):
            self._manager = manager
            self._method_name = method_name
            self._iterable: Optional[Iterable[_T_co]] = None

        def __iter__(self):
            return iter(self.dataloader)

        def __len__(self):
            return len(self.dataloader)

        def __getitem__(self, index):
            return self.dataloader.__getitem__(index)

        @property
        def dataloader(self) -> Iterable[_T_co]:
            if self._iterable is None:
                self._iterable = getattr(self._manager.source, self._method_name)()
            return self._iterable

    def __init__(self, source: "reax.data.DataSource[_T_co]"):
        self._source: "reax.data.DataSource[_T_co]" = source
        self._ready: bool = False

    @property
    def ready(self) -> bool:
        return self._ready

    def get_loader_proxy(self, method_name: str) -> LoaderProxy[_T_co]:
        return DataSourceManager.LoaderProxy(self, method_name)

    @property
    def source(self) -> "reax.data.DataSource[_T_co]":
        if not self._ready:
            raise RuntimeError(
                "`prepare_and_setup` has not been called, this must be done before accessing the "
                "dataset"
            )
        return self._source

    def prepare_and_setup(self, stage) -> None:
        if self.ready:
            # Already done
            return

        self._source.prepare_data()
        self._source.setup(stage)
        self._ready = True


def get_datasource(
    datamodule: "Optional[reax.DataModule[_T_co]]" = None,
    module: Optional["reax.Module"] = None,
) -> Optional[DataSourceManager[_T_co]]:
    if datamodule is not None:
        return DataSourceManager(datamodule)

    if module is not None:
        return DataSourceManager(module)

    return None
