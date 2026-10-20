from collections.abc import Callable, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    TypedDict,
    TypeVar,
    cast,
)

import beartype
import jax
import jaxtyping as jt
import numpy as np

from .. import data, keys
from ..utils import events

if TYPE_CHECKING:
    import reax

__all__ = ("MetricResults", "StageListener")

_T_co = TypeVar("_T_co", covariant=True)
_U = TypeVar("_U")
_StageT = TypeVar("_StageT", bound="reax.Stage")
_LoopT = TypeVar("_LoopT", bound="reax.stages.Loop")
StageEvents = events.EventGenerator["StageListener"]
ScalarLike = jax.Array | np.ndarray | np.bool_ | np.number | complex


class StageListener:
    # Stage lifecycle
    def on_stage_start(self, stage: _StageT, /):
        """The stage is starting."""

    def on_stage_started(self, stage: _StageT, /):
        """The stage has started."""

    def on_stage_finishing(self, stage: _StageT, /):
        """The stage is ending."""

    def on_stage_finished(self, stage: _StageT, /):
        """The stage has ended."""

    # Loop lifecycle
    def on_loop_starting(self, stage: _StageT, loop: _LoopT, /):
        """A loop is starting."""

    def on_loop_started(self, stage: _StageT, loop: _LoopT, max_iters: int | float | None, /):
        """A loop has started."""

    def on_loop_iter_starting(self, stage: _StageT, loop: _LoopT, /):
        """A loop iteration is about to start."""

    def on_loop_iter_started(self, stage: _StageT, loop: _LoopT, /):
        """A loop iteration is about to start."""

    def on_loop_iter_finishing(self, stage: _StageT, loop: _LoopT, outputs: Any, /):
        """A loop iteration has just finished processing."""

    def on_loop_iter_finished(self, stage: _StageT, loop: _LoopT, outputs: Any, /):
        """A loop iteration has completed."""

    def on_loop_finishing(self, stage: _StageT, loop: _LoopT, /):
        """A loop is ending."""

    def on_loop_finished(self, stage: _StageT, loop: _LoopT, /):
        """A loop has ended."""


class MetricResults(TypedDict):
    listener: dict[str, ScalarLike]
    log: dict[str, ScalarLike]
    pbar: dict[str, float]


def empty_metric_results() -> MetricResults:
    return MetricResults(listener={}, log={}, pbar={})


def clear_metric_results(results: MetricResults) -> None:
    results[keys.LISTENER].clear()
    results[keys.LOG].clear()
    results[keys.PBAR].clear()


class Stopper:
    """A hierarchical cancellation token with per-level conditions.

    A stopper may have a *parent* (a stopper from an ancestor, forming a chain
    from parent stage down to child loops). ``set()`` records a stop *request*
    only at that level; ``is_requested`` reports whether a request exists anywhere
    up the chain, and ``can_stop`` reports whether that level's own conditions and
    every ancestor's conditions are met.

    ``do_stop()`` is true only when a stop has been requested somewhere in the
    chain AND the conditions at every level are met. This lets a child loop defer
    stopping until an ancestor (e.g. ``min_epochs``) is satisfied, without the
    child needing a reference to that ancestor.
    """

    def __init__(
        self, conditions: Sequence[Callable[[], bool]] = tuple(), *, parent: "Stopper | None" = None
    ):
        """Init function."""
        self._stop_requested = False
        self._parent = parent
        self._conditions: list[Callable[[], bool]] = list(conditions)

    @property
    def is_requested(self) -> bool:
        """True if a stop has been requested here or by any ancestor in the chain."""
        return self._stop_requested or (self._parent is not None and self._parent.is_requested)

    @property
    def can_stop(self) -> bool:
        """True if this stopper's own conditions and all ancestors' conditions are met."""
        if self._parent is not None and not self._parent.can_stop:
            return False
        return all(condition() for condition in self._conditions)

    def do_stop(self) -> bool:
        """True when a stop has been requested somewhere in the chain and every level's
        conditions are met.

        This is the decision point used by loops: a child returns ``False`` (and so keeps
        running) until an ancestor's conditions (e.g. ``min_epochs``) are satisfied, even if
        its own stop was requested.
        """
        return self.is_requested and self.can_stop

    def set(self) -> None:
        """Request a stop at this level (visible to descendants as ``is_requested``)."""
        self._stop_requested = True

    def get(self) -> bool:
        """Whether a stop was requested *at this level* (does not consider ancestors)."""
        return self._stop_requested

    def add_condition(self, condition: Callable[[], bool]) -> None:
        """Add a condition that must be met (with its ancestors') before this stopper can
        actually stop."""
        self._conditions.append(condition)


@jt.jaxtyped(typechecker=beartype.beartype)
def batches_limit(batch_limit: int | float | None, dataloader: "reax.DataLoader") -> int | None:
    """Return the maximum number of batches that will be drawn from a dataloader.

    The batch limit is interpreted as:

    * ``None``: no limit, the whole dataloader is used.
    * ``int``: an absolute number of batches, capped at the length of the dataloader.
    * ``float``: a fraction of the length of the dataloader, where ``1.0`` is the whole dataloader.

    Args:
        batch_limit (int | float | None): The batches limit.
        dataloader ("reax.DataLoader"): The dataloader.

    Returns:
        The maximum number of batches, or ``None`` if this isn't known in advance, i.e. the
        dataloader has no length and the limit doesn't bound it.

    Raises:
        ValueError: If a fractional limit (other than ``1.0``) is used with a dataloader that has
            no length.
    """
    dataloader_size = data.sized_len(dataloader)

    if isinstance(batch_limit, int):
        if dataloader_size is None:
            return batch_limit
        return min(batch_limit, dataloader_size)

    if batch_limit is None or batch_limit == 1.0:
        # The whole dataloader, however long that is
        return dataloader_size

    # A fraction of the dataloader
    if dataloader_size is None:
        raise ValueError(
            f"Cannot use a fractional batch limit ({batch_limit}) with a dataloader that has no "
            f"length"
        )
    return int(round(batch_limit * dataloader_size))


class LoaderProxy(data.DataLoader[_T_co, _U]):
    @jt.jaxtyped(typechecker=beartype.beartype)
    def __init__(self, manager: "DataSourceManager[_T_co, _U]", method_name: str):
        self._manager = manager
        self._method_name = method_name
        self._iterable: reax.data.DataLoader[_T_co, _U] | None = None

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)

    def __getitem__(self, index: int) -> _U:
        loader = self.dataloader
        try:
            return cast(Sequence[_U], loader)[index]
        except TypeError as exc:
            raise TypeError(
                f"{type(loader).__name__} does not support indexing; "
                f"this dataloader can only be iterated"
            ) from exc

    @property
    def sampler(self) -> "reax.data.Sampler":
        return self.dataloader.sampler

    @property
    def dataset(self) -> "reax.data.Dataset[_T_co]":
        return self.dataloader.dataset

    @property
    def dataloader(self) -> "reax.data.DataLoader[_T_co, _U]":
        if self._iterable is None:
            self._iterable = getattr(self._manager.source, self._method_name)()
        return self._iterable

    def with_new_sampler(self, sampler: "reax.data.Sampler") -> "reax.data.DataLoader[_T_co, _U]":
        return self.dataloader.with_new_sampler(sampler)


class DataSourceManager(Generic[_T_co, _U]):
    def __init__(
        self,
        source: "reax.data.DataSource[_T_co, _U]",
        engine: "reax.Engine | None" = None,
    ):
        self._source = source
        self._engine = engine
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def get_loader_proxy(self, method_name: str) -> LoaderProxy[_T_co, _U]:
        return LoaderProxy(self, method_name)

    @property
    def source(self) -> "reax.data.DataSource[_T_co, _U]":
        if not self._ready:
            raise RuntimeError(
                "`prepare_and_setup` has not been called, this must be done before accessing the "
                "dataset"
            )
        return self._source

    def prepare_and_setup(self, engine: "reax.Engine", /, *, stage: str | None = None) -> None:
        if self.ready:
            # Already done
            return

        self._source.prepare_data()
        self._source.setup(engine, stage=stage)
        self._ready = True
