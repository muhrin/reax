import abc
from typing import TYPE_CHECKING

from lightning_utilities.core import overrides

from .. import modules
from . import _datasources, datamodules

if TYPE_CHECKING:
    import reax


__all__ = ("DataSourceManager", "create_manager")


class DataSourceManager(abc.ABC):  # noqa: B024
    """Coordinator that fronts a :class:`~reax.data.DataSource` (a :class:`~reax.DataModule` or
    :class:`~reax.Module`) and hands out ready-to-use dataloaders.

    The manager is bound to a :class:`reax.Engine` once at construction time. It is responsible
    for running the source's lifecycle hooks (:meth:`prepare_data`, :meth:`setup`,
    :meth:`teardown`, :meth:`on_exception`) and for wrapping any dataloaders returned by the
    source with the engine's device/distributed logic via
    :meth:`~reax.Engine.setup_dataloaders`.

    Attributes:
        source: The wrapped :class:`~reax.data.DataSource` (or ``None`` if the manager was
            created with pre-supplied loaders only).
    """

    def __init__(
        self,
        source: _datasources.DataSource | None,
        engine: "reax.Engine",
        **loaders: "reax.data.DataLoader",
    ) -> None:
        """Create a manager.

        Args:
            source: A :class:`~reax.DataModule` or :class:`~reax.Module` providing the dataloaders
                and lifecycle hooks, or ``None`` if dataloaders are passed directly.
            engine: The :class:`reax.Engine` to use for device placement and distributed setup.
                Bound once here and reused for every :meth:`setup`/:meth:`teardown` call.
            **loaders: Optional pre-supplied dataloaders keyed by stage name (e.g.
                ``train=..., val=...``). These bypass the source and are wrapped directly.
        """
        self._datasource: _datasources.DataSource | None = source
        self._engine = engine
        self._loaders: dict[str, reax.data.DataLoader] = {
            name: self._setup_dataloader(loaders) for name, loaders in loaders.items()
        }
        self._from_datasource: dict[str, reax.data.DataLoader] = {}

    @property
    def _source_base_type(
        self,
    ) -> "type[reax.DataModule] | type[reax.Module]":
        if isinstance(self._datasource, datamodules.DataModule):
            return datamodules.DataModule
        if isinstance(self._datasource, modules.Module):
            return modules.Module

        raise RuntimeError("expected datasource to be a DataModule or Module")

    def has_dataloader(self, name: str) -> bool:
        """Returns `True` if this source provides the name dataloader, `False` otherwise"""
        if name in self._loaders:
            return True

        if name in self._from_datasource:
            return True

        return overrides.is_overridden(
            f"{name}_dataloader", self._datasource, self._source_base_type
        )

    def get_dataloader(self, name: str) -> "reax.data.DataLoader":
        """Get the dataloader of the given name"""
        try:
            return self._loaders[name]
        except KeyError:
            pass

        try:
            return self._from_datasource[name]
        except KeyError:
            pass

        loader = self._request_dataloader(name)
        self._from_datasource[name] = loader
        return loader

    @property
    def source(self) -> "reax.data.DataSource | None":
        """The original source of the dataloader (if there is one)"""
        return self._datasource

    def prepare_data(self) -> None:
        """Tell the data source to prepare the data for use"""
        if self._datasource is not None and overrides.is_overridden(
            "prepare_data", self._datasource, self._source_base_type
        ):
            self._datasource.prepare_data()

    def setup(self, /, *, stage: str | None = None) -> None:
        """Call ``setup`` on the wrapped source.

        The :class:`reax.Engine` bound at construction is forwarded to the source.

        Args:
            stage: Name of the stage being set up (e.g. ``"fit"``, ``"validate"``, ``"test"``).
        """
        if self._datasource is None:
            return
        self._datasource.setup(self._engine, stage=stage)

    def on_exception(self, exception: BaseException) -> None:
        """Tell the data source that an exception has occurred"""

        if self._datasource is not None:
            self._datasource.on_exception(exception)

    def teardown(self, /, *, stage: str | None = None) -> None:
        """Call ``teardown`` on the wrapped source.

        Args:
            stage: Name of the stage being torn down (e.g. ``"fit"``, ``"validate"``, ``"test"``).
        """
        if self._datasource is None:
            return
        self._datasource.teardown(self._engine, stage=stage)

    def reset(self):
        """Reset the cache so dataloaders get reloaded"""
        self._from_datasource = {}

    def _request_dataloader(self, name: str) -> "reax.DataLoader":
        """Get the dataloader directly from the source"""
        loader_name = f"{name}_dataloader"
        loader = getattr(self._datasource, loader_name)()
        loader = self._setup_dataloader(loader)
        return loader

    def _setup_dataloader(self, loader: "reax.DataLoader") -> "reax.DataLoader":
        """Wrap a dataloader with the bound engine's device/distributed logic."""
        return self._engine.setup_dataloaders(loader)


def create_manager(
    engine: "reax.Engine",
    module: _datasources.DataSource | None = None,
    datamodule: datamodules.DataModule | None = None,
    **loaders,
) -> DataSourceManager:
    """Create a :class:`DataSourceManager` for a source and engine.

    Args:
        engine: The :class:`reax.Engine` to bind to the manager for device placement.
            Required — the manager will call ``engine.setup_dataloaders`` for every
            dataloader it hands out.
        module: Optional :class:`~reax.Module` to source dataloaders from.
        datamodule: Optional :class:`~reax.DataModule` to source dataloaders from.
            Takes priority over ``module`` if both are provided.
        **loaders: Pre-supplied dataloaders keyed by name (e.g. ``train=dl``) that
            bypass the source and are wrapped directly by the engine.

    Returns:
        A :class:`DataSourceManager` ready to be used with the :class:`reax.Trainer`.
    """
    # Filter out any loaders that are `None` as this makes calling this method easier
    passed_loaders = {name: loader for name, loader in loaders.items() if loader is not None}

    if datamodule is not None:
        source = datamodule
    elif module is not None:
        source = module
    else:
        source = None

    return DataSourceManager(source, engine, **passed_loaders)
