import abc
from typing import TYPE_CHECKING, Optional, Union

from lightning_utilities.core import overrides

from . import _datasources, datamodules
from .. import modules

if TYPE_CHECKING:
    import reax


___all__ = ("DataSourceManager", "create_manager")


class DataSourceManager(abc.ABC):
    """Manager for coordinating getting data from a source"""

    def __init__(self, source: Optional[_datasources.DataSource], **loaders):
        self._datasource: Optional[_datasources.DataSource] = source
        self._loaders: dict[str, "reax.data.DataLoader"] = loaders
        self._from_datasource: dict[str, "reax.data.DataLoader"] = {}

    @property
    def _source_base_type(
        self,
    ) -> Union[type[datamodules.DataModule], type[modules.Module]]:
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

        loader_name = f"{name}_dataloader"
        loader = getattr(self._datasource, loader_name)()
        self._from_datasource[name] = loader
        return loader

    @property
    def source(self) -> Optional["reax.data.DataSource"]:
        """The original source of the dataloader (if there is one)"""

        return self._datasource

    def prepare_data(self) -> None:
        """Tell the data source to prepare the data for use"""

        if self._datasource is not None and overrides.is_overridden(
            "prepare_data", self._datasource, self._source_base_type
        ):
            self._datasource.prepare_data()

    def setup(self, stage) -> None:
        """Tell the data source to set itself up"""

        if self._datasource is not None:
            self._datasource.setup(stage)

    def on_exception(self, exception: BaseException) -> None:
        """Tell the data source that an exception has occurred"""

        if self._datasource is not None:
            self._datasource.on_exception(exception)

    def teardown(self, stage) -> None:
        """Tell the data source to tear down"""

        if self._datasource is not None:
            self._datasource.teardown(stage)

    def reset(self):
        """Reset the cache so dataloaders get reloaded"""
        self._from_datasource = {}


def create_manager(
    module: _datasources.DataSource = None,
    datamodule: datamodules.DataModule = None,
    **loaders,
) -> DataSourceManager:
    # Filter out any loaders that are `None` as this makes calling this method easier
    passed_loaders = {name: loader for name, loader in loaders.items() if loader is not None}

    if datamodule is not None:
        source = datamodule
    elif module is not None:
        source = module
    else:
        source = None

    return DataSourceManager(source, **passed_loaders)
