from typing import TYPE_CHECKING, Generic, Optional, TypeVar

from .. import exceptions

if TYPE_CHECKING:
    import reax

__all__ = ("DataSource",)

_T_co = TypeVar("_T_co", covariant=True)


class DataSource(Generic[_T_co]):
    """Interface for classes that provide datasets."""

    def __init__(self) -> None:
        """
        Attributes:
            prepare_data_per_node:
                If True, each LOCAL_RANK=0 will call prepare data.
                Otherwise, only NODE_RANK=0, LOCAL_RANK=0 will prepare data.
            allow_zero_length_dataloader_with_multiple_devices:
                If True, dataloader with zero length within local rank is allowed.
                Default value is False.
        """
        super().__init__()
        self.prepare_data_per_node: bool = True
        self.allow_zero_length_dataloader_with_multiple_devices: bool = False

    # Region Events

    def prepare_data(self) -> None:
        """Use this to download and prepare data. Downloading and saving data with multiple
        processes (distributed settings) will result in corrupted data. Lightning ensures this
        method is called only within a single process, so you can safely add your downloading logic
        within.

        .. warning:: DO NOT set state to the model (use ``setup`` instead)
            since this is NOT called on every device

        Example::

            def prepare_data(self):
                # good
                download_data()
                tokenize()
                etc()

                # bad
                self.split = data_split
                self.some_state = some_other_state()

        In a distributed environment, ``prepare_data`` can be called in two ways
        (using :ref:`prepare_data_per_node<common/lightning_module:prepare_data_per_node>`)

        1. Once per node. This is the default and is only called on LOCAL_RANK=0.
        2. Once in total. Only called on GLOBAL_RANK=0.

        Example::

            # DEFAULT
            # called once per node on LOCAL_RANK=0 of that node
            class LitDataModule(LightningDataModule):
                def __init__(self):
                    super().__init__()
                    self.prepare_data_per_node = True


            # call on GLOBAL_RANK=0 (great for shared file systems)
            class LitDataModule(LightningDataModule):
                def __init__(self):
                    super().__init__()
                    self.prepare_data_per_node = False

        This is called before requesting the dataloaders:

        .. code-block:: python

            model.prepare_data()
            initialize_distributed()
            model.setup(stage)
            model.train_dataloader()
            model.val_dataloader()
            model.test_dataloader()
            model.predict_dataloader()

        """

    def setup(self, stage: "reax.Stage", /) -> None:
        """Called at the beginning of a trainer stage. This is a good hook when you need to
        build models dynamically or adjust something about them. This hook is called on every
        process when using DDP.

        Example::

            class LitModel(...):
                def __init__(self):
                    self.l1 = None

                def prepare_data(self):
                    download_data()
                    tokenize()

                    # don't do this
                    self.something = else

                def setup(self, stage):
                    data = load_data(...)
                    self.l1 = nn.Linear(28, data.num_classes)

        """

    def teardown(self, stage: "reax.Stage", /) -> None:
        """Called at the end of a trainer stage"""

    def on_exception(self, exception: BaseException, /) -> None:
        """Called when the stage execution is interrupted by an exception."""

    # endregion

    def train_dataloader(self) -> "reax.DataLoader[_T_co]":
        """An iterable or collection of iterables specifying training samples.

        For more information about multiple dataloaders, see :ref:`section <multiple-dataloaders>`.

        The dataloader you return will not be reloaded unless you set
        :paramref:`~reax.Trainer.fit(reload_dataloaders_every_n_epochs=k)` to a positive integer.

        For data processing use the following pattern:

            - download in :meth:`prepare_data`
            - process and split in :meth:`setup`

        However, the above are only necessary for distributed processing.

        .. warning:: do not assign state in prepare_data

        - :meth:`~reax.Trainer.fit`
        - :meth:`prepare_data`
        - :meth:`setup`

        Note:
            Lightning tries to add the correct sampler for distributed and arbitrary hardware.
            There is no need to set it yourself.

        """
        raise exceptions.MisconfigurationException(
            "`train_dataloader` must be implemented to be used with the Lightning Trainer"
        )

    def val_dataloader(self) -> Optional["reax.DataLoader[_T_co]"]:
        r"""An iterable or collection of iterables specifying validation samples.

        For more information about multiple dataloaders, see :ref:`section <multiple-dataloaders>`.

        The dataloader you return will not be reloaded unless you set
        :paramref:`~lightning.pytorch.trainer.trainer.Trainer.reload_dataloaders_every_n_epochs` to
        a positive integer.

        It's recommended that all data downloads and preparation happen in :meth:`prepare_data`.

        - :meth:`~reax.Trainer.fit`
        - :meth:`~reax.Trainer.validate`
        - :meth:`prepare_data`
        - :meth:`setup`

        Note:
            REAX tries to add the correct sampler for distributed and arbitrary hardware
            There is no need to set it yourself.

        Note:
            If you don't need a validation dataset and a :meth:`validation_step`, you don't need to
            implement this method.

        """

    def test_dataloader(self) -> "reax.DataLoader[_T_co]":
        r"""An iterable or collection of iterables specifying test samples.

        For more information about multiple dataloaders, this :ref:`section <multiple-dataloaders>`.

        For data processing use the following pattern:

            - download in :meth:`prepare_data`
            - process and split in :meth:`setup`

        However, the above are only necessary for distributed processing.

        .. warning:: do not assign state in prepare_data


        - :meth:`~lightning.pytorch.trainer.trainer.Trainer.test`
        - :meth:`prepare_data`
        - :meth:`setup`

        Note:
            Lightning tries to add the correct sampler for distributed and arbitrary hardware.
            There is no need to set it yourself.

        Note:
            If you don't need a test dataset and a :meth:`test_step`, you don't need to implement
            this method.

        """
        raise exceptions.MisconfigurationException(
            "`test_dataloader` must be implemented to be used with the Lightning Trainer"
        )

    def predict_dataloader(self) -> "reax.DataLoader[_T_co]":
        r"""An iterable or collection of iterables specifying prediction samples.

        For more information about multiple dataloaders, see :ref:`section <multiple-dataloaders>`.

        It's recommended that all data downloads and preparation happen in :meth:`prepare_data`.

        - :meth:`~reax.Trainer.predict`
        - :meth:`prepare_data`
        - :meth:`setup`

        Note:
            REAX tries to add the correct sampler for distributed and arbitrary hardware
            There is no need to set it yourself.

        Return:
            A :class:`reax.DataLoader` or a sequence of them specifying prediction samples.

        """
        raise exceptions.MisconfigurationException(
            "`predict_dataloader` must be implemented to be used with the REAX Trainer"
        )
