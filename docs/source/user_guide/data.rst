Data Handling
=============

REAX provides flexible tools for managing data using :class:`~reax.data.DataLoader` and
:class:`~reax.DataModule`.

DataLoaders
-----------

REAX works seamlessly with JAX data. Use :class:`~reax.data.ReaxDataLoader` together with
:class:`~reax.data.ArrayDataset` (or your own :class:`~reax.data.Dataset`), or any iterable that
yields batches of numpy/JAX arrays.

DataModules
-----------

A :class:`~reax.DataModule` encapsulates all steps needed to process data: downloading,
tokenising, and splitting. It ensures reproducibility and makes data handling reusable across
projects.

A DataModule is defined by the following steps:

1.  **prepare_data**: Download, tokenise, etc. (runs only on 1 CPU in distributed settings).
2.  **setup**: Split data, apply transforms (runs on every device).
3.  **train_dataloader**: Returns the training dataloader.
4.  **val_dataloader**: Returns the validation dataloader.
5.  **test_dataloader**: Returns the test dataloader.
6.  **predict_dataloader**: Returns the predict dataloader.

Example
-------

.. code-block:: python

    from reax.data import ReaxDataLoader

    class MNISTDataModule(reax.DataModule):
        def prepare_data(self):
            # Download MNIST
            ...

        def setup(self, stage):
            # Split dataset
            ...

        def train_dataloader(self):
            return ReaxDataLoader(self.data_train, batch_size=64, shuffle=True)

        def val_dataloader(self):
            return ReaxDataLoader(self.data_val, batch_size=64)

Using a DataModule
------------------

Pass the DataModule to the Trainer:

.. code-block:: python

    dm = MNISTDataModule()
    trainer.fit(model, datamodule=dm)
