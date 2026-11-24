Data Handling
=============

REAX provides flexible tools for managing data using :class:`~reax.data.DataLoader` and
:class:`~reax.DataModule`.

DataLoaders
-----------

REAX works seamlessly with JAX data. Use :class:`~reax.data.ReaxDataLoader` together with
:class:`~reax.data.ArrayDataset` (or your own :class:`~reax.data.Dataset`), or any iterable that
yields batches of numpy/JAX arrays.

Where your data lives
---------------------

REAX moves every batch to the engine's device for you, so you only need to decide where the
*dataset* lives.  The rule is simple: **NumPy arrays live on the host, JAX arrays live on the
engine's device.**

**On the host (the default choice).**  Build your dataset from NumPy arrays.  Sampling,
indexing and collation then all happen on the CPU, and REAX copies each batch to the device
once, just before the step.  This works for datasets of any size and keeps accelerator memory
free for the model.

.. code-block:: python

    dataset = reax.data.ArrayDataset(np.load("x.npy"), np.load("y.npy"))
    loader = reax.data.ReaxDataLoader(dataset, batch_size=64, shuffle=True)

**On the device.**  If the whole dataset fits in accelerator memory, keeping it there avoids the
per-batch copy.  Create it as JAX arrays and use :class:`~reax.data.ArrayLoader`, which gathers
each batch directly on the device:

.. code-block:: python

    x, y = jax.device_put((np.load("x.npy"), np.load("y.npy")), engine.device)
    loader = reax.data.ArrayLoader((x, y), batch_size=64, shuffle=True)

.. note::

    :class:`~reax.data.ReaxDataLoader` indexes one sample at a time and collates on the host
    with NumPy.  Given a dataset of JAX arrays, it copies every sample back to the host and then
    sends the batch to the device again, which is slower than either option above.  Use
    :class:`~reax.data.ArrayLoader` for device-resident data.

Inside a stage (e.g. in :meth:`~reax.DataModule.setup`), arrays created with ``jnp`` are placed on
the engine's device, as the engine's device is the default JAX device while a stage runs.  This
matters when it differs from JAX's own default, e.g. ``accelerator="cpu"`` on a GPU machine.

Randomness used for data (splits, shuffling, k-fold partitions) is always computed with NumPy on
the host, whichever option you choose, see :doc:`randomness`.

DataModules
-----------

A :class:`~reax.DataModule` encapsulates all steps needed to process data: downloading,
tokenising, and splitting. It ensures reproducibility and makes data handling reusable across
projects.

A DataModule is defined by the following steps:

1.  **prepare_data**: Download, tokenise, etc. (runs only on 1 node in distributed settings).
2.  **setup**: Split data, apply transforms (runs on every device).
3.  **train_dataloader**: Returns the training dataloader.
4.  **val_dataloader**: Returns the validation dataloader.
5.  **test_dataloader**: Returns the test dataloader.
6.  **predict_dataloader**: Returns the predict dataloader.
7.  **teardown**: Clean up resources after a stage ends (runs on every device).

Example
-------

.. code-block:: python

    from reax.data import ReaxDataLoader

    class MNISTDataModule(reax.DataModule):
        def prepare_data(self):
            # Download MNIST
            ...

        def setup(self, engine, *, stage):
            # Split dataset
            # Draws from the engine's `data` stream, so the split follows the trainer seed
            self.data_train, self.data_val = random_split(engine.rngs, dataset, [0.9, 0.1])
            ...

        def train_dataloader(self):
            return ReaxDataLoader(self.data_train, batch_size=64, shuffle=True)

        def val_dataloader(self):
            return ReaxDataLoader(self.data_val, batch_size=64)

        def teardown(self, engine, *, stage):
            # Free resources
            ...

Using a DataModule
------------------

Pass the DataModule to the Trainer:

.. code-block:: python

    dm = MNISTDataModule()
    trainer.fit(model, datamodule=dm)
