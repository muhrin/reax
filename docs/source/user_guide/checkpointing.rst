Checkpointing
=============

REAX provides flexible checkpointing to save and restore your training progress. This guide covers
checkpoint strategies, resuming training, and best practices.

Automatic Checkpointing
------------------------

The :class:`~reax.Trainer` automatically saves checkpoints when you enable checkpointing:

.. code-block:: python

    trainer = reax.Trainer(
        enable_checkpointing=True  # Enabled by default
    )
    trainer.fit(model, train_loader, val_loader, max_epochs=10)

By default, REAX saves checkpoints in ``./reax_logs/version_X/checkpoints/``.

Custom Checkpoint Directory
----------------------------

Specify where to save checkpoints:

.. code-block:: python

    trainer = reax.Trainer(
        default_root_dir="./my_experiments",
        enable_checkpointing=True
    )

Checkpoints will be saved in ``./my_experiments/version_X/checkpoints/``.

ModelCheckpoint Listener
-------------------------

For fine-grained control, use the :class:`~reax.listeners.ModelCheckpoint` listener:

.. code-block:: python

    from reax.listeners import ModelCheckpoint

    # Save the best model based on validation loss
    checkpoint_listener = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=3,  # Keep the 3 best checkpoints
        filename="best-{epoch:02d}-{val_loss:.2f}"
    )

    trainer = reax.Trainer(
        listeners=[checkpoint_listener]
    )

    trainer.fit(model, train_loader, val_loader, max_epochs=10)

Monitor Options
~~~~~~~~~~~~~~~

*   **monitor**: Metric to track (e.g., ``"val_loss"``, ``"val_acc"``)
*   **mode**: ``"min"`` for loss, ``"max"`` for accuracy
*   **save_top_k**: Number of best checkpoints to keep (``-1`` keeps all)
*   **filename**: Checkpoint filename pattern

Save Every N Epochs
~~~~~~~~~~~~~~~~~~~~

Save checkpoints at regular intervals:

.. code-block:: python

    checkpoint_listener = ModelCheckpoint(
        every_n_epochs=5,  # Save every 5 epochs
        save_top_k=-1      # Keep all checkpoints
    )

Save Last Checkpoint
~~~~~~~~~~~~~~~~~~~~

Always save the most recent checkpoint:

.. code-block:: python

    checkpoint_listener = ModelCheckpoint(
        save_last=True,
        filename="last-{epoch:02d}"
    )

Resuming Training
-----------------

Load a checkpoint to resume training:

.. code-block:: python

    # Load checkpoint
    checkpoint = trainer.checkpointing.load("path/to/checkpoint.ckpt")

    # Restore model parameters
    model.set_parameters(checkpoint["parameters"])

    # Resume training
    trainer.fit(model, train_loader, val_loader)

The checkpoint contains:

*   ``parameters``: Model parameters
*   ``state_dict``: Any additional module state (the optimiser is restored automatically)
*   ``epoch``: Current epoch
*   ``global_step``: Global training step
*   ``reax_version``: The REAX version the checkpoint was written with

Manual Checkpointing
--------------------

Save checkpoints manually:

.. code-block:: python

    # During training
    checkpoint_data = {
        "parameters": model.parameters(),
        "state_dict": model.state_dict(),
        "epoch": current_epoch,
    }

    # ``save`` takes the checkpoint dict first, then the filepath
    trainer.checkpointing.save(checkpoint_data, "my_checkpoint.ckpt")

Load manual checkpoints:

.. code-block:: python

    checkpoint = trainer.checkpointing.load("my_checkpoint.ckpt")
    model.set_parameters(checkpoint["parameters"])

Checkpoint Format
-----------------

REAX serialises checkpoints with MessagePack (a fast, compact binary format) by default. The
:class:`~reax.Checkpointing` backend is a small abstract interface with
:meth:`~reax.Checkpointing.save` and :meth:`~reax.Checkpointing.load`, and
:class:`~reax.MsgpackCheckpointing` is the built-in implementation. You can pass an
alternate backend to the Trainer:

.. code-block:: python

    from reax import MsgpackCheckpointing

    trainer = reax.Trainer(checkpointing=MsgpackCheckpointing())

Best Practices
--------------

Save Based on Validation Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Always monitor validation metrics to avoid overfitting:

.. code-block:: python

    checkpoint_listener = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1
    )

Keep Multiple Checkpoints
~~~~~~~~~~~~~~~~~~~~~~~~~~

Save several checkpoints in case the best one is corrupted:

.. code-block:: python

    checkpoint_listener = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=3  # Keep top 3
    )

Use Descriptive Filenames
~~~~~~~~~~~~~~~~~~~~~~~~~~

Include metrics in checkpoint filenames for easy identification:

.. code-block:: python

    checkpoint_listener = ModelCheckpoint(
        filename="epoch={epoch:02d}-val_loss={val_loss:.4f}-val_acc={val_acc:.4f}"
    )

Checkpoint Large Models Efficiently
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For large models, consider:

*   Saving less frequently (``every_n_epochs=10``)
*   Keeping fewer checkpoints (``save_top_k=1``)
*   Saving weights only with ``save_weights_only=True`` to skip the extra module state

Example: Complete Checkpointing Setup
--------------------------------------

.. code-block:: python

    from reax import Trainer
    from reax.listeners import ModelCheckpoint

    # Save best model based on validation accuracy
    best_checkpoint = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename="best-model-{epoch:02d}-{val_acc:.4f}"
    )

    # Save checkpoint every 10 epochs
    periodic_checkpoint = ModelCheckpoint(
        every_n_epochs=10,
        save_top_k=-1,
        filename="periodic-{epoch:02d}"
    )

    # Always save the last checkpoint
    last_checkpoint = ModelCheckpoint(
        save_last=True,
        filename="last"
    )

    trainer = Trainer(
        listeners=[best_checkpoint, periodic_checkpoint, last_checkpoint],
        default_root_dir="./experiments/my_model"
    )

    trainer.fit(model, train_loader, val_loader, max_epochs=100)

    # After training, load the best model
    best_path = best_checkpoint.best_model_path
    checkpoint = trainer.checkpointing.load(best_path)
    model.set_parameters(checkpoint["parameters"])

See Also
--------

*   :doc:`trainer` - Trainer configuration
*   :doc:`listeners` - Custom listeners
*   :doc:`logging` - Experiment logging
