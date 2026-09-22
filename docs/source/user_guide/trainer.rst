The Trainer
===========

The :class:`~reax.Trainer` automates the training loop. It handles the boring details of the
training process, such as iterating over epochs, validation checks, creating checkpoints, and
logging.

Basic Usage
-----------

To use the Trainer, you simply initialise it and call :meth:`~reax.Trainer.fit`.

.. code-block:: python

    model = MyModel(din=32, dout=10, rngs=nnx.Rngs(0))
    trainer = reax.Trainer(accelerator='auto')
    trainer.fit(model, train_dataloader, val_dataloader, max_epochs=10)

Under the Hood
--------------

The Trainer uses an :class:`~reax.Engine` to execute the training. The Engine abstracts away the
hardware and distributed strategy details.

Key Arguments
-------------

The ``Trainer`` constructor takes the following keyword arguments:

*   **accelerator**: The hardware accelerator to use (e.g., ``'cpu'``, ``'gpu'``, ``'tpu'``, or
    ``'auto'``).
*   **strategy**: The distributed strategy to use (e.g., ``'ddp'`` or ``'auto'``).
*   **devices**: The number of devices or specific device indices to use.
*   **logger**: The logger to use (e.g., :class:`~reax.loggers.CsvLogger`).
*   **listeners**: A list of listeners to extend the Trainer's behaviour.
*   **enable_checkpointing**: Whether to automatically save checkpoints (default ``True``).
*   **default_root_dir**: Root directory for logs and checkpoints.
*   **checkpointing**: A :class:`~reax.Checkpointing` instance controlling serialisation.

The following limits and scheduling options are **not** constructor arguments -- they are passed to
:meth:`~reax.Trainer.fit` instead:

*   **max_epochs**: The maximum number of epochs to train for.
*   **min_epochs**: The minimum number of epochs to train for.
*   **max_updates**: The maximum number of optimizer updates.
*   **max_time**: The maximum amount of wall-clock time to train for.
*   **limit_train_batches** / **limit_val_batches**: Limits the fraction of batches per epoch.

Methods
-------

Fit
~~~

:meth:`~reax.Trainer.fit` runs the full training routine, including validation loops.

.. code-block:: python

    trainer.fit(model, train_loader, val_loader)

max_time
~~~~~~~~

Set the maximum amount of time for training. Training will get interrupted
mid-epoch. ``max_time`` is a keyword argument to :meth:`~reax.Trainer.fit` and accepts a
``"DD:HH:MM:SS"`` string, a :class:`datetime.timedelta`, or a ``dict`` of calendar fields.

.. code-block:: python

    # Default (disabled)
    trainer.fit(model, train_dataloader, val_dataloader, max_time=None)

    # Stop after 12 hours of training or when reaching 10 epochs (string)
    trainer.fit(model, train_dataloader, val_dataloader, max_time="00:12:00:00", max_epochs=10)

    # Stop after 1 day and 5 hours (dict)
    trainer.fit(model, train_dataloader, val_dataloader, max_time={"days": 1, "hours": 5})

Test
~~~~

:meth:`~reax.Trainer.test` runs the test loop on the given dataloader.

.. code-block:: python

    trainer.test(model, test_loader)

Predict
~~~~~~~

:meth:`~reax.Trainer.predict` runs inference on the given dataloader.

.. code-block:: python

    predictions = trainer.predict(model, predict_loader)

Automatic Optimization
----------------------

By default, the Trainer handles backward passes and optimizer steps automatically. This simplifies
the ``training_step`` to just returning the loss.
