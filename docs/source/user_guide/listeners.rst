Listeners
=========

Listeners allow you to customise the behaviour of the training loop at specific points, such as the
start of an epoch or after a batch.

Built-in Listeners
------------------

REAX comes with several built-in listeners:

*   :class:`~reax.listeners.ModelCheckpoint`: Automatically save model checkpoints based on a
    monitored metric.
*   :class:`~reax.listeners.EarlyStopping`: Stop training early if a metric stops improving.
*   :class:`~reax.listeners.ProgressBar`: Display a progress bar during training.

Custom Listeners
----------------

You can create your own listener by subclassing :class:`~reax.TrainerListener` and overriding any
of its hooks.

Each hook receives the ``Trainer`` instance and the ``Stage`` running. Loop-level hooks --
``on_train_start``, ``on_train_end`` and the epoch/batch hooks such as
``on_train_epoch_start``, ``on_train_batch_start`` and ``on_train_batch_end`` (and their
validation/test/predict variants) -- also receive the ``Loop`` object driving that part of the run,
e.g. a :class:`~reax.stages.FitLoop` or :class:`~reax.stages.TrainEpoch`. Use it to inspect or
control the iteration, such as reading ``loop.iteration``.

.. code-block:: python

    import reax

    class MyPrintingListener(reax.TrainerListener):
        def on_train_start(self, trainer, stage, loop):
            print("Training is starting!")

        def on_train_end(self, trainer, stage, loop):
            print("Training is ending!")

Using Listeners
---------------

Pass your listeners to the Trainer:

.. code-block:: python

    trainer = reax.Trainer(listeners=[MyPrintingListener()])

Order of Execution
------------------

Listeners are executed in the order they are passed to the Trainer.
