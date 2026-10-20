Hooks
=====

Hooks let you run code at a specific point in a :class:`~reax.Trainer` call
(:meth:`~reax.Trainer.fit`, :meth:`~reax.Trainer.validate`,
:meth:`~reax.Trainer.test`, or :meth:`~reax.Trainer.predict`) without touching
the core training logic.

REAX splits its hooks into two layers:

*   **Module hooks** -- short, model-specific methods you override on your
    :class:`~reax.Module` (e.g. :meth:`~reax.Module.training_step`,
    :meth:`~reax.Module.on_train_batch_start`,
    :meth:`~reax.Module.on_before_optimizer_step`).
*   **Trainer hooks** -- generic lifecycle hooks on a
    :class:`~reax.TrainerListener` you pass to the
    :class:`~reax.Trainer` (e.g. :meth:`~reax.TrainerListener.on_train_start`,
    :meth:`~reax.TrainerListener.on_batch_end`).

Both hook types see the same run, but they run in a fixed, predictable order
described below.

Which hook goes where
----------------------

A hook belongs in the **Module** when it is about the model's behaviour at a
particular point (e.g. computing a loss in :meth:`~reax.Module.training_step`,
or logging a quantity in :meth:`~reax.Module.on_train_batch_end`, or adjusting
the loss in :meth:`~reax.Module.on_before_optimizer_step`).

A hook belongs in a **Trainer listener** when it is a cross-cutting concern
that does not depend on a specific model (e.g. checkpointing, early stopping,
progress bars, custom logging).

The two hook families share names (``on_train_start``, ``on_train_batch_start``,
and so on) but have **different signatures**:

*   Trainer hooks always receive the ``trainer`` and the current ``stage``;
    loop-level hooks additionally receive the driving ``loop`` (for example
    :class:`~reax.stages.TrainEpoch`, :class:`~reax.stages.ValidationEpoch`,
    :class:`~reax.stages.TestEpoch`, or
    :class:`~reax.stages.PredictionEpoch`).
*   Module hooks receive the current ``stage`` and, where useful, the
    ``batch`` and its ``batch_idx``.

The ordered execution of hooks
------------------------------

When :meth:`~reax.Trainer.fit` (or a similar call) runs, the following general
lifecycle wraps the stage-specific hooks below:

.. code-block:: text

    setup(trainer, stage)
    on_stage_start(trainer, stage)
    ... stage-specific hooks ...
    on_stage_end(trainer, stage)
    teardown(trainer, stage)

Training (``Trainer.fit(...)``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hooks fired during :meth:`~reax.Trainer.fit`. ``T`` denotes a
:class:`~reax.TrainerListener`; ``M`` denotes a :class:`~reax.Module` hook.
Module hooks are called directly by the stage; listener hooks are dispatched by
the trainer.

.. code-block:: text

    T.setup(trainer, stage)
    T.on_stage_start(trainer, stage)
    T.on_fit_start(trainer, stage)
    T.on_train_start(trainer, stage, loop)
    for epoch in 1..max_epochs:
        T.on_train_epoch_start(trainer, stage, loop)
        M.on_train_epoch_start(stage)
        for batch, batch_idx in dataloader:
            M.on_train_batch_start(stage, batch, batch_idx)
            T.on_train_batch_start(trainer, stage, loop, batch)
            M.training_step(batch, batch_idx)
            M.on_before_optimizer_step(optimizer, grad)
            optimizer.update_module(...)
            T.on_train_batch_end(trainer, stage, loop, outputs)
            M.on_train_batch_end(stage, outputs, batch, batch_idx)
            # if enabled: validation loop runs here (see below)
        T.on_train_epoch_end(trainer, stage, loop)
        M.on_train_epoch_end(stage)
    T.on_train_end(trainer, stage, loop)
    T.on_fit_end(trainer, stage)
    T.on_stage_end(trainer, stage)
    T.teardown(trainer, stage)

Validation (``Trainer.validate(...)``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    T.setup(trainer, stage)
    T.on_stage_start(trainer, stage)
    T.on_validation_start(trainer, stage)
    T.on_validation_epoch_start(trainer, stage, loop)
    M.on_validation_epoch_start(stage)
    for batch, batch_idx in dataloader:
        M.on_validation_batch_start(stage, batch, batch_idx)
        T.on_validation_batch_start(trainer, stage, loop, batch)
        M.validation_step(batch, batch_idx)
        T.on_validation_batch_end(trainer, stage, loop, outputs)
        M.on_validation_batch_end(stage, outputs, batch, batch_idx)
    T.on_validation_epoch_end(trainer, stage, loop)
    M.on_validation_epoch_end(stage)
    T.on_validation_end(trainer, stage)
    T.on_stage_end(trainer, stage)
    T.teardown(trainer, stage)

Test (``Trainer.test(...)``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    T.setup(trainer, stage)
    T.on_stage_start(trainer, stage)
    T.on_test_start(trainer, stage)
    T.on_test_epoch_start(trainer, stage, loop)
    M.on_test_epoch_start(stage)
    for batch, batch_idx in dataloader:
        M.on_test_batch_start(stage, batch, batch_idx)
        T.on_test_batch_start(trainer, stage, loop, batch)
        M.test_step(batch, batch_idx)
        T.on_test_batch_end(trainer, stage, loop, outputs)
        M.on_test_batch_end(stage, outputs, batch, batch_idx)
    T.on_test_epoch_end(trainer, stage, loop)
    M.on_test_epoch_end(stage)
    T.on_test_end(trainer, stage)
    T.on_stage_end(trainer, stage)
    T.teardown(trainer, stage)

Predict (``Trainer.predict(...)``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    T.setup(trainer, stage)
    T.on_stage_start(trainer, stage)
    T.on_predict_start(trainer, stage)
    T.on_predict_epoch_start(trainer, stage, loop)
    M.on_predict_epoch_start(stage)
    for batch, batch_idx in dataloader:
        M.on_predict_batch_start(stage, batch, batch_idx)
        T.on_predict_batch_start(trainer, stage, loop, batch)
        M.predict_step(batch, batch_idx)
        T.on_predict_batch_end(trainer, stage, loop, outputs)
        M.on_predict_batch_end(stage, outputs, batch, batch_idx)
    T.on_predict_epoch_end(trainer, stage, loop)
    M.on_predict_epoch_end(stage)
    T.on_predict_end(trainer, stage)
    T.on_stage_end(trainer, stage)
    T.teardown(trainer, stage)

Reference
---------

.. autosummary::

    reax.TrainerListener
    reax.Module
