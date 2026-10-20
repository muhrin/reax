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
and so on) and have the **same signatures**, except that module hooks don't
receive the ``trainer`` (it is available as :attr:`reax.Module.trainer`):

*   Stage-level hooks receive the current ``stage``.
*   Loop-level hooks additionally receive the driving ``loop`` (for example
    :class:`~reax.stages.TrainEpoch`, :class:`~reax.stages.ValidationEpoch`,
    :class:`~reax.stages.TestEpoch`, or
    :class:`~reax.stages.PredictionEpoch`).
*   Batch hooks additionally receive the ``batch`` (at the start) or the
    ``outputs`` of the step (at the end). The batch index is available as
    ``loop.batch_idx``.

The ordered execution of hooks
------------------------------

When :meth:`~reax.Trainer.fit` (or a similar call) runs, the following general
lifecycle wraps the stage-specific hooks below:

.. code-block:: text

    setup(trainer, stage)
    on_stage_start(trainer, stage)
    on_<stage>_start(trainer, stage)    # e.g. on_fit_start
    ... loop hooks ...
    on_stage_end(trainer, stage)
    on_<stage>_end(trainer, stage)      # e.g. on_fit_end
    teardown(trainer, stage)

Every loop additionally fires the generic ``on_loop_start``,
``on_loop_iter_start``, ``on_loop_iter_end`` and ``on_loop_end`` listener hooks.
These are fired just before the corresponding loop-specific listener hook, and are
omitted from the listings below for clarity, as are the per-iteration hooks of
the loop over epochs.

Module hooks are called directly by the stage, while listener hooks are
dispatched by the trainer. For each event the module hook is called first.

Training (``Trainer.fit(...)``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Hooks fired during :meth:`~reax.Trainer.fit`. ``T`` denotes a
:class:`~reax.TrainerListener`; ``M`` denotes a :class:`~reax.Module` hook.

.. code-block:: text

    T.setup(trainer, stage)
    M.on_fit_start(stage)
    T.on_stage_start(trainer, stage)
    T.on_fit_start(trainer, stage)
    # if enabled: the validation sanity check runs here
    M.on_train_start(stage, loop)
    T.on_train_start(trainer, stage, loop)
    for epoch in range(max_epochs):
        M.on_train_epoch_start(stage, loop)
        T.on_train_epoch_start(trainer, stage, loop)
        for batch in dataloader:
            M.on_train_batch_start(stage, loop, batch)
            T.on_train_batch_start(trainer, stage, loop, batch)
            M.training_step(batch, batch_idx)
            M.on_before_optimizer_step(optimizer, grad)
            optimizer.update_module(...)
            M.on_train_batch_end(stage, loop, outputs)
            # if due: the validation loop runs here (see below)
            T.on_train_batch_end(trainer, stage, loop, outputs)
        M.on_train_epoch_end(stage, loop)
        T.on_train_epoch_end(trainer, stage, loop)
    M.on_train_end(stage, loop)
    T.on_train_end(trainer, stage, loop)
    M.on_fit_end(stage)
    T.on_stage_end(trainer, stage)
    T.on_fit_end(trainer, stage)
    T.teardown(trainer, stage)

Validation during fitting runs the validation epoch hooks (from
``on_validation_epoch_start`` to ``on_validation_epoch_end``) shown below, with
the :class:`~reax.stages.Fit` stage as the ``stage``.

Validation (``Trainer.validate(...)``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    T.setup(trainer, stage)
    M.on_validation_start(stage)
    T.on_stage_start(trainer, stage)
    T.on_validation_start(trainer, stage)
    M.on_validation_epoch_start(stage, loop)
    T.on_validation_epoch_start(trainer, stage, loop)
    for batch in dataloader:
        M.on_validation_batch_start(stage, loop, batch)
        T.on_validation_batch_start(trainer, stage, loop, batch)
        M.validation_step(batch, batch_idx)
        M.on_validation_batch_end(stage, loop, outputs)
        T.on_validation_batch_end(trainer, stage, loop, outputs)
    M.on_validation_epoch_end(stage, loop)
    T.on_validation_epoch_end(trainer, stage, loop)
    M.on_validation_end(stage)
    T.on_stage_end(trainer, stage)
    T.on_validation_end(trainer, stage)
    T.teardown(trainer, stage)

Test (``Trainer.test(...)``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    T.setup(trainer, stage)
    M.on_test_start(stage)
    T.on_stage_start(trainer, stage)
    T.on_test_start(trainer, stage)
    M.on_test_epoch_start(stage, loop)
    T.on_test_epoch_start(trainer, stage, loop)
    for batch in dataloader:
        M.on_test_batch_start(stage, loop, batch)
        T.on_test_batch_start(trainer, stage, loop, batch)
        M.test_step(batch, batch_idx)
        M.on_test_batch_end(stage, loop, outputs)
        T.on_test_batch_end(trainer, stage, loop, outputs)
    M.on_test_epoch_end(stage, loop)
    T.on_test_epoch_end(trainer, stage, loop)
    M.on_test_end(stage)
    T.on_stage_end(trainer, stage)
    T.on_test_end(trainer, stage)
    T.teardown(trainer, stage)

Predict (``Trainer.predict(...)``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

    T.setup(trainer, stage)
    M.on_predict_start(stage)
    T.on_stage_start(trainer, stage)
    T.on_predict_start(trainer, stage)
    M.on_predict_epoch_start(stage, loop)
    T.on_predict_epoch_start(trainer, stage, loop)
    for batch in dataloader:
        M.on_predict_batch_start(stage, loop, batch)
        T.on_predict_batch_start(trainer, stage, loop, batch)
        M.predict_step(batch, batch_idx)
        M.on_predict_batch_end(stage, loop, outputs)
        T.on_predict_batch_end(trainer, stage, loop, outputs)
    M.on_predict_epoch_end(stage, loop)
    T.on_predict_epoch_end(trainer, stage, loop)
    M.on_predict_end(stage)
    T.on_stage_end(trainer, stage)
    T.on_predict_end(trainer, stage)
    T.teardown(trainer, stage)

Reference
---------

.. autosummary::

    reax.TrainerListener
    reax.Module
