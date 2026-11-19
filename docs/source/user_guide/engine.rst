The Engine
==========

The :class:`~reax.Engine` is the low-level execution component of REAX. While the
:class:`~reax.Trainer` provides a strict, opinionated structure, the Engine offers flexible
primitives for building custom training loops while still benefiting from REAX's distributed
capabilities.

Trainer vs. Engine
------------------

*   **Trainer**: High-level, "batteries included". Manages loops, logging, checkpointing, and
    listeners automatically. Best for standard training workflows.
*   **Engine**: Low-level, explicit. You write the loops. It handles device placement, distributed
    communication, and optimizer wrapping. Best for research and non-standard loops.

Initialising the Engine
-----------------------

.. code-block:: python

    engine = reax.Engine(accelerator="gpu", devices=4, strategy="ddp")

Randomness
~~~~~~~~~~

The Engine is the single source of randomness.  Pass ``seed`` for a reproducible run, or leave it
as ``None`` to get a random seed, shared by all processes and available as
:attr:`~reax.Engine.seed`:

.. code-block:: python

    engine = reax.Engine(seed=42)
    params = model.init(engine.rngs.params(), example_input)   # same on every process
    train, val = reax.data.random_split(engine.rngs, dataset, [0.9, 0.1])  # `data` stream

:attr:`~reax.Engine.rngs` is a :class:`flax.nnx.Rngs` with independent ``params``, ``data``,
``dropout`` and ``default`` streams.  Only ``dropout`` differs between processes.  Shuffling
loaders passed to :meth:`~reax.Engine.setup_dataloaders` are seeded from the ``data`` stream.  In
a custom loop, draw keys outside your jitted update function and pass them in:

.. code-block:: python

    for batch in dataloader:
        loss = update_step(model, batch, engine.rngs.dropout())

See :doc:`randomness` for the full picture.

Using the Engine
----------------

The Engine provides helper methods to setup your environment.

Setup
~~~~~

:meth:`~reax.Engine.setup` prepares your model and optimizers for distributed training.

.. code-block:: python

    model, optimizer = engine.setup(model, optimizer)

Setup DataLoaders
~~~~~~~~~~~~~~~~~

:meth:`~reax.Engine.setup_dataloaders` prepares your dataloaders, handling sharding and device
placement.

.. code-block:: python

    train_loader = engine.setup_dataloaders(train_loader)

Distributed Primitives
~~~~~~~~~~~~~~~~~~~~~~

The Engine exposes methods for cross-device communication:

*   :meth:`~reax.Engine.all_reduce`: Average or sum a tensor across all devices.
*   :meth:`~reax.Engine.broadcast`: Send a tensor from one device to all others.
*   :meth:`~reax.Engine.barrier`: Synchronize all processes.

Example Custom Loop
-------------------

.. code-block:: python

    engine = reax.Engine()
    model, optimizer = engine.setup(model, optimizer)
    dataloader = engine.setup_dataloaders(dataloader)

    for batch in dataloader:
        # Your custom update logic
        loss = update_step(model, batch)

        # Logging
        if engine.is_global_zero:
             log(loss)
