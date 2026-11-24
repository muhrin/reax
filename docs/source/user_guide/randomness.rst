Randomness
==========

All randomness in a REAX run comes from **one seed**, held by the :class:`~reax.Engine` (and so the
:class:`~reax.Trainer`).  From it, REAX builds a single :class:`flax.nnx.Rngs`, available as
``trainer.rngs`` (or ``engine.rngs``, or ``self.rngs`` inside a :class:`~reax.Module`), and draws
everything else from it.

Seeding a run
-------------

.. code-block:: python

    trainer = reax.Trainer(seed=42)  # Reproducible: the same seed always gives the same run
    trainer = reax.Trainer()         # Different every time: a random seed is drawn

In both cases the seed is logged, saved in checkpoints and available as ``trainer.seed``, so you
can always replay a run with ``reax.Trainer(seed=<logged seed>)``.  In a distributed run the seed
is drawn on rank 0 and shared with every process.

The engine also seeds the global ``random`` and ``numpy.random`` generators (and ``torch``, if it
has been imported), so older code that relies on them follows the same seed.

Streams
-------

``trainer.rngs`` has several *independent* named streams:

=============  =================================================  ==========================
Stream         Used for                                           Across processes
=============  =================================================  ==========================
``params``     Initialising model parameters                      identical
``data``       Splitting datasets, shuffling, k-fold partitions   identical
``dropout``    Dropout and other stochastic layers                **different** per process
``default``    Anything else                                      identical
=============  =================================================  ==========================

The streams are independent, so drawing from one never shifts another.  For example, adding a
dropout layer won't change how your data is split.  ``params`` and ``data`` must be the same on
every process (so that every process starts from the same parameters and sees a consistent shard
of the same shuffled data), while ``dropout`` differs so that data-parallel processes don't use
identical masks.

What REAX does for you
~~~~~~~~~~~~~~~~~~~~~~

* Any ``shuffle=True`` loader, or unseeded :class:`~reax.data.samplers.RandomSampler`, set up by the
  engine gets a seed from the ``data`` stream.  Each epoch reshuffles, reproducibly.
* :func:`reax.data.random_split` draws from the ``data`` stream of the RNGs you pass it (usually
  ``engine.rngs``).
* :class:`reax.data.KFold` with ``seed=None`` and an ``engine`` draws its seed from the ``data``
  stream.
* Resuming ``fit`` from a checkpoint restores the RNG state.

All of this data-side randomness is *seeded* from the JAX ``data`` stream but *computed* with
NumPy on the host, where the data lives.  This avoids device round trips and recompilation for
every new dataset size, and keeps data handling independent of the accelerator backend.  If you
write your own data utilities, do the same with :func:`reax.random.numpy_rng`:

.. code-block:: python

    rng = reax.random.numpy_rng(engine.rngs)  # A numpy.random.Generator from the `data` stream
    indices = rng.permutation(len(dataset))

Using the RNGs in your code
---------------------------

**1. Create models from the trainer's RNGs**, not from a separate ``nnx.Rngs(...)``.  Otherwise you
have a second seed that the trainer knows nothing about.

.. code-block:: python

    trainer = reax.Trainer(seed=42)
    model = MyModel(din=32, dout=10, rngs=trainer.rngs)  # nnx layers draw from `params`

If parameters are created lazily (e.g. Flax Linen, or Equinox), draw from the ``params`` stream in
:meth:`~reax.Module.configure_model`:

.. code-block:: python

    def configure_model(self, stage, batch):
        if self.parameters() is None:
            self.set_parameters(self.model.init(self.rngs.params(), batch[0]))

**2. Draw keys outside jit and pass them in.**  Calling a stream (``self.rngs.dropout()``) is a
Python-side operation that advances a counter.  Do it in your (un-jitted) step method and pass
the resulting key into your ``jax.jit``-ed function.  You don't need to split or merge anything:

.. code-block:: python

    @functools.partial(jax.jit, static_argnums=0)
    def loss_and_grad(model: linen.Module, params, batch, key):
        def loss_fn(params):
            preds = model.apply(params, batch[0], rngs={"dropout": key})
            return optax.losses.squared_error(preds, batch[1]).mean()

        return jax.value_and_grad(loss_fn)(params)

    class MyModule(reax.Module):
        ...

        def training_step(self, batch, batch_idx):
            key = self.rngs.dropout()  # Outside jit
            loss, grads = loss_and_grad(self.model, self.parameters(), batch, key)
            return {"loss": loss, "grad": grads}

**3. With** ``nnx.jit`` you can instead pass the RNGs (or an nnx model holding stochastic layers
built from them) straight into the jitted function.  nnx tracks and propagates the counter
updates.

.. warning::

    Never call ``self.rngs`` *inside* a ``jax.jit``-traced function.  The key would be baked into
    the compiled function, or a tracer would leak, and the counter wouldn't advance between calls.

**4. Pick the stream by purpose**: ``params`` for initialisation, ``dropout`` for stochastic
layers, ``data`` for anything that must be identical on every process, and ``default`` otherwise.
Need a random number that must agree across processes?  Use a shared stream, not ``dropout``.

Advanced: explicit RNGs
-----------------------

You can pass your own :class:`flax.nnx.Rngs` to the engine or trainer with ``rngs=...`` instead
of a ``seed``.  In that case REAX uses them as-is.  Any stream your ``Rngs`` doesn't define falls
back to ``default``, so the streams are no longer independent, and nothing is folded per process.
:func:`reax.random.create_rngs` builds RNGs with the standard streams if you need them outside an
engine.
