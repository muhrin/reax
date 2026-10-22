REAX
====

.. image:: https://codecov.io/gh/muhrin/reax/branch/develop/graph/badge.svg
    :target: https://codecov.io/gh/muhrin/reax
    :alt: Coverage

.. image:: https://github.com/camml-lab/reax/actions/workflows/ci.yml/badge.svg
    :target: https://github.com/camml-lab/reax/actions/workflows/ci.yml
    :alt: Tests

.. image:: https://readthedocs.org/projects/reax/badge/
    :target: https://reax.readthedocs.io/
    :alt: Documentation

.. image:: https://img.shields.io/pypi/v/reax.svg
    :target: https://pypi.python.org/pypi/reax/
    :alt: Latest Version

.. image:: https://img.shields.io/pypi/pyversions/reax.svg
    :target: https://pypi.python.org/pypi/reax/

.. image:: https://img.shields.io/pypi/l/reax.svg
    :target: https://pypi.python.org/pypi/reax/

.. image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
    :target: https://github.com/astral-sh/ruff
    :alt: Ruff


**Scalable, flexible training for JAX, inspired by the simplicity of PyTorch Lightning.**

REAX is a lightweight framework for training JAX models. It takes care of the training loop,
logging, checkpointing and scaling across devices, so that your code can focus on the model and
the science. 🚀

Why REAX? 🌟
------------

- **Bring your own model** 🔧: REAX works with any JAX library, e.g. Flax Linen, Flax NNX or
  Equinox. Your model stays plain JAX code.
- **Minimal boilerplate** 🧩: write a ``reax.Module`` with your training step and optimiser, and
  let the ``reax.Trainer`` run fitting, validation, testing and prediction.
- **Batteries included** 💾: checkpointing, early stopping, progress bars, metrics, and loggers
  for CSV, TensorBoard and MLflow.
- **Scales with JAX** ⚡: run on multiple devices and processes with minimal code changes.
- **Stay in control** 🎯: hook into every stage of training with module hooks and listeners, or
  drop down to the ``reax.Engine`` to write your own training loop.

Installation 🛠️
---------------

With pip:

.. code-block:: shell

    pip install reax

or with `uv <https://docs.astral.sh/uv/>`_, to add REAX to your project:

.. code-block:: shell

    uv add reax

Example
-------

Train an autoencoder on MNIST, using a Flax Linen model:

.. code-block:: python

    from functools import partial

    from flax import linen
    import jax
    import optax

    import reax
    from reax.demos import mnist


    class Autoencoder(linen.Module):
        def setup(self):
            self.encoder = linen.Sequential([linen.Dense(128), linen.relu, linen.Dense(3)])
            self.decoder = linen.Sequential([linen.Dense(128), linen.relu, linen.Dense(28 * 28)])

        def __call__(self, x):
            return self.decoder(self.encoder(x))


    # --------------------------------
    # Step 1: Define a REAX Module
    # --------------------------------
    # A reax.Module defines a full *system* (e.g. an LLM, diffusion model, autoencoder or
    # image classifier): the model, how to train it and how to optimise it.
    class ReaxAutoEncoder(reax.Module):
        def __init__(self):
            super().__init__()
            self.ae = Autoencoder()

        def configure_model(self, stage: reax.Stage, batch, /):
            if self.parameters() is None:
                # Initialise the parameters using an example batch
                x, _ = batch
                params = self.ae.init(self.rngs(), x.reshape(x.shape[0], -1))
                self.set_parameters(params)

        def __call__(self, x):
            return self.ae.apply(self.parameters(), x)

        def training_step(self, batch, batch_idx):
            x, _ = batch
            x = x.reshape(x.shape[0], -1)
            loss, grads = jax.value_and_grad(self.loss_fn)(self.parameters(), x, self.ae.apply)
            self.log("train_loss", loss, on_step=True, prog_bar=True)
            # Return the gradients and REAX takes care of the optimiser update
            return loss, grads

        def validation_step(self, batch, batch_idx):
            x, _ = batch
            x = x.reshape(x.shape[0], -1)
            self.log("val_loss", self.loss_fn(self.parameters(), x, self.ae.apply), prog_bar=True)

        @staticmethod
        @partial(jax.jit, static_argnums=2)
        def loss_fn(params, x, apply_fn):
            predictions = apply_fn(params, x)
            return optax.losses.squared_error(predictions, x).mean()

        def configure_optimizers(self):
            opt = optax.adam(learning_rate=1e-3)
            return opt, opt.init(self.parameters())


    # -------------------
    # Step 2: Define data
    # -------------------
    trainer = reax.Trainer()
    dataset = mnist.MnistDataset(download=True)
    train, val = reax.data.random_split(trainer.rngs, dataset, [55_000, 5_000])

    # -------------------
    # Step 3: Train
    # -------------------
    autoencoder = ReaxAutoEncoder()
    trainer.fit(
        autoencoder,
        reax.ReaxDataLoader(train, batch_size=64, shuffle=True),
        reax.ReaxDataLoader(val, batch_size=64),
        max_epochs=5,
    )

More examples, including Flax NNX and Equinox models, are in the
`documentation <https://reax.readthedocs.io/>`_.

Coming from PyTorch Lightning?
------------------------------

REAX ports the abstractions and best practices of PyTorch Lightning to JAX, and parts of its
code are adapted directly from Lightning. If you know Lightning, the ``Trainer``, ``Module``,
hooks, listeners (Lightning's callbacks) and loggers will feel familiar. Where it matters, REAX
departs from Lightning to fit JAX: parameters and optimiser state are explicit, and training steps
return gradients rather than relying on ``loss.backward()``.
