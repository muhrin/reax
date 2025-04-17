REAX
====

.. image:: https://codecov.io/gh/muhrin/reax/branch/develop/graph/badge.svg
    :target: https://codecov.io/gh/muhrin/reax
    :alt: Coverage

.. image:: https://github.com/camml-lab/reax/actions/workflows/ci.yml/badge.svg
    :target: https://github.com/camml-lab/reax/actions/workflows/ci.yml
    :alt: Tests

.. image:: https://img.shields.io/pypi/v/reax.svg
    :target: https://pypi.python.org/pypi/reax/
    :alt: Latest Version

.. image:: https://img.shields.io/pypi/wheel/reax.svg
    :target: https://pypi.python.org/pypi/reax/

.. image:: https://img.shields.io/pypi/pyversions/reax.svg
    :target: https://pypi.python.org/pypi/reax/

.. image:: https://img.shields.io/pypi/l/reax.svg
    :target: https://pypi.python.org/pypi/reax/


REAX — Scalable, flexible training for JAX, inspired by the simplicity of PyTorch Lightning.

REAX - Scalable Training for JAX
================================

REAX is a minimal and high-performance framework for training JAX models, designed to simplify
research workflows. Inspired by PyTorch Lightning, it brings similar high-level abstractions and
scalability to JAX users, making it easier to scale models across multiple GPUs with minimal
boilerplate. 🚀

A Port of PyTorch Lightning to JAX
----------------------------------

Much of REAX is built by porting the best practices and abstractions of **PyTorch Lightning** to
the **JAX** ecosystem. If you're familiar with PyTorch Lightning, you'll recognize concepts like:

- Training loops ⚡
- Multi-GPU training 🖥️
- Logging and checkpointing 💾

However, REAX has been designed with JAX-specific optimizations, ensuring high performance without
sacrificing flexibility.

Why REAX? 🌟
------------

- **Scalable**: Built to leverage JAX’s parallelism and scalability. ⚡
- **Minimal Boilerplate**: Simplifies the training process with just enough structure. 🧩
- **Familiar**: For users who have experience with frameworks like PyTorch Lightning, the
  transition to REAX is seamless. 🔄

Installation 🛠️
---------------

To install REAX, run the following command:

.. code-block:: shell

    pip install reax


REAX example
------------

Define the training workflow. Here's a toy example:

.. code-block:: python

    # main.py
    # ! pip install torchvision
    from functools import partial
    import jax, optax, reax, flax.linen as linen
    import torch.utils.data as data, torchvision as tv


    class Autoencoder(linen.Module):
        def setup(self):
            super().__init__()
            self.encoder = linen.Sequential([linen.Dense(128), linen.relu, linen.Dense(3)])
            self.decoder = linen.Sequential([linen.Dense(128), linen.relu, linen.Dense(28 * 28)])

        def __call__(self, x):
            z = self.encoder(x)
            return self.decoder(z)


    # --------------------------------
    # Step 1: Define a REAX Module
    # --------------------------------
    # A ReaxModule (nn.Module subclass) defines a full *system*
    # (ie: an LLM, diffusion model, autoencoder, or simple image classifier).
    class ReaxAutoEncoder(reax.Module):
        def __init__(self):
            super().__init__()
            self.ae = Autoencoder()

        def setup(self, stage: "reax.Stage", batch) -> None:
            if self.parameters() is None:
                x = batch[0].reshape(len(batch[0]), -1)
                params = self.ae.init(self.rng_key(), x)
                self.set_parameters(params)

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

        def forward(self, x):
            embedding = jax.jit(self.ae.encoder.apply)(self.parameters()["params"]["encoder"], x)
            return embedding

        def training_step(self, batch, batch_idx):
            x = batch[0].reshape(len(batch[0]), -1)
            loss, grads = jax.value_and_grad(self.loss_fn, argnums=0)(self.parameters(), x, self.ae)
            self.log("train_loss", loss, on_step=True, prog_bar=True)
            return loss, grads

        @staticmethod
        @partial(jax.jit, static_argnums=2)
        def loss_fn(params, x, model):
            predictions = model.apply(params, x)
            return optax.losses.squared_error(predictions, x).mean()

        def configure_optimizers(self):
            opt = optax.adam(learning_rate=1e-3)
            state = opt.init(self.parameters())
            return opt, state


    # -------------------
    # Step 2: Define data
    # -------------------
    dataset = tv.datasets.MNIST(".", download=True, transform=jax.numpy.asarray)
    train, val = data.random_split(dataset, [55000, 5000])

    # -------------------
    # Step 3: Train
    # -------------------
    autoencoder = ReaxAutoEncoder()
    trainer = reax.Trainer(autoencoder)
    trainer.fit(reax.ReaxDataLoader(train), reax.ReaxDataLoader(val))

Here, we reproduce an example from PyTorch Lightning, so we use torch vision to fetch the data,
but for real models there's no need to use this or pytorch at all.


Disclaimer ⚠️
-------------

REAX takes inspiration from PyTorch Lightning, and large portions of its core functionality are
directly ported from Lightning. If you are already familiar with Lightning, you'll feel right at
home with REAX, but we’ve tailored it to work seamlessly with JAX's performance optimizations.
