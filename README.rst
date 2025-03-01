
REAX
====

.. image:: https://codecov.io/gh/muhrin/reax/branch/develop/graph/badge.svg
    :target: https://codecov.io/gh/muhrin/reax
    :alt: Coverage

.. image:: https://github.com/muhrin/reax/actions/workflows/ci.yml/badge.svg
    :target: https://github.com/muhrin/reax/actions/workflows/ci.yml
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


REAX: A simple training framework for JAX-based projects

REAX is based on PyTorch Lightning and tries to bring a similar level of easy-of-use and
customizability to the world of training JAX models. Much of lightning's API has been adopted
with some modifications being made to accommodate JAX's pure function based approach.


Quick start
-----------

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

Here, we reproduce an example from PyTorch Lightning, so we use torch vision to fetch the data, but for real models
there's no need to use this or pytorch at all.
Run the model on the terminal


.. code-block:: bash

    pip install reax torchvision
    python main.py
