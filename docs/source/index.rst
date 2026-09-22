.. REAX documentation master file

.. _reax: https://github.com/camml-lab/reax

Welcome to REAX
===============

.. image:: https://codecov.io/gh/muhrin/reax/branch/develop/graph/badge.svg
    :target: https://codecov.io/gh/camml-lab/reax
    :alt: Coveralls

.. image:: https://img.shields.io/pypi/v/reax.svg
    :target: https://pypi.python.org/pypi/reax/
    :alt: Latest Version

.. image:: https://img.shields.io/pypi/wheel/reax.svg
    :target: https://pypi.python.org/pypi/reax/

.. image:: https://img.shields.io/pypi/pyversions/reax.svg
    :target: https://pypi.python.org/pypi/reax/

.. image:: https://img.shields.io/pypi/l/reax.svg
    :target: https://pypi.python.org/pypi/reax/


**REAX** is a lightweight, flexible training framework for JAX that removes boilerplate whilst
preserving maximum flexibility and enabling performance at scale.

Why REAX?
---------

🔧 **Framework Agnostic**
    REAX works with **any** JAX neural network library: Flax Linen, Flax NNX, Equinox, Haiku, and
    more. Bring your own models.

⚡ **Scale Effortlessly**
    Built-in support for multi-GPU and multi-node training with minimal code changes.

🎯 **Stay in Control**
    Choose your level of abstraction: use the high-level Trainer or drop down to the Engine for
    custom training loops.

📊 **Experiment Tracking**
    Integrated logging, checkpointing, and experiment management out of the box.

Quick Example
-------------

.. code-block:: python

    import reax
    from flax import nnx
    import optax

    # Works with any JAX library - Flax NNX example
    class MyModel(reax.Module):
        def __init__(self, rngs):
            super().__init__()
            self.linear = nnx.Linear(784, 10, rngs=rngs)

        def __call__(self, x):
            return self.linear(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
            self.log("train_loss", loss)
            return loss

        def configure_optimizers(self):
            return optax.adam(1e-3)

    # Train on multiple GPUs with one line
    trainer = reax.Trainer(accelerator="gpu", devices=4)
    trainer.fit(model, train_loader, val_loader)

Installation 🛠️
---------------

.. code-block:: shell

    pip install reax

Or with conda:

.. code-block:: shell

    conda install reax -c conda-forge

Get Started
-----------

New to REAX? Start here:

* :doc:`get_started/introduction` - Learn REAX in 15 minutes
* :doc:`get_started/installation` - Detailed installation guide
* :doc:`get_started/quick_start` - Interactive tutorial

Documentation
-------------

.. toctree::
   :maxdepth: 1
   :caption: Get Started

   get_started/index

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   user_guide/index

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/index

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/index

Community & Support
-------------------

* **GitHub**: `camml-lab/reax <https://github.com/camml-lab/reax>`_
* **Issues**: Report bugs and request features on GitHub

Versioning
----------

This software follows `Semantic Versioning`_

.. _Semantic Versioning: http://semver.org/
