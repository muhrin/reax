REAX Module
===========

The :class:`~reax.Module` is the central building block of your model in REAX. It organises your
JAX code into 5 specific sections:

1.  **Computations** (``__init__``, ``__call__``, etc.)
2.  **Train Loop** (``training_step``)
3.  **Validation Loop** (``validation_step``)
4.  **Test Loop** (``test_step``)
5.  **Optimizers** (``configure_optimizers``)

REAX Modules are fully compatible with Flax NNX.

Basic Example
-------------

Here is a minimal example of a REAX Module:

.. code-block:: python

    import reax
    from flax import nnx
    import optax

    class MyModel(reax.Module):
        def __init__(self, din, dout, rngs: nnx.Rngs):
            super().__init__()
            self.linear = nnx.Linear(din, dout, rngs=rngs)

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

The Life Cycle
--------------

The methods in the module are called by the :class:`~reax.Trainer` in a specific order.

Training Step
~~~~~~~~~~~~~

The :meth:`~reax.Module.training_step` method is the heart of the training loop. It receives a
batch of data and an index. It should return the loss (scalar) or a dictionary containing the loss
under the key ``'loss'``.

Validation Step
~~~~~~~~~~~~~~~

The :meth:`~reax.Module.validation_step` is called during validation. It is used to evaluate the
model's performance on unseen data. You can log metrics here to track progress.

.. code-block:: python

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        self.log("val_loss", loss)

Test Step
~~~~~~~~~

The :meth:`~reax.Module.test_step` is similar to validation but is typically used after training is
complete to evaluate the final model on a test set.

Optimization
~~~~~~~~~~~~

The :meth:`~reax.Module.configure_optimizers` method defines the optimizers and learning rate
schedulers. REAX uses Optax for optimization.

.. code-block:: python

    def configure_optimizers(self):
        optimizer = optax.adamw(learning_rate=1e-4, weight_decay=1e-5)
        return optimizer

Organizing Code
---------------

By using :class:`~reax.Module`, your code becomes more organised and readable. It separates the
model definition from the training logic, making it easier to share and reproduce experiments.
