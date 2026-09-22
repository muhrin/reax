REAX in 15 Minutes
==================

**REAX** is a lightweight training framework for JAX that works with any neural network library.
This guide will show you the essential concepts in 15 minutes.

What Makes REAX Different?
---------------------------

🔧 **Library Agnostic**
    Unlike other frameworks, REAX doesn't force you to use a specific neural network library. Use
    Flax Linen, Flax NNX, Equinox, Haiku, or any JAX-based library you prefer.

⚡ **Minimal Boilerplate**
    REAX handles the training loop, distributed training, logging, and checkpointing so you can
    focus on your model.

🎯 **Flexible Abstraction**
    Use the high-level :class:`~reax.Trainer` for standard workflows, or drop down to the
    :class:`~reax.Engine` for custom training loops.

The 7 Key Steps
---------------

1. Install REAX
~~~~~~~~~~~~~~~

.. code-block:: bash

    pip install reax

2. Define Your Model
~~~~~~~~~~~~~~~~~~~~

REAX works with **any** JAX neural network library. Here's an example using Flax NNX:

.. code-block:: python

    import reax
    from flax import nnx
    import optax

    class ImageClassifier(reax.Module):
        def __init__(self, num_classes: int, rngs: nnx.Rngs):
            super().__init__()
            self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
            self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
            self.linear = nnx.Linear(64 * 5 * 5, num_classes, rngs=rngs)

        def __call__(self, x):
            x = nnx.relu(self.conv1(x))
            x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
            x = nnx.relu(self.conv2(x))
            x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
            x = x.reshape((x.shape[0], -1))
            return self.linear(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
            self.log("train_loss", loss)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
            acc = (logits.argmax(axis=1) == y).mean()
            self.log("val_loss", loss)
            self.log("val_acc", acc)

        def configure_optimizers(self):
            return optax.adam(learning_rate=1e-3)

The :class:`~reax.Module` organises your code into clear sections:

- **Model definition** (``__init__``, ``__call__``)
- **Training logic** (``training_step``)
- **Validation logic** (``validation_step``)
- **Optimiser configuration** (``configure_optimizers``)

3. Prepare Your Data
~~~~~~~~~~~~~~~~~~~~

REAX works with any iterable. The most common case -- plain in-memory arrays of features and
labels -- is handled by :class:`~reax.data.ArrayDataset` and :class:`~reax.data.ReaxDataLoader`:

.. code-block:: python

    from reax.data import ArrayDataset, ReaxDataLoader

    # e.g. 55000 MNIST images of shape (28, 28) and integer labels of shape (55000,)
    train_dataset = ArrayDataset(train_x, train_y)
    val_dataset = ArrayDataset(val_x, val_y)

    train_loader = ReaxDataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = ReaxDataLoader(val_dataset, batch_size=64)

4. Train Your Model
~~~~~~~~~~~~~~~~~~~

The :class:`~reax.Trainer` handles the training loop automatically:

.. code-block:: python

    # Initialise the model
    model = ImageClassifier(num_classes=10, rngs=nnx.Rngs(42))

    # Create a trainer
    trainer = reax.Trainer()

    # Train! (max_epochs is a fit() argument)
    trainer.fit(model, train_loader, val_loader, max_epochs=10)

That's it! REAX handles:

- Training and validation loops
- Progress bars
- Logging metrics
- Checkpointing

5. Scale to Multiple GPUs
~~~~~~~~~~~~~~~~~~~~~~~~~~

Want to train on 4 GPUs? Just change one line:

.. code-block:: python

    # Single GPU
    trainer = reax.Trainer(accelerator="gpu", devices=1)

    # 4 GPUs with Data Distributed Parallel
    trainer = reax.Trainer(accelerator="gpu", devices=4, strategy="ddp")

REAX automatically handles:

- Model replication
- Data sharding
- Gradient synchronisation
- Device placement

6. Add Logging and Checkpointing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Track your experiments with built-in logger support:

.. code-block:: python

    from reax.loggers import TensorBoardLogger

    logger = TensorBoardLogger("logs/", name="my_experiment")
    trainer = reax.Trainer(
        logger=logger,
        enable_checkpointing=True,  # Automatically saves a checkpoint each epoch
    )

7. Use Your Trained Model
~~~~~~~~~~~~~~~~~~~~~~~~~~

After training, use your model for inference:

.. code-block:: python

    # Load the best checkpoint
    best_model_path = trainer.checkpoint_listeners[0].best_model_path
    checkpoint = trainer.checkpointing.load(best_model_path)
    model.set_parameters(checkpoint["parameters"])

    # Make predictions
    predictions = trainer.predict(model, test_loader)

Works with Any JAX Library
---------------------------

The examples above use Flax NNX, but REAX works equally well with:

**Flax Linen**

.. code-block:: python

    from flax import linen as nn
    import jax
    import optax

    class LinenModel(reax.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Dense(10)

        def configure_model(self, stage: reax.Stage, batch, /):
            """Initialise parameters with example batch."""
            if self.parameters() is None:
                x, _ = batch
                # Flax Linen: init() returns parameters
                params = self.model.init(self.rngs(), x)
                self.set_parameters(params)

        def __call__(self, x):
            # Flax Linen: apply() uses parameters
            return self.model.apply(self.parameters(), x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            # Pass apply function to static method
            loss, grads = jax.value_and_grad(self.loss_fn, argnums=0)(
                self.parameters(), x, y, self.model.apply
            )
            self.log("train_loss", loss)
            return loss, grads

        @staticmethod
        @jax.jit
        def loss_fn(params, x, y, apply_fn):
            logits = apply_fn(params, x)
            return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

        def configure_optimizers(self):
            opt = optax.adam(learning_rate=1e-3)
            state = opt.init(self.parameters())
            return opt, state

**Equinox**

.. code-block:: python

    import equinox as eqx
    import jax
    import optax

    class EquinoxModel(reax.Module):
        def __init__(self):
            super().__init__()

        def configure_model(self, stage: reax.Stage, batch, /):
            """Initialise model with example batch."""
            if self.parameters() is None:
                x, _ = batch
                # Equinox: model IS the parameters (callable PyTree)
                model = eqx.nn.MLP(
                    in_size=x.shape[-1],
                    out_size=10,
                    width_size=128,
                    depth=2,
                    key=self.rngs()
                )
                self.set_parameters(model)

        def __call__(self, x):
            # Equinox: call the model directly
            return jax.vmap(self.parameters())(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            # Pass model directly to static method
            loss, grads = jax.value_and_grad(self.loss_fn, argnums=0)(
                self.parameters(), x, y
            )
            self.log("train_loss", loss)
            return loss, grads

        @staticmethod
        @jax.jit
        def loss_fn(model, x, y):
            # Equinox models are callable
            logits = jax.vmap(model)(x)
            return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

        def configure_optimizers(self):
            opt = optax.adam(learning_rate=1e-3)
            state = opt.init(self.parameters())
            return opt, state

**Haiku**

.. code-block:: python

    import haiku as hk
    import jax
    import optax

    class HaikuModel(reax.Module):
        def __init__(self):
            super().__init__()
            # Define and transform the forward function
            def forward_fn(x):
                mlp = hk.nets.MLP(output_sizes=[128, 10])
                return mlp(x)
            # Transform returns init and apply methods
            self.forward_transformed = hk.without_apply_rng(hk.transform(forward_fn))

        def configure_model(self, stage: reax.Stage, batch, /):
            """Initialise parameters with example batch."""
            if self.parameters() is None:
                x, _ = batch
                # Haiku: init() returns parameters
                params = self.forward_transformed.init(rng=self.rngs(), x=x)
                self.set_parameters(params)

        def __call__(self, x):
            # Haiku: apply() uses parameters
            return self.forward_transformed.apply(params=self.parameters(), x=x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            # Pass apply function to static method
            loss, grads = jax.value_and_grad(self.loss_fn, argnums=0)(
                self.parameters(), x, y, self.forward_transformed.apply
            )
            self.log("train_loss", loss)
            return loss, grads

        @staticmethod
        @jax.jit
        def loss_fn(params, x, y, apply_fn):
            logits = apply_fn(params=params, x=x)
            return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

        def configure_optimizers(self):
            opt = optax.adam(learning_rate=1e-3)
            state = opt.init(self.parameters())
            return opt, state

Next Steps
----------

Now that you understand the basics, explore:

**Level Up Your Skills**

- :doc:`../user_guide/module` - Deep dive into REAX Modules
- :doc:`../user_guide/trainer` - Master the Trainer
- :doc:`../user_guide/distributed` - Scale to multiple nodes
- :doc:`../user_guide/checkpointing` - Advanced checkpointing strategies

**See Examples**

- :doc:`../examples/index` - Real-world examples across different domains

**API Reference**

- :doc:`../api/index` - Detailed API documentation
