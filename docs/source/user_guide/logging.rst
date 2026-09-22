Logging
=======

REAX supports logging metrics to various backends.

Supported Loggers
-----------------

*   :class:`~reax.loggers.CsvLogger`: Logs metrics to a CSV file.
*   :class:`~reax.loggers.TensorBoardLogger`: Logs metrics to TensorBoard.
*   :class:`~reax.loggers.MlflowLogger`: Logs metrics to MLflow.

Configuration
-------------

To use a logger, pass it to the Trainer:

.. code-block:: python

    from reax.loggers import TensorBoardLogger

    logger = TensorBoardLogger("tb_logs", name="my_model")
    trainer = reax.Trainer(logger=logger)

Logging Metrics
---------------

Inside your :class:`~reax.Module`, you can log metrics using ``self.log()``:

.. code-block:: python

    def training_step(self, batch, batch_idx):
        loss = ...
        self.log("train_loss", loss)
        return loss

Console Logging
---------------

By default, the :class:`~reax.listeners.ProgressBar` will display logged metrics in the console
progress bar. You can control this with the ``prog_bar`` argument:

.. code-block:: python

    self.log("acc", acc, prog_bar=True)
