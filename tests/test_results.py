import jax.numpy as jnp
import numpy as np
import pytest

from reax import results

# Deliberately ragged: the final batch is smaller, as it is whenever the dataset size isn't a
# multiple of the batch size.  Nothing about the reduction should depend on that.
BATCH_SIZES = (32, 32, 32, 11)
VALUES = (0.10, 0.20, 0.30, 0.25)


def _log_epoch(values, batch_sizes, **kwargs):
    """Log one value per batch, as a module would during an epoch, and compute the result"""
    collection = results.ResultCollection()
    for batch_idx, (value, batch_size) in enumerate(zip(values, batch_sizes)):
        collection.log(
            "train",
            "loss",
            value,
            batch_idx=batch_idx,
            on_epoch=True,
            batch_size=batch_size,
            **kwargs,
        )

    return collection["train.loss"].metric.compute()


def test_mean_is_the_default():
    """Values are averaged over the batches they were logged in"""
    expected = np.mean(VALUES)

    assert jnp.isclose(_log_epoch(VALUES, BATCH_SIZES, reduce_fn="mean"), expected)
    assert jnp.isclose(_log_epoch(VALUES, BATCH_SIZES), expected)


def test_sum():
    """Values that are parts of a whole are totalled instead"""
    assert jnp.isclose(_log_epoch(VALUES, BATCH_SIZES, reduce_fn="sum"), sum(VALUES))


def test_batch_size_does_not_affect_the_result():
    """A raw value says nothing about how many samples are behind it, so the batch size can't be
    used to weight it"""
    even = _log_epoch(VALUES, (16, 16, 16, 16), reduce_fn="mean")
    ragged = _log_epoch(VALUES, BATCH_SIZES, reduce_fn="mean")

    assert jnp.isclose(even, ragged)


def test_unknown_reduce_fn():
    with pytest.raises(ValueError, match="reduce_fn"):
        _log_epoch(VALUES, BATCH_SIZES, reduce_fn="average")


def test_cannot_merge_different_reductions():
    """The same key logged two ways has no single answer"""
    summed = results.ArrayResultMetric.create(1.0, reduce_fn="sum")
    averaged = results.ArrayResultMetric.create(1.0, reduce_fn="mean")

    with pytest.raises(ValueError, match="reduced differently"):
        summed.merge(averaged)
