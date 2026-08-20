import jax.numpy as jnp
import numpy as np
import pytest

from reax import results

# Deliberately ragged: the final batch is smaller, as it is whenever the dataset size isn't a
# multiple of the batch size
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
    """Values are averaged over the samples they were computed from"""
    expected = np.average(VALUES, weights=BATCH_SIZES)

    assert jnp.isclose(_log_epoch(VALUES, BATCH_SIZES, reduce_fn="mean"), expected)
    assert jnp.isclose(_log_epoch(VALUES, BATCH_SIZES), expected)


def test_mean_weights_by_batch_size():
    """A short final batch counts for the samples it holds, not as a whole batch"""
    weighted = _log_epoch(VALUES, BATCH_SIZES, reduce_fn="mean")

    assert jnp.isclose(weighted, np.average(VALUES, weights=BATCH_SIZES))
    # The ragged batch makes this differ from the plain mean over batches
    assert not jnp.isclose(weighted, np.mean(VALUES))


def test_mean_without_batch_sizes():
    """Where the batch size isn't known, every value counts once"""
    sizes = (None,) * len(VALUES)

    assert jnp.isclose(_log_epoch(VALUES, sizes, reduce_fn="mean"), np.mean(VALUES))


def test_equal_batches_are_a_plain_mean():
    """Weighting only shows up when the batches differ in size"""
    equal = _log_epoch(VALUES, (16,) * len(VALUES), reduce_fn="mean")

    assert jnp.isclose(equal, np.mean(VALUES))


def test_sum_ignores_batch_size():
    """Values that are parts of a whole are totalled as they are"""
    assert jnp.isclose(_log_epoch(VALUES, BATCH_SIZES, reduce_fn="sum"), sum(VALUES))
    assert jnp.isclose(_log_epoch(VALUES, (1,) * len(VALUES), reduce_fn="sum"), sum(VALUES))


def test_unknown_reduce_fn():
    with pytest.raises(ValueError, match="reduce_fn"):
        _log_epoch(VALUES, BATCH_SIZES, reduce_fn="average")


def test_cannot_merge_different_reductions():
    """The same key logged two ways has no single answer"""
    summed = results.ArrayResultMetric.create(1.0, reduce_fn="sum")
    averaged = results.ArrayResultMetric.create(1.0, reduce_fn="mean")

    with pytest.raises(ValueError, match="reduced differently"):
        summed.merge(averaged)
