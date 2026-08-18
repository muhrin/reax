import jax.numpy as jnp
import numpy as np
import pytest

from reax import results

# Deliberately ragged: the final batch is smaller, as it is whenever the dataset size isn't a
# multiple of the batch size
BATCH_SIZES = (32, 32, 32, 11)
BATCH_MEANS = (0.10, 0.20, 0.30, 0.25)
BATCH_TOTALS = tuple(mean * size for mean, size in zip(BATCH_MEANS, BATCH_SIZES))


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


def test_sum_is_the_default():
    """A raw value is taken to be the total over its batch unless we're told otherwise"""
    expected = sum(BATCH_TOTALS) / sum(BATCH_SIZES)

    assert jnp.isclose(_log_epoch(BATCH_TOTALS, BATCH_SIZES, reduce_fx="sum"), expected)
    # Not passing `reduce_fx` at all must keep doing exactly the same thing
    assert jnp.isclose(_log_epoch(BATCH_TOTALS, BATCH_SIZES), expected)


def test_mean_is_weighted_by_batch_size():
    """Values that are already averages are weighted by the number of samples behind them"""
    expected = np.average(BATCH_MEANS, weights=BATCH_SIZES)

    assert jnp.isclose(_log_epoch(BATCH_MEANS, BATCH_SIZES, reduce_fx="mean"), expected)


def test_mean_and_sum_agree():
    """Logging the mean of a batch and logging its total are two ways of saying the same thing"""
    assert jnp.isclose(
        _log_epoch(BATCH_MEANS, BATCH_SIZES, reduce_fx="mean"),
        _log_epoch(BATCH_TOTALS, BATCH_SIZES, reduce_fx="sum"),
    )


def test_mean_logged_as_a_total_is_scaled_down():
    """The bug this guards against: a mean logged as if it were a total comes out roughly the batch
    size times too small"""
    correct = _log_epoch(BATCH_MEANS, BATCH_SIZES, reduce_fx="mean")
    as_if_total = _log_epoch(BATCH_MEANS, BATCH_SIZES, reduce_fx="sum")

    assert as_if_total < correct
    assert jnp.isclose(as_if_total, sum(BATCH_MEANS) / sum(BATCH_SIZES))


def test_unknown_reduce_fx():
    with pytest.raises(ValueError, match="reduce_fx"):
        _log_epoch(BATCH_MEANS, BATCH_SIZES, reduce_fx="average")
