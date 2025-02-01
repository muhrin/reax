import jax
from jax import random
import jax.numpy as jnp
import optax
import pytest

from reax import metrics
import reax.data


def test_aggregation(rng_key):
    values = random.uniform(rng_key, (100,))
    avg = metrics.Average()
    assert jnp.allclose(avg.update(values).compute(), values.mean())


def test_mean_square_error(rng_key):
    rng_key, *keys = random.split(rng_key, 3)
    n_batches = 4
    values = random.uniform(keys[0], (n_batches, 10))
    targets = random.uniform(keys[1], (n_batches, 10))

    mse = metrics.MeanSquaredError()
    for prediction, target in zip(values, targets):
        mse = mse.update(prediction, target)

    assert jnp.isclose(mse.compute(), optax.squared_error(values, targets).mean())
    # Check the convenience function gives us the right type
    assert isinstance(metrics.get("mse"), metrics.MeanSquaredError)


@pytest.mark.parametrize("shape", [(4, 10), (4, 3, 3)])
def test_root_mean_square_error(shape, rng_key):
    rng_key, *keys = random.split(rng_key, 3)
    predictions = random.uniform(keys[0], shape)
    targets = random.uniform(keys[1], shape)

    mse = metrics.RootMeanSquareError()
    for prediction, target in zip(predictions, targets):
        mse = mse.update(prediction, target)

    assert jnp.isclose(mse.compute(), jnp.sqrt(optax.squared_error(predictions, targets).mean()))
    # Check the convenience function gives us the right type
    assert isinstance(metrics.get("rmse"), metrics.RootMeanSquareError)


def test_mae(rng_key):
    rng_key, *keys = random.split(rng_key, 3)
    n_batches = 4
    predictions = random.uniform(keys[0], (n_batches, 10))
    targets = random.uniform(keys[1], (n_batches, 10))

    mse = metrics.MeanAbsoluteError()
    for prediction, target in zip(predictions, targets):
        mse = mse.update(prediction, target)

    assert jnp.isclose(mse.compute(), jnp.abs(predictions - targets).mean())
    # Check the convenience function gives us the right type
    assert isinstance(metrics.get("mae"), metrics.MeanAbsoluteError)


def test_from_fn(rng_key):
    n_batches = 4
    values = random.uniform(rng_key, (n_batches, 10))

    # Let's create a fake function, where we calculate the mean of the squares
    metric = metrics.Average.from_fun(lambda values: values**2)()
    mean = jnp.mean(values**2)

    for batch in values:
        metric = metric.update(batch)

    assert jnp.isclose(metric.compute(), mean)


def test_stats_evaluator(rng_key):
    batch_size = 10
    values = random.uniform(rng_key, (40,))
    stats = {
        "avg": metrics.Average(),
        "min": metrics.Min(),
        "max": metrics.Max(),
        "std": metrics.Std(),
    }

    results = reax.evaluate_stats(stats, reax.data.ArrayLoader(values, batch_size=batch_size))

    assert isinstance(results, dict)
    assert jnp.isclose(results["avg"], values.mean())
    assert jnp.isclose(results["min"], values.min())
    assert jnp.isclose(results["max"], values.max())
    assert jnp.isclose(results["std"], values.flatten().std())

    # Check that `evaluate_stats` produces the same result
    evaluated = reax.evaluate_stats(stats, values)

    comparison = jax.tree.map(lambda a, b: jnp.isclose(a, b), results, evaluated)
    assert jnp.all(jnp.stack(jax.tree.flatten(comparison)[0]))


def test_num_unique(rng_key):
    batch_size = 9
    values = random.randint(rng_key, (40,), minval=0, maxval=9)
    res = reax.evaluate_stats(
        metrics.NumUnique(), reax.data.ArrayLoader(values, batch_size=batch_size)
    )
    assert res["NumUnique"] == len(jnp.unique(values))

    # Test the masking functionality
    mask = values == 2
    res = reax.evaluate_stats(
        metrics.NumUnique(), reax.data.ArrayLoader((values, mask), batch_size=batch_size)
    )
    assert res["NumUnique"] == len(jnp.unique(values[mask]))


def test_unique(rng_key):
    unique = metrics.Unique.create(jnp.array([1, 1, 1]))
    assert unique.compute().tolist() == [1]

    unique = unique.update(jnp.array([1]))
    assert unique.compute().tolist() == [1]

    unique = unique.update(jnp.array([1, 2]))
    assert unique.compute().tolist() == [1, 2]

    values = random.randint(rng_key, (40,), minval=0, maxval=10)
    res = reax.evaluate_stats(metrics.Unique(), reax.data.ArrayLoader(values, batch_size=9))
    assert jnp.all(jnp.array(res["Unique"]) == jnp.unique(values))


def test_metric_collection(rng_key):
    batch_size = 9
    collection = reax.metrics.MetricCollection(
        dict(mean=reax.metrics.Average(), std=reax.metrics.Std())
    )

    values = random.uniform(rng_key, (40,))
    loader = reax.data.ArrayLoader(values, batch_size=batch_size)

    accumulator = collection.empty()
    for batch in loader:
        accumulator = accumulator.update(batch)
    res = accumulator.compute()

    assert "mean" in res
    assert jnp.isclose(res["mean"], values.mean())

    assert "std" in res
    assert jnp.isclose(res["std"], values.std())
