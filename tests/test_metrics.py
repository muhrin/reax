import jax
from jax import random
import jax.numpy as jnp
import numpy as np
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

    rmse = metrics.RootMeanSquareError.empty()
    for prediction, target in zip(predictions, targets):
        rmse = rmse.update(prediction, target)

    assert jnp.isclose(rmse.compute(), jnp.sqrt(optax.squared_error(predictions, targets).mean()))
    # Check the convenience function gives us the right type
    assert isinstance(metrics.get("rmse"), metrics.RootMeanSquareError)

    # Test that masking works
    masks = np.random.randint(0, 2, size=shape[:2], dtype=bool)
    rmse = metrics.RootMeanSquareError.create(predictions[0], targets[0], mask=masks[0])
    for prediction, target, mask in zip(predictions[1:], targets[1:], masks[1:]):
        rmse = rmse.update(prediction, target, mask=mask)

    expected = jnp.sqrt(optax.squared_error(predictions[masks], targets[masks]).mean())
    assert jnp.isclose(rmse.compute(), expected)
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
    MeanSq = metrics.Average.from_fun(lambda values: values**2)
    mean = jnp.mean(values**2)

    metric = MeanSq.empty()
    for batch in values:
        metric = metric.update(batch)

    assert jnp.isclose(metric.compute(), mean)

    # Now try by creating the first one using the `create` classmethod
    metric = MeanSq.create(values[0])
    for batch in values[1:]:
        metric = metric.update(batch)

    assert jnp.isclose(metric.compute(), mean)


def test_stats_evaluator(rng_key, test_trainer):
    batch_size = 10
    values = random.uniform(rng_key, (40,))
    stats = {
        "avg": metrics.Average(),
        "min": metrics.Min(),
        "max": metrics.Max(),
        "std": metrics.Std(),
    }

    results = test_trainer.eval_stats(
        stats, reax.data.ArrayLoader(values, batch_size=batch_size)
    ).logged_metrics

    assert isinstance(results, dict)
    assert jnp.isclose(results["avg"], values.mean())
    assert jnp.isclose(results["min"], values.min())
    assert jnp.isclose(results["max"], values.max())
    assert jnp.isclose(results["std"], values.flatten().std())

    # Check that `evaluate_stats` produces the same result
    evaluated = test_trainer.eval_stats(stats, values).logged_metrics

    comparison = jax.tree.map(lambda a, b: jnp.isclose(a, b), results, evaluated)
    assert jnp.all(jnp.stack(jax.tree.flatten(comparison)[0]))


def test_vmap_evaluator_parity(rng_key):
    # 1. Setup batched data: (Batch, Features)
    # Total 40 elements, but explicitly batched as 4 samples of 10
    batch_size = 4
    obs_per_sample = 10
    values = random.normal(rng_key, (batch_size, obs_per_sample))

    # Define our test metrics
    stats = {
        "avg": metrics.Average(),
        "min": metrics.Min(),
        "max": metrics.Max(),
        "std": metrics.Std(),
    }

    vmap_eval = metrics.VmapEvaluator()
    default_eval = metrics.DefaultEvaluator()

    for name, metric in stats.items():
        # 2. Compute via VmapEvaluator (Lifts create, then reduces)
        # This simulates: reduce(vmap(metric.create)(values))
        vmapped_metric = vmap_eval.create(metric, values)
        vmapped_result = vmapped_metric.compute()

        # 3. Compute via DefaultEvaluator (Treats the whole (4, 10) as one block)
        # This is our 'ground truth' for the aggregation logic
        standard_metric = default_eval.create(metric, values)
        standard_result = standard_metric.compute()

        # 4. Assert mathematical parity
        # We use a slight tolerance for Std due to sum-of-squares precision
        assert jnp.isclose(
            vmapped_result, standard_result, atol=1e-6
        ), f"Parity failed for {name}: vmap={vmapped_result}, std={standard_result}"

    # 5. Regression check against raw JNP
    # Ensure our Metric logic itself matches the JAX primitives
    assert jnp.isclose(vmap_eval.create(stats["avg"], values).compute(), values.mean())
    assert jnp.isclose(vmap_eval.create(stats["min"], values).compute(), values.min())


def test_num_unique(rng_key, test_trainer):
    batch_size = 4
    values = random.randint(rng_key, (10,), minval=0, maxval=3)
    res = test_trainer.eval_stats(
        metrics.NumUnique(), reax.data.ArrayLoader(values, batch_size=batch_size)
    ).logged_metrics["NumUnique"]
    assert res == len(jnp.unique(values))

    # Test the masking functionality
    mask = values == 2
    res = test_trainer.eval_stats(
        metrics.NumUnique(), reax.data.ArrayLoader((values, mask), batch_size=batch_size)
    ).logged_metrics["NumUnique"]
    assert res == len(jnp.unique(values[mask]))


def test_unique(rng_key, test_trainer):
    unique = metrics.Unique.create(jnp.array([1, 1, 1]))
    assert unique.compute().tolist() == [1]

    unique = unique.update(jnp.array([1]))
    assert unique.compute().tolist() == [1]

    unique = unique.update(jnp.array([1, 2]))
    assert unique.compute().tolist() == [1, 2]

    values = random.randint(rng_key, (40,), minval=0, maxval=10)
    res = test_trainer.eval_stats(
        metrics.Unique(), reax.data.ArrayLoader(values, batch_size=9)
    ).logged_metrics["Unique"]
    assert jnp.all(jnp.array(res) == jnp.unique(values))


@pytest.mark.parametrize(
    "dtype", [jnp.int32, jnp.int64, jnp.uint32, jnp.uint64, jnp.float32, jnp.float64]
)
def test_unique_vmap_reduce(rng_key, dtype):
    """Test that Unique handles vmap and reduction correctly."""
    # 1. Create a vectorized batch of data: (4, 5) -> 4 batches, 5 items each
    # Some overlaps exist within batches and between batches
    data = jnp.array(
        [[1, 2, 2, 3, 4], [3, 4, 4, 5, 6], [6, 6, 7, 8, 8], [1, 5, 9, 9, 0]], dtype=dtype
    )
    mask = data != 4

    # 2. vmap the creation/update
    # We create a Unique instance for each row
    def create_and_update(row, row_mask):
        return metrics.Unique.create(row, mask=row_mask)

    # Batch create states
    vectorized_unique = jax.vmap(create_and_update)(data, mask)

    # 3. Perform the reduction
    # This collapses the (4, max_size) accumulator into (1, max_size)
    reduced_unique = vectorized_unique.reduce(axis=0)

    # 4. Verify result
    # The union of {1,2,3,4}, {3,4,5,6}, {6,7,8}, {1,5,9,0} is {0,1,2,3,4,5,6,7,8,9} (except 4 which is masked)
    expected = jnp.array([0, 1, 2, 3, 5, 6, 7, 8, 9], dtype=dtype)
    result = reduced_unique.compute()

    assert jnp.all(jnp.sort(result) == expected)


def test_unique_jit_vmap(rng_key):
    """Ensure it works inside a JIT compiled function with vmap."""

    @jax.jit
    def run_vmap_accumulate(data):
        # We perform the creation and the reduction entirely within JIT.
        # We return the raw accumulator, keeping shapes static.
        states = jax.vmap(metrics.Unique.create)(data)
        return states.reduce(axis=0).accumulator

    data = jnp.array([[1, 1], [2, 2]])

    # Run the JIT-ed part
    raw_accumulator = run_vmap_accumulate(data)

    # Now, run the 'compute' logic on the host (outside JIT)
    # Since raw_accumulator is a concrete JAX array on the host,
    # we can easily filter it.
    valid_mask = raw_accumulator != metrics.Unique.create(data).fill_value
    actual = raw_accumulator[valid_mask]

    assert jnp.array_equal(jnp.sort(actual), jnp.array([1, 2]))


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


def test_metrics_registry():
    registry = metrics.get_registry()

    expected_metrics = {
        "mean": metrics.Average,
        "min": metrics.Min,
        "max": metrics.Max,
        "num_unique": metrics.NumUnique,
        "unique": metrics.Unique,
        "std": metrics.Std,
        "sum": metrics.Sum,
        "mse": metrics.MeanSquaredError,
        "rmse": metrics.RootMeanSquareError,
        "mae": metrics.MeanAbsoluteError,
    }
    assert not set(expected_metrics).difference(set(registry))


def test_least_squares_vmap_reduction(rng_key):
    # Setup: 4 samples in a batch, each sample has 10 observations of 3 features
    batch_size = 4
    n_obs = 10
    n_features = 3

    k1, k2, k3 = random.split(rng_key, 3)
    inputs = random.normal(k1, (batch_size, n_obs, n_features))
    # Create a simple linear relationship: y = Xw + noise
    true_w = random.normal(k2, (n_features, 1))
    outputs = inputs @ true_w + random.normal(k3, (batch_size, n_obs, 1)) * 0.1

    # 1. Initialize Metric
    ls_metric = metrics.LeastSquaresEstimate()
    vmap_eval = metrics.VmapEvaluator()
    default_eval = metrics.DefaultEvaluator()

    # 2. Compute via Vmap (This tests your custom reduce_fn)
    # vmap(create) produces values of shape (4, 10, 3)
    # reduce(axis=0) should turn it into (40, 3)
    vmapped_metric = vmap_eval.create(ls_metric, inputs, outputs)

    # 3. Compute via Default (The 'Ground Truth')
    # Reshape manually to compare
    flat_inputs = inputs.reshape(-1, n_features)
    flat_outputs = outputs.reshape(-1, 1)
    standard_metric = default_eval.create(ls_metric, flat_inputs, flat_outputs)

    # 4. Assertions
    # Check shapes first - this is where the reduce_fn usually fails
    assert vmapped_metric.values.shape == (batch_size * n_obs, n_features)
    assert vmapped_metric.targets.shape == (batch_size * n_obs, 1)

    # Check the actual result (the weights w)
    vmap_w = vmapped_metric.compute()
    std_w = standard_metric.compute()

    assert jnp.allclose(vmap_w, std_w)
