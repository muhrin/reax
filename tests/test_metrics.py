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


# ---------------------------------------------------------------------------
# jm.utils internals
# ---------------------------------------------------------------------------


def test_jm_select_topk():
    from reax.metrics import jm

    res = jm.select_topk(jnp.array([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1]]), topk=2)
    assert res.tolist() == [[1, 1, 0], [1, 1, 0]]
    assert res.dtype in (jnp.uint32, jnp.int32)

    with pytest.raises(NotImplementedError):
        jm.select_topk(jnp.zeros((2, 3, 4)), topk=2)


def test_jm_stat_scores_update():
    from reax.metrics import jm
    import reax.metrics.jm.utils as jmu

    # multiclass (N, C) float preds, (N,) int target
    preds = jnp.array([[0.5, 0.5, 0.0], [0.1, 0.6, 0.3]])
    target = jnp.array([0, 1])

    tp, fp, tn, fn = jmu.stat_scores_update(
        preds, target, intended_mode=jm.DataType.MULTICLASS, average_method=jm.AverageMethod.MICRO
    )
    # row1 -> class0, row2 -> class1: 2 correct, 2*3 - 2 = 4 negatives, 0 wrong
    assert (int(tp), int(fp), int(tn), int(fn)) == (2, 0, 4, 0)
    assert int(tp) + int(fp) + int(tn) + int(fn) == 6

    with pytest.raises(ValueError):
        jmu.stat_scores_update(
            preds, target, intended_mode=jm.DataType.BINARY, average_method=jm.AverageMethod.MICRO
        )


def test_jm_accuracy_compute():
    from reax.metrics import jm

    AM, MD, DT = jm.AverageMethod, jm.MDMCAverageMethod, jm.DataType

    # micro: tp / (tp + fn)
    assert jnp.isclose(
        jm.accuracy_compute(
            jnp.array(3),
            jnp.array(1),
            jnp.array(2),
            jnp.array(1),
            AM.MICRO,
            MD.GLOBAL,
            DT.MULTICLASS,
        ),
        0.75,
    )
    # binary/multilabel: (tp + tn) / (tp + tn + fp + fn) = (2+3)/7
    assert jnp.isclose(
        jm.accuracy_compute(
            jnp.array(2), jnp.array(1), jnp.array(3), jnp.array(1), AM.MICRO, MD.GLOBAL, DT.BINARY
        ),
        5 / 7,
    )
    # weighted
    assert jnp.isclose(
        jm.accuracy_compute(
            jnp.array([3, 0]),
            jnp.zeros(2),
            jnp.zeros(2),
            jnp.array([1, 2]),
            AM.WEIGHTED,
            MD.GLOBAL,
            DT.MULTICLASS,
        ),
        0.5,
    )
    # macro
    assert jnp.isclose(
        jm.accuracy_compute(
            jnp.array([3, 0]),
            jnp.array([1, 2]),
            jnp.zeros(2),
            jnp.array([1, 0]),
            AM.MACRO,
            MD.GLOBAL,
            DT.MULTICLASS,
        ),
        0.375,
    )


def test_jm_basic_input_validation_raises():
    import reax.metrics.jm.utils as jmu

    # target must be integer
    with pytest.raises(ValueError, match="target.*integer"):
        jmu._basic_input_validation(jnp.array([1, 2]), jnp.array([0.5, 0.5]), 0.5, None)
    # first dim must match
    with pytest.raises(ValueError, match="first dimension"):
        jmu._basic_input_validation(jnp.array([[1, 2], [3, 4]]), jnp.array([0, 1, 2]), 0.5, None)
    # multiclass=False but target > 1
    with pytest.raises(ValueError, match="target.*1"):
        jmu._basic_input_validation(jnp.array([1, 2]), jnp.array([0, 2]), 0.5, False)
    # multiclass=False and integer preds > 1
    with pytest.raises(ValueError, match="preds.*1"):
        jmu._basic_input_validation(jnp.array([1, 2]), jnp.array([0, 1]), 0.5, False)


def test_jm_shape_type_consistency_raises():
    from reax.metrics import jm
    import reax.metrics.jm.utils as jmu

    # 1D target with 2D float preds in binary mode -> not allowed
    with pytest.raises(ValueError, match="should not be `binary`"):
        jmu._check_shape_and_type_consistency(
            jnp.array([[0.5, 0.5]]), jnp.array([0]), jm.DataType.BINARY
        )
    # integer preds with extra dim -> not allowed
    with pytest.raises(ValueError, match="float"):
        jmu._check_shape_and_type_consistency(
            jnp.array([[1, 2]]), jnp.array([0]), jm.DataType.MULTICLASS
        )
    # mismatched ndim (> 1 difference)
    with pytest.raises(ValueError):
        jmu._check_shape_and_type_consistency(
            jnp.zeros((2, 3, 4)), jnp.zeros(2), jm.DataType.MULTICLASS
        )


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------


def test_accuracy_multiclass():
    preds = jnp.array([[0.5, 0.5, 0.0], [0.1, 0.6, 0.3], [0.2, 0.2, 0.6], [0.9, 0.1, 0.0]])
    target = jnp.array([0, 1, 2, 1])

    acc = metrics.Accuracy()
    # tp/fp/tn/fn are dataclass fields
    for field in ("tp", "fp", "tn", "fn"):
        assert hasattr(acc, field)

    acc1 = acc.update(preds, target)
    acc2 = acc.update(preds, target)
    merged = acc1.merge(acc2)
    # rows -> classes [0,1,2,0], so target [0,1,2,1] gives 3/4 correct
    assert jnp.isclose(merged.compute(), 0.75)


def test_accuracy_binary_and_multilabel():
    # regression: jnp.where threshold path (previously crashed with `.int()`)
    preds = jnp.array([0.1, 0.6, 0.8, 0.3])
    target = jnp.array([0, 1, 1, 0])
    acc = metrics.Accuracy(mode="binary").update(preds, target)
    assert jnp.isclose(acc.compute(), 1.0)  # 0.1->0, 0.6->1, 0.8->1, 0.3->0 == target

    # multilabel
    preds = jnp.array([[0.9, 0.1], [0.2, 0.7]])
    target = jnp.array([[1, 0], [0, 1]])
    acc = metrics.Accuracy(mode="multilabel").update(preds, target)
    assert jnp.isclose(acc.compute(), 1.0)

    # samplewise mdmc_average no longer dead-code
    with pytest.raises(ValueError, match="not yet supported"):
        metrics.Accuracy(mdmc_average="samplewise")


def test_accuracy_invalid_args():
    with pytest.raises(ValueError, match="number of classes"):
        metrics.Accuracy(average="macro")  # macro -> needs num_classes
    with pytest.raises(ValueError, match="top_k"):
        metrics.Accuracy(top_k=0)
    with pytest.raises(ValueError, match="is not valid"):
        metrics.Accuracy(num_classes=3, ignore_index=5)


# ---------------------------------------------------------------------------
# utils.prepare_mask / concat
# ---------------------------------------------------------------------------


def test_prepare_mask():
    from reax.metrics import utils as mu

    vals = jnp.array([[1.0, 2.0], [3.0, 4.0]])

    assert mu.prepare_mask(vals, None) is None
    mask, count = mu.prepare_mask(vals, None, return_count=True)
    assert mask is None and count == vals.size

    mask, count = mu.prepare_mask(vals, jnp.array([True, False]), return_count=True)
    assert mask.shape == (2, 1) and count == 2

    mask, count = mu.prepare_mask(vals, jnp.array([[True, False], [True, True]]), return_count=True)
    assert count == 3

    with pytest.raises(ValueError):
        mu.prepare_mask(vals, jnp.array([True]))
    with pytest.raises(ValueError):
        mu.prepare_mask(vals, jnp.zeros((2, 3), dtype=bool))


def test_utils_concat():
    from reax.metrics import utils as mu

    assert mu.concat((jnp.array(1), jnp.array([2.0, 3.0]))).tolist() == [1, 2, 3]


# ---------------------------------------------------------------------------
# Aggregation primitives (Sum / Min / Max)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls,expected",
    [(metrics.Sum, 21.0), (metrics.Min, 1.0), (metrics.Max, 6.0)],
)
def test_aggregation_sum_min_max(cls, expected):
    m = cls.create(jnp.array([1.0, 2.0, 3.0]))
    assert m.empty().accumulator is None
    m = m.update(jnp.array([4.0, 5.0]))
    other = cls.create(jnp.array([6.0]))
    assert jnp.isclose(m.merge(other).compute(), expected)


def test_num_unique_update_and_saturation():
    nu = metrics.NumUnique.create(jnp.array([1, 2, 2, 3]))
    assert nu.compute() == 3
    assert nu.update(jnp.array([3, 4])).compute() == 4

    with pytest.raises(RuntimeError):
        _ = metrics.Unique.empty().accumulator

    sat = metrics.Unique.create(jnp.array([1, 2, 3])).saturation()
    assert sat == 3.0 / metrics.Unique.create(jnp.array([1, 2, 3])).max_size


# ---------------------------------------------------------------------------
# Std
# ---------------------------------------------------------------------------


def test_std():
    vals = jnp.array([1.0, 2.0, 3.0, 4.0])
    s = metrics.Std.empty()
    assert s.count == 0
    s = metrics.Std.create(vals)
    assert jnp.isclose(s.compute(), vals.std())
    # Masked version only averages the first element
    masked = metrics.Std.create(vals, mask=jnp.array([True, True, False, False]))
    assert jnp.isclose(masked.compute(), 0.0)  # single value -> no variance


# ---------------------------------------------------------------------------
# get / registry / collections
# ---------------------------------------------------------------------------


def test_globals_get_dispatch():
    assert isinstance(metrics.get("mse"), metrics.MeanSquaredError)
    assert isinstance(metrics.get(metrics.Sum), metrics.Sum)
    inst = metrics.Sum()
    assert metrics.get(inst) is inst
    with pytest.raises(TypeError):
        metrics.get(123)


def test_registry_getitem_and_register():
    reg = metrics.get_registry()
    assert "mse" in reg
    with pytest.raises(KeyError, match="Metric not found"):
        reg["does_not_exist_zzz"]

    fresh = metrics.Registry()
    with pytest.raises(ValueError, match="reax.Metric"):
        fresh.register("bad", 42)
    fresh.register("avg_cls", metrics.Average)
    assert "avg_cls" in fresh
    assert isinstance(fresh["avg_cls"], metrics.Average)


def test_set_registry_roundtrip():
    original = metrics.get_registry()
    try:
        fresh = metrics.Registry()
        fresh.register("mean", metrics.Average)
        metrics.set_registry(fresh)
        assert isinstance(metrics.get("mean"), metrics.Average)
    finally:
        metrics.set_registry(original)


def test_build_collection():
    col = metrics.build_collection("mean")
    assert isinstance(col, metrics.MetricCollection)
    assert "Average" in {name for name, _ in col.items()}

    col = metrics.build_collection({"a": "mean", "b": "std"})
    names = {name for name, _ in col.items()}
    assert "a" in names or "Average" in names

    assert isinstance(metrics.build_collection(["mean"]), metrics.MetricCollection)

    with pytest.raises(TypeError, match="Unknown metrics type"):
        metrics._registry._get_metrics(123)


def test_metric_collection_single_and_sequence():
    single = metrics.MetricCollection(metrics.Average())
    assert "Average" in {name for name, _ in single.items()}

    seq = metrics.MetricCollection([metrics.Average(), metrics.Std()])
    names = {name for name, _ in seq.items()}
    assert names == {"Average", "Std"}

    import reax.metrics.collections as coll

    with pytest.raises(TypeError, match="reax.Matric"):
        coll._ensure_metric(42)


def test_metric_collection_merge_and_combine():
    c1 = metrics.MetricCollection(dict(avg=metrics.Average())).empty()
    c2 = metrics.MetricCollection(dict(std=metrics.Std())).empty()
    merged = c1.merge(c2)
    names = {name for name, _ in merged.items()}
    assert names == {"avg", "std"}

    combined = metrics.combine(metrics.Average(), metrics.Std())
    assert isinstance(combined, metrics.MetricCollection)


def test_regression_empty_and_merge(rng_key):
    keys = random.split(rng_key, 3)
    v1 = random.uniform(keys[0], (2, 3))
    v2 = random.uniform(keys[1], (2, 3))
    t = random.uniform(keys[2], (2, 3))

    # Default constructor is empty
    rmse = metrics.RootMeanSquareError()
    assert rmse.is_empty

    # Each update uses targets `t` for its own batch
    expected = jnp.sqrt(jnp.mean(jnp.concat([jnp.square(v1 - t), jnp.square(v2 - t)])))
    rmse = metrics.RootMeanSquareError.create(v1, t).update(v2, t)
    assert jnp.isclose(rmse.compute(), expected)

    # Merge two independently created metrics -> same as a single update
    m1 = metrics.RootMeanSquareError.create(v1, t)
    m2 = metrics.RootMeanSquareError.create(v2, t)
    merged = m1.merge(m2)
    assert jnp.isclose(merged.compute(), expected)

    mae = metrics.MeanAbsoluteError()
    mae_c = mae.create(v1, t)
    assert jnp.isclose(mae_c.compute(), jnp.mean(jnp.abs(v1 - t)))
    assert jnp.isclose(
        metrics.MeanSquaredError.create(v1, t).compute(), jnp.mean(jnp.square(v1 - t))
    )
