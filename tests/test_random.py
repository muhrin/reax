import flax.serialization
import jax
import numpy as np
import pytest

import reax
from reax import demos
from reax.data import samplers


def _key_data(key):
    return np.asarray(jax.random.key_data(key))


def test_same_seed_same_streams():
    rngs1 = reax.Engine(seed=5, logger=False).rngs
    rngs2 = reax.Engine(seed=5, logger=False).rngs
    for stream in reax.random.STREAMS:
        assert np.array_equal(
            _key_data(getattr(rngs1, stream)()), _key_data(getattr(rngs2, stream)())
        )


def test_different_seed_different_streams():
    rngs1 = reax.Engine(seed=5, logger=False).rngs
    rngs2 = reax.Engine(seed=6, logger=False).rngs
    assert not np.array_equal(_key_data(rngs1.params()), _key_data(rngs2.params()))


def test_seed_and_rngs_are_exclusive():
    from flax import nnx

    with pytest.raises(ValueError):
        reax.Engine(seed=1, rngs=nnx.Rngs(0), logger=False)


def test_explicit_rngs():
    from flax import nnx

    rngs = nnx.Rngs(0)
    engine = reax.Engine(rngs=rngs, logger=False)
    assert engine.rngs is rngs
    assert engine.seed is None


def test_random_seed_is_recorded_and_reproducible():
    engine1 = reax.Engine(logger=False)
    engine2 = reax.Engine(logger=False)
    assert isinstance(engine1.seed, int)
    assert engine1.seed != engine2.seed  # Vanishingly unlikely to collide

    replay = reax.Engine(seed=engine1.seed, logger=False)
    assert np.array_equal(_key_data(engine1.rngs.params()), _key_data(replay.rngs.params()))


def test_streams_are_independent():
    rngs1 = reax.random.create_rngs(0)
    rngs2 = reax.random.create_rngs(0)
    # Drawing extra keys from one stream must not affect the others
    rngs1.dropout()
    rngs1.params()
    assert np.array_equal(_key_data(rngs1.data()), _key_data(rngs2.data()))


def test_per_process_streams():
    rank0 = reax.random.create_rngs(0, process_index=0)
    rank1 = reax.random.create_rngs(0, process_index=1)
    for stream in reax.random.SHARED_STREAMS:
        assert np.array_equal(
            _key_data(getattr(rank0, stream)()), _key_data(getattr(rank1, stream)())
        )
    assert not np.array_equal(_key_data(rank0.dropout()), _key_data(rank1.dropout()))


def test_rngs_state_dict_roundtrip():
    rngs = reax.random.create_rngs(0)
    rngs.params()
    rngs.data()
    state = flax.serialization.msgpack_restore(
        flax.serialization.msgpack_serialize(reax.random.rngs_state_dict(rngs))
    )
    expected = [_key_data(rngs.params()), _key_data(rngs.data())]

    restored = reax.random.create_rngs(1)
    reax.random.load_rngs_state_dict(restored, state)
    assert np.array_equal(_key_data(restored.params()), expected[0])
    assert np.array_equal(_key_data(restored.data()), expected[1])


def test_seeded_random_sampler():
    sampler1 = samplers.RandomSampler(20, seed=3)
    sampler2 = samplers.RandomSampler(20, seed=3)
    epoch0, epoch1 = list(sampler1), list(sampler1)
    assert epoch0 == list(sampler2)
    assert epoch0 != epoch1  # Each epoch reshuffles
    assert sorted(epoch0) == list(range(20))

    # Resuming at a given epoch reproduces that epoch's order
    sampler3 = samplers.RandomSampler(20, seed=3)
    sampler3.set_epoch(1)
    assert list(sampler3) == epoch1


def test_seeded_random_sampler_with_replacements():
    sampler1 = samplers.RandomSampler(10, replacements=True, num_samples=50, seed=3)
    sampler2 = samplers.RandomSampler(10, replacements=True, num_samples=50, seed=3)
    assert list(sampler1) == list(sampler2)


def _batch_order(seed: int) -> list[np.ndarray]:
    engine = reax.Engine(seed=seed, logger=False)
    loader = reax.data.ReaxDataLoader(np.arange(32), batch_size=4, shuffle=True)
    loader = engine.setup_dataloaders(loader)
    return [np.asarray(batch) for batch in loader]


def test_engine_seeds_loader_shuffling():
    order1 = _batch_order(11)
    order2 = _batch_order(11)
    order3 = _batch_order(12)
    assert all(np.array_equal(a, b) for a, b in zip(order1, order2, strict=True))
    assert not all(np.array_equal(a, b) for a, b in zip(order1, order3, strict=True))


def test_engine_respects_explicit_sampler_seed():
    sampler = samplers.RandomSampler(16, seed=99)
    loader = reax.data.ReaxDataLoader(np.arange(16), batch_size=4, sampler=sampler)
    reax.Engine(seed=0, logger=False).setup_dataloaders(loader)
    assert sampler.seed == 99


def _fit_params(seed: int, tmp_path):
    trainer = reax.Trainer(
        seed=seed,
        default_root_dir=tmp_path,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    model = demos.BoringModel()
    data = np.random.default_rng(0).normal(size=(64, 32)).astype(np.float32)
    loader = reax.data.ReaxDataLoader(data, batch_size=8, shuffle=True)
    trainer.fit(model, loader, max_epochs=2)
    return jax.tree.leaves(model.parameters())


def test_trainer_seed_reproducible_fit(tmp_path):
    params1 = _fit_params(21, tmp_path / "a")
    params2 = _fit_params(21, tmp_path / "b")
    params3 = _fit_params(22, tmp_path / "c")
    assert all(np.array_equal(a, b) for a, b in zip(params1, params2, strict=True))
    assert not all(np.array_equal(a, b) for a, b in zip(params1, params3, strict=True))


def test_numpy_rng_is_deterministic():
    key = jax.random.key(7)
    assert (
        reax.random.numpy_rng(key).permutation(100) == reax.random.numpy_rng(key).permutation(100)
    ).all()
    # The integer seed and the corresponding key give the same generator
    assert (
        reax.random.numpy_rng(7).permutation(100) == reax.random.numpy_rng(key).permutation(100)
    ).all()
    # Different keys, or extra entropy, give different generators
    assert (
        reax.random.numpy_rng(key).permutation(100)
        != reax.random.numpy_rng(jax.random.key(8)).permutation(100)
    ).any()
    assert (
        reax.random.numpy_rng(key).permutation(100)
        != reax.random.numpy_rng(key, 1).permutation(100)
    ).any()


def test_numpy_rng_draws_from_data_stream():
    rngs1 = reax.random.create_rngs(3)
    rngs2 = reax.random.create_rngs(3)
    expected = reax.random.numpy_rng(rngs2.data()).permutation(50)
    assert (reax.random.numpy_rng(rngs1).permutation(50) == expected).all()
    # Drawing advanced only the `data` stream
    assert int(rngs1.data.count.get_value()) == 1
    assert int(rngs1.params.count.get_value()) == 0


def test_numpy_rng_is_the_same_across_processes():
    # Shared streams don't depend on the process index, so neither do splits drawn from them
    perms = [
        reax.random.numpy_rng(reax.random.create_rngs(5, process_index=idx)).permutation(50)
        for idx in range(3)
    ]
    for perm in perms[1:]:
        assert (perm == perms[0]).all()


def test_numpy_rng_uses_whole_key():
    key = jax.random.key(0)
    # Keys that differ only in their first word must give different generators
    data = jax.random.key_data(key)
    other = jax.random.wrap_key_data(data.at[0].add(1))
    assert (
        reax.random.numpy_rng(key).permutation(100) != reax.random.numpy_rng(other).permutation(100)
    ).any()
