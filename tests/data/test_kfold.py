import jax
import numpy as np
import pytest

from reax.data import KFold


def test_kfold_basic_split():
    kf = KFold(n_splits=3, shuffle=False)
    splits = list(kf.split(6))
    assert len(splits) == 3

    # Fold 0: val 0,1, train 2,3,4,5
    assert np.array_equal(splits[0][0], [2, 3, 4, 5])
    assert np.array_equal(splits[0][1], [0, 1])

    # Fold 1: val 2,3, train 0,1,4,5
    assert np.array_equal(splits[1][0], [0, 1, 4, 5])
    assert np.array_equal(splits[1][1], [2, 3])


def test_kfold_uneven_distribution():
    kf = KFold(n_splits=3, shuffle=False)
    splits = list(kf.split(7))
    assert len(splits) == 3

    # Sizes should be 3, 2, 2
    assert len(splits[0][1]) == 3
    assert len(splits[1][1]) == 2
    assert len(splits[2][1]) == 2


def test_kfold_determinism():
    kf1 = KFold(n_splits=3, shuffle=True, seed=42)
    kf2 = KFold(n_splits=3, shuffle=True, seed=42)
    kf3 = KFold(n_splits=3, shuffle=True, seed=43)

    splits1 = list(kf1.split(10))
    splits2 = list(kf2.split(10))
    splits3 = list(kf3.split(10))

    assert np.array_equal(splits1[0][0], splits2[0][0])
    assert not np.array_equal(splits1[0][0], splits3[0][0])


def test_kfold_get_fold():
    dataset = ["a", "b", "c", "d", "e", "f"]
    kf = KFold(n_splits=3, shuffle=False)
    train_sub, val_sub = kf.get_fold(dataset, 1)

    assert list(val_sub) == ["c", "d"]
    assert list(train_sub) == ["a", "b", "e", "f"]


def test_kfold_invalid_n_splits():
    with pytest.raises(ValueError):
        KFold(n_splits=1)


def test_kfold_invalid_fold_index():
    dataset = [1, 2, 3, 4]
    kf = KFold(n_splits=2)
    with pytest.raises(ValueError):
        kf.get_fold(dataset, 2)


def test_kfold_n_splits_gt_n_samples():
    kf = KFold(n_splits=5)
    with pytest.raises(ValueError):
        list(kf.split(3))


class MockEngine:
    def __init__(self, is_zero: bool):
        self._is_zero = is_zero
        self.broadcast_called = False

    @property
    def is_global_zero(self):
        return self._is_zero

    def broadcast(self, obj, src):
        self.broadcast_called = True
        return 99

    class MockRngs:
        def default(self):
            return jax.random.key(123)

    rngs = MockRngs()


def test_kfold_with_engine_seed_broadcast():
    engine = MockEngine(is_zero=True)
    kf = KFold(n_splits=3, shuffle=True, seed=None, engine=engine)

    assert engine.broadcast_called
    assert kf.seed == 99


def test_kfold_with_engine_explicit_seed():
    engine = MockEngine(is_zero=True)
    kf = KFold(n_splits=3, shuffle=True, seed=42, engine=engine)

    assert not engine.broadcast_called
    assert kf.seed == 42
