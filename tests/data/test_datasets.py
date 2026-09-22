from flax import nnx
import jax.numpy as jnp
import numpy as np
import pytest

from reax.data import datasets


def test_concatdataset_init_empty_raises():
    with pytest.raises(ValueError, match="empty iterable"):
        datasets.ConcatDataset([])


def test_concatdataset_init_non_sequence_raises():
    # a non-Sequence is rejected (by beartype's typechecker, which raises a
    # TypeCheckError that is a subclass of TypeError)
    with pytest.raises(TypeError):
        datasets.ConcatDataset([iter([1, 2, 3])])


def test_concatdataset_len_and_getitem():
    ds1 = [1, 2]
    ds2 = [3, 4, 5]
    ds3 = [6]
    concat = datasets.ConcatDataset([ds1, ds2, ds3])
    assert len(concat) == 6
    assert concat.cumulative_sizes == [2, 5, 6]
    assert concat[0] == 1
    assert concat[1] == 2
    assert concat[2] == 3
    assert concat[5] == 6
    # negative indexing
    assert concat[-1] == 6
    assert concat[-6] == 1
    with pytest.raises(ValueError, match="absolute value"):
        concat[-10]


def test_arraydataset_size_mismatch_raises():
    with pytest.raises(ValueError, match="Size mismatch"):
        datasets.ArrayDataset(np.array([1, 2]), np.array([1, 2, 3]))


def test_arraydataset_getitem_and_len():
    a = np.array([[1.0], [2.0], [3.0]])
    b = np.array([10.0, 20.0, 30.0])
    ds = datasets.ArrayDataset(a, b)
    assert len(ds) == 3
    x, y = ds[0]
    assert x == pytest.approx(jnp.array([1.0]))
    assert y == 10.0
    x2, y2 = ds[2]
    assert x2 == pytest.approx(jnp.array([3.0]))
    assert y2 == 30.0


def test_subset_with_list_indexed_dataset():
    base = ["a", "b", "c", "d"]
    sub = datasets.Subset(base, [0, 2])
    assert len(sub) == 2
    assert sub[0] == "a"
    assert sub[1] == "c"
    # list indexing
    assert sub[[0, 1]] == ["a", "c"]


def test_subset_with_dataset_getitems_method():
    class WithGetItems(list):
        def __getitems__(self, idxs):
            return [self[i] for i in idxs]

    base = WithGetItems([10, 20, 30, 40])
    sub = datasets.Subset(base, [1, 3])
    # dataset has __getitems__, so it should be delegated
    assert sub[[0, 1]] == [20, 40]


def test_random_split_with_int_lengths():
    rngs = nnx.Rngs(0)
    ds = list(range(10))
    splits = datasets.random_split(rngs, ds, [4, 6])
    assert [len(s) for s in splits] == [4, 6]
    # union should be full dataset
    all_items = [item for s in splits for item in s]
    assert sorted(all_items) == sorted(ds)


def test_random_split_with_int_lengths_mismatch_raises():
    rngs = nnx.Rngs(0)
    ds = list(range(10))
    with pytest.raises(ValueError, match="Sum of input lengths"):
        datasets.random_split(rngs, ds, [3, 4])


def test_random_split_with_fractions():
    rngs = nnx.Rngs(0)
    ds = list(range(10))
    splits = datasets.random_split(rngs, ds, [0.3, 0.7])
    total = sum(len(s) for s in splits)
    assert total == 10
    # individual lengths: floor(3)=3, floor(7)=7, remainder 0
    assert [len(s) for s in splits] == [3, 7]


def test_random_split_with_fractions_remainder_distributed():
    rngs = nnx.Rngs(0)
    ds = list(range(10))
    # 0.33 * 10 = 3.3 -> 3, 0.34 * 10 = 3.4 -> 3, 0.33 * 10 = 3.3 -> 3
    # sum = 9, remainder 1 -> first gets +1 -> [4, 3, 3]
    splits = datasets.random_split(rngs, ds, [0.33, 0.34, 0.33])
    assert sum(len(s) for s in splits) == 10
    # 2 of the 3 splits should be 3 (the one without the +1 is the second or third)
    assert sorted(len(s) for s in splits) == [3, 3, 4]


def test_random_split_with_fractions_out_of_range_raises():
    rngs = nnx.Rngs(0)
    ds = list(range(10))
    with pytest.raises(ValueError, match="between 0 and 1"):
        datasets.random_split(rngs, ds, [1.5, -0.5])


def test_random_split_zero_length_warns():
    rngs = nnx.Rngs(0)
    ds = list(range(10))
    # 0.05 * 10 = 0.5 -> 0
    with pytest.warns(UserWarning, match="is 0"):
        splits = datasets.random_split(rngs, ds, [0.0, 1.0])
    assert len(splits[0]) == 0


def test_len_dataset_sized():
    assert datasets.len_dataset([1, 2, 3]) == 3
    assert datasets.len_dataset("abc") == 3
    assert datasets.len_dataset(np.array([1, 2, 3])) == 3


def test_len_dataset_tuple():
    # tuple containing one non-None entry: use len of first non-None
    assert datasets.len_dataset((None, [1, 2, 3])) == 3


def test_len_dataset_unsized_returns_inf():
    # a generator has no len
    def gen():
        yield 1
        yield 2

    assert datasets.len_dataset(gen()) == float("inf")
