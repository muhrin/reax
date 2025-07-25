from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
import pytest

from reax.data import samplers


def test_sequential_sampler():
    indices = np.arange(10).tolist()
    sampler = samplers.SequentialSampler(len(indices))
    assert list(sampler) == indices

    with pytest.raises(TypeError):
        samplers.SequentialSampler(iter(indices))


def test_random_sampler():
    indices = np.arange(10).tolist()
    sampler = samplers.RandomSampler(len(indices))
    assert sorted(list(sampler)) == indices

    sampler = samplers.RandomSampler(len(indices), replacements=True)
    assert all(sample in indices for sample in list(sampler))


class MockDataset:
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length


@pytest.fixture
def sampler_factory():
    """Factory fixture to create samplers with consistent setup"""

    def _create_sampler(dataset_length, **kwargs):
        dataset = MockDataset(dataset_length)
        return samplers.DistributedSampler(dataset, **kwargs)

    return _create_sampler


@pytest.mark.parametrize(
    "dataset_length, drop_last, expected_total_size",
    [
        (10, False, 5),  # Divisible by 2
        (11, False, 6),  # Not divisible - padding
        (10, True, 5),  # Divisible - no drop needed
        (11, True, 5),  # Not divisible - drop excess
        (1, True, 0),  # Smaller than replicas - drop all
        (5, True, 2),  # (5-2)=3 -> ceil(3/2)=2 -> 4
    ],
)
def test_total_size_calculation(sampler_factory, dataset_length, drop_last, expected_total_size):
    """Test total size calculation with various configurations"""
    sampler = sampler_factory(dataset_length, drop_last=drop_last, num_replicas=2)
    assert len(sampler) == expected_total_size


def test_init_parameters(sampler_factory):
    """Test initialization with various parameters"""
    # Default parameters
    sampler = sampler_factory(10, num_replicas=2)
    assert sampler._num_replicas == 2
    assert sampler._process_index == 0
    assert sampler._shuffle
    assert not sampler._drop_last

    # Explicit parameters
    sampler = sampler_factory(10, num_replicas=4, process_index=2, shuffle=True, drop_last=True)
    assert sampler._num_replicas == 4
    assert sampler._process_index == 2
    assert sampler._shuffle
    assert sampler._drop_last


def test_invalid_parameters():
    """Test invalid initialization parameters"""
    with pytest.raises(ValueError):
        samplers.DistributedSampler(MockDataset(10), num_replicas=0)

    with pytest.raises(ValueError):
        samplers.DistributedSampler(
            MockDataset(10), process_index=2
        )  # process_index > num_replicas


def test_sample_indices_no_shuffle(sampler_factory):
    """Test sampling without shuffle"""
    sampler = sampler_factory(10, num_replicas=2, shuffle=False)
    indices = list(sampler)
    # Expected for 10 samples, 2 replicas, rank=0: [0, 2, 4, 6, 8]
    assert indices == [0, 2, 4, 6, 8]
    assert len(indices) == 5  # ceil(10/2) = 5


def test_sample_indices_with_drop_last(sampler_factory):
    """Test sampling with drop_last=True"""
    sampler = sampler_factory(11, num_replicas=2, drop_last=True, shuffle=False)
    indices = list(sampler)
    # For 11 samples, drop_last=True: should get first 10 samples
    assert indices == [0, 2, 4, 6, 8]
    assert len(indices) == 5


def test_sample_indices_with_shuffle(sampler_factory):
    """Test sampling with shuffle (using mock)"""
    with patch("jax.random.permutation") as mock_perm:
        # Mock permutation to return predictable sequence
        mock_perm.return_value = jnp.array([0, 2, 4, 6, 8, 1, 3, 5, 7, 9])

        sampler = sampler_factory(10, shuffle=True, num_replicas=2)
        indices = list(sampler)

        # Rank 0 should get indices 0,2,4,6,8 of permutation
        # [0, 4, 8, 1, 5] -> wait, let's compute:
        # permutation: [0,2,4,6,8,1,3,5,7,9]
        # rank0: indices 0,2,4,6,8 -> values: 0,4,8,3,7
        assert indices == [0, 4, 8, 3, 7]


def test_rank_1_sampling(sampler_factory):
    """Test sampling for rank=1"""
    with patch("jax.process_index", return_value=1):
        sampler = sampler_factory(10, num_replicas=2, shuffle=False)
        indices = list(sampler)
        # Expected for rank 1: [1,3,5,7,9]
        assert indices == [1, 3, 5, 7, 9]


def test_empty_dataset(sampler_factory):
    """Test with empty dataset"""
    sampler = sampler_factory(0)
    assert len(list(sampler)) == 0


def test_shuffle_consistency(sampler_factory):
    """Test shuffle consistency with seed"""
    sampler1 = sampler_factory(10, shuffle=True, seed=42)
    sampler2 = sampler_factory(10, shuffle=True, seed=42)
    sampler3 = sampler_factory(10, shuffle=True, seed=43)

    assert list(sampler1) == list(sampler2)
    assert list(sampler1) != list(sampler3)


def test_iterable(sampler_factory):
    """Test that sampler is iterable"""
    sampler = sampler_factory(10, num_replicas=2)
    assert isinstance(iter(sampler), type(iter([])))
    assert len(list(sampler)) == 5


def test_dataset_length_zero(sampler_factory):
    """Test with dataset length zero"""
    sampler = sampler_factory(0)
    assert len(list(sampler)) == 0
