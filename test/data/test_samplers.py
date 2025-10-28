from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from reax.data import samplers


@pytest.fixture
def mock_dataset_small():
    return [None] * 10


@pytest.fixture
def mock_dataset_large():
    return [None] * 23


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


class TestDistributedSamplerInit:
    """Tests initialization and property calculations."""

    def test_init_invalid_replicas(self, mock_dataset_small):
        with pytest.raises(ValueError, match="Number of replicas cannot be 0."):
            samplers.DistributedSampler(mock_dataset_small, num_replicas=0)

    @pytest.mark.parametrize("replica_count, index", [(4, 4), (2, 3), (1, 1)])
    def test_init_invalid_process_index(self, mock_dataset_small, replica_count, index):
        with pytest.raises(
            ValueError, match="Process index.*must be less than the number of replicas"
        ):
            samplers.DistributedSampler(
                mock_dataset_small, num_replicas=replica_count, process_index=index
            )

    def test_len_even_split(self, mock_dataset_small):
        # Dataset size 10, 2 replicas -> 5 samples per replica
        sampler = samplers.DistributedSampler(
            mock_dataset_small, num_replicas=2, process_index=0, drop_last=False
        )
        assert len(sampler) == 5
        assert sampler.total_size == 10

    def test_len_uneven_split_no_drop(self, mock_dataset_large):
        # Dataset size 23, 4 replicas
        # math.ceil(23 / 4) = 6
        sampler = samplers.DistributedSampler(
            mock_dataset_large, num_replicas=4, process_index=0, drop_last=False
        )
        assert len(sampler) == 6
        assert sampler.total_size == 24  # 6 * 4

    def test_len_uneven_split_drop_last(self, mock_dataset_large):
        # Dataset size 23, 4 replicas, drop_last=True
        # Total size must be divisible by 4: 20 (23 // 4 * 4)
        # num_samples = 20 / 4 = 5
        sampler = samplers.DistributedSampler(
            mock_dataset_large, num_replicas=4, process_index=0, drop_last=True
        )
        assert len(sampler) == 5
        assert sampler.total_size == 20  # 5 * 4


class TestDistributedSamplerIteration:
    """Tests the __iter__ method for different data distributions."""

    # Scenario: 10 items, 3 replicas, process_index=0, no shuffle, no drop
    # len=10, num_replicas=3. total_size = math.ceil(10/3)*3 = 4*3 = 12
    # Padded indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1] (Indices 0, 1 padded)
    # Subsampled (every 3rd index, starting at 0): [0, 3, 6, 9]
    def test_uneven_padding_proc0(self, mock_dataset_small):
        sampler = samplers.DistributedSampler(
            mock_dataset_small, num_replicas=3, process_index=0, shuffle=False, drop_last=False
        )
        assert len(sampler) == 4
        assert list(sampler) == [0, 3, 6, 9]

    # Subsampled (every 3rd index, starting at 1): [1, 4, 7, 0]
    def test_uneven_padding_proc1(self, mock_dataset_small):
        sampler = samplers.DistributedSampler(
            mock_dataset_small, num_replicas=3, process_index=1, shuffle=False, drop_last=False
        )
        assert len(sampler) == 4
        assert list(sampler) == [1, 4, 7, 0]

    # Subsampled (every 3rd index, starting at 2): [2, 5, 8, 1]
    def test_uneven_padding_proc2(self, mock_dataset_small):
        sampler = samplers.DistributedSampler(
            mock_dataset_small, num_replicas=3, process_index=2, shuffle=False, drop_last=False
        )
        assert len(sampler) == 4
        assert list(sampler) == [2, 5, 8, 1]

    # Scenario: 23 items, 4 replicas, drop_last=True
    # total_size = 20. num_samples = 5.
    # Original indices: [0, 1, ..., 22]. Truncated to: [0, 1, ..., 19]
    # Subsample Proc 0: [0, 4, 8, 12, 16]
    def test_drop_last_proc0(self, mock_dataset_large):
        sampler = samplers.DistributedSampler(
            mock_dataset_large, num_replicas=4, process_index=0, shuffle=False, drop_last=True
        )
        assert len(sampler) == 5
        assert list(sampler) == [0, 4, 8, 12, 16]

    # Subsample Proc 3: [3, 7, 11, 15, 19]
    def test_drop_last_proc3(self, mock_dataset_large):
        sampler = samplers.DistributedSampler(
            mock_dataset_large, num_replicas=4, process_index=3, shuffle=False, drop_last=True
        )
        assert len(sampler) == 5
        assert list(sampler) == [3, 7, 11, 15, 19]

    # Scenario: Shuffling and epoch change
    def test_shuffling_and_determinism(self, mock_dataset_small):
        # 10 items, 2 replicas, process 0, shuffle=True, seed=123
        sampler = samplers.DistributedSampler(
            mock_dataset_small, num_replicas=2, process_index=0, shuffle=True, seed=123
        )
        assert len(sampler) == 5

        indices = jax.random.permutation(jax.random.key(123), len(mock_dataset_small)).tolist()

        # Epoch 0 (Seed 123 + 0)
        # Full shuffled indices (from Python random):
        # indices = [1, 3, 5, 9, 8, 0, 4, 6, 7, 2]
        # Padded indices: [1, 3, 5, 9, 8, 0, 4, 6, 7, 2] (length 10)
        # Proc 0 (every 2nd index, starting at 0): [1, 5, 8, 4, 7]
        epoch0_indices = list(sampler)
        assert len(epoch0_indices) == 5
        assert epoch0_indices == indices[::2]

        # Epoch 1 (Seed 123 + 1)
        sampler.set_epoch(1)
        indices = jax.random.permutation(jax.random.key(123 + 1), len(mock_dataset_small)).tolist()

        # Full shuffled indices (from Python random):
        # indices = [3, 6, 1, 9, 7, 8, 5, 4, 2, 0]
        # Proc 0: [3, 1, 7, 5, 2]
        epoch1_indices = list(sampler)
        assert epoch1_indices != epoch0_indices  # Ensure the shuffle changed
        assert epoch1_indices == indices[::2]

    # Scenario: Test with default jax process count (mocked to 4)
    def test_default_jax_mock_values(self, mock_dataset_small):
        # Dataset size 10, default index=0, no drop
        # len=10, num_replicas=4. total_size = math.ceil(10/4)*4 = 3*4 = 12
        # Padded indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1]
        # Subsampled (every 4th index, starting at 0): [0, 4, 8]
        sampler = samplers.DistributedSampler(
            mock_dataset_small, num_replicas=4, shuffle=False, drop_last=False
        )  # Uses mock jax defaults
        assert sampler._num_replicas == 4
        assert sampler._process_index == 0
        assert len(sampler) == 3
        assert list(sampler) == [0, 4, 8]


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
