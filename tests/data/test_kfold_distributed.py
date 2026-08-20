import jax
import jax.numpy as jnp
import numpy as np

from reax import Engine, testing
from reax.data import KFold


def kfold_distributed_test():
    from reax.strategies import JaxDdpStrategy

    # Initialize engine with JAX DDP strategy on 1 device, simulating the distributed setup
    strategy = JaxDdpStrategy(platform="cpu", devices=1)
    engine = Engine(strategy=strategy)

    assert engine.strategy.process_count == 1
    assert engine.is_global_zero

    # Use KFold with None seed, expecting it to generate on rank 0 and broadcast
    kf = KFold(n_splits=3, shuffle=True, seed=None, engine=engine)

    # Gather the seed from all processes using the engine
    gathered_seeds = engine.strategy.all_gather(jnp.array(kf.seed))

    # In devices=1, gathered_seeds is shape (1,)
    assert gathered_seeds[0] == kf.seed

    # Also verify that the actual splits are identical across processes
    splits = list(kf.split(10))
    train_idx, val_idx = splits[0]
    gathered_train_idx = engine.strategy.all_gather(jnp.array(train_idx))

    # verify the gathered shape matches the local shape prepended with process_count (1)
    assert gathered_train_idx.shape == (1,) + train_idx.shape
    assert np.array_equal(gathered_train_idx[0], train_idx)


test_kfold_distributed = testing.in_subprocess(kfold_distributed_test)
