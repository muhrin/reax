import pytest

import reax
from reax.demos import boring_classes

NUM_SAMPLES = 64
BATCH_SIZE = 4
NUM_BATCHES = NUM_SAMPLES // BATCH_SIZE


class BatchCounter(reax.TrainerListener):
    def __init__(self):
        super().__init__()
        self.batches = {"train": 0, "validation": 0, "test": 0, "predict": 0}

    def on_train_batch_start(self, *_):
        self.batches["train"] += 1

    def on_validation_batch_start(self, *_):
        self.batches["validation"] += 1

    def on_test_batch_start(self, *_):
        self.batches["test"] += 1

    def on_predict_batch_start(self, *_):
        self.batches["predict"] += 1


def _sizeless_loader() -> reax.ReaxDataLoader:
    """A dataloader over an iterable dataset, so its length isn't known in advance."""
    loader = reax.ReaxDataLoader(
        boring_classes.RandomIterableDataset(32, NUM_SAMPLES), batch_size=BATCH_SIZE
    )
    assert reax.data.utils.sized_len(loader) is None
    return loader


@pytest.mark.parametrize("limit_batches", [None, 1.0])
def test_loops_consume_all_batches_of_sizeless_dataloader(tmp_path, limit_batches):
    """All batches are used when the dataloader has no length and no limit bounds it."""
    counter = BatchCounter()
    trainer = reax.Trainer(default_root_dir=tmp_path, listeners=[counter])
    model = boring_classes.BoringModel()

    trainer.fit(
        model,
        train_dataloaders=_sizeless_loader(),
        val_dataloaders=_sizeless_loader(),
        max_epochs=1,
        num_sanity_val_steps=0,
        limit_train_batches=limit_batches,
        limit_val_batches=limit_batches,
    )
    trainer.validate(model, dataloaders=_sizeless_loader(), limit_batches=limit_batches)
    trainer.test(model, dataloaders=_sizeless_loader(), limit_batches=limit_batches)
    trainer.predict(model, dataloaders=_sizeless_loader(), limit_batches=limit_batches)

    assert counter.batches == {
        "train": NUM_BATCHES,
        # Once during fit, and once for the standalone validate
        "validation": 2 * NUM_BATCHES,
        "test": NUM_BATCHES,
        "predict": NUM_BATCHES,
    }
