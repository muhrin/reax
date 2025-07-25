import reax
from reax.demos import boring_classes


def test_data_loader():
    engine = reax.Engine()

    ds = boring_classes.RandomDataset(32, 16)
    loader = reax.ReaxDataLoader(ds, batch_size=2)

    engine_loader = engine.setup(loader)
    assert isinstance(engine_loader, reax.DataLoader)
