import pytest

from reax import strategies, testing
from reax.strategies._single_device import SingleDevice


def test_create_single_by_name():
    strategy = strategies.create("single", "cpu")
    assert isinstance(strategy, SingleDevice)
    assert strategy.device.platform == "cpu"


def test_create_auto_with_explicit_devices():
    # 'auto' with a single device must resolve to the single device strategy
    strategy = strategies.create("auto", "cpu", devices=1)
    assert isinstance(strategy, SingleDevice)


def test_create_auto_with_auto_devices():
    # This machine may only expose a single CPU device, so the probe may return 1
    # and we end up with a single device strategy. Both are acceptable outcomes.
    strategy = strategies.create("auto", "cpu")
    assert isinstance(strategy, (SingleDevice, strategies.JaxDdpStrategy))


def _create_auto_with_many_devices_uses_ddp():
    # Explicit device count of 2 forces the distributed DDP strategy
    strategy = strategies.create("auto", "cpu", devices=2)
    assert isinstance(strategy, strategies.JaxDdpStrategy)
    assert strategy.process_count == 2


test_create_auto_with_many_devices_uses_ddp = pytest.mark.multiproc(
    testing.in_subprocess(_create_auto_with_many_devices_uses_ddp)
)


def test_create_unknown_strategies_raises():
    with pytest.raises(ValueError, match="Unknown strategy"):
        strategies.create("bogus", "cpu")
