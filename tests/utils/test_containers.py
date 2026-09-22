import pytest

from reax.utils import containers


class _A:
    pass


class _B(_A):
    pass


def test_baseregistry_register_and_getitem():
    reg = containers.BaseRegistry()
    assert len(reg) == 0
    reg.register("k1", 1)
    reg.register("k2", 2)
    assert len(reg) == 2
    assert reg["k1"] == 1
    assert reg["k2"] == 2
    with pytest.raises(KeyError):
        _ = reg["missing"]


def test_baseregistry_init_and_register_many():
    reg = containers.BaseRegistry({"x": 10, "y": 20})
    assert len(reg) == 2
    reg.register_many({"a": 1, "b": 2})
    assert reg["a"] == 1
    assert reg["b"] == 2


def test_baseregistry_iter_and_items():
    reg = containers.BaseRegistry({"x": 10, "y": 20})
    assert set(iter(reg)) == {"x", "y"}
    assert dict(reg.items()) == {"x": 10, "y": 20}
    assert reg == {"x": 10, "y": 20}


def test_baseregistry_unregister():
    reg = containers.BaseRegistry({"x": 10})
    assert reg.unregister("x") == 10
    assert len(reg) == 0
    with pytest.raises(KeyError):
        reg.unregister("x")


def test_registry_find():
    reg = containers.Registry()
    reg.register("fit", 1)
    reg.register("fit_val", 2)
    reg.register("test", 3)
    assert dict(reg.find("fit")) == {"fit": 1, "fit_val": 2}
    assert list(reg.find("nope")) == []


def test_typeregistry_find_exact():
    reg = containers.TypeRegistry()
    reg.register(int, "int-obj")
    reg.register(str, "str-obj")
    assert reg.find(5) == "int-obj"


def test_typeregistry_find_isinstance():
    reg = containers.TypeRegistry()
    reg.register(_A, "base-obj")
    b = _B()
    assert reg.find(b) == "base-obj"


def test_typeregistry_find_missing():
    reg = containers.TypeRegistry()
    assert reg.find(_A()) is None
