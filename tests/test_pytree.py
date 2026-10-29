"""Tests for the vendored simple-pytree and the `reax.pytree.PyTree` base class built on it.

`TestVendoredPytree` and `TestMutablePytree` are adapted from simple-pytree's own test suite
(``tests/test_pytree.py``), using ``dataclasses`` directly instead of ``simple_pytree.dataclass``.
"""

import dataclasses
from typing import Generic, TypeVar

from flax import serialization
import jax
import jax.numpy as jnp
import optax
import pytest

import reax
from reax import metrics, pytree
from reax._vendor.simple_pytree import Pytree


class TestVendoredPytree:
    def test_immutable_pytree(self):
        class Foo(Pytree):
            x: int = pytree.field(static=True)

            def __init__(self, y) -> None:
                self.x = 2
                self.y = y

        tree = Foo(y=3)

        leaves = jax.tree_util.tree_leaves(tree)
        assert leaves == [3]

        tree = jax.tree.map(lambda x: x * 2, tree)
        assert tree.x == 2
        assert tree.y == 6

        tree = tree.replace(x=3)
        assert tree.x == 3
        assert tree.y == 6

        with pytest.raises(AttributeError, match="is immutable, trying to update field"):
            tree.x = 4

    def test_immutable_pytree_dataclass(self):
        @dataclasses.dataclass(frozen=True)
        class Foo(Pytree):
            y: int = dataclasses.field()
            x: int = pytree.field(static=True, default=2)

        tree = Foo(y=3)

        leaves = jax.tree_util.tree_leaves(tree)
        assert leaves == [3]

        tree = jax.tree.map(lambda x: x * 2, tree)
        assert tree.x == 2
        assert tree.y == 6

        tree = tree.replace(x=3)
        assert tree.x == 3
        assert tree.y == 6

        with pytest.raises(AttributeError, match="cannot assign to field"):
            tree.x = 4

    def test_jit(self):
        @dataclasses.dataclass
        class Foo(Pytree):
            a: int
            b: int = pytree.field(static=True)

        module = Foo(a=1, b=2)

        @jax.jit
        def f(m: Foo):
            return m.a + m.b

        assert f(module) == 3

    def test_pytree_node_false_metadata(self):
        """The upstream way of marking static fields still works"""

        @dataclasses.dataclass
        class Foo(Pytree):
            a: int
            b: int = dataclasses.field(metadata={"pytree_node": False})

        assert jax.tree_util.tree_leaves(Foo(a=1, b=2)) == [1]

    def test_flax_serialization(self):
        class Bar(Pytree):
            a: int = pytree.field(static=True)

            def __init__(self, a, b):
                self.a = a
                self.b = b

        @dataclasses.dataclass
        class Foo(Pytree):
            bar: Bar
            c: int
            d: int = pytree.field(static=True)

        foo: Foo = Foo(bar=Bar(a=1, b=2), c=3, d=4)

        state_dict = serialization.to_state_dict(foo)

        assert state_dict == {
            "bar": {
                "b": 2,
            },
            "c": 3,
        }

        state_dict["bar"]["b"] = 5

        foo = serialization.from_state_dict(foo, state_dict)

        assert foo.bar.b == 5

        del state_dict["bar"]["b"]

        with pytest.raises(ValueError, match="Missing field"):
            serialization.from_state_dict(foo, state_dict)

        state_dict["bar"]["b"] = 5

        # add unknown field
        state_dict["x"] = 6

        with pytest.raises(ValueError, match="Unknown field"):
            serialization.from_state_dict(foo, state_dict)

    def test_generics(self):
        T = TypeVar("T")

        class MyClass(Pytree, Generic[T]):
            def __init__(self, x: T):
                self.x = x

        MyClass[int]  # pylint: disable=pointless-statement

    def test_key_paths(self):
        @dataclasses.dataclass
        class Bar(Pytree):
            a: int = 1
            b: int = pytree.field(static=True, default=2)

        @dataclasses.dataclass
        class Foo(Pytree):
            x: int = 3
            y: int = pytree.field(static=True, default=4)
            z: Bar = dataclasses.field(default_factory=Bar)

        foo = Foo()

        path_values, _ = jax.tree_util.tree_flatten_with_path(foo)
        path_values = [(list(map(str, path)), value) for path, value in path_values]

        assert path_values[0] == ([".x"], 3)
        assert path_values[1] == ([".z", ".a"], 1)

    def test_setter_attribute_allowed(self):
        n = None

        class SetterDescriptor:
            def __set__(self, _, value):
                nonlocal n
                n = value

        class Foo(Pytree):
            x = SetterDescriptor()

        foo = Foo()
        foo.x = 1

        assert n == 1

    def test_replace_unknown_fields_error(self):
        class Foo(Pytree):
            pass

        with pytest.raises(ValueError, match="Trying to replace unknown fields"):
            Foo().replace(y=1)

    def test_dataclass_inheritance(self):
        @dataclasses.dataclass
        class A(Pytree):
            a: int = 1
            b: int = pytree.field(static=True, default=2)

        @dataclasses.dataclass
        class B(A):
            c: int = 3

        tree = B()
        leaves = jax.tree_util.tree_leaves(tree)
        assert leaves == [1, 3]

    def test_pytree_with_new(self):
        class A(Pytree):
            def __init__(self, a):
                self.a = a

            def __new__(cls, a):
                del a
                return super().__new__(cls)

        tree = A(a=1)

        jax.tree.map(lambda x: x * 2, tree)

    def test_deterministic_order(self):
        class A(Pytree):
            def __init__(self, order: bool):
                if order:
                    self.a = 1
                    self.b = 2
                else:
                    self.b = 2
                    self.a = 1

        p1 = A(order=True)
        p2 = A(order=False)

        leaves1 = jax.tree_util.tree_leaves(p1)
        leaves2 = jax.tree_util.tree_leaves(p2)

        assert leaves1 == leaves2

    def test_treedef_is_hashable(self):
        @dataclasses.dataclass
        class Foo(Pytree):
            a: int
            b: int = pytree.field(static=True)

        treedef = jax.tree.structure(Foo(a=1, b=2))
        assert hash(treedef) == hash(jax.tree.structure(Foo(a=5, b=2)))
        assert treedef == jax.tree.structure(Foo(a=5, b=2))
        assert treedef != jax.tree.structure(Foo(a=5, b=3))


class TestMutablePytree:
    def test_pytree(self):
        class Foo(Pytree, mutable=True):
            x: int = pytree.field(static=True)

            def __init__(self, y) -> None:
                self.x = 2
                self.y = y

        tree = Foo(y=3)

        leaves = jax.tree_util.tree_leaves(tree)
        assert leaves == [3]

        tree = jax.tree.map(lambda x: x * 2, tree)
        assert tree.x == 2
        assert tree.y == 6

        tree = tree.replace(x=3)
        assert tree.x == 3
        assert tree.y == 6

        # test mutation
        tree.x = 4
        assert tree.x == 4

    def test_no_new_fields_after_init(self):
        class Foo(Pytree, mutable=True):
            def __init__(self, x):
                self.x = x

        foo = Foo(x=1)
        foo.x = 2

        with pytest.raises(AttributeError, match=r"Cannot add new fields to"):
            foo.y = 2  # ty: ignore[unresolved-attribute]

    def test_pytree_dataclass(self):
        @dataclasses.dataclass
        class Foo(Pytree, mutable=True):
            y: int = dataclasses.field()
            x: int = pytree.field(static=True, default=2)

        tree: Foo = Foo(y=3)

        leaves = jax.tree_util.tree_leaves(tree)
        assert leaves == [3]

        tree = jax.tree.map(lambda x: x * 2, tree)
        assert tree.x == 2
        assert tree.y == 6

        tree = tree.replace(x=3)
        assert tree.x == 3
        assert tree.y == 6

        # test mutation
        tree.x = 4
        assert tree.x == 4

    def test_dataclass_inheritance(self):
        A = dataclasses.make_dataclass(  # pylint: disable=invalid-name
            "A",
            [("x", int), "y", ("z", int, pytree.field(static=True, default=5))],
        )

        @dataclasses.dataclass
        class B(Pytree, A): ...  # ty: ignore[unsupported-base]

        b = B(1, 2)  # ty: ignore[too-many-positional-arguments]

        assert b.x == 1
        assert b.y == 2
        assert b.z == 5

        assert jax.tree_util.tree_leaves(b) == [1, 2]


class TestPyTree:
    def test_generated_init(self):
        class Foo(pytree.PyTree):
            a: int
            b: int = 2
            c: str = pytree.field(static=True, default="c")

        foo = Foo(1)
        assert (foo.a, foo.b, foo.c) == (1, 2, "c")
        assert jax.tree.leaves(foo) == [1, 2]
        assert dataclasses.is_dataclass(foo)

    def test_generated_init_inheritance(self):
        class Foo(pytree.PyTree):
            a: int = 1

        class Bar(Foo):
            b: int = 2

        assert jax.tree.leaves(Bar(3, 4)) == [3, 4]

    def test_immutable(self):
        class Foo(pytree.PyTree):
            a: int = 1

        foo = Foo()
        with pytest.raises(AttributeError):
            foo.a = 2

    def test_custom_init(self):
        """A hand-written `__init__` can transform its parameters, only what is assigned is
        stored."""

        class Foo(pytree.PyTree):
            total: int

            def __init__(self, values=(1, 2)):
                self.total = sum(values)

        foo = Foo((3, 4))
        assert foo.total == 7
        assert vars(foo).keys() == {"total", "_pytree__node_fields"}

    def test_custom_init_inherited(self):
        """A subclass without its own `__init__` must use the parent's hand-written one rather than
        a generated one."""

        class Foo(pytree.PyTree):
            total: int

            def __init__(self, values=(1, 2)):
                self.total = sum(values)

        class Bar(Foo):
            extra: int = pytree.field(static=True, default=0)

        bar = Bar((3, 4))  # ty: ignore[invalid-argument-type]  # checkers don't see the inherited __init__
        assert bar.total == 7
        assert jax.tree.leaves(bar) == [7]

    def test_custom_init_field_ordering(self):
        """Fields without defaults can follow ones with if there is a custom `__init__`"""

        class Foo(pytree.PyTree):
            a: int = 0
            b: int  # ty: ignore[dataclass-field-order]

            def __init__(self):
                self.a = 1
                self.b = 2

        assert jax.tree.leaves(Foo()) == [1, 2]

    def test_init_not_rerun(self):
        calls = []

        class Foo(pytree.PyTree):
            total: jax.Array

            def __init__(self, values):
                calls.append(None)
                self.total = jnp.sum(jnp.asarray(values))

        foo = Foo([1.0, 2.0])
        assert len(calls) == 1

        foo = jax.tree.map(lambda x: x * 2, foo)
        foo = jax.jit(lambda f: f)(foo)
        foo = foo.replace(total=foo.total + 1)
        assert len(calls) == 1
        assert foo.total == 7.0

    def test_jit_static_fields(self):
        traces = []

        class Foo(pytree.PyTree):
            a: jax.Array
            mode: str = pytree.field(static=True, default="double")

        @jax.jit
        def fn(foo: Foo):
            traces.append(None)
            if foo.mode == "double":  # Python control flow on a static field
                return foo.a * 2
            return foo.a

        assert fn(Foo(jnp.array(1.0))) == 2.0
        assert fn(Foo(jnp.array(2.0))) == 4.0
        assert len(traces) == 1
        assert fn(Foo(jnp.array(2.0), mode="single")) == 2.0
        assert len(traces) == 2

    def test_equinox_static_field(self):
        """Fields declared with `equinox.field(static=True)` are also static"""
        eqx = pytest.importorskip("equinox")

        class Foo(pytree.PyTree):
            a: int = 1
            b: int = eqx.field(static=True, default=2)

        assert jax.tree.leaves(Foo()) == [1]

    def test_partition_combine(self):
        class Foo(pytree.PyTree):
            a: jax.Array
            b: str = "b"

        foo = Foo(jnp.array([1.0, 2.0]))
        dynamic, static = pytree.partition(foo, pytree.is_array)
        assert jax.tree.leaves(dynamic) == [foo.a]
        assert jax.tree.leaves(static) == ["b"]

        combined = pytree.combine(dynamic, static)
        assert isinstance(combined, Foo)
        assert combined.b == "b"
        assert jnp.array_equal(combined.a, foo.a)


def test_optimizer_update_jit():
    """`Optimizer.update` is jitted with plain `jax.jit` and shouldn't retrace each step"""
    params = {"w": jnp.ones(3)}
    optimizer = reax.Optimizer.from_params(optax.adam(1e-2), params)

    def step(params, optimizer):
        params, optimizer = optimizer.update(params, params)
        # Python floats should be traced not static (otherwise every new value would recompile)
        return optimizer.update(params, params, value=float(jnp.sum(params["w"])))

    # The very first call sees `update_count` as a Python int, after that it is an array, so warm
    # up with two steps to see all combinations of int/array count and `None`/float value
    for _ in range(2):
        params, optimizer = step(params, optimizer)
    cache_size = reax.Optimizer.update._cache_size()  # ty: ignore[unresolved-attribute]  # pylint: disable=protected-access
    for _ in range(3):
        params, optimizer = step(params, optimizer)

    assert optimizer.update_count == 10
    assert reax.Optimizer.update._cache_size() == cache_size  # ty: ignore[unresolved-attribute]  # pylint: disable=protected-access


def test_distributed_optimizer_constructs():
    """Regression: this used to fail with equinox as `_engine` wasn't a declared field"""
    transform = optax.sgd(0.1)
    engine = object()
    optimizer = reax.optimizers.DistributedOptimizer(
        transform,
        transform.init({"w": jnp.ones(2)}),
        engine=engine,  # ty: ignore[invalid-argument-type]
    )
    assert optimizer._engine is engine  # pylint: disable=protected-access
    assert len(jax.tree.leaves(optimizer)) == 1  # Only the update count


def test_accuracy_jit():
    preds = jnp.array([[0.5, 0.5, 0.0], [0.1, 0.6, 0.3], [0.2, 0.2, 0.6], [0.9, 0.1, 0.0]])
    target = jnp.array([0, 1, 2, 1])

    @jax.jit
    def step(acc: metrics.Accuracy, preds, target) -> metrics.Accuracy:
        return acc.merge(acc.update(preds, target))

    acc = step(metrics.Accuracy().update(preds, target), preds, target)
    assert isinstance(acc, metrics.Accuracy)
    assert jnp.isclose(acc.compute(), 0.75)


def test_metric_collection_jit():
    collection = metrics.MetricCollection(dict(mean=metrics.Average(), std=metrics.Std()))
    values = jnp.arange(10.0)

    @jax.jit
    def create(values) -> metrics.MetricCollection:
        return collection.create(values)

    result = create(values).compute()
    assert jnp.isclose(result["mean"], jnp.mean(values))
    assert jnp.isclose(result["std"], jnp.std(values))


def test_from_fun_jit():
    mean_sq = metrics.Average.from_fun(lambda values: values**2)
    values = jnp.arange(4.0)

    metric = jax.jit(mean_sq.create)(values)
    metric = jax.jit(lambda m, v: m.update(v))(metric, values)
    assert jnp.isclose(metric.compute(), jnp.mean(values**2))
