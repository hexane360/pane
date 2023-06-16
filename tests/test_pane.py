
import inspect
from dataclasses import field, dataclass
import typing as t

import pytest

import pane
from pane.convert import ErrorNode, ProductErrorNode, DuplicateKeyError


def check_ord(obj, other, ordering: t.Literal[-1, 0, 1]):
    assert (obj < other) == (ordering < 0)
    assert (obj > other) == (ordering > 0)
    assert (obj <= other) == (ordering <= 0)
    assert (obj >= other) == (ordering >= 0)
    assert (obj == other) == (ordering == 0)
    assert (obj != other) == (ordering != 0)


def f(*args, **kwargs):
    return (args, kwargs)


class TestClass(pane.PaneBase):
    x: int = 3
    y: float = 5.

    __test__ = False


@pytest.mark.parametrize(('args', 'result'), [
    (f(), {'x': 3, 'y': 5.}),
    (f(4), {'x': 4, 'y': 5.}),
    (f(y=8), {'x': 3, 'y': 8.0}),
])
def test_pane_construction(args, result):
    (args, kwargs) = args
    assert TestClass(*args, **kwargs).__dict__ == result


@pytest.mark.parametrize(('args', 'result'), [
    (f(), {'x': 3, 'y': 5.}),
    (f('st'), {'x': 'st', 'y': 5.}),
    (f(y=None), {'x': 3, 'y': None}),
])
def test_pane_unchecked(args, result):
    (args, kwargs) = args
    assert TestClass.make_unchecked(*args, **kwargs).__dict__ == result


@pytest.mark.parametrize(('args', 'result'), [
    (f(), "TestClass(x=3, y=5.0)"),
    (f(4), "TestClass(x=4, y=5.0)"),
    (f(y=8), "TestClass(x=3, y=8.0)"),
])
def test_pane_repr(args, result):
    (args, kwargs) = args
    assert repr(TestClass(*args, **kwargs)) == result


def test_pane_ord():
    check_ord(TestClass(x=1, y=3.), TestClass(x=2, y=1.), -1)
    check_ord(TestClass(x=2, y=3.), TestClass(x=2, y=1.), 1)
    check_ord(TestClass(x=2, y=1.), TestClass(x=2, y=1.), 0)
    with pytest.raises(TypeError):
        check_ord(TestClass(x=1, y=3.), 5, -1)


class TestClass2(pane.PaneBase):
    x: int = 1
    z: int = pane.field(default=3, kw_only=True)
    y: int = 2
    _: pane.KW_ONLY
    w: int = pane.field(default=4, aliases=('W', 'P'))

    __test__ = False


class TestClass3(pane.PaneBase, kw_only=True):
    x: int = 1
    z: int = pane.field(default=3, kw_only=True)
    y: int = 2
    _: pane.KW_ONLY
    w: int = 4

    __test__ = False


@pytest.mark.parametrize(('cls', 'sig'), [
    (TestClass, '(x: int = 3, y: float = 5.0) -> None'),
    (TestClass2, '(x: int = 1, y: int = 2, *, z: int = 3, w: int = 4) -> None'),
    (TestClass3, '(*, x: int = 1, z: int = 3, y: int = 2, w: int = 4) -> None'),
])
def test_init_signature(cls, sig):
    assert str(inspect.signature(cls.__init__)) == sig
    assert str(inspect.signature(cls)) == sig


@pytest.mark.parametrize(('cls', 'sig'), [
    (TestClass, '(x: int = 3, y: float = 5.0) -> test_pane.TestClass'),
    (TestClass2, '(x: int = 1, y: int = 2, *, z: int = 3, w: int = 4) -> test_pane.TestClass2'),
    (TestClass3, '(*, x: int = 1, z: int = 3, y: int = 2, w: int = 4) -> test_pane.TestClass3'),
])
def test_make_unchecked_signature(cls, sig):
    assert str(inspect.signature(cls.make_unchecked)) == sig


@pytest.mark.parametrize(('cls', 'val', 'result'), [
    (TestClass, {'x': 3, 'y': 5.}, TestClass(3, 5.)),
    (TestClass2, {'x': 3, 'w': 3, 'W': 4}, ProductErrorNode('TestClass2', {'W': DuplicateKeyError('W', ('w', 'W', 'P'))}, {'x': 3, 'w': 3, 'W': 4})),
])
def test_pane_convert(cls, val, result):
    if isinstance(result, ErrorNode):
        with pytest.raises(pane.ConvertError) as e:
            pane.convert(val, cls)
        assert e.value.tree == result
    else:
        assert pane.convert(val, cls) == result


T = t.TypeVar('T')


class GenericPane(pane.PaneBase, t.Generic[T]):
    x: T


def test_generic_pane():
    with pytest.raises(pane.ConvertError):
        GenericPane[int]('str')  # type: ignore

    GenericPane[int](5)
    with pytest.warns(UserWarning, match="Unbound TypeVar '~T'. Will be interpreted as Any."):
        GenericPane('any value')