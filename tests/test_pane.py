
import inspect

import pytest
from dataclasses import field, dataclass

import pane


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


class TestClass2(pane.PaneBase):
    x: int = 1
    z: int = pane.field(default=3, kw_only=True)
    y: int = 2
    _: pane.KW_ONLY
    w: int = 4

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


@pytest.mark.parametrize(('cls', 'val'), [
    (TestClass, {'x': 3, 'y': 5.})
])
def test_pane_convert(cls, val):
    pane.convert(val, cls)