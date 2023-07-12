
import inspect
import typing as t

import pytest

import pane
from pane.errors import ErrorNode, ProductErrorNode, DuplicateKeyError, WrongTypeError


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
    assert TestClass(*args, **kwargs).dict() == result


@pytest.mark.parametrize(('args', 'result', 'set_only_result'), [
    (f(), {'x': 3, 'y': 5.}, {}),
    (f('st'), {'x': 'st', 'y': 5.}, {'x': 'st'}),
    (f(y=None), {'x': 3, 'y': None}, {'y': None}),
])
def test_pane_unchecked(args, result, set_only_result):
    (args, kwargs) = args
    assert TestClass.make_unchecked(*args, **kwargs).dict() == result
    assert TestClass.make_unchecked(*args, **kwargs).dict(set_only=True) == set_only_result


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


class TestClassInherit(TestClass):
    z: float = pane.field(default=3., kw_only=True)


class TestClassInherit2(TestClassInherit):
    w: int = 4


class TestClassModifyFields(TestClassInherit):
    x: float = 9.  # type: ignore


class TestClass2(pane.PaneBase):
    x: int = 1
    z: int = pane.field(default=3, kw_only=True)
    y: int = 2
    _: pane.KW_ONLY
    w: int = pane.field(default=4, aliases=('W', 'P'))

    __test__ = False


class TestClass3(pane.PaneBase, kw_only=True, allow_extra=True):
    x: int = 1
    z: int = pane.field(default=3, kw_only=True)
    y: int = 2
    _: pane.KW_ONLY
    w: int

    __test__ = False


@pytest.mark.parametrize(('cls', 'sig'), [
    (TestClass, '(x: int = 3, y: float = 5.0) -> None'),
    (TestClass2, '(x: int = 1, y: int = 2, *, z: int = 3, w: int = 4) -> None'),
    (TestClass3, '(*, x: int = 1, z: int = 3, y: int = 2, w: int) -> None'),
    (TestClassInherit, '(x: int = 3, y: float = 5.0, *, z: float = 3.0) -> None'),
    (TestClassInherit2, '(x: int = 3, y: float = 5.0, w: int = 4, *, z: float = 3.0) -> None'),
    (TestClassModifyFields, '(x: float = 9.0, y: float = 5.0, *, z: float = 3.0) -> None'),
])
def test_init_signature(cls, sig):
    assert str(inspect.signature(cls.__init__)) == sig
    assert str(inspect.signature(cls)) == sig


@pytest.mark.parametrize(('cls', 'sig'), [
    (TestClass, '(x: int = 3, y: float = 5.0) -> test_pane.TestClass'),
    (TestClass2, '(x: int = 1, y: int = 2, *, z: int = 3, w: int = 4) -> test_pane.TestClass2'),
    (TestClass3, '(*, x: int = 1, z: int = 3, y: int = 2, w: int) -> test_pane.TestClass3'),
])
def test_make_unchecked_signature(cls, sig):
    assert str(inspect.signature(cls.make_unchecked)) == sig


@pytest.mark.parametrize(('cls', 'val', 'result'), [
    (TestClass, {'x': 3, 'y': 5.}, TestClass(3, 5.)),
    # extra fields
    (TestClass, {'x': 3, 'y': 5., 'zz': 5}, ProductErrorNode('TestClass', {}, {'x': 3, 'y': 5., 'zz': 5}, extra={'zz'})),
    # duplicate keys
    (TestClass2, {'x': 3, 'w': 3, 'W': 4}, ProductErrorNode('TestClass2', {'W': DuplicateKeyError('W', ('w', 'W', 'P'))}, {'x': 3, 'w': 3, 'W': 4})),
    # extra fields, allow_extra=True
    (TestClass3, {'x': 2, 'y': 3, 'w': 4, 'zz': 5}, TestClass3(x=2, y=3, w=4)),
    # missing fields
    (TestClass3, {}, ProductErrorNode('TestClass3', {}, {}, missing={'w'}))
])
def test_pane_convert(cls, val, result):
    if isinstance(result, ErrorNode):
        with pytest.raises(pane.ConvertError) as e:
            pane.convert(val, cls)
        assert e.value.tree == result
    else:
        assert pane.convert(val, cls) == result


T = t.TypeVar('T')
U = t.TypeVar('U')


class GenericPane(pane.PaneBase, t.Generic[T]):
    x: T

class GenericInherit(GenericPane[U]):
    y: U

class ConcreteGenericInherit(GenericPane[int]):
    y: float

class GenericGenericInherit(GenericInherit[T]):
    z: T

class GenericConcreteInherit(GenericInherit[int], t.Generic[T]):
    z: T


@pytest.mark.parametrize(('cls', 'sig'), [
    (GenericPane, '(x: ~T) -> None'),
    (GenericPane[int], '(x: int) -> None'),
    (ConcreteGenericInherit, '(x: int, y: float) -> None'),
    (GenericInherit, '(x: ~U, y: ~U) -> None'),
    (GenericInherit[float], '(x: float, y: float) -> None'),
    (GenericGenericInherit, '(x: ~T, y: ~T, z: ~T) -> None'),
    (GenericGenericInherit[float], '(x: float, y: float, z: float) -> None'),
    (GenericConcreteInherit, '(x: int, y: int, z: ~T) -> None'),
    (GenericConcreteInherit[float], '(x: int, y: int, z: float) -> None'),
])
def test_generic_signatures(cls, sig):
    assert str(inspect.signature(cls.__init__)) == sig
    assert str(inspect.signature(cls)) == sig


@pytest.mark.parametrize(('cls', 'args', 'error'), [
    (GenericPane[int], f('s'), WrongTypeError('an int', 's')),
    (GenericPane[int], f(5), None),
    (GenericPane, f('any value'), UserWarning("Unbound TypeVar '~T'. Will be interpreted as Any.")),
    (GenericInherit[float], f('s', 5.), WrongTypeError('a float', 's')),
    (GenericInherit[float], f(5., 's'), WrongTypeError('a float', 's')),
])
def test_generic_pane_init(cls, args, error):
    (args, kwargs) = args

    if isinstance(error, ErrorNode):
        with pytest.raises(pane.ConvertError) as e:
            cls(*args, **kwargs)
        assert e.value.tree == error
    elif isinstance(error, Warning):
        with pytest.warns(type(error), match=error.args[0]):
            cls(*args, **kwargs)
    else:
        cls(*args, **kwargs)


class PaneRename(pane.PaneBase):
    snake_case: int
    SCREAM_CASE: float
    camelCase: int
    PascalCase: float

class PaneRenameKebab(PaneRename, rename='kebab'):
    ...


def test_pane_rename():
    obj = {'snake-case': 1, 'scream-case': 2., 'camel-case': 3, 'pascal-case': 4.}
    assert PaneRenameKebab.from_data(obj) == PaneRenameKebab(1, 2., 3, 4.)
    assert PaneRenameKebab(1, 2., 3, 4.).into_data() == obj