from __future__ import annotations

import inspect
import io
import typing as t

import pytest

import pane
from pane.errors import ErrorNode, ProductErrorNode, DuplicateKeyError, WrongTypeError
from pane.convert import make_converter
from pane.annotations import Tagged

from test_converters import DoubleIntConverter, DoubleStrConverter


def check_ord(obj, other, ordering: t.Literal[-1, 0, 1]):
    assert (obj < other) == (ordering < 0)
    assert (obj > other) == (ordering > 0)
    assert (obj <= other) == (ordering <= 0)
    assert (obj >= other) == (ordering >= 0)
    assert (obj == other) == (ordering == 0)
    assert (obj != other) == (ordering != 0)


def f(*args, **kwargs):
    return (args, kwargs)


def test_pane_required_after_default():
    with pytest.raises(TypeError, match="Mandatory field 'y' follows optional field"):
        class RequiredAfterDefault(pane.PaneBase):
            x: int = 3
            y: int

    class RequiredAfterDefaultKwOnly(pane.PaneBase):
        _: pane.KW_ONLY
        x: int = 3
        y: int


def test_pane_required_kw_only():
    class RequiredKwOnly(pane.PaneBase):
        x: int
        z: int = pane.field(kw_only=True)
        y: int

    with pytest.raises(TypeError, match="Field 'z' is kw_only but mandatory. This is incompatible with the 'tuple' in_format."):
        class RequiredKwOnlyTuple(RequiredKwOnly, in_format='tuple'):
            ...


class TestClass(pane.PaneBase, in_format=('tuple, struct'), out_format='tuple'):
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





class TestClassInherit(TestClass, out_format='struct'):
    z: float = pane.field(default=3., kw_only=True)


class TestClassInherit2(TestClassInherit):
    w: int = 4


class TestClassModifyFields(TestClassInherit):
    x: float = 9.  # type: ignore


class TestClass2(pane.PaneBase, in_format=('tuple', 'struct')):
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


class NestedClass(pane.PaneBase, in_format=('tuple',), out_format='tuple'):
    x: TestClass
    y: int
    z: TestClass


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
    (TestClass, {'x': 3, 'y': 5., 'zz': 5}, ProductErrorNode('struct TestClass', {}, {'x': 3, 'y': 5., 'zz': 5}, extra={'zz'})),
    # duplicate keys
    (TestClass2, {'x': 3, 'w': 3, 'W': 4}, ProductErrorNode('struct TestClass2', {'W': DuplicateKeyError('W', ('w', 'W', 'P'))}, {'x': 3, 'w': 3, 'W': 4})),
    # extra fields, allow_extra=True
    (TestClass3, {'x': 2, 'y': 3, 'w': 4, 'zz': 5}, TestClass3(x=2, y=3, w=4)),
    # missing fields
    (TestClass3, {}, ProductErrorNode('struct TestClass3', {}, {}, missing={'w'})),
    (NestedClass, ({'x': 2, 'y': 4.}, 10, {'x': 4, 'y': 8.}), NestedClass(x=TestClass(x=2, y=4.0), y=10, z=TestClass(x=4, y=8.0))),
])
def test_pane_convert(cls, val, result):
    if isinstance(result, ErrorNode):
        with pytest.raises(pane.ConvertError) as e:
            pane.convert(val, cls)
        assert e.value.tree == result
    else:
        assert pane.convert(val, cls) == result


@pytest.mark.parametrize(('val', 'result'), [
    # out_format tuple
    (TestClass(x=3, y=5.), (3, 5.)),
    # out_format struct
    (TestClass2(x=3, w=3), {'x': 3, 'y': 2, 'w': 3, 'z': 3}),
    (NestedClass(TestClass(2, 4.), 10, TestClass(4, 8.)), ((2, 4.), 10, (4, 8.))),
    ({'x': 5, 'y': TestClass(2, 4.)}, {'x': 5, 'y': (2, 4.)}),
])
def test_pane_into_data(val: pane.PaneBase, result: pane.DataType):
    if isinstance(val, pane.PaneBase):
        assert val.into_data() == result
    assert pane.into_data(val) == result


def test_pane_write_json():
    val = TestClassInherit(x=5, y=10., z=5.0)

    expected = """{"x": 5, "y": 10.0, "z": 5.0}"""

    buf = io.StringIO()
    val.write_json(buf)
    assert buf.getvalue() == expected
    assert val.into_json() == expected


def test_pane_write_yaml():
    val = TestClassInherit(x=5, y=10., z=5.0)

    expected = """---
x: 5
y: 10.0
z: 5.0
"""

    buf = io.StringIO()
    val.write_yaml(buf, default_flow_style=False)
    assert buf.getvalue() == expected
    assert val.into_yaml(default_flow_style=False) == expected


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


@pytest.mark.parametrize(('lhs', 'rhs', 'result'), [
    (GenericPane[int](5), GenericPane[t.Any](5), True),
    (TestClass3(x=1, z=3, w=3), TestClass3(w=3), True),
    (TestClass(), TestClass2(), False),
])
def test_pane_eq(lhs, rhs, result):
    if result:
        assert lhs == rhs
    else:
        assert lhs != rhs


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
    (GenericPane, f('any value'), UserWarning("Unbound TypeVar '~T'. Will be interpreted as 'typing.Any'.")),
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

    assert PaneRename(1, 2., 3, 4.).dict(rename='scream') == {
        'SNAKE_CASE': 1, 'SCREAM_CASE': 2., 'CAMEL_CASE': 3, 'PASCAL_CASE': 4.
    }


class PaneTag1(pane.PaneBase, kw_only=True):
    tag: t.Literal['tag1'] = 'tag1'
    x: int
    y: float


class PaneTag2(pane.PaneBase, kw_only=True):
    tag: t.Literal['tag2'] = 'tag2'
    x: int


class PaneTag3(pane.PaneBase, kw_only=True):
    tag: t.Literal['tag3'] = 'tag3'
    x: int


class PaneTagBase(pane.PaneBase):
    x: t.Annotated[t.Union[PaneTag1, PaneTag2, PaneTag3], Tagged('tag')]


@pytest.mark.parametrize(('val', 'result'), [
    ({'x': {'tag': 'tag1', 'x': 5, 'y': 5.0}}, PaneTagBase(x=PaneTag1(x=5, y=5.0))),
    ({'x': {'tag': 'tag3', 'x': 8}}, PaneTagBase(x=PaneTag3(x=8))),
])
def test_pane_convert_tagged_union(val, result):
    cls = PaneTagBase
    if isinstance(result, ErrorNode):
        with pytest.raises(pane.ConvertError) as e:
            pane.convert(val, cls)
        assert e.value.tree == result
    else:
        assert pane.convert(val, cls) == result


def test_manual_slots():
    class SlotsClass(pane.PaneBase, frozen=False):
        __slots__ = ("x", "z")
        x: int

    obj = SlotsClass(5)
    obj.z = 8
    with pytest.raises(AttributeError, match="'SlotsClass' object has no attribute 'y'"):
        obj.y = 8


class PaneCustomParent(pane.PaneBase, custom={int: DoubleIntConverter()}):
    x: int
    y: str


class PaneCustomMember(pane.PaneBase, custom={str: make_converter(str)}):
    inner: str


class PaneCustomChild(PaneCustomParent, custom={str: DoubleStrConverter()}):
    z: str = pane.field(converter=make_converter(str))
    member: PaneCustomMember


def test_pane_custom_converters():
    # custom from class definition
    assert PaneCustomParent.from_data({'x': 5, 'y': 'test'}) == PaneCustomParent.make_unchecked(x=10, y='test')

    # parent custom definition is overridden by child custom definition 
    # outer custom definition is overridden by inner custom definition
    # field definition overrides class definitions
    assert PaneCustomChild.from_data(
        {'x': 5, 'y': 'test', 'z': 'str', 'member': {'inner': 'str'}}
    ) == PaneCustomChild.make_unchecked(x=5, y='testtest', z='str', member=PaneCustomMember.make_unchecked('str'))

    # custom passed to from_data takes precedence over everything except field converters
    # this also tests that make_converter is memoized correctly
    assert pane.from_data(
        {'x': 5, 'y': 'str', 'z': 'str', 'member': {'inner': 'str'}},
        PaneCustomChild, custom={int: make_converter(int), str: DoubleStrConverter()}
    ) == PaneCustomChild.make_unchecked(x=5, y='strstr', z='str', member=PaneCustomMember.make_unchecked('strstr'))