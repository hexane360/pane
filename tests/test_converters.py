from __future__ import annotations

import re
import datetime
import collections
from decimal import Decimal
from fractions import Fraction
import pathlib
import os
import typing as t

import pytest

from pane.errors import ErrorNode, SumErrorNode, ProductErrorNode, WrongTypeError, ConditionFailedError, ParseInterrupt
from pane.convert import convert, from_data, into_data, make_converter, ConvertError
from pane.converters import Converter, ScalarConverter, TupleConverter, SequenceConverter, TaggedUnionConverter, AnyConverter
from pane.converters import StructConverter, UnionConverter, LiteralConverter, ConditionalConverter
from pane.converters import PatternConverter, DatetimeConverter
from pane.annotations import Condition, Tagged, val_range, len_range


class TestConvertible():
    @classmethod
    def _converter(cls, *args, handlers=None) -> Converter[TestConvertible]:
        return TestConverter()  # type: ignore

    def __hash__(self):
        return hash(self.__class__)

    def __eq__(self, other):
        return self.__class__ == other.__class__


class TestConverter(Converter[TestConvertible]):
    def try_convert(self, val: t.Any):
        return TestConvertible()

    def expected(self, plural: bool = False) -> str:
        return "TestConvertible"

    def collect_errors(self, val: t.Any) -> None:
        return None

    def __hash__(self):
        return hash(self.__class__)

    def __eq__(self, other):
        return self.__class__ == other.__class__


@pytest.mark.parametrize(('input', 'conv'), [
    (int, ScalarConverter(int, int, 'an int', 'ints', int)),
    ({'x': int, 'y': float}, StructConverter(dict, {'x': int, 'y': float})),
    (t.Tuple[int, ...], SequenceConverter(tuple, int)),
    (list[str], SequenceConverter(list, str)),
    (tuple, SequenceConverter(tuple, t.Any)),
    (t.Tuple[int, str], TupleConverter(tuple, (int, str))),
    (t.Tuple[()], TupleConverter(tuple, ())),
    (TestConvertible, TestConverter()),
    (t.Union[str, int], UnionConverter((str, int))),
    (t.Literal['a', 'b', 'c'], LiteralConverter(('a', 'b', 'c'))),
    (t.Literal['a'], LiteralConverter(('a',))),
    (re.Pattern[bytes], PatternConverter(bytes)),
    (t.Pattern, PatternConverter(str)),
    (datetime.datetime, DatetimeConverter(datetime.datetime)),
    (datetime.time, DatetimeConverter(datetime.time)),
])
def test_make_converter(input, conv: Converter):
    assert make_converter(input) == conv


@pytest.mark.parametrize(('input', 'error'), [
    (t.Pattern[int], TypeError("Pattern only accepts a 'str' or 'bytes' type argument, instead got '<class 'int'>'")),
    # tag not found, tag matches multiple types
])
def test_make_converter_raises(input, error: Exception):
    with pytest.raises(type(error)) as exc_info:
        make_converter(input)
    assert str(exc_info.value) == str(error)


@pytest.mark.parametrize(('input', 'conv', 's'), [
    (t.TypeVar('T'), AnyConverter(), "Unbound TypeVar '~T'. Will be interpreted as 'typing.Any'."),
    (t.TypeVar('U', int, str), UnionConverter((int, str)), "Unbound TypeVar '~U'. Will be interpreted as 'typing.Union[int, str]'."),
    (t.TypeVar('V', bound=tuple), SequenceConverter(tuple, t.Any), "Unbound TypeVar '~V'. Will be interpreted as '<class 'tuple'>'."),
])
def test_typevar_converters(input, conv, s):
    with pytest.warns(UserWarning, match=re.escape(s)):
        assert make_converter(input) == conv


cond1 = Condition(lambda v: True, 'true', lambda exp, plural: exp)
cond2 = Condition(lambda v: v > 0, 'v > 0', lambda exp, plural: f"{exp} > 0")


class Variant1(int):
    tag: int = 3

class Variant2(float):
    tag: int = 4

class Variant3(dict):
    tag: int = 3

class Variant4(dict):
    tag: int = 4


def test_make_converter_annotated():
    inner_conv = ScalarConverter(int, int, 'an int', 'ints', int)

    conv = make_converter(t.Annotated[int, cond1])
    assert conv == ConditionalConverter(
        int, cond1.f, 'true', cond1.make_expected
    )
    assert conv.inner == inner_conv

    conv = make_converter(t.Annotated[int, cond2])
    assert conv == ConditionalConverter(
        int, cond2.f, 'v > 0', cond2.make_expected
    )
    assert conv.inner == inner_conv

    compound = make_converter(t.Annotated[int, cond1, cond2])
    assert isinstance(compound, ConditionalConverter)
    assert compound.condition_name == 'true and v > 0'

    assert make_converter(t.Annotated[t.Union[Variant1, Variant2], Tagged('tag')]) \
        == TaggedUnionConverter((Variant1, Variant2), 'tag', external=False)

    assert make_converter(t.Annotated[t.Union[Variant1, Variant2], Tagged('tag', ('t', 'c'))]) \
        == TaggedUnionConverter((Variant1, Variant2), 'tag', external=('t', 'c'))

    with pytest.raises(TypeError, match="Tag value '3' matches multiple types"):
        make_converter(t.Annotated[t.Union[Variant1, Variant3], Tagged('tag')])

@pytest.mark.parametrize(('conv', 'plural', 'expected'), [
    (int, False, 'an int'),
    (t.Optional[str], True, 'strings or null values'),
    (t.Sequence[str], True, 'sequences of strings'),
    (t.Dict[str, int], False, 'mapping of strings => ints'),
    (t.Literal['a', 'b', 'c'], False, "'a', 'b', or 'c'"),
    (t.Annotated[int, cond2], False, "an int > 0"),
    (t.Annotated[int, cond2], True, "ints > 0"),
    (t.Annotated[t.Union[Variant1, Variant2], Tagged('tag')], False, "an int or a float"),
    (t.Annotated[t.Union[Variant1, Variant2], Tagged('tag', external=True)], False, "a mapping '3 or 4' => an int or a float"),
    (t.Annotated[t.Union[Variant1, Variant2], Tagged('tag', external=('t', 'c'))], False, "a mapping 't' => 3 or 4, 'c' => an int or a float"),
    (t.Pattern[str], True, 'string regex patterns'),
    (datetime.datetime, True, 'datetimes'),
    (datetime.time, False, 'a time'),
])
def test_converter_expected(conv: Converter, plural: bool, expected: str):
    if not isinstance(conv, Converter):
        conv = make_converter(conv)
    assert conv.expected(plural) == expected


@pytest.mark.parametrize(('ty', 'val', 'result'), [
    # scalar converters
    (int, 's', WrongTypeError('an int', 's')),
    (float, 5, 5.0),
    (complex, 6, 6.0+0j),
    # bytes types
    (bytes, b'bytestring', b'bytestring'),
    (str, b'bytestring', WrongTypeError('a string', b'bytestring')),
    (bytes, 'string', WrongTypeError('a bytestring', 'string')),
    (bytearray, b'bytestring', bytearray(b'bytestring')),
    (bytes, bytearray(b'bytestring'), b'bytestring'),
    # be conservative for now
    (t.List[int], bytearray(b'bytestring'), WrongTypeError('sequence of ints', bytearray(b'bytestring'))),
    (bytearray, [98, 121, 116, 101, 115, 116, 114, 105, 110, 103], WrongTypeError('a bytearray', [98, 121, 116, 101, 115, 116, 114, 105, 110, 103])),
    # sequence converters
    (t.Tuple[int, ...], [1, 2], (1, 2)),
    (t.Sequence[int], [1, 2], (1, 2)),  # same as tuple
    (t.List[int], (1, 2), [1, 2]),
    # exotic sequences
    (t.Set[float], (1, 2, 3), {1., 2., 3.}),
    (collections.deque, (1, 2, 3), collections.deque((1, 2, 3))),
    (t.MutableSet[int], (1, 2, 3), {1, 2, 3}),
    # dict converters
    (t.Dict[str, float], {'a': 1}, {'a': 1.0}),
    (collections.abc.Mapping[str, float], {'a': 1}, {'a': 1.0}),
    (t.MutableMapping[str, str], {'a': 'b'}, {'a': 'b'}),
    (t.Counter[str], {'a': 5, 'b': 1, 'c': 2}, collections.Counter('aaaaabcc')),
    (collections.defaultdict, {'a': 'b'}, collections.defaultdict(None, {'a': 'b'})),
    # tuple converters
    (t.Tuple[str], ['int'], ('int',)),
    (t.Tuple[str], ('s1', 's2'), WrongTypeError('tuple of length 1', ('s1', 's2'))),
    # union converters (left to right)
    (t.Union[int, float, str], 5., 5.),
    (t.Union[str, float, int], 5, 5.),
    # struct converter
    ({'x': int, 'y': float}, {'x': 5, 'y': 4}, {'x': 5, 'y': 4.}),
    ({'x': t.Union[str, int], 'y': t.Tuple[t.Union[str, int], int]}, {'x': 5., 'y': (0., 's')},
     ProductErrorNode('struct', {
         'x': SumErrorNode([WrongTypeError('a string', 5.), WrongTypeError('an int', 5.)]),
         'y': ProductErrorNode('tuple of length 2', {
               0: SumErrorNode([WrongTypeError('a string', 0.), WrongTypeError('an int', 0.)]),
               1: WrongTypeError('an int', 's'),
         },  (0., 's')),
     }, {'x': 5., 'y': (0., 's')})
    ),
    # custom convertible type
    (TestConvertible, 's', TestConvertible()),
    # conditions
    (t.Annotated[int, cond2], 1, 1),
    (t.Annotated[int, cond2], 0, ConditionFailedError('an int > 0', 0., 'v > 0')),
    (t.Annotated[int, cond1, cond2], 0, ConditionFailedError('an int satisfying true and v > 0', 0., 'true and v > 0')),
    (t.Annotated[int, val_range(min=0)], 0, 0),
    (t.Annotated[int, val_range(min=0)], -1, ConditionFailedError('an int satisfying v >= 0', -1., 'v >= 0')),
    (t.Annotated[float, val_range(min=0, max=5)], 5, 5.),
    (t.Annotated[float, val_range(min=0, max=5)], 5.05, ConditionFailedError('a float satisfying v >= 0 and v <= 5', 5.05, 'v >= 0 and v <= 5')),
    (t.Annotated[t.Sequence[int], len_range(min=1)], [], ConditionFailedError('sequence of ints with at least 1 elem', [], 'at least 1 elem')),
    # tagged unions
    (t.Annotated[t.Union[Variant1, Variant2], Tagged('tag', True)], {3: 4}, Variant1(4)),
    (t.Annotated[t.Union[Variant1, Variant2], Tagged('tag', ('t', 'c'))], {'t': 4, 'c': 4.}, Variant2(4.)),
    (t.Annotated[t.Union[Variant1, Variant2], Tagged('tag', ('t', 'c'))], {'t': 5, 'c': 4.}, WrongTypeError("tag 'tag' one of 3 or 4", 5)),
    (t.Annotated[t.Union[Variant1, Variant2], Tagged('tag', ('t', 'c'))], {'c': 4.}, WrongTypeError("mapping with keys 't' and 'c'", {'c': 4.})),
    (t.Annotated[t.Union[Variant3, Variant4], Tagged('tag')], {'tag': 3, 'val1': 4, 'val2': 3.}, Variant3({'val1': 4, 'val2': 3.})),
    (t.Annotated[t.Union[Variant3, Variant4], Tagged('tag')], {'tag': 5, 'v': 4}, WrongTypeError("tag 'tag' one of 3 or 4", 5)),
    # regex patterns
    (re.Pattern, 'abcde', re.compile('abcde')),
    (re.Pattern[bytes], 'abcde', WrongTypeError('a bytes regex pattern', 'abcde')),
    (re.Pattern[bytes], b'abcde', re.compile(b'abcde')),
    (re.Pattern[str], '(', WrongTypeError('a string regex pattern', '(', cause=re.error('missing ), unterminated subpattern at position 0'))),
    (re.Pattern[bytes], re.compile(b'abcde'), re.compile(b'abcde')),
    # datetime (from str)
    (datetime.datetime, "2023-09-05 11:11:11", datetime.datetime(2023, 9, 5, 11, 11, 11)),
    (datetime.time, "11:11:11", datetime.time(11, 11, 11)),
    (datetime.date, "2023-09-05", datetime.date(2023, 9, 5)),
    (datetime.date, "11:11:11", WrongTypeError('a date', "11:11:11", cause=ValueError("Invalid isoformat string: '11:11:11'"))),
    (datetime.date, "09/05/2023", WrongTypeError('a date', "09/05/2023", cause=ValueError("Invalid isoformat string: '09/05/2023'"))),
    # TODO should we accept this?
    (datetime.time, "2023-09-05 11:11:11", WrongTypeError('a time', "2023-09-05 11:11:11", cause=ValueError("Invalid isoformat string: '2023-09-05 11:11:11'"))),
    # fraction/decimal
    (Decimal, 5, Decimal('5')),
    (Decimal, '5.123', Decimal('5.123')),
    (Decimal, 1.1, Decimal('1.100000000000000088817841970012523233890533447265625')),
    (Fraction, '1/5', Fraction(1, 5)),
    (Fraction, 5.1345, Fraction(5780933071683453, 1125899906842624)),
    (Fraction, '1/0', WrongTypeError('a fraction', '1/0', cause=ZeroDivisionError('Fraction(1, 0)'))),
    # path
    (pathlib.PurePosixPath, "/test/path", pathlib.PurePosixPath("/test/path")),
    (os.PathLike, "test/path", pathlib.PurePath("test/path")),
    # subclasses
    (Variant1, 5, Variant1(5)),
    (Variant1, '5', WrongTypeError('an int', '5')),
])
def test_convert(ty, val, result):
    if isinstance(result, ErrorNode):
        with pytest.raises(ConvertError) as exc_info:
            convert(val, ty)
        assert exc_info.value.tree == result
    else:
        assert convert(val, ty) == result
        assert make_converter(ty).collect_errors(val) is None

@pytest.mark.parametrize(('conv', 'val', 'result'), [
    # to time
    (datetime.time, datetime.time.fromisoformat("11:11:11"), datetime.time.fromisoformat("11:11:11")),
    (datetime.time, datetime.date.fromisoformat("2023-09-01"), WrongTypeError('a time', datetime.date.fromisoformat("2023-09-01"))),
    (datetime.time, datetime.datetime.fromisoformat("2023-09-01 11:11:11"), datetime.time.fromisoformat("11:11:11")),
    # to date
    (datetime.date, datetime.date.fromisoformat("2023-09-01"), datetime.date.fromisoformat("2023-09-01")),
    (datetime.date, datetime.time.fromisoformat("11:11:11"), WrongTypeError('a date', datetime.time.fromisoformat("11:11:11"))),
    (datetime.date, datetime.datetime.fromisoformat("2023-09-01 11:11:11"), datetime.date.fromisoformat("2023-09-01")),
    # to datetime
    (datetime.datetime, datetime.date.fromisoformat("2023-09-01"), datetime.datetime.fromisoformat("2023-09-01 00:00:00")),
    (datetime.datetime, datetime.time.fromisoformat("12:05:23"), WrongTypeError('a datetime', datetime.time.fromisoformat("12:05:23"))),
    (datetime.datetime, datetime.datetime.fromisoformat("2023-09-01 11:11:11"), datetime.datetime.fromisoformat("2023-09-01 11:11:11")),
])
def test_conv_convert(conv, val, result):
    # this test is useful for testing how Converters respond to
    # values other than data interchange values

    if not isinstance(conv, Converter):
        conv = make_converter(conv)

    if isinstance(result, ErrorNode):
        with pytest.raises(ConvertError) as exc_info:
            conv.convert(val)
        assert exc_info.value.tree == result
    else:
        assert conv.convert(val) == result
        assert conv.collect_errors(val) is None

# TODO test converter idempotence

@pytest.mark.parametrize(('err', 's'), [
    (WrongTypeError('an int', 3.), "Expected an int, instead got `3.0` of type `float`"),
    # TODO more
])
def test_error_print(err, s):
    assert str(err) == s


class DoubleIntConverter(Converter[int]):
    def into_data(self, val: t.Any) -> t.Any:
        return val // 2

    def expected(self, plural: bool = False) -> str:
        return 'ints' if plural else 'an int'

    def try_convert(self, val: t.Any) -> int:
        if isinstance(val, int):
            return val * 2
        raise ParseInterrupt()

    def collect_errors(self, val: t.Any) -> t.Optional[WrongTypeError]:
        if isinstance(val, int):
            return None
        return WrongTypeError(self.expected(), val)


class DoubleStrConverter(Converter[str]):
    def into_data(self, val: t.Any) -> t.Any:
        return val[:len(val)//2]

    def expected(self, plural: bool = False) -> str:
        return 'strings' if plural else 'a string'

    def try_convert(self, val: t.Any) -> str:
        if isinstance(val, str):
            return val * 2
        raise ParseInterrupt()

    def collect_errors(self, val: t.Any) -> t.Optional[WrongTypeError]:
        if isinstance(val, str):
            return None
        return WrongTypeError(self.expected(), val)


def manual_handler(ty: t.Any, args, handlers) -> Converter[t.Any]:
    if issubclass(ty, int):
        return DoubleIntConverter()

    return NotImplemented


def other_handler(ty: t.Any, args, handlers) -> Converter[t.Any]:
    if issubclass(ty, int):
        return make_converter(int, handlers=handlers)

    return NotImplemented


@pytest.mark.parametrize('custom', [
    manual_handler,
    (manual_handler, other_handler),
    {int: DoubleIntConverter()},
])
def test_custom_converter(custom):
    assert from_data(5, int, custom=custom) == 10
    assert into_data(10, custom=custom) == 5
    assert convert(10, int, custom=custom) == 10

    # make sure it works nested
    assert from_data([5, 10, 15], t.Set[int], custom=custom) == {10, 20, 30}
    assert into_data([10], list[int], custom=custom) == [5]