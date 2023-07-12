from __future__ import annotations

import typing as t

import pytest

from .errors import ErrorNode, SumErrorNode, ProductErrorNode, WrongTypeError, ConditionFailedError
from .convert import convert, make_converter, ConvertError
from .converters import Converter, ScalarConverter, TupleConverter, SequenceConverter
from .converters import StructConverter, UnionConverter, LiteralConverter, ConditionalConverter
from .annotations import Condition, range, len_range


class TestConvertible():
    @classmethod
    def _converter(cls, *args,
                   annotations: t.Optional[t.Tuple[t.Any, ...]] = None) -> Converter[TestConvertible]:
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
    (int, ScalarConverter(int, int, 'an int', 'ints')),
    ({'x': int, 'y': float}, StructConverter(dict, {'x': int, 'y': float})),
    (t.Tuple[int, ...], SequenceConverter(tuple, int)),
    (list[str], SequenceConverter(list, str)),
    (t.Tuple[int, str], TupleConverter(tuple, (int, str))),
    (TestConvertible, TestConverter()),
    (t.Union[str, int], UnionConverter((str, int))),
    (t.Literal['a', 'b', 'c'], LiteralConverter(('a', 'b', 'c'))),
    (t.Literal['a'], LiteralConverter(('a',))),
])
def test_make_converter(input, conv: Converter):
    assert make_converter(input) == conv


cond1 = Condition(lambda v: True, 'true', lambda exp, plural: exp)
cond2 = Condition(lambda v: v > 0, 'v > 0', lambda exp, plural: f"{exp} > 0")


def test_make_converter_annotated():
    inner = int
    inner_conv = ScalarConverter(int, int, 'an int', 'ints')

    assert make_converter(t.Annotated[int, cond1]) == ConditionalConverter(
        inner_conv, cond1.f, 'true', cond1.make_expected
    )

    assert make_converter(t.Annotated[int, cond2]) == ConditionalConverter(
        inner_conv, cond2.f, 'v > 0', cond2.make_expected
    )

    compound = make_converter(t.Annotated[int, cond1, cond2])
    assert isinstance(compound, ConditionalConverter)
    assert compound.condition_name == 'true and v > 0'



@pytest.mark.parametrize(('conv', 'plural', 'expected'), [
    (int, False, 'an int'),
    (t.Optional[str], True, 'strings or null values'),
    (t.Sequence[str], True, 'sequences of strings'),
    (t.Dict[str, int], False, 'mapping of strings => ints'),
    (t.Literal['a', 'b', 'c'], False, "'a', 'b', or 'c'"),
    (t.Annotated[int, cond2], False, "an int > 0"),
    (t.Annotated[int, cond2], True, "ints > 0"),
])
def test_converter_expected(conv: Converter, plural: bool, expected: str):
    if not isinstance(conv, Converter):
        conv = make_converter(conv)
    assert conv.expected(plural) == expected


@pytest.mark.parametrize(('ty', 'val', 'result'), [
    # scalar converter
    (int, 's', WrongTypeError('an int', 's')),
    # sequence converters
    (t.Tuple[int, ...], [1, 2], (1, 2)),
    (t.Sequence[int], [1, 2], (1, 2)),  # same as tuple
    (t.List[int], (1, 2), [1, 2]),
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
     (t.Annotated[int, range(min=0)], 0, 0),
     (t.Annotated[int, range(min=0)], -1, ConditionFailedError('an int satisfying v >= 0', -1., 'v >= 0')),
     (t.Annotated[float, range(min=0, max=5)], 5, 5.),
     (t.Annotated[float, range(min=0, max=5)], 5.05, ConditionFailedError('a float satisfying v >= 0 and v <= 5', 5.05, 'v >= 0 and v <= 5')),
     (t.Annotated[t.Sequence[int], len_range(min=1)], [], ConditionFailedError('sequence of ints with at least 1 elem', [], 'at least 1 elem')),
])
def test_convert(ty, val, result):
    if isinstance(result, ErrorNode):
        with pytest.raises(ConvertError) as exc_info:
            convert(val, ty)
        assert exc_info.value.tree == result
    else:
        assert convert(val, ty) == result
        assert make_converter(ty).collect_errors(val) is None


@pytest.mark.parametrize(('err', 's'), [
    (WrongTypeError('an int', 3.), "Expected an int, instead got `3.0` of type `float`"),
])
def test_error_print(err, s):
    assert str(err) == s