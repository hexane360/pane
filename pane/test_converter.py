from __future__ import annotations

import typing as t

import pytest

from .convert import convert, make_converter, ConvertError
from .convert import ScalarConverter, StructConverter, TupleConverter, SequenceConverter
from .convert import UnionConverter, LiteralConverter, Converter
from .convert import ErrorNode, SumErrorNode, ProductErrorNode, WrongTypeError
from .convert import ErrorNode


class TestConvertible():
    @classmethod
    def _converter(cls, *args,
                   annotations: t.Optional[t.Tuple[t.Any, ...]] = None) -> Converter[TestConvertible]:
        return TestConverter()  # type: ignore


class TestConverter(Converter[TestConvertible]):
    def try_convert(self, val: t.Any) -> t.NoReturn:
        raise NotImplementedError()

    def collect_errors(self, val: t.Any) -> t.NoReturn:
        raise NotImplementedError()

    def __hash__(self):
        return hash(self.__class__)

    def __eq__(self, other):
        return self.__class__ == other.__class__


@pytest.mark.parametrize(('input', 'conv'), [
    (int, ScalarConverter(int, int, 'an int')),
    ({'x': int, 'y': float}, StructConverter('dict', dict, {'x': int, 'y': float})),
    (t.Tuple[int, ...], SequenceConverter(tuple, int)),
    (list[str], SequenceConverter(list, str)),
    (t.Tuple[int, str], TupleConverter(tuple, (int, str))),
    (TestConvertible, TestConverter()),
    (t.Union[str, int], UnionConverter((str, int))),
    (t.Literal['a', 'b', 'c'], LiteralConverter(('a', 'b', 'c')))
])
def test_make_converter(input, conv: Converter):
    assert make_converter(input) == conv


@pytest.mark.parametrize(('ty', 'val', 'result'), [
    (int, 's', WrongTypeError('an int', 's')),
    ({'x': int, 'y': float}, {'x': 5, 'y': 4}, {'x': 5, 'y': 4.}),
    (t.Union[int, float, str], 5., 5.),
    (t.Union[str, float, int], 5, 5.),
    ({'x': t.Union[str, int], 'y': t.Tuple[t.Union[str, int], int]}, {'x': 5., 'y': (0., 's')},
     ProductErrorNode('dict', {
         'x': SumErrorNode([WrongTypeError('a str', 5.), WrongTypeError('an int', 5.)]),
         'y': ProductErrorNode('tuple of length 2', {
               0: SumErrorNode([WrongTypeError('a str', 0.), WrongTypeError('an int', 0.)]),
               1: WrongTypeError('an int', 's'),
         },  (0., 's')),
     }, {'x': 5., 'y': (0., 's')})
     )
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