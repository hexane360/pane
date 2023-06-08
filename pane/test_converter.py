from __future__ import annotations

import typing as t

import pytest

from .convert import convert, make_converter, ConvertError
from .converter import ScalarConverter, StructConverter, TupleConverter, SequenceConverter, UnionConverter, Converter
from .converter import ErrorNode, SumErrorNode, SimpleErrorNode, ProductErrorNode
from .converter import ErrorNode


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
    ({'x': int, 'y': float}, StructConverter('dict', {'x': int, 'y': float})),
    (t.Tuple[int, ...], SequenceConverter(tuple, int)),
    (list[str], SequenceConverter(list, str)),
    (t.Tuple[int, str], TupleConverter(tuple, (int, str))),
    (TestConvertible, TestConverter()),
    (t.Union[str, int], UnionConverter((str, int))),
])
def test_make_converter(input, conv: Converter):
    assert make_converter(input) == conv


@pytest.mark.parametrize(('ty', 'val', 'result'), [
    (int, 's', SimpleErrorNode('an int', 's')),
    ({'x': int, 'y': float}, {'x': 5, 'y': 4}, {'x': 5, 'y': 4.})
])
def test_convert(ty, val, result):
    if isinstance(result, ErrorNode):
        with pytest.raises(ConvertError) as exc_info:
            convert(val, ty)
        assert exc_info.value.tree == result
    else:
        assert convert(val, ty) == result