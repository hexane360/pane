from __future__ import annotations

import typing as t

import pytest

try:
    import numpy
    from numpy.typing import NDArray
    from numpy.testing import assert_array_equal

except ImportError:
    pytest.skip("skipping numpy tests", allow_module_level=True)

from pane.errors import WrongTypeError, ProductErrorNode, ErrorNode, ConditionFailedError
from pane.convert import convert, into_data, make_converter, ConvertError
from pane.converters import Converter, NestedSequenceConverter
from pane.annotations import broadcastable

@pytest.mark.parametrize(('input', 'conv'), [
    (numpy.ndarray, NestedSequenceConverter(t.Any, numpy.array, ragged=False)),
    (NDArray[numpy.generic], NestedSequenceConverter(numpy.generic, numpy.array, ragged=False)),
    (NDArray[numpy.int_], NestedSequenceConverter(numpy.int_, numpy.array, ragged=False)),
])
def test_make_converter_numpy(input, conv: Converter):
    assert make_converter(input) == conv


@pytest.mark.parametrize(('conv', 'plural', 'expected'), [
    (numpy.ndarray, False, 'a n-d array of any values'),
    (NestedSequenceConverter(int, numpy.array, ragged=False), True, 'n-d arrays of ints'),
    (NestedSequenceConverter(int, numpy.array, ragged=True), False, 'a nested sequence of ints'),
    (t.Annotated[NDArray[numpy.int_], broadcastable((2, 2))], False, 'a n-d array of ints broadcastable to (2, 2)'),
])
def test_numpy_expected(conv, plural, expected):
    if not isinstance(conv, Converter):
        conv = make_converter(conv)
    assert conv.expected(plural) == expected


@pytest.mark.parametrize(('ty', 'val', 'result'), [
     (NDArray[int], [[5, 6], [7, 8]], numpy.array([[5, 6], [7, 8]])),
     (NDArray[int], [[5, 6, 7], [7, 8]], WrongTypeError('a n-d array of ints', [[5, 6, 7], [7, 8]], info='shape mismatch at dim 0. Sub-shapes: [(3,), (2,)]')),
     (NDArray[numpy.float64], [1.+1j, 2.], ProductErrorNode('a n-d array of floats', {0: WrongTypeError('a float', 1.+1j)}, [1.+1j, 2.])),
     (t.Annotated[NDArray[numpy.int_], broadcastable((2, 2))], [1, 2], numpy.array([1, 2])),
     (t.Annotated[NDArray[numpy.int_], broadcastable((2, 2))], [1, 2, 3], ConditionFailedError('a n-d array of ints broadcastable to (2, 2)', [1, 2, 3], 'broadcastable to (2, 2)')),
])
def test_convert_numpy(ty, val, result):
    if isinstance(result, ErrorNode):
        with pytest.raises(ConvertError) as exc_info:
            convert(val, ty)
        assert exc_info.value.tree == result
    else:
        assert_array_equal(convert(val, ty), result)
        assert make_converter(ty).collect_errors(val) is None


@pytest.mark.parametrize(('ty', 'val', 'result'), [
    (NDArray[int], numpy.array([1, 2, 3, 4]), [1, 2, 3, 4]),
    (int, numpy.int32(5), 5),
    (complex, numpy.complex128(5.+3.j), 5.+3.j),
])
def test_into_data_numpy(ty, val, result):
    if isinstance(result, ErrorNode):
        with pytest.raises(ConvertError) as exc_info:
            into_data(val, ty)
        assert exc_info.value.tree == result
    else:
        actual = into_data(val, ty)
        assert actual == result
        assert type(actual) == type(result)