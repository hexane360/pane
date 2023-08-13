from __future__ import annotations

import typing as t

import pytest

from pane.convert import convert, make_converter, ConvertError
from pane.types import PositiveInt, ListNotEmpty, ValueOrList, Range
from pane.errors import ConvertError, ProductErrorNode, SumErrorNode, WrongTypeError, WrongLenError, ConditionFailedError, ErrorNode


@pytest.mark.parametrize(('ty', 'val', 'result'), [
    (PositiveInt, 5, 5),
    (PositiveInt, -5, ConditionFailedError('a positive int', -5, 'positive')),
    (ListNotEmpty[int], [], ConditionFailedError('sequence of ints with at least 1 elem', [], 'at least 1 elem')),
    (Range[float], (2., 4., 2), Range[t.Any](2., 4., 2)),
    (Range[float], (2., 4., 2, 5, 8), WrongLenError('tuple Range', (2, 3), (2., 4., 2, 5, 8), 5)),
    (Range[float], {'start': 2., 'end': 4., 'step': 1.}, Range[t.Any](start=2., end=4., step=1.)),
    (Range[float], {'start': 2., 'end': 4.}, WrongTypeError('struct Range', {'start': 2., 'end': 4.}, cause=TypeError("Either 'n' or 'step' must be specified"))),
    (Range[float], {'start': 2., 'end': 4., 'n': 1, 'step': 3.}, WrongTypeError('struct Range', {'start': 2., 'end': 4., 'n': 1, 'step': 3.}, cause=TypeError("Either 'n' or 'step' may be specified, but not both"))),
    (ValueOrList[int], 5, ValueOrList.from_val(5)),
    (ValueOrList[int], (5, 6, 7), ValueOrList.from_list([5, 6, 7])),
    (ValueOrList[int], (5, 6.5, 7), SumErrorNode([WrongTypeError('an int', (5, 6.5, 7)), ProductErrorNode('sequence of ints', {1: WrongTypeError('an int', 6.5)}, (5, 6.5, 7))])),
    (ValueOrList[int], {}, SumErrorNode([WrongTypeError('an int', {}), WrongTypeError('sequence of ints', {})])),
])
def test_convert_types(ty, val, result):
    if isinstance(result, ErrorNode):
        with pytest.raises(ConvertError) as exc_info:
            convert(val, ty)
        assert exc_info.value.tree == result
    else:
        assert convert(val, ty) == result
        assert make_converter(ty).collect_errors(val) is None
