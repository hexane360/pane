import collections.abc
import typing as t

import pytest

from .util import replace_typevars, collect_typevars


T = t.TypeVar('T')
U = t.TypeVar('U')
K = t.TypeVar('K')
V = t.TypeVar('V')
P = t.ParamSpec('P')


@pytest.mark.parametrize(('input', 'output'), [
    (t.Dict[K, V], dict[int, V]),
    (t.Union[int, K], int),  # deduplicate union and extract single-member union
    (t.Union[int, U], t.Union[int, str]),  # flatten union and deduplicate, preserving order
    (t.Tuple[T, ...], tuple[t.Tuple[int, ...], ...]),
    (t.Tuple[T, U, K], tuple[t.Tuple[int, ...], t.Union[int, str], int]),
    (t.Callable[[int, T], U], collections.abc.Callable[[int, t.Tuple[int, ...]], t.Union[int, str]]),
])
def test_replace_typevars(input, output):
    replacements: t.Dict[t.Union[t.TypeVar, t.ParamSpec], type] = {
        K: int,
        T: t.Tuple[int, ...],
        U: t.Union[int, str]
    }
    assert replace_typevars(input, replacements) == output


@pytest.mark.parametrize(('input', 'output'), [
    ((T, t.Callable[P, T]), (T, P)),
    (t.Tuple[int, float, T], (T,))
])
def tset_collect_typevars(input, output):
    assert collect_typevars(input) == output