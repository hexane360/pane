import typing as t

import pytest

from .util import replace_typevars


T = t.TypeVar('T')
U = t.TypeVar('U')
K = t.TypeVar('K')
V = t.TypeVar('V')


@pytest.mark.parametrize(('input', 'output'), [
    (t.Dict[K, V], dict[int, V]),
    (t.Union[int, K], int),  # deduplicate union and extract single-member union
    (t.Union[int, U], t.Union[int, str]),  # flatten union and deduplicate, preserving order
    (t.Tuple[T, ...], tuple[t.Tuple[int, ...], ...]),
    (t.Tuple[T, U, K], tuple[t.Tuple[int, ...], t.Union[int, str], int])
])
def test_replace_typevars(input, output):
    replacements = {
        K: int,
        T: t.Tuple[int, ...],
        U: t.Union[int, str]
    }
    assert replace_typevars(input, replacements) == output