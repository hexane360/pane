import collections.abc
import typing as t

from typing_extensions import ParamSpec
import pytest

from pane.util import replace_typevars, collect_typevars, flatten_union_args, get_type_hints
from pane.util import resolve_type_aliases, TypeAliasType
from pane.util import pluralize, list_phrase, remove_article


T = t.TypeVar('T')
U = t.TypeVar('U')
K = t.TypeVar('K')
V = t.TypeVar('V')
P = ParamSpec('P')

# mock `type X = ...` syntax,
# type IntList = t.List[int]
IntList = TypeAliasType('IntList', t.List[int])
# type ListAlias[T] = t.List[T]
ListAlias = TypeAliasType('ListAlias', t.List[T], type_params=(T,))
# type NestedAlias[U] = t.Dict[str, ListAlias[U]]
NestedAlias = TypeAliasType('NestedAlias', t.Dict[str, ListAlias[U]], type_params=(U,))


@pytest.mark.parametrize(('input', 'output'), [
    (t.Dict[K, V], dict[int, V]),
    (t.Union[int, K], int),  # deduplicate union and extract single-member union
    (t.Union[int, U], t.Union[int, str]),  # flatten union and deduplicate, preserving order
    (t.Tuple[T, ...], tuple[t.Tuple[int, ...], ...]),
    (t.Tuple[T, U, K], tuple[t.Tuple[int, ...], t.Union[int, str], int]),
    (t.Callable[[int, T], U], collections.abc.Callable[[int, t.Tuple[int, ...]], t.Union[int, str]]),
    (t.Literal['test'], t.Literal['test']),
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
def test_collect_typevars(input, output):
    assert collect_typevars(input) == output


@pytest.mark.parametrize(('input', 'output'), [
    (int, int),  # non-alias types are returned unchanged
    (IntList, list[int]),  # non-generic alias
    (ListAlias[int], list[int]),  # generic alias
    (t.List[ListAlias[int]], list[list[int]]),  # alias nested inside another generic
    (NestedAlias[ListAlias[float]], dict[str, list[list[float]]]),  # alias nested inside another alias
])
def test_resolve_type_aliases(input, output):
    assert resolve_type_aliases(input) == output


def test_resolve_type_aliases_too_many_args():
    with pytest.raises(TypeError, match='Too many type arguments'):
        resolve_type_aliases(ListAlias[int, str])


class TestClass:
    x: int = 5
    y: float
    z: 'str'
    w: t.Annotated[int, ()]


@pytest.mark.parametrize(('cls', 'types'), [
    (TestClass, {'x': int, 'y': float, 'z': str, 'w': t.Annotated[int, ()]}),
])
def test_get_type_hints(cls, types):
    assert types == get_type_hints(cls)


@pytest.mark.parametrize(('input', 'output'), [
    ((int, float, str), (int, float, str)),
    ((t.Union[int, float], t.Union[float, str], T), (int, float, float, str, T)),
])
def test_flatten_union_args(input, output):
    assert tuple(flatten_union_args(input)) == output


@pytest.mark.parametrize(('input', 'conj', 'output'), [
    (('word',), None, 'word'),
    (('a', 'b'), 'xor', 'a xor b'),
    (('foo', 'bar', 'baz'), None, 'foo, bar, or baz'),
    (('foo', 'bar', 'baz'), 'and', 'foo, bar, and baz'),
])
def test_list_phrase(input, conj, output):
    if conj is not None:
        assert list_phrase(input, conj) == output
    else:
        assert list_phrase(input) == output


@pytest.mark.parametrize(('input', 'output'), [
    ('the a word', 'a word'),
    ('an article', 'article'),
    ('  the break', 'break'),
    ('article', 'article'),
])
def test_remove_article(input, output):
    assert remove_article(input) == output
