import sys

import functools
from itertools import zip_longest
import operator
import typing as t
from threading import RLock
from typing_extensions import ParamSpec


T = t.TypeVar('T')
P = ParamSpec('P')


# mock KW_ONLY on python <3.10
try:
    from dataclasses import KW_ONLY
except ImportError:
    class _KW_ONLY_TYPE:
        pass
    KW_ONLY = _KW_ONLY_TYPE()


def partition(f: t.Callable[[T], bool], iter: t.Iterable[T]) -> t.Tuple[t.Tuple[T, ...], t.Tuple[T, ...]]:
    """Partition `iter` into values that satisfy `f` and those which don't."""
    true: t.List[T] = []
    false: t.List[T] = []
    for val in iter:
        if f(val):
            true.append(val)
        else:
            false.append(val)
    return (tuple(true), tuple(false))


def pluralize(word: str, plural: t.Union[bool, int], suffix: str = 's', article: t.Optional[str] = None) -> str:
    """Pluralize `word` based on the value of `plural`."""
    if not isinstance(plural, bool):
        plural = plural != 1
    article = article + " " if article is not None and len(article) else ""
    return (word + suffix) if plural else (article + word)


def list_phrase(words: t.Sequence[str], conj: str = 'or') -> str:
    """
    Form an english list phrase from `words`, using the conjunction `conj`.
    """
    if len(words) <= 2:
        return f" {conj} ".join(words)
    return ", ".join(words[:-1]) + f", {conj} {words[-1]}"


def remove_article(s: str) -> str:
    """Remove an article from `s`, if present."""
    s = s.lstrip()
    for article in ('a ', 'an ', 'the '):
        if s.startswith(article):
            return s[len(article):]
    return s


def _collect_typevars(d: t.Dict[t.Union[t.TypeVar, ParamSpec], None], ty: t.Any):
    if isinstance(ty, type):
        pass
    elif isinstance(ty, (tuple, t.Sequence)):
        ty = t.cast(t.Sequence[t.Any], ty)
        for arg in ty:
            _collect_typevars(d, arg)
    elif hasattr(ty, '__typing_subst__') or isinstance(ty, (t.TypeVar, ParamSpec)):
        d.setdefault(ty)
    else:
        for ty in getattr(ty, '__parameters__', ()):
            d.setdefault(ty)


def collect_typevars(args: t.Any) -> t.Tuple[t.Union[t.TypeVar, ParamSpec], ...]:
    """
    Collect a list of type variables in `args`

    Preserves order but removes duplicates (i.e. type variables are returned
    in the order they are encountered, but no type variable is returned twice).

    Loosely based on `typing._collect_parameters`.
    """
    d: t.Dict[t.Union[t.TypeVar, ParamSpec], None] = {}  # relies on dicts preserving insertion order
    _collect_typevars(d, args)
    return tuple(d)


def type_union(types: t.Iterable[type]) -> type:
    return functools.reduce(operator.or_, types)


def flatten_union_args(types: t.Iterable[T]) -> t.Iterator[T]:
    """Flatten nested unions, returning a single sequence of possible union types."""
    for ty in types:
        if t.get_origin(ty) is t.Union:
            yield from flatten_union_args(t.get_args(ty))
        else:
            yield ty


def replace_typevars(ty: t.Any,
                     replacements: t.Mapping[t.Union[t.TypeVar, ParamSpec], type]) -> t.Any:
    """
    Apply a list of type-variable replacements to `ty`, and return the modified type.
    """
    if isinstance(ty, (t.TypeVar, ParamSpec)):
        return replacements.get(ty, ty)
    if isinstance(ty, t.Sequence) and not isinstance(ty, (str, bytes)):
        return type(ty)(replace_typevars(t, replacements) for t in ty)  # type: ignore

    base = t.get_origin(ty) or ty
    args = t.get_args(ty)

    if not len(args):
        return ty

    args = (replace_typevars(ty, replacements) for ty in args)

    if base is t.Union:
        args = tuple(flatten_union_args(args))
        # deduplicate union
        args = dict.fromkeys(args).keys()

        if len(args) == 1:
            # single-element union, return as value
            return next(iter(args))

    return base[tuple(args)]  # type: ignore


def get_type_hints(cls: type) -> t.Dict[str, t.Any]:
    """
    Extract a dict of type hints from `cls`. Evaluate forward refs if possible.

    This is a slightly modified version of [typing.get_type_hints]().
    """

    globalns = getattr(sys.modules.get(cls.__module__, None), '__dict__', {})
    localns = dict(vars(cls))

    d: t.Dict[str, t.Any] = {}
    for name, value in cls.__dict__.get('__annotations__', {}).items():
        if value is None:
            value = type(None)
        if isinstance(value, str):
            value = t.ForwardRef(value, is_argument=False, is_class=True)
        if isinstance(value, t.ForwardRef):
            # hack to handle top-level KW_ONLY
            val = value.__forward_value__ if value.__forward_evaluated__ else eval(value.__forward_code__, globalns, localns)
            if val is KW_ONLY:
                d[name] = KW_ONLY
                continue
        # private access inside typing module
        value = t._eval_type(value, globalns, localns)  # type: ignore
        d[name] = value

    return d


def broadcast_shapes(*args: t.Sequence[int]) -> t.Tuple[int, ...]:
    """
    Attempt to broadcast the given shapes together using numpy semantics.

    Defers to `numpy.broadcast_shapes` if numpy is available.
    """
    try:
        import numpy
        return numpy.broadcast_shapes(*map(tuple, args))
    except ImportError:
        pass

    # our own implementation, with worse error messages
    out_shape: t.List[int] = []
    for ax_lens in zip_longest(*(reversed(arg) for arg in args), fillvalue=1):
        bcast = max(ax_lens)
        if not all(ax_len in (1, bcast) for ax_len in ax_lens):
            shapes = [f"'{tuple(arg)!r}'" for arg in args]
            raise ValueError(f"Couldn't broadcast shapes {list_phrase(shapes, 'and')}")
        out_shape.append(bcast)
    return tuple(out_shape)


def is_broadcastable(*args: t.Sequence[int]) -> bool:
    """Return whether `args` are broadcastable together using numpy semantics."""
    try:
        broadcast_shapes(*args)
        return True
    except ValueError:
        return False


PREV, NEXT, KEY, RESULT = 0, 1, 2, 3

class KeyCache(t.Generic[P, T]):
    _missing = object()

    def __init__(self, f: t.Callable[P, T], key_f: t.Callable[P, t.Any], maxsize: t.Optional[int] = None):
        self.maxsize: t.Optional[int] = maxsize
        self.key_f: t.Callable[P, t.Any] = key_f
        self.inner_f: t.Callable[P, T] = f
        self.cache: t.Dict[t.Tuple[t.Tuple[t.Any, ...], t.Tuple[t.Tuple[str, t.Any], ...]], t.Any] = {}

        self._root: t.List[t.Any] = []
        self._root[:] = [self._root, self._root, None, None]
        self._lock = RLock()

        self.full = self.maxsize == 0

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        if self.maxsize is None:
            key = self.key_f(*args, **kwargs)
            result = self.cache.get(key, self._missing)
            if result is not self._missing:
                return t.cast(T, result)
            result = self.inner_f(*args, **kwargs)
            self.cache[key] = result
            return result

        key = self.key_f(*args, **kwargs)
        with self._lock:
            link = self.cache.get(key, None)
            if link is not None:
                # extract this link
                prev_link, next_link, _key, result = link
                prev_link[NEXT] = next_link
                next_link[PREV] = prev_link

                # and move it to the end of the list
                last = self._root[PREV]
                last[NEXT] = self._root[PREV] = link
                link[PREV] = last
                link[NEXT] = self._root
                return t.cast(T, result)

        result = self.inner_f(*args, **kwargs)
        with self._lock:
            if key in self.cache:
                pass
            elif self.full:
                # turn the oldest link into the new root
                # and reuse oldroot on the end of the list
                oldroot = self._root
                oldroot[KEY] = key
                oldroot[RESULT] = result

                self._root = oldroot[NEXT]
                oldkey = self._root[KEY]
                oldresult = self._root[RESULT]  # type: ignore # noqa: F841 (we want to keep this around for a bit)
                self._root[KEY] = self._root[RESULT] = None
                del self.cache[oldkey]
                self.cache[key] = oldroot
            else:
                last = self._root[PREV]
                link = [last, self._root, key, result]
                last[NEXT] = self._root[PREV] = self.cache[key] = link
                self.full = (len(self.cache) >= self.maxsize)
        return result


# TODO support maxsize
def key_cache(key_f: t.Callable[P, t.Any], *, maxsize: t.Optional[int] = None) -> t.Callable[[t.Callable[P, T]], KeyCache[P, T]]:
    def inner(f: t.Callable[P, T]) -> KeyCache[P, T]:
        return t.cast(KeyCache[P, T], functools.update_wrapper(KeyCache(f, key_f, maxsize), f))

    return inner


__all__ = [
    'list_phrase', 'pluralize', 'remove_article',
    'flatten_union_args', 'collect_typevars', 'replace_typevars', 'get_type_hints',
    'KW_ONLY',
]
