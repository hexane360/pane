
import sys
from pathlib import Path
from io import TextIOBase, IOBase, TextIOWrapper, BufferedIOBase
from contextlib import AbstractContextManager, nullcontext
from itertools import chain
import typing as t


T = t.TypeVar('T')
FileOrPath = t.Union[str, Path, TextIOBase, t.TextIO]


def _validate_file(f: t.Union[t.IO[t.AnyStr], IOBase], mode: t.Union[t.Literal['r'], t.Literal['w']]):
    if f.closed:
        raise IOError("Error: Provided file is closed.")

    if mode == 'r':
        if not f.readable():
            raise IOError("Error: Provided file not readable.")
    elif mode == 'w':
        if not f.writable():
            raise IOError("Error: Provided file not writable.")


def open_file(f: FileOrPath,
              mode: t.Literal['r', 'w'] = 'r',
              newline: t.Optional[str] = None,
              encoding: t.Optional[str] = 'utf-8') -> AbstractContextManager[TextIOBase]:
    """
    Open the given file for text I/O.

    If given a path-like, opens it with the specified settings.
    Otherwise, make an effort to reconfigure the encoding, and
    check that it is readable/writable as specified.
    """
    if not isinstance(f, (IOBase, t.BinaryIO, t.TextIO)):
        return open(f, mode, newline=newline, encoding=encoding)

    if isinstance(f, TextIOWrapper):
        f.reconfigure(newline=newline, encoding=encoding)
    elif isinstance(f, t.TextIO):
        f = TextIOWrapper(f.buffer, newline=newline, encoding=encoding)
    elif isinstance(f, (BufferedIOBase, t.BinaryIO)):
        f = TextIOWrapper(t.cast(t.BinaryIO, f), newline=newline, encoding=encoding)

    _validate_file(f, mode)
    return nullcontext(f)  # don't close a f we didn't open


def partition(f: t.Callable[[T], bool], iter: t.Iterable[T]) -> t.Tuple[t.Tuple[T, ...], t.Tuple[T, ...]]:
    """Partition ``iter`` into values that satisfy ``f`` and those which don't."""
    true: t.List[T] = []
    false: t.List[T] = []
    for val in iter:
        if f(val):
            true.append(val)
        else:
            false.append(val)
    return (tuple(true), tuple(false))


def pluralize(word: str, plural: t.Union[bool, int], suffix: str = 's'):
    """Pluralize ``word`` based on the value of ``plural``."""
    if not isinstance(plural, bool):
        plural = plural != 1
    return (word + suffix) if plural else word


def list_phrase(words: t.Sequence[str], conj: str = 'or') -> str:
    """
    Form an english list phrase from ``words``, using the conjunction ``conj``.
    """
    if len(words) <= 2:
        return f" {conj} ".join(words)
    return ", ".join(words[:-1]) + f", {conj} {words[-1]}"


def remove_article(s: str) -> str:
    """Remove an article from ``s``, if present."""
    return s.removeprefix('a ').removeprefix('an ').removeprefix('the ')


def _collect_typevars(d: t.Dict[t.Union[t.TypeVar, t.ParamSpec], None], ty: t.Any):
    if isinstance(ty, type):
        pass
    elif isinstance(ty, (tuple, t.Sequence)):
        ty = t.cast(t.Sequence[t.Any], ty)
        for arg in ty:
            _collect_typevars(d, arg)
    elif hasattr(ty, '__typing_subst__') or isinstance(ty, (t.TypeVar, t.ParamSpec)):
        d.setdefault(ty)
    else:
        for ty in getattr(ty, '__parameters__', ()):
            d.setdefault(ty)


def collect_typevars(args: t.Any) -> tuple[t.Union[t.TypeVar, t.ParamSpec]]:
    # loosely based on typing._collect_parameters
    d: t.Dict[t.Union[t.TypeVar, t.ParamSpec], None] = {}  # relies on dicts preserving insertion order
    _collect_typevars(d, args)
    return tuple(d)


def _union_args(ty: type) -> t.Sequence[type]:
    base = t.get_origin(ty) or ty
    args = t.get_args(ty)

    if base is t.Union:
        return args

    return (ty,)


def replace_typevars(ty: t.Any,
                     replacements: t.Mapping[t.Union[t.TypeVar, t.ParamSpec], type]) -> t.Any:
    if isinstance(ty, (t.TypeVar, t.ParamSpec)):
        return replacements.get(ty, ty)
    if isinstance(ty, t.Sequence):
        return type(ty)(replace_typevars(t, replacements) for t in ty)  # type: ignore

    base = t.get_origin(ty) or ty
    args = t.get_args(ty)

    if not len(args):
        return ty

    args = (replace_typevars(ty, replacements) for ty in args)

    if base is t.Union:
        # deduplicate union
        args = tuple(chain.from_iterable(map(_union_args, args)))
        args = dict.fromkeys(args).keys()

        if len(args) == 1:
            # single-element union, return as value
            return next(iter(args))

    return base[tuple(args)]  # type: ignore


def get_type_hints(cls: type) -> t.Dict[str, t.Any]:
    # modified version of typing.get_type_hints

    globalns = getattr(sys.modules.get(cls.__module__, None), '__dict__', {})
    localns = dict(vars(cls))

    d: t.Dict[str, t.Any] = {}
    for name, value in cls.__dict__.get('__annotations__', {}).items():
        if value is None:
            value = type(None)
        if isinstance(value, str):
            value = t.ForwardRef(value, is_argument=False, is_class=True)
        # private access inside typing.
        value = t._eval_type(value, globalns, localns)  # type: ignore
        d[name] = value

    return d


__all__ = [
    'open_file', 'list_phrase', 'pluralize',
    'collect_typevars', 'replace_typevars', 'get_type_hints'
]
