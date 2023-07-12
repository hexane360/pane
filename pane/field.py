from __future__ import annotations

from dataclasses import dataclass, replace
import itertools
import re
import typing as t

from typing_extensions import Self

from .util import replace_typevars


class _Missing:
    pass


T = t.TypeVar('T')
_MISSING = _Missing()


RenameStyle = t.Literal['snake', 'camel', 'pascal', 'kebab', 'scream']


CONVERT_FNS: t.Dict[RenameStyle, t.Callable[[t.Sequence[str]], str]] = {
    'snake': lambda parts: '_'.join(part.lower() for part in parts),
    'scream': lambda parts: '_'.join(part.upper() for part in parts),
    'kebab': lambda parts: '-'.join(part.lower() for part in parts),
    'camel': lambda parts: ''.join(part.lower() if i == 0 else part.title() for (i, part) in enumerate(parts)),
    'pascal': lambda parts: ''.join(part.title() for part in parts),
}


def _pairwise(it: t.Iterable[T]) -> t.Iterator[t.Tuple[T, T]]:
    it = iter(it)
    while True:
        try:
            a = next(it)
            b = next(it)
        except StopIteration:
            break
        yield (a, b)


def _split_field_name(field: str) -> t.Sequence[str]:
    parts = re.split(r'[_-]', field)
    if not all(parts):
        raise ValueError(f"Unable to interpret field '{field}' for automatic rename")

    def split_case(field: str):
        if field.isupper() or field.islower() or field.istitle():
            yield field
            return
        seps = re.split(r'([A-Z])', field)
        if seps[0] != '':
            yield seps[0]
        for (s1, s2) in _pairwise(seps[1:]):
            yield s1 + s2

    return tuple(itertools.chain.from_iterable(map(split_case, parts)))


def rename_field(field: str, style: RenameStyle) -> str:
    return CONVERT_FNS[style](_split_field_name(field))


@dataclass(kw_only=True)
class Field:
    name: str
    type: type
    in_names: t.Sequence[str]
    out_name: str
    init: bool = True
    default: t.Union[t.Any, _Missing] = _MISSING
    default_factory: t.Optional[t.Callable[[], t.Any]] = None
    kw_only: bool = False

    @classmethod
    def make(cls, name: str, ty: type,
             in_rename: t.Optional[t.Sequence[RenameStyle]] = None,
             out_rename: t.Optional[RenameStyle] = None) -> Field:
        in_names = tuple(rename_field(name, style) for style in in_rename) if in_rename is not None else (name,)
        out_name = rename_field(name, out_rename) if out_rename is not None else name
        return cls(name=name, type=ty, in_names=in_names, out_name=out_name)

    def is_optional(self) -> bool:
        return self.default is not _MISSING or self.default_factory is not None


@dataclass(kw_only=True)
class FieldSpec:
    rename: t.Optional[str] = None
    in_names: t.Optional[t.Sequence[str]] = None
    aliases: t.Optional[t.Sequence[str]] = None
    out_name: t.Optional[str] = None
    init: bool = True
    default: t.Union[t.Any, _Missing] = _MISSING
    default_factory: t.Optional[t.Callable[[], t.Any]] = None
    kw_only: bool = False
    ty: t.Union[type, _Missing] = _MISSING

    def __post_init__(self):
        if isinstance(self.aliases, str):
            self.aliases = [self.aliases]

    def replace_typevars(self, replacements: t.Mapping[t.Union[t.TypeVar, t.ParamSpec], t.Type[t.Any]]) -> Self:
        if self.ty is _MISSING:
            return replace(self)
        return replace(self, ty=replace_typevars(t.cast(type, self.ty), replacements))

    def make_field(self, name: str,
                   in_rename: t.Optional[t.Sequence[RenameStyle]] = None,
                   out_rename: t.Optional[RenameStyle] = None) -> Field:
        # out_name
        if self.out_name is not None:
            out_name = self.out_name
        elif self.rename is not None:
            out_name = self.rename
        else:
            out_name = rename_field(name, out_rename) if out_rename is not None else name

        if sum(p is not None for p in (self.rename, self.aliases, self.in_names)) > 1:
            raise TypeError("Can only specify one of 'rename', 'aliases', and 'in_names'")

        if self.rename is not None:
            in_names = (self.rename,)
        elif self.aliases is not None:
            in_names = (name, *(alias for alias in self.aliases if alias != name))
        elif self.in_names is not None:
            in_names = self.in_names
        else:
            in_names = tuple(rename_field(name, style) for style in in_rename) if in_rename is not None else (name,)

        ty = t.cast(type, t.Any if self.ty is _MISSING else self.ty)
        return Field(name=name, type=ty, out_name=out_name, in_names=in_names,
                     init=self.init, default=self.default, default_factory=self.default_factory,
                     kw_only=self.kw_only)


# TODO overloads here
def field(*,
    rename: t.Optional[str] = None,
    in_names: t.Optional[t.Sequence[str]] = None,
    aliases: t.Optional[t.Sequence[str]] = None,
    out_name: t.Optional[str] = None,
    init: bool = True,
    default: t.Union[T, _Missing] = _MISSING,
    default_factory: t.Optional[t.Callable[[], T]] = None,
    kw_only: bool = False,
) -> t.Any:
    return FieldSpec(
        rename=rename, in_names=in_names, aliases=aliases, out_name=out_name,
        init=init, default=default, default_factory=default_factory, kw_only=kw_only
    )


__all__ = [
    'Field', 'FieldSpec', 'field',
    'rename_field',
]
