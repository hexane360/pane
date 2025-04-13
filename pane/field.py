from __future__ import annotations

import dataclasses
import itertools
import re
import typing as t

from typing_extensions import ParamSpec, Self, TypeAlias

from .util import replace_typevars, KW_ONLY
from .converters import Converter


class _Missing:
    pass


T = t.TypeVar('T')
_MISSING = _Missing()


RenameStyle: TypeAlias = t.Literal['snake', 'camel', 'pascal', 'kebab', 'scream']
"""List of supported field-renaming styles"""


_CONVERT_FNS: t.Dict[RenameStyle, t.Callable[[t.Sequence[str]], str]] = {
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
    """Split `field` into parts for renaming"""
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


def rename_field(field: str, style: t.Optional[RenameStyle] = None) -> str:
    """
    Rename `field` to match style `style`.

    Parameters:
        field: Field name to rename
        style: Style to match
    """
    if style is None:
        return field

    return _CONVERT_FNS[style](_split_field_name(field))


@dataclasses.dataclass
class Field:
    """
    Represents a materialized dataclass field.

    Typically instantiated from a [`FieldSpec`][pane.field.FieldSpec].
    """

    _: KW_ONLY = dataclasses.field(init=False, repr=False, compare=False)
    name: str
    """Name of field"""
    type: t.Any
    """Type of field. Must be [`Convertible`][pane.convert.Convertible]."""
    in_names: t.Sequence[str]
    """List of names which convert to this field."""
    out_name: str
    """Name this field converts into."""
    init: bool = True
    """Whether to add this field to __init__ methods (and conversion)"""
    default: t.Union[t.Any, _Missing] = _MISSING
    """Default value for field"""
    default_factory: t.Optional[t.Callable[[], t.Any]] = None
    """Default value factory for field"""
    kw_only: bool = False
    """Whether field is keyword only"""
    converter: t.Optional[Converter[t.Any]] = None
    """Custom converter to use for this field."""

    @classmethod
    def make(cls, name: str, ty: type,
             in_rename: t.Optional[t.Sequence[RenameStyle]] = None,
             out_rename: t.Optional[RenameStyle] = None) -> Field:
        in_names = tuple(rename_field(name, style) for style in in_rename) if in_rename is not None else (name,)
        out_name = rename_field(name, out_rename) if out_rename is not None else name
        return cls(name=name, type=ty, in_names=in_names, out_name=out_name)

    def has_default(self) -> bool:
        """Return whether this field has a default value"""
        return self.default is not _MISSING or self.default_factory is not None


@dataclasses.dataclass
class FieldSpec:
    """
    Represents a field specification.

    This hasn't been applied to a class yet, so some information is missing.

    In most cases, end users should use the [`field()`][pane.field.field] function instead.
    """

    _: KW_ONLY = dataclasses.field(init=False, repr=False, compare=False)
    rename: t.Optional[str] = None
    """Rename this field. Affects both `in_names` and `out_name`."""
    in_names: t.Optional[t.Sequence[str]] = None
    """Complete list of names which convert to this field."""
    aliases: t.Optional[t.Sequence[str]] = None
    """Additional list of names which convert to this field (excluding the name in Python)."""
    out_name: t.Optional[str] = None
    """Name this field converts into."""
    init: bool = True
    """Whether to add this field to __init__ methods (and conversion)"""
    default: t.Union[t.Any, _Missing] = _MISSING
    """Default value for field"""
    default_factory: t.Optional[t.Callable[[], t.Any]] = None
    """Default value factory for field"""
    kw_only: bool = False
    """Whether field is keyword only"""
    ty: t.Union[t.Any, _Missing] = _MISSING
    """Type of field, if known. Must be Convertible."""
    converter: t.Optional[Converter[t.Any]] = None
    """Custom converter to use for this field."""

    def __post_init__(self):
        if isinstance(self.aliases, str):
            self.aliases = [self.aliases]

    def replace_typevars(self, replacements: t.Mapping[t.Union[t.TypeVar, ParamSpec], t.Type[t.Any]]) -> Self:
        """
        Apply type variable replacements to `self`.
        """
        if self.ty is _MISSING:
            return dataclasses.replace(self)
        return dataclasses.replace(self, ty=replace_typevars(t.cast(type, self.ty), replacements))

    def make_field(self, name: str,
                   in_rename: t.Optional[t.Sequence[RenameStyle]] = None,
                   out_rename: t.Optional[RenameStyle] = None) -> Field:
        """
        Make a [`Field`][pane.field.Field] from this [`FieldSpec`][pane.field.FieldSpec].
        """
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
                     kw_only=self.kw_only, converter=self.converter)


# only allow one of rename, in_names, and aliases
@t.overload
def field(*,
    rename: t.Optional[str] = None,
    in_names: None = None,
    aliases: None = None,
    out_name: t.Optional[str] = None,
    init: bool = True,
    default: t.Union[T, _Missing] = _MISSING,
    default_factory: t.Optional[t.Callable[[], T]] = None,
    kw_only: bool = False,
    converter: t.Optional[Converter[T]] = None,
) -> t.Any:
    ...

@t.overload
def field(*,
    rename: None = None,
    in_names: t.Sequence[str],
    aliases: None = None,
    out_name: t.Optional[str] = None,
    init: bool = True,
    default: t.Union[T, _Missing] = _MISSING,
    default_factory: t.Optional[t.Callable[[], T]] = None,
    kw_only: bool = False,
    converter: t.Optional[Converter[T]] = None,
) -> t.Any:
    ...

@t.overload
def field(*,
    rename: None = None,
    in_names: None = None,
    aliases: t.Sequence[str],
    out_name: t.Optional[str] = None,
    init: bool = True,
    default: t.Union[T, _Missing] = _MISSING,
    default_factory: t.Optional[t.Callable[[], T]] = None,
    kw_only: bool = False,
    converter: t.Optional[Converter[T]] = None,
) -> t.Any:
    ...

@t.overload
def field(*,
    rename: t.Optional[str] = None,
    in_names: t.Optional[t.Sequence[str]] = None,
    aliases: t.Optional[t.Sequence[str]] = None,
    out_name: t.Optional[str] = None,
    init: bool = True,
    default: t.Union[T, _Missing] = _MISSING,
    default_factory: t.Optional[t.Callable[[], T]] = None,
    kw_only: bool = False,
    converter: t.Optional[Converter[T]] = None,
) -> t.Any:
    ...

def field(*,
    rename: t.Optional[str] = None,
    in_names: t.Optional[t.Sequence[str]] = None,
    aliases: t.Optional[t.Sequence[str]] = None,
    out_name: t.Optional[str] = None,
    init: bool = True,
    default: t.Union[T, _Missing] = _MISSING,
    default_factory: t.Optional[t.Callable[[], T]] = None,
    kw_only: bool = False,
    converter: t.Optional[Converter[T]] = None,
) -> t.Any:
    """
    Annotate a dataclass field.

    Parameters:
      rename: Name to rename this field as. Used for both input and output. Useful when a field name should be different inside vs. outside of Python.
      in_names: List of names which should convert into this field. If specified, the field name inside Python will be excluded (unlike `aliases`).
      aliases: List of aliases (additional names) for this field. Includes the field name inside Python (unlike `in_names`).
      out_name: Name which this field should convert into.
      init: If `False`, this field won't be touched by `pane`, and it's up to the class to initialize it in `__post_init__`.
      default: Default value for field
      default_factory: Default value factory for field
      kw_only: Whether the field is keyword-only.
    """
    return FieldSpec(
        rename=rename, in_names=in_names, aliases=aliases, out_name=out_name,
        init=init, default=default, default_factory=default_factory, kw_only=kw_only,
        converter=converter
    )


__all__ = [
    'Field', 'FieldSpec', 'field',
    'rename_field',
]
