from __future__ import annotations

from dataclasses import dataclass
import itertools
import re
import typing as t


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


def pairwise(it: t.Iterable[T]) -> t.Iterator[t.Tuple[T, T]]:
    it = iter(it)
    while True:
        try:
            a = next(it)
            b = next(it)
        except StopIteration:
            break
        yield (a, b)


def split_field_name(field: str) -> t.Sequence[str]:
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
        for (s1, s2) in pairwise(seps[1:]):
            yield s1 + s2

    return tuple(itertools.chain.from_iterable(map(split_case, parts)))


def rename_field(field: str, style: RenameStyle) -> str:
    return CONVERT_FNS[style](split_field_name(field))


@dataclass(kw_only=True)
class FieldSpec:
    aliases: t.Optional[t.Sequence[str]] = None
    save_name: t.Optional[str] = None
    init: bool = True
    default: t.Union[t.Any, _Missing] = _MISSING
    default_factory: t.Optional[t.Callable[[], t.Any]] = None
    kw_only: bool = False
    flatten: bool = False

    def __post_init__(self):
        if isinstance(self.aliases, str):
            self.aliases = [self.aliases]

        if self.flatten:
            raise NotImplementedError()

    def make_field(self, name: str, ty: t.Union[type, _Missing] = _MISSING) -> Field:
        py_name = name = name
        ty = t.cast(type, t.Any if ty is _MISSING else ty)
        return Field(name=name, py_name=py_name, type=ty, save_name=self.save_name, aliases=self.aliases,
                     init=self.init, default=self.default, default_factory=self.default_factory,
                     kw_only=self.kw_only, flatten=self.flatten)

    def is_optional(self) -> bool:
        return self.default is not _MISSING or self.default_factory is not None


@dataclass(kw_only=True)
class Field(FieldSpec):
    name: str
    type: type
    py_name: str


def field(*,
    aliases: t.Optional[t.Sequence[str]] = None,
    save_name: t.Optional[str] = None,
    init: bool = True,
    default: t.Union[T, _Missing] = _MISSING,
    default_factory: t.Optional[t.Callable[[], T]] = None,
    kw_only: bool = False,
    flatten: bool = False,
) -> t.Any:
    return FieldSpec(
        aliases=aliases, save_name=save_name, flatten=flatten, init=init,
        default=default, default_factory=default_factory, kw_only=kw_only
    )