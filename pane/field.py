from __future__ import annotations

from dataclasses import dataclass
import typing as t
from typing import Generic, Optional, TypeVar, Type, Callable, Any, Union, cast
from typing import Sequence, List


T = TypeVar('T')


class _Missing:
    pass


_MISSING = _Missing()


@dataclass(kw_only=True)
class FieldSpec:
    aliases: Optional[List[str]] = None
    save_name: t.Optional[str] = None
    init: bool = True
    default: Union[t.Any, _Missing] = _MISSING
    default_factory: Optional[Callable[[], t.Any]] = None
    kw_only: bool = False
    flatten: bool = False

    def __post_init__(self):
        if isinstance(self.aliases, str):
            self.aliases = [self.aliases]

        if self.flatten:
            raise NotImplementedError()

    def make_field(self, name: str, ty: Union[type, _Missing] = _MISSING) -> Field:
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
    aliases: Optional[List[str]] = None,
    save_name: t.Optional[str] = None,
    init: bool = True,
    default: Union[T, _Missing] = _MISSING,
    default_factory: Optional[Callable[[], T]] = None,
    kw_only: bool = False,
    flatten: bool = False,
) -> t.Any:
    return FieldSpec(
        aliases=aliases, save_name=save_name, flatten=flatten, init=init,
        default=default, default_factory=default_factory, kw_only=kw_only
    )