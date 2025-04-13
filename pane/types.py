"""Helper types for use with ``pane.convert`` and dataclasses."""
from __future__ import annotations

import math
import typing as t

from typing_extensions import TypeAlias

from pane.classes import PaneBase, field
from pane.converters import UnionConverter
from pane.convert import Convertible, DataType, into_data, ConverterHandlers
from pane.annotations import (
    Positive, NonNegative, Negative, NonPositive, Finite,
    len_range,
)


T = t.TypeVar('T', bound=Convertible)
U = t.TypeVar('U', bound=Convertible)
Num = t.TypeVar('Num', bound=t.Union[int, float])

PositiveInt: TypeAlias = t.Annotated[int, Positive]
NonNegativeInt: TypeAlias = t.Annotated[int, NonNegative]
NegativeInt: TypeAlias = t.Annotated[int, Negative]
NonPositiveInt: TypeAlias = t.Annotated[int, NonPositive]

PositiveFloat: TypeAlias = t.Annotated[float, Positive]
NonNegativeFloat: TypeAlias = t.Annotated[float, NonNegative]
NegativeFloat: TypeAlias = t.Annotated[float, Negative]
NonPositiveFloat: TypeAlias = t.Annotated[float, NonPositive]
FiniteFloat: TypeAlias = t.Annotated[float, Finite]

ListNotEmpty: TypeAlias = t.Annotated[t.List[T], len_range(min=1)]


class Range(PaneBase, t.Generic[Num],
            in_format=('tuple', 'struct'),
            out_format='struct'):
    start: Num
    end: Num

    n: t.Optional[NonNegativeInt] = field(default=None)
    step: t.Optional[Num] = field(default=None, kw_only=True)

    def __post_init__(self):
        s = sum((self.step is None, self.n is None))
        if s == 0:
            raise TypeError("Either 'n' or 'step' may be specified, but not both")
        if s == 2:
            raise TypeError("Either 'n' or 'step' must be specified")
        span = self.end - self.start
        if self.step is not None:
            if math.isclose(self.step, 0.):
                raise ValueError("'step' should be nonzero")
            n = 1 + math.ceil(span / self.step - 1e-6) if span > 0 else 0
            object.__setattr__(self, 'n', n)
        else:
            assert self.n is not None
            if not isinstance(self.start, float) and span % (self.n - 1):
                raise ValueError("Range must be evenly divisible by 'n'")
            step = type(self.start)(span / (self.n - 1)) if self.n > 1 else None
            object.__setattr__(self, 'step', step)

    def __len__(self) -> int:
        return t.cast(int, self.n)

    def __iter__(self) -> t.Iterator[Num]:
        assert self.n is not None
        if self.n == 0:
            return
        val: Num = self.start
        for _ in range(self.n - 1):
            yield val
            val = t.cast(Num, val + self.step)
        yield self.end


class ValueOrList(t.Generic[T]):
    _inner: t.Union[T, t.List[T]]
    _is_val: bool

    def __init__(self, val: t.Union[T, t.List[T]], _is_val: bool):
        self._inner = val
        self._is_val = _is_val

    @classmethod
    def from_val(cls, val: T) -> ValueOrList[T]:
        return cls(val, True)

    @classmethod
    def from_list(cls, list_val: t.List[T]) -> ValueOrList[T]:
        return cls(list_val, False)

    def __repr__(self) -> str:
        return f"ValueOrList({self._inner!r})"

    def __str__(self) -> str:
        return str(self._inner)

    def __eq__(self, other: t.Any) -> bool:
        if not self.__class__ == other.__class__:
            return False
        return self._is_val == other._is_val and self._inner == other._inner

    @classmethod
    def _converter(cls: t.Type[T], *args: t.Type[Convertible],
                   handlers: ConverterHandlers) -> ValueOrListConverter:
        arg = t.cast(t.Type[Convertible], args[0] if len(args) > 0 else t.Any)
        return ValueOrListConverter(arg, handlers=handlers)

    def __len__(self) -> int:
        return 1 if self._is_val else len(t.cast(t.List[T], self._inner))

    def map(self, f: t.Callable[[T], U]) -> ValueOrList[U]:
        if self._is_val:
            return ValueOrList(f(t.cast(T, self._inner)), True)
        return ValueOrList(list(map(f, t.cast(t.List[T], self._inner))), False)

    def __iter__(self) -> t.Iterator[T]:
        if self._is_val:
            yield t.cast(T, self._inner)
        else:
            yield from t.cast(t.List[T], self._inner)


class ValueOrListConverter(UnionConverter):
    def __init__(self, ty: t.Type[Convertible], handlers: ConverterHandlers):
        types = t.cast(t.Sequence[t.Type[Convertible]], (ty, t.List[ty]))
        super().__init__(types, constructor=lambda v, i: ValueOrList(v, i == 0), handlers=handlers)
        self.ty = ty

    def expected(self, plural: bool = False) -> str:
        inner = self.converters[0].expected(plural)
        return f"{inner} or sequence of {inner}"

    def into_data(self, val: t.Any) -> DataType:
        if not isinstance(val, ValueOrList):
            return into_data(val)
        return t.cast(ValueOrList[t.Any], val).map(
            lambda v: into_data(v, self.ty)
        )._inner


class YAMLDocList(list):  # type: ignore
    """
    `list` subclass representing a list of objects from YAML documents.
    """
    ...


__all__ = [
    'PositiveInt', 'NonNegativeInt', 'NegativeInt', 'NonPositiveInt',
    'PositiveFloat', 'NonNegativeFloat', 'NegativeFloat', 'NonPositiveFloat', 'FiniteFloat',
]