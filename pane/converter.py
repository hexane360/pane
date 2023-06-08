from __future__ import annotations

import abc
from types import NotImplementedType

import typing as t

DataType = t.Union[str, int, bool, float, complex, None, t.Mapping, t.Sequence]
DataTypes = (str, int, bool, float, complex, type(None), t.Mapping, t.Sequence)
T_co = t.TypeVar('T_co', covariant=True)
missing = object()


class Converter(abc.ABC, t.Generic[T_co]):
    def expected(self) -> str:
        return str(T_co)

    @abc.abstractmethod
    def try_convert(self, val: DataType, annotation: t.Any = missing) -> t.Union[T_co, NotImplementedType]:
        ...

    def convert(self, val: DataType) -> T_co:
        result = self.try_convert(val)
        if result is NotImplemented:
            raise TypeError(f"Can't convert value of type '{type(val).__name__}'. Expected {self.expected()}.")
        return result  # type: ignore


@t.runtime_checkable
class FromData(t.Protocol):
    @classmethod
    def _converter(cls: t.Type[FromDataT], *args: t.Type[Convertible],
                   annotation: t.Any = missing) -> Converter[FromDataT]:
        ...


@t.runtime_checkable
class IntoData(t.Protocol):
    @classmethod
    def into_data(cls) -> DataType:
        ...


FromDataT = t.TypeVar('FromDataT', bound=FromData)
Convertible = t.Union[DataType, FromData]
