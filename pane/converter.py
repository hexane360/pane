from __future__ import annotations

import abc
from types import NotImplementedType

from typing import Type, TypeVar, Generic, Union, Mapping, Sequence
from typing import Protocol, runtime_checkable

DataType = Union[str, int, bool, float, complex, None, Mapping, Sequence]
DataTypes = (str, int, bool, float, complex, type(None), Mapping, Sequence)
T_co = TypeVar('T_co', covariant=True)


class Converter(abc.ABC, Generic[T_co]):
	def expected(self) -> str:
		return str(T_co)

	def convert_int(self, val: int) -> Union[T_co, NotImplementedType]:
		return NotImplemented

	def convert_float(self, val: float) -> Union[T_co, NotImplementedType]:
		return NotImplemented

	def convert_complex(self, val: complex) -> Union[T_co, NotImplementedType]:
		return NotImplemented

	def convert_bool(self, val: bool) -> Union[T_co, NotImplementedType]:
		return NotImplemented

	def convert_none(self, val: None) -> Union[T_co, NotImplementedType]:
		return NotImplemented

	def convert_str(self, val: str) -> Union[T_co, NotImplementedType]:
		return NotImplemented

	def convert_map(self, val: Mapping) -> Union[T_co, NotImplementedType]:
		return NotImplemented

	def convert_seq(self, val: Sequence) -> Union[T_co, NotImplementedType]:
		return NotImplemented

	def try_convert(self, val: DataType) -> Union[T_co, NotImplementedType]:
		if isinstance(val, str):
			return self.convert_str(val)
		if isinstance(val, int):
			return self.convert_int(val)
		if isinstance(val, float):
			return self.convert_float(val)
		if isinstance(val, complex):
			return self.convert_complex(val)
		if isinstance(val, bool):
			return self.convert_bool(val)
		if val is None:
			return self.convert_none(val)
		if isinstance(val, Sequence):
			return self.convert_seq(val)
		if isinstance(val, Mapping):
			return self.convert_map(val)
		raise TypeError(f"Unknown data type '{type(val)}' passed.")

	def convert(self, val: DataType) -> T_co:
		result = self.try_convert(val)
		if result is NotImplemented:
			raise TypeError(f"Can't convert value of type '{type(val)}'. Expected {self.expected()}.")
		return result  # type: ignore


@runtime_checkable
class FromData(Protocol):
	@classmethod
	def _converter(cls: Type[FromDataT], *args: Type[Convertible]) -> Converter[FromDataT]:
		...


@runtime_checkable
class IntoData(Protocol):
	@classmethod
	def into_data(cls) -> DataType:
		...


FromDataT = TypeVar('FromDataT', bound=FromData)
Convertible = Union[DataType, FromData]
