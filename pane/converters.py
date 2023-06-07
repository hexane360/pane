from __future__ import annotations

from collections import abc
import typing
from typing import Union, Mapping, Sequence, Any
from typing import Type, TypeVar, Generic
from typing import Dict, List

from .converter import Converter, Convertible, FromData, DataType, DataTypes


ConvertibleT = TypeVar('ConvertibleT', bound=Convertible)
ConvertibleTco = TypeVar('ConvertibleTco', bound=Convertible, covariant=True)
Int = TypeVar('Int', bound=int)
Float = TypeVar('Float', bound=float)
Complex = TypeVar('Complex', bound=complex)
Str = TypeVar('Str', bound=str)
DictT = TypeVar('DictT', bound=Union[dict, Dict])
ListT = TypeVar('ListT', bound=Union[list, List])
K = TypeVar('K', bound=Convertible)
V = TypeVar('V', bound=Convertible)


class AnyConverter(Converter[Any]):
	def expected(self) -> str:
		return "anything"

	def convert(self, val) -> Any:
		return val


class IntConverter(Generic[Int], Converter[Int]):
	def __init__(self, ty: Type[Int]):
		self.ty: Type[Int] = ty

	def expected(self) -> str:
		return "an int"

	def convert_int(self, val: int) -> Int:
		return self.ty(val)


class FloatConverter(Generic[Float], Converter[Float]):
	def __init__(self, ty: Type[Float]):
		self.ty: Type[Float] = ty

	def expected(self) -> str:
		return "a float"

	def convert_float(self, val: float) -> Float:
		return self.ty(val)

	def convert_int(self, val: int) -> Float:
		return self.ty(val)


class ComplexConverter(Generic[Complex], Converter[Complex]):
	def __init__(self, ty: Type[Complex]):
		self.ty: Type[Complex] = ty

	def expected(self) -> str:
		return "a complex float"

	def convert_complex(self, val: complex) -> Complex:
		return self.ty(val.real, val.imag)

	def convert_float(self, val: float) -> Complex:
		return self.ty(val)

	def convert_int(self, val: int) -> Complex:
		return self.ty(val)


class StrConverter(Generic[Str], Converter[Str]):
	def __init__(self, ty: Type[Str]):
		self.ty: Type[Str] = ty

	def expected(self) -> str:
		return "a string"

	def convert_str(self, val: str) -> Str:
		return self.ty(val)


class NoneConverter(Converter[None]):
	def expected(self) -> str:
		return "null value"

	def convert_none(self, val: None) -> None:
		return None


class DictConverter(Generic[K, V], Converter[Dict[K, V]]):
	def __init__(self, ty: Type[Dict], k: Type[K] = Any, v: Type[V] = Any):  # type: ignore
		self.ty: Type[Dict] = ty
		self.K: Type[K] = k
		self.V: Type[V] = v

	def expected(self) -> str:
		return "a dictionary"

	def convert_map(self, val: Mapping) -> Dict[K, V]:
		if self.K is Any and self.V is Any:
			return self.ty(val.items())
		return self.ty((from_data(k, self.K), from_data(v, self.V)) for (k, v) in val.items())


class ListConverter(Generic[K], Converter[List[K]]):
	def __init__(self, ty: Type[List], k: Type[K] = Any):  # type: ignore
		self.ty: Type[List] = ty
		self.K: Type[K] = k

	def expected(self) -> str:
		return "a list"

	def convert_seq(self, val: Sequence) -> List[K]:
		if self.K is Any:
			return self.ty(val)
		return self.ty(from_data(v, self.K) for v in val)


class UnionConverter(Converter):
	def __init__(self, ty, *args: Type[Convertible]):
		self.args = args

	def expected(self) -> str:
		return " or ".join(make_converter(ty).expected() for ty in self.args)

	def convert(self, val):
		exc = None
		for ty in self.args:
			try:
				return from_data(val, ty)
			except Exception as e:
				exc = e
		if exc is None:
			raise TypeError("Can't convert value to empty union.")
		raise exc


class MyData(Generic[K]):
	def __init__(self, val: K):
		self.inner: K = val

	@classmethod
	def _converter(cls, *args: Type[Convertible]) -> MyDataConverter:
		return MyDataConverter(cls, args[0] if len(args) > 0 else Any)


class MyDataConverter(Generic[K], Converter[MyData[K]]):
	def __init__(self, ty: Type[MyData], k: Type[K]):
		self.ty: Type[MyData] = ty
		self.K: Type[K] = k

	def convert(self, val) -> MyData[K]:
		return MyData(from_data(val, self.K))


def from_data(val: DataType, ty: Type[ConvertibleT]) -> ConvertibleT:
	if not isinstance(val, DataTypes):
		raise TypeError(f"Type {type(val)} is not a valid data interchange type.")

	if ty is Any:
		return val  # type: ignore

	converter = make_converter(ty)
	return converter.convert(val)


def make_converter(ty: Type[ConvertibleT]) -> Converter[ConvertibleT]:
	if ty is Any:
		return AnyConverter()

	base = typing.get_origin(ty) or ty
	args = typing.get_args(ty)

	if issubclass(base, FromData):
		return base._converter(*args)

	if issubclass(base, str):
		return StrConverter(ty)  # type: ignore
	if issubclass(base, complex):
		return ComplexConverter(ty)  # type: ignore
	if issubclass(base, float):
		return FloatConverter(ty)  # type: ignore
	if issubclass(base, int):
		return IntConverter(ty)  # type: ignore
	if base is None:
		return NoneConverter()  # type: ignore
	if issubclass(base, (list, abc.Sequence)):
		return ListConverter(base, args[0] if len(args) > 0 else Any)  # type: ignore
	if issubclass(base, (dict, abc.Mapping)):
		args = typing.get_args(ty)
		return DictConverter(base,  # type: ignore
		                     args[0] if len(args) > 0 else Any,
		                     args[1] if len(args) > 1 else Any)  # type: ignore

	raise TypeError(f"Can't convert data into type '{ty}'")
