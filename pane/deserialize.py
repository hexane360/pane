from __future__ import annotations

import abc
import collections.abc
from types import GenericAlias
import typing
from typing import Mapping, Sequence, Generic, Union, TypeVar
from typing import Type, Any
from typing import Dict, Tuple, List


T = TypeVar('T')


class Deserialize(abc.ABC):

	@classmethod
	@abc.abstractmethod
	def _de_visitor(cls: T) -> DeVisitor[T]:
		...


class DeVisitor(abc.ABC, Generic[T]):
	def expected(self) -> Union[str, None]:
		return None

	def deserialize_int(self, val: int) -> Union[T, NotImplemented]:
		return NotImplemented

	def deserialize_float(self, val: float) -> Union[T, NotImplemented]:
		return NotImplemented

	def deserialize_complex(self, val: complex) -> Union[T, NotImplemented]:
		return NotImplemented

	def deserialize_bool(self, val: bool) -> Union[T, NotImplemented]:
		return NotImplemented

	def deserialize_none(self, val: None) -> Union[T, NotImplemented]:
		return NotImplemented

	def deserialize_str(self, val: str) -> Union[T, NotImplemented]:
		return NotImplemented

	def deserialize_map(self, val: Mapping) -> Union[T, NotImplemented]:
		return NotImplemented

	def deserialize_seq(self, val: Sequence) -> Union[T, NotImplemented]:
		return NotImplemented

	def deserialize_struct(self, val: Mapping[str, Any]) -> Union[T, NotImplemented]:
		return NotImplemented

	def deserialize_tuple(self, val: Tuple) -> Union[T, NotImplemented]:
		return NotImplemented


class IntVisitor(DeVisitor[int]):
	def expected(self) -> str:
		return "an integer"

	def deserialize_int(self, val: int) -> int:
		return val


class FloatVisitor(DeVisitor[float]):
	def expected(self) -> str:
		return "a float"

	def deserialize_float(self, val: float) -> float:
		return val

	def deserialize_int(self, val: int) -> float:
		return float(val)


class ComplexVisitor(DeVisitor[complex]):
	def expected(self) -> str:
		return "a complex float"

	def deserialize_complex(self, val: complex) -> complex:
		return val

	def deserialize_float(self, val: float) -> complex:
		return complex(val)

	def deserialize_int(self, val: int) -> complex:
		return complex(val)


class StrVisitor(DeVisitor[str]):
	def expected(self) -> str:
		return "a string"

	def deserialize_str(self, val: str) -> str:
		return val


class DictVisitor(DeVisitor[Dict]):
	def __init__(self, ty):
		self.ty = ty
		self.args = typing.get_args(ty)
		self.T = self.args[0] if len(self.args) > 0 else Any
		self.U = self.args[1] if len(self.args) > 1 else Any

	def expected(self) -> str:
		return "a dictionary"

	def deserialize_map(self, val: Mapping) -> Dict:
		if self.T is Any and self.U is Any:
			return dict(val.items())
		return {deserialize(k, self.T): deserialize(v, self.U) for (k, v) in val.items()}

	def deserialize_struct(self, val: Mapping) -> Dict:
		return self.deserialize_map(val)


class ListVisitor(DeVisitor[List]):
	def __init__(self, ty):
		self.ty = ty
		self.args = typing.get_args(ty)
		self.T = self.args[0] if len(self.args) > 0 else Any

	def expected(self) -> str:
		return "a list"

	def deserialize_seq(self, val: Sequence) -> List:
		if self.T is Any:
			return list(val)
		return [deserialize(v, self.T) for v in val]

	def deserialize_tuple(self, val: Tuple) -> List:
		return self.deserialize_seq(val)


Deserializable = Union[Deserialize, int, float, complex, str, bool, Dict, Tuple, List, None]


def deserialize(val: Any, ty: Type[Deserializable]):
	if ty == Any:
		return val

	visitor = _get_visitor(ty)

	if val is None:
		r = visitor.deserialize_none(val)
	elif isinstance(val, str):
		r = visitor.deserialize_str(val)
	elif isinstance(val, complex):
		r = visitor.deserialize_complex(val)
	elif isinstance(val, float):
		r = visitor.deserialize_float(val)
	elif isinstance(val, int):
		r = visitor.deserialize_int(val)
	elif isinstance(val, bool):
		r = visitor.deserialize_bool(val)
	elif isinstance(val, dict):
		r = visitor.deserialize_map(val)
	elif isinstance(val, list):
		r = visitor.deserialize_seq(val)
	elif isinstance(val, tuple):
		r = visitor.deserialize_tuple(val)
	else:
		raise TypeError(f"Don't know how to deserialize from type {type(val)}")

	if r is NotImplemented:
		expected = visitor.expected()
		if expected is None:
			raise TypeError(f"Unexpected value '{val}'")
		raise TypeError(f"Expected {expected}, instead got '{val}'")

	return r


_STR_VISITOR = StrVisitor()
_INT_VISITOR = IntVisitor()
_FLOAT_VISITOR = FloatVisitor()
_COMPLEX_VISITOR = ComplexVisitor()


def _get_visitor(ty: Type[Deserializable]):
	if hasattr(ty, '_de_visitor'):
		return ty._de_visitor()

	base = typing.get_origin(ty) or ty

	if issubclass(base, str):
		return _STR_VISITOR
	if issubclass(base, int):
		return _INT_VISITOR
	if issubclass(base, float):
		return _FLOAT_VISITOR
	if issubclass(base, complex):
		return _COMPLEX_VISITOR
	if issubclass(base, list):
		return ListVisitor(ty)
	if issubclass(base, (dict, collections.abc.Mapping)):
		return DictVisitor(ty)

	raise TypeError(f"Type '{ty}' does not implement Deserialize")
