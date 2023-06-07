from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Optional, TypeVar, Type, Callable, Any, Union, cast
from typing import Sequence, List


T = TypeVar('T')


class _Missing:
	pass


_MISSING = _Missing()


@dataclass(init=False)
class Field(Generic[T]):
	name: Optional[str]
	type: Type[T]
	ser_name: Optional[str]
	de_name: Optional[str]
	aliases: Optional[List[str]]
	default: Union[T, _Missing]
	default_factory: Optional[Callable[[], T]]
	flatten: bool

	def __init__(self, type: Type[T] = Any, /, *,
	             name: Optional[str] = None,
	             ser_name: Optional[str] = None,
	             de_name: Optional[str] = None,
	             aliases: Union[str, Sequence[str], None] = None,
	             flatten: bool = False,
	             default: Union[T, _Missing] = _MISSING,
	             default_factory: Optional[Callable[[], T]] = None):
		self.name = None
		self.type = type
		if name is not None:
			if self.ser_name is not None or self.de_name is not None:
				raise ValueError("`name` overrides `ser_name` and `de_name`.")
			self.ser_name = name
			self.de_name = name
		else:
			self.ser_name = ser_name
			self.de_name = de_name
		self.default = default
		self.default_factory = default_factory
		self.flatten = flatten

		if isinstance(aliases, str):
			aliases = [aliases]
		elif aliases is not None:
			aliases = list(aliases)
		self.aliases = aliases

	@staticmethod
	def with_name(name: str, ty: Union[Field[T], Type[T]]) -> Field[T]:
		if isinstance(ty, Field):
			field = ty
		else:
			field = cast(Field[T], Field(ty))

		field.name = name
		return field
