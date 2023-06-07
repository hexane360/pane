#!/usr/bin/env python3
from typing import Generic, Optional, TypeVar

from .convert import Converter, Type, Convertible, make_converter


T = TypeVar('T', bound=Convertible)


class OptionalConverter(Generic[T], Converter[Optional[T]]):
	def __init__(self, t: Type[T]):
		self.T = t
		self.inner = make_converter(self.T)

	def expected(self) -> str:
		return f"{self.inner.expected()} or None"

	def convert_none(self, val: None) -> None:
		return None

	def convert(self, val) -> Optional[T]:
		if val is None:
			return self.convert_none(val)
		return self.inner.convert(val)
