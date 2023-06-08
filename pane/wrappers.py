
import typing as t

from .convert import Converter, Convertible
from .converters import make_converter


T = t.TypeVar('T', bound=Convertible)


class OptionalConverter(t.Generic[T], Converter[t.Optional[T]]):
    def __init__(self, t: t.Type[T]):
        self.T = t
        self.inner = make_converter(self.T)

    def expected(self) -> str:
        return f"{self.inner.expected()} or None"

    def convert_none(self, val: None) -> None:
        return None

    def convert(self, val) -> t.Optional[T]:
        if val is None:
            return self.convert_none(val)
        return self.inner.convert(val)
