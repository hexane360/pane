from __future__ import annotations

from collections import abc
from types import NotImplementedType
import typing as t

from .converter import Converter, Convertible, FromData, DataType, DataTypes, missing


ConvertibleT = t.TypeVar('ConvertibleT', bound=Convertible)
ConvertibleTco = t.TypeVar('ConvertibleTco', bound=Convertible, covariant=True)
T = t.TypeVar('T')
DictT = t.TypeVar('DictT', bound=t.Union[dict, t.Dict])
ListT = t.TypeVar('ListT', bound=t.Union[list, t.List])
K = t.TypeVar('K', bound=Convertible)
V = t.TypeVar('V', bound=Convertible)


class AnyConverter(Converter[t.Any]):
    def expected(self) -> str:
        return "anything"

    def try_convert(self, val, annotation=missing) -> t.Any:
        return val


class ScalarConverter(t.Generic[T], Converter[T]):
    def __init__(self, to_type: t.Type[T], from_types: t.Union[type, t.Tuple[type, ...]], name: str):
        self.to_type = to_type
        self.from_types = from_types
        self.name = name

    def expected(self) -> str:
        return self.name

    def try_convert(self, val, annotation=missing) -> t.Union[T, NotImplementedType]:
        if not isinstance(val, self.from_types):
            return NotImplemented
        return self.to_type(val)


class NoneConverter(Converter[None]):
    def expected(self) -> str:
        return "null value"

    def try_convert(self, val, annotation=missing) -> t.Union[None, NotImplementedType]:
        if val is None:
            return val
        return NotImplemented


class DictConverter(t.Generic[K, V], Converter[t.Dict[K, V]]):
    def __init__(self, ty: t.Type[t.Dict], k: t.Type[K] = t.Any, v: t.Type[V] = t.Any):
        self.ty: t.Type[t.Dict] = ty
        self.K: t.Type[K] = k
        self.V: t.Type[V] = v

    def expected(self) -> str:
        return "a dictionary"

    def convert_map(self, val: t.Mapping) -> t.Dict[K, V]:
        if self.K is t.Any and self.V is t.Any:
            return self.ty(val.items())
        return self.ty((from_data(k, self.K), from_data(v, self.V)) for (k, v) in val.items())


class ListConverter(t.Generic[K], Converter[t.Sequence[K]]):
    def __init__(self, ty: t.Type[t.Sequence], k: t.Type[K] = t.Any):
        self.ty: t.Type[t.Sequence] = ty
        self.K: t.Type[K] = k

    def expected(self) -> str:
        return "a list"

    def try_convert(self, val, annotation=missing) -> t.Union[t.Sequence, NotImplementedType]:
        if isinstance(val, abc.Sequence) and not isinstance(val, str):
            return self.ty(from_data(v, self.K) for v in val)  # type: ignore
        return NotImplemented


class UnionConverter(Converter):
    def __init__(self, ty, *args: t.Type[Convertible]):
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


def from_data(val: DataType, ty: t.Type[ConvertibleT]) -> ConvertibleT:
    if not isinstance(val, DataTypes):
        raise TypeError(f"Type {type(val)} is not a valid data interchange type.")

    if ty is t.Any:
        return val  # type: ignore

    converter = make_converter(ty)
    return converter.convert(val)


_BASIC_CONVERTERS = {
    str: ScalarConverter(str, str, 'str'),
    int: ScalarConverter(int, int, 'an int'),
    float: ScalarConverter(float, (int, float), 'an int'),
    complex: ScalarConverter(complex, (int, float, complex), 'a complex float'),
}


def make_converter(ty: t.Type[ConvertibleT]) -> Converter[ConvertibleT]:
    if ty is t.Any:
        return AnyConverter()

    base = t.get_origin(ty) or ty
    args = t.get_args(ty)

    if issubclass(base, FromData):
        return base._converter(*args, annotation=None)

    if ty in _BASIC_CONVERTERS:
        return _BASIC_CONVERTERS[ty]
    if base is None:
        return NoneConverter()  # type: ignore

    if issubclass(base, (list, abc.Sequence)):
        return ListConverter(base, args[0] if len(args) > 0 else t.Any)  # type: ignore
    if issubclass(base, (dict, abc.Mapping)):
        args = t.get_args(ty)
        return DictConverter(base,  # type: ignore
                             args[0] if len(args) > 0 else t.Any,
                             args[1] if len(args) > 1 else t.Any)  # type: ignore

    raise TypeError(f"Can't convert data into type '{ty}'")
