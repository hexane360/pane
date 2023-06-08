import typing as t

from .converter import IntoData, FromData, DataType, DataTypes, Convertible, ConvertError
from .converter import make_converter

T = t.TypeVar('T', bound=Convertible)

def into_data(val: t.Union[IntoData, DataType]) -> DataType:
    if isinstance(val, DataTypes):
        return val
    if isinstance(val, IntoData):
        return val.into_data()
    raise TypeError(f"Can't convert {type(val)} to data.")


def from_data(val: DataType, ty: t.Type[T]) -> T:
    if not isinstance(val, DataTypes):
        raise TypeError(f"Type {type(val)} is not a valid data interchange type.")

    converter = make_converter(ty)
    return converter.convert(val)


def convert(val: t.Union[IntoData, DataType], ty: t.Type[T]) -> T:
    data = into_data(val)
    return from_data(data, ty)

__all__ = [
	'IntoData', 'FromData', 'DataType', 'ConvertError',
    'from_data', 'into_data', 'make_converter', 'convert',
]