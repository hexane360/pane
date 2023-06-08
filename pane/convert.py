import typing as t

from .converter import Convertible, Converter, IntoData, DataType, DataTypes
from .converters import from_data


ConvertibleT = t.TypeVar('ConvertibleT', bound=Convertible)


def into_data(val: t.Union[IntoData, DataType]) -> DataType:
    if isinstance(val, DataTypes):
        return val
    if isinstance(val, IntoData):
        return val.into_data()
    raise TypeError(f"Can't convert {type(val)} to data.")


def convert(val: t.Union[IntoData, DataType], ty: t.Type[ConvertibleT]) -> ConvertibleT:
    data = into_data(val)
    return from_data(data, ty)
