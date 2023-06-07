import abc
import collections.abc
from typing import Union, Mapping, Sequence, Any
from typing import Type, TypeVar, Generic, Protocol, runtime_checkable
from typing import Dict, List
import typing

from .converter import Converter, Convertible, FromData, IntoData, DataType, DataTypes
from .converters import from_data, make_converter


ConvertibleT = TypeVar('ConvertibleT', bound=Convertible)


def into_data(val: Union[IntoData, DataType]) -> DataType:
	if isinstance(val, DataTypes):
		return val
	if isinstance(val, IntoData):
		return val.into_data()
	raise TypeError(f"Can't convert {type(val)} to data.")


def convert(val: Union[IntoData, DataType], ty: Type[ConvertibleT]) -> ConvertibleT:
	data = into_data(val)
	return from_data(data, ty)
