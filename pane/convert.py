"""
High-level interface to `pane`.
"""

from __future__ import annotations

import warnings
import collections.abc
import typing as t

from .errors import ConvertError, UnsupportedAnnotation
from .addons import numpy as numpy

if t.TYPE_CHECKING:
    from .converters import Converter

T = t.TypeVar('T', bound='Convertible')


@t.runtime_checkable
class HasConverter(t.Protocol):
    """
    Protocol to add [`convert`][pane.convert.convert] functionality into an arbitrary type from data.
    """

    @classmethod
    def _converter(cls: t.Type[T], *args: t.Type[Convertible]) -> Converter[T]:
        """
        Return a [`Converter`][pane.converters.Converter] capable of constructing `cls`.

        Any given type arguments are passed as positional arguments.
        This function should error when passed unknown type arguments.
        """
        ...


DataType = t.Union[str, bytes, int, bool, float, complex, None, t.Mapping['DataType', 'DataType'], t.Sequence['DataType'], numpy.NDArray[numpy.generic]]
"""Common data interchange type. [`into_data`][pane.convert.into_data] converts to this."""
_DataType = (str, bytes, int, bool, float, complex, type(None), t.Mapping, t.Sequence, numpy.ndarray)  # type: ignore
"""[`DataType`][pane.convert.DataType] for use in [`isinstance`][isinstance]."""
Convertible = t.Union[DataType, HasConverter]
"""Types supported by [`from_data`][pane.convert.from_data]. [`DataType`][pane.convert.DataType] + [`HasConverter`][pane.convert.HasConverter]"""
IntoConverter = t.Union[
    t.Type[DataType], t.Type[HasConverter],
    t.Mapping[str, 'IntoConverter'],
    t.Sequence['IntoConverter']
]
"""
Types supported by [`make_converter`][pane.convert.make_converter].
Consists of `t.Type[DataType]`, mappings (struct types), and sequences (tuple types).
"""


_CONVERTER_HANDLERS: t.Sequence[t.Callable[[t.Any, t.Tuple[t.Any]], Converter[t.Any]]] = []


@t.overload
def make_converter(ty: t.Type[T]) -> Converter[T]:
    ...

@t.overload
def make_converter(ty: IntoConverter) -> Converter[t.Any]:
    ...

def make_converter(ty: IntoConverter) -> Converter[t.Any]:
    """
    Make a [`Converter`][pane.convert.Converter] for `ty`.

    Supports types, mappings of types, and sequences of types.
    """

    from .converters import AnyConverter, StructConverter, SequenceConverter, UnionConverter
    from .converters import LiteralConverter, DictConverter, TupleConverter
    from .converters import _BASIC_CONVERTERS

    if ty is t.Any:
        return AnyConverter()
    if isinstance(ty, t.TypeVar):
        warnings.warn(f"Unbound TypeVar '{ty}'. Will be interpreted as Any.")
        return AnyConverter()
    if isinstance(ty, (dict, t.Mapping)):
        return StructConverter(type(ty), ty)
    if isinstance(ty, (tuple, t.Tuple)):
        return TupleConverter(type(ty), ty)
    if isinstance(ty, t.ForwardRef) or isinstance(ty, str):
        raise TypeError(f"Unresolved forward reference '{ty}'")

    base = t.get_origin(ty) or ty
    args = t.get_args(ty)

    # special types

    # handle annotations
    if base is t.Annotated:
        return _annotated_converter(args[0], args[1:])

    # union converter
    if base is t.Union:
        return UnionConverter(args)
    # literal converter
    if base is t.Literal:
        return LiteralConverter(args)

    if not isinstance(base, type):
        raise TypeError(f"Unsupported special type '{base}'")

    # custom converter
    if issubclass(base, HasConverter):
        return base._converter(*args)

    # simple/scalar converters
    if base in _BASIC_CONVERTERS:
        return _BASIC_CONVERTERS[base]

    # add-on handlers
    for handler in _CONVERTER_HANDLERS:
        try:
            result = handler(base, args)
            if result is not NotImplemented:
                return result
        except NotImplementedError:
            pass

    # tuple converter
    if issubclass(base, (tuple, t.Tuple)) and len(args) > 0 and args[-1] != Ellipsis:
        return TupleConverter(base, args)
    if issubclass(base, (list, t.Sequence)):
        if base is t.Sequence or base is collections.abc.Sequence:
            # t.Sequence => tuple
            base = tuple
        return SequenceConverter(base, args[0] if len(args) > 0 else t.Any)  # type: ignore
    # homogenous mapping converter
    if issubclass(base, (dict, t.Mapping)):
        if base is t.Mapping or base is collections.abc.Mapping:
            # t.Mapping => dict
            base = dict
        return DictConverter(base,  # type: ignore
                             args[0] if len(args) > 0 else t.Any,
                             args[1] if len(args) > 1 else t.Any)  # type: ignore

    # after we've handled common cases, look for subclasses of basic types
    for (ty, conv) in _BASIC_CONVERTERS.items():
        if issubclass(base, ty):
            return conv

    raise TypeError(f"No converter for type '{ty}'")


def register_converter_handler(handler: t.Callable[[t.Any, t.Tuple[t.Any, ...]], Converter[t.Any]]) -> None:
    """
    Register a handler for make_converter.

    This allows extending `pane` to handle third-party types, not
    defined by your code or by `pane`. Use sparingly, as this will
    add runtime to [`make_converter`][pane.convert.make_converter].
    """
    _CONVERTER_HANDLERS.append(handler)


def _annotated_converter(ty: IntoConverter, args: t.Sequence[t.Any]) -> Converter[t.Any]:
    """
    Make an annotated converter.

    Wraps `ty` in `args` from left to right. However, [`Condition`][pane.annotations.Condition] annotations
    are handled separately (bundled together).
    """
    from .converters import Converter
    from .annotations import Condition, ConvertAnnotation

    conv: t.Union[IntoConverter, Converter[t.Any]] = ty

    conditions: t.List[Condition] = []  # buffer of conditions to combine
    for arg in args:
        if isinstance(arg, Condition):
            conditions.append(arg)
            continue

        if not isinstance(arg, ConvertAnnotation):
            raise UnsupportedAnnotation(arg)

        # dump list of conditions
        if len(conditions):
            if len(conditions) > 1:
                conv = Condition.all(*conditions)._converter(conv)
            else:
                conv = conditions[0]._converter(conv)

        conv = arg._converter(conv)

    # dump list of conditions
    if len(conditions):
        if len(conditions) > 1:
            conv = Condition.all(*conditions)._converter(conv)
        else:
            conv = conditions[0]._converter(conv)

    return conv if isinstance(conv, Converter) else make_converter(conv)


def into_data(val: Convertible, ty: t.Optional[IntoConverter] = None) -> DataType:
    """
    Convert `val` of type `ty` into a data interchange format.
    """
    from .converters import data_is_sequence, data_is_mapping

    if ty is not None:
        # use specialized implementation
        converter = make_converter(ty)
        return converter.into_data(val)

    # without type information, convert as best we can
    if isinstance(val, HasConverter):
        converter = val._converter()
        return converter.into_data(val)
    if data_is_mapping(val):
        return {into_data(k): into_data(v) for (k, v) in val.items()}
    if isinstance(val, tuple):
        return type(val)(map(into_data, val))
    if data_is_sequence(val):
        return list(map(into_data, val))
    if isinstance(val, _DataType):
        return val

    raise TypeError(f"Can't convert type '{type(val)}' into data.")


def from_data(val: DataType, ty: t.Type[T]) -> T:
    """
    Convert `val` from a data interchange format into type `ty`.
    """

    if not isinstance(val, _DataType):
        raise TypeError(f"Type {type(val)} is not a valid data interchange type.")

    converter = make_converter(ty)
    return converter.convert(val)


def convert(val: Convertible, ty: t.Type[T]) -> T:
    """
    Convert `val` into type `ty`, passing through a data interchange format.
    """
    data = into_data(val)
    return from_data(data, ty)


# register some handlers
register_converter_handler(numpy.numpy_converter_handler)


__all__ = [
    'Convertible', 'HasConverter', 'IntoConverter',
    'Converter', 'DataType', 'ConvertError',
    'from_data', 'into_data', 'make_converter', 'convert',
]
