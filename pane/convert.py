"""
High-level interface to `pane`.
"""

# pyright: reportUnknownMemberType=none

from __future__ import annotations

import warnings
import collections
import collections.abc
import enum
import datetime
from decimal import Decimal
from fractions import Fraction
import inspect
import os
import pathlib
import typing as t

from .errors import ConvertError, UnsupportedAnnotation
from .addons import numpy as numpy
from .util import key_cache

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


ScalarType = t.Union[str, bytes, int, bool, float, complex, None]

DataType = t.Union[str, bytes, int, bool, float, complex, None, t.Mapping['DataType', 'DataType'], t.Sequence['DataType'], numpy.NDArray[numpy.generic]]
"""Common data interchange type. [`into_data`][pane.convert.into_data] converts to this."""

_ScalarType = (str, bytes, int, bool, float, complex, type(None))  # type: ignore
"""Scalar [`DataType`][pane.convert.DataType]s for use in [`isinstance`][isinstance] checks."""
_DataType = (*_ScalarType, t.Mapping, t.Sequence, numpy.ndarray)  # type: ignore
"""[`DataType`][pane.convert.DataType] for use in [`isinstance`][isinstance] checks."""

Convertible = t.Union[
    DataType, HasConverter,
    t.AbstractSet[DataType],
    Fraction, Decimal,
    datetime.datetime, datetime.date, datetime.time,
    os.PathLike[str],
    t.Pattern[str], t.Pattern[bytes],
    enum.Enum
]
"""
Types supported by [`from_data`][pane.convert.from_data].

Consists of [`DataType`][pane.convert.DataType] + [`HasConverter`][pane.convert.HasConverter] + supported stdlib types.
"""

IntoConverter = t.Union[
    t.Type[Convertible],
    t.Type[t.Any],
    t.Mapping[str, 'IntoConverter'],
    t.Sequence['IntoConverter']
]
"""
Inputs supported by [`make_converter`][pane.convert.make_converter].
Consists of `t.Type[Convertible]`, mappings (struct types), and sequences (tuple types).
"""


_CONVERTER_HANDLERS: t.Sequence[t.Callable[[t.Any, t.Tuple[t.Any, ...]], Converter[t.Any]]] = []


_ABSTRACT_MAPPING: t.Mapping[type, type] = t.cast(t.Mapping[type, type], {
    t.Sequence: tuple,
    collections.abc.Sequence: tuple,
    t.MutableSequence: list,
    collections.abc.MutableSequence: list,

    collections.abc.Mapping: dict,
    t.Mapping: dict,
    collections.abc.MutableMapping: dict,
    t.MutableMapping: dict,

    collections.abc.MutableSet: set,
    collections.abc.Set: frozenset,

    os.PathLike: pathlib.PurePath,
})
"""Mapping to attempt to choose a simple concrete type for abstract/base collection types"""


@t.overload
def make_converter(ty: t.Type[T]) -> Converter[T]:
    ...

@t.overload
def make_converter(ty: IntoConverter) -> Converter[t.Any]:
    ...

@key_cache(id)
def make_converter(ty: IntoConverter) -> Converter[t.Any]:
    """
    Make a [`Converter`][pane.convert.Converter] for `ty`.

    Supports types, mappings of types, and sequences of types.
    """

    from .converters import AnyConverter, StructConverter, SequenceConverter, UnionConverter
    from .converters import LiteralConverter, DictConverter, TupleConverter, ScalarConverter
    from .converters import EnumConverter, _BASIC_CONVERTERS, _BASIC_WITH_ARGS

    if ty is t.Any:
        return AnyConverter()
    if isinstance(ty, t.TypeVar):
        if ty.__bound__ is not None:  # type: ignore
            # bound typevar
            var_ty = ty.__bound__
        elif len(ty.__constraints__) == 1:
            # typevar with constraints
            var_ty = ty.__constraints__
        elif len(ty.__constraints__) > 1:
            # typevar with multiple constraints
            var_ty = t.Union[ty.__constraints__]  # type: ignore
        else:
            # unbound typevar
            var_ty = t.Any

        warnings.warn(f"Unbound TypeVar '{ty}'. Will be interpreted as '{var_ty}'.")
        return make_converter(var_ty)
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

    if base in _BASIC_WITH_ARGS:
        return _BASIC_WITH_ARGS[base](*args)

    # add-on handlers
    for handler in _CONVERTER_HANDLERS:
        try:
            result = handler(base, args)
            if result is not NotImplemented:
                return result
        except NotImplementedError:
            pass

    if issubclass(base, enum.Enum):
        return EnumConverter(base)

    # pathlike converter
    if issubclass(base, os.PathLike):
        new_base = _ABSTRACT_MAPPING.get(base, base)  # type: ignore
        if inspect.isabstract(new_base):
            raise TypeError(f"No converter for abstract type '{ty}'")
        return ScalarConverter(new_base, (str, os.PathLike), 'a path', 'paths', str)  # type: ignore

    # tuple converter
    if issubclass(base, (tuple, t.Tuple)):
        # treat tuple[int, ...] and tuple[()] correctly
        if len(args) > 0 and args[-1] != Ellipsis \
              or args == () and hasattr(ty, '__args__'):
            if args == ((),):  # tuple[()] on python <3.11
                args = ()
            return TupleConverter(base, args)
        # fall through to sequence converter

    # homogenous sequence converter
    # concrete t.Set/t.List/etc are already converted to set/list/etc by t.get_origin
    if issubclass(base, (collections.abc.Sequence, collections.abc.Set)):
        # map abstract to concrete types
        new_base = _ABSTRACT_MAPPING.get(base, base)  # type: ignore
        if inspect.isabstract(new_base):
            raise TypeError(f"No converter for abstract type '{ty}'")
        return SequenceConverter(new_base, args[0] if len(args) > 0 else t.Any)  # type: ignore

    # homogenous mapping converter
    # this also handles dict subclasses like Counter & OrderedDict
    if issubclass(base, (dict, t.Mapping)):
        # map abstract to concrete types
        new_base = _ABSTRACT_MAPPING.get(base, base)  # type: ignore
        if inspect.isabstract(new_base):
            raise TypeError(f"No converter for abstract type '{ty}'")
        if issubclass(new_base, collections.Counter):
            # counter takes one type argument, handle it specially
            return DictConverter(new_base, args[0] if len(args) > 0 else t.Any, int)

        # defaultdict needs a special constructor
        constructor: t.Optional[t.Callable[[t.Mapping[t.Any, t.Any]], collections.defaultdict[t.Any, t.Any]]]
        constructor = (lambda d: collections.defaultdict(None, d)) if issubclass(new_base, collections.defaultdict) else None
        return DictConverter(new_base,  # type: ignore
                             args[0] if len(args) > 0 else t.Any,
                             args[1] if len(args) > 1 else t.Any,
                             constructor)

    # after we've handled common cases, look for subclasses of basic types
    for (conv_ty, conv) in _BASIC_CONVERTERS.items():
        if issubclass(base, conv_ty):
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
    if ty is None:
        if isinstance(val, _ScalarType):
            # we can bypass the converter for scalar types
            return val
        ty = type(val)

    try:
        conv = make_converter(ty)
    except TypeError:
        raise TypeError(f"Can't convert type '{type(val)}' into data.") from None

    return conv.into_data(val)


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
