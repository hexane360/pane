"""
High-level interface to ``pane``.
"""

from __future__ import annotations

import warnings
import collections.abc
import typing as t

from .errors import ConvertError
from .util import partition

if t.TYPE_CHECKING:
    from .converters import Converter

T = t.TypeVar('T', bound='FromData')


@t.runtime_checkable
class HasFromData(t.Protocol):
    """
    Protocol to add ``convert`` functionality into an arbitrary type from data.
    """

    @classmethod
    def _converter(cls: t.Type[T], *args: t.Type[FromData],
                   annotations: t.Optional[t.Tuple[t.Any, ...]] = None) -> Converter[T]:
        """
        Return a ``Converter`` capable of constructing ``cls``.

        Any given type arguments are passed as positional arguments,
        as well as unsupported ``Annotation``s.

        This function should error when passed unknown type arguments
        or unknown annotations.
        """
        ...


@t.runtime_checkable
class HasIntoData(t.Protocol):
    """
    Protocol to add ``convert`` functionality from an arbitrary type into data.
    """

    def into_data(self) -> DataType:
        """Convert ``self`` into ``DataType``"""
        ...


DataType = t.Union[str, int, bool, float, complex, None, t.Mapping['DataType', 'DataType'], t.Sequence['DataType']]
"""Common data interchange type. ``IntoData`` converts to this, and ``FromData`` converts from this."""
_DataType: type = t.Union[str, int, bool, float, complex, None, t.Mapping, t.Sequence]  # type: ignore
"""``DataType`` for use in ``isinstance``."""
FromData = t.Union[DataType, HasFromData]
"""Types supported by ``from_data``."""
IntoData = t.Union[DataType, HasIntoData]
"""Types supported by ``into_data``."""
IntoConverter = t.Union[
    t.Type[FromData],
    t.Mapping[str, 'IntoConverter'],
    t.Sequence['IntoConverter']
]
"""
Types supported by `make_converter``.
Consists of ``FromData``, mappings, and sequences.
"""


@t.overload
def make_converter(ty: t.Type[T]) -> Converter[T]:
    ...

@t.overload
def make_converter(ty: IntoConverter) -> Converter[t.Any]:
    ...

def make_converter(ty: IntoConverter) -> Converter[t.Any]:
    """
    Make a ``Converter`` for ``ty``.

    Supports types, mappings of types, and sequences of types.
    """

    from .annotations import ConvertAnnotation, Condition
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

    if base is t.Annotated:
        base = args[0]
        annotations = args[1:]  # TODO what order to extend in
        # if annotated, we first split into annotations handled by us and those handled by the subtype
        known, unknown = partition(lambda a: isinstance(a, ConvertAnnotation), annotations)
        known = t.cast(t.Tuple[ConvertAnnotation, ...], known)

        # unknown annotations are passed to the subtype
        if len(unknown):
            if not issubclass(base, HasFromData):
                raise TypeError(f"Unsupported annotation(s) '{annotations}'")
            inner = base._converter(*args, annotations=unknown)
        else:
            inner = make_converter(base)

        # and then we surround with known annotations
        if len(known) == 0:
            return inner
        if len(known) == 1:
            return known[0]._converter(inner)
        if all(isinstance(cond, Condition) for cond in known):  # special case for and conditions
            return Condition.all(*(cond for cond in t.cast(t.Sequence[Condition], known)))._converter(inner)
        # otherwise just nest conditions
        for cond in known:
            inner = cond._converter(inner)
        return inner

    # union converter
    if base is t.Union:
        return UnionConverter(args)
    # literal converter
    if base is t.Literal:
        return LiteralConverter(args)

    if not isinstance(base, type):
        raise TypeError(f"Unsupported special type '{base}'")

    # custom converter
    if issubclass(base, HasFromData):
        return base._converter(*args)

    # simple/scalar converters
    if ty in _BASIC_CONVERTERS:
        return _BASIC_CONVERTERS[ty]

    # tuple converter
    if issubclass(base, (tuple, t.Tuple)) and len(args) > 0 and args[-1] != Ellipsis:
        return TupleConverter(base, args)
    # homogenous sequence converter
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

    raise TypeError(f"Can't convert data into type '{ty}'")


def into_data(val: IntoData) -> DataType:
    if isinstance(val, (dict, t.Mapping)):
        return {into_data(k): into_data(v) for (k, v) in val.items()}
    if isinstance(val, tuple):
        return type(val)(map(into_data, val))
    if isinstance(val, t.Sequence) and not isinstance(val, str):
        return list(map(into_data, val))
    if isinstance(val, _DataType):
        return val
    if isinstance(val, HasIntoData):
        return val.into_data()

    raise TypeError(f"Can't convert type '{type(val)}' into data.")


def from_data(val: DataType, ty: t.Type[T]) -> T:
    if not isinstance(val, _DataType):
        raise TypeError(f"Type {type(val)} is not a valid data interchange type.")

    converter = make_converter(ty)
    return converter.convert(val)


def convert(val: IntoData, ty: t.Type[T]) -> T:
    data = into_data(val)
    return from_data(data, ty)


__all__ = [
    'HasFromData', 'HasIntoData', 'FromData', 'IntoData',
    'Converter', 'DataType', 'ConvertError',
    'from_data', 'into_data', 'make_converter', 'convert',
]
