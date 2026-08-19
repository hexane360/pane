
import typing as t

# pyright: reportUnknownMemberType=none

try:
    import numpy as _numpy
    from numpy import array, dtype, generic, ndarray
    from numpy.typing import NDArray

    if t.TYPE_CHECKING:
        from ..convert import ConverterHandlers
        from ..converters import Converter


    def _dtype_map(ty: t.Union[type[t.Any], type[generic]]) -> type:
        # TODO add a lookup table here
        # TODO add conditions to some types
        # e.g. unsigned int -> NonNegativeInt
        if ty in (_numpy.generic, _numpy.object_, t.Any):
            return type(t.Any)
        if issubclass(ty, (_numpy.integer, int)):
            return int
        if issubclass(ty, (_numpy.floating, float)):
            return float
        if issubclass(ty, (_numpy.complexfloating, complex)):
            return complex
        if issubclass(ty, (_numpy.bool_, bool)):
            return bool
        if issubclass(ty, (_numpy.bytes_, bytes)):
            return bytes
        if issubclass(ty, (_numpy.str_, str)):
            return str
        raise TypeError(f"Don't know how to handle numpy dtype '{ty}'")


    def _check_shape_typevar(ty: t.Any):
        if ty is t.Any:
            return

        base = t.get_origin(ty) or ty
        args = t.get_args(ty)

        # let tuple[int, ...], tuple[t.Any, ...] through
        if issubclass(base, (tuple, t.Tuple)) and args[0] in (int, t.Any) and args[1] == Ellipsis:   # noqa: UP006
            return

        raise TypeError("Numpy shape types are currently unsupported.")


    def _is_ndarray(val: t.Any) -> bool:
        return isinstance(val, _numpy.ndarray)


    def numpy_converter_handler(ty: t.Any, args: t.Sequence[t.Any], *,
                                handlers: 'ConverterHandlers') -> 'Converter[t.Any]':
        from ..convert import make_converter

        if issubclass(ty, generic):
            # dtype converters
            return t.cast('Converter[t.Any]', make_converter(
                _dtype_map(t.cast('type[generic[t.Any]]', ty)),
                handlers=handlers
            ))

        if not (ty is NDArray or issubclass(ty, ndarray)):
            return NotImplemented

        if issubclass(ty, ndarray):
            arg1 = t.Any if len(args) < 1 else args[0]
            dtype = t.Any if len(args) < 2 else args[1]

            _check_shape_typevar(arg1)

            if dtype is not t.Any:
                dtype_ty, dtype_args = t.get_origin(dtype), t.get_args(dtype)
                if dtype_ty is not _numpy.dtype:
                    raise TypeError(f"ndarray type argument should be 'numpy.dtype[<type>]', not '{dtype}'")
                dtype = t.Any if len(dtype_args) < 1 else dtype_args[0]
        else:
            dtype = t.Any if len(args) < 1 else args[0]

        from ..converters import NestedSequenceConverter

        return NestedSequenceConverter(dtype, array, ragged=False, handlers=handlers,  # type: ignore
                                       isinstance_check=_is_ndarray)

except ImportError:
    if not t.TYPE_CHECKING:
        class generic:
            pass

        _DTypeScalar_co = t.TypeVar("_DTypeScalar_co", covariant=True, bound=generic)

        class dtype(t.Generic[_DTypeScalar_co]):
            ...

        _ShapeType = t.TypeVar('_ShapeType', bound=dtype[t.Any])
        _DType_co = t.TypeVar('_DType_co', bound=t.Any, covariant=True)

        class ndarray(t.Generic[_ShapeType, _DType_co]):
            pass

        ScalarType_co = t.TypeVar("ScalarType_co", covariant=True, bound=generic)
        NDArray = ndarray[t.Any, dtype[ScalarType_co]]

        # dummy handler
        def numpy_converter_handler(ty: t.Any, args: t.Sequence[t.Any], *,
                                    custom: t.Optional['ConverterHandlers'] = None):
            return NotImplemented
