from __future__ import annotations

import abc
import sys
from io import StringIO
import warnings
import traceback
import dataclasses
from itertools import chain
import typing as t

from .util import list_phrase


@t.runtime_checkable
class HasFromData(t.Protocol):
    @classmethod
    def _converter(cls: t.Type[HasFromDataT], *args: t.Type[FromData],
                   annotations: t.Optional[t.Tuple[t.Any, ...]] = None) -> Converter[HasFromDataT]:
        ...


@t.runtime_checkable
class HasIntoData(t.Protocol):
    def into_data(self) -> DataType:
        ...


DataType = t.Union[str, int, bool, float, complex, None, t.Mapping, t.Sequence]
FromData = t.Union[DataType, HasFromData]
IntoData = t.Union[DataType, HasIntoData]

HasFromDataT = t.TypeVar('HasFromDataT', bound=HasFromData)
T_co = t.TypeVar('T_co', covariant=True)
TupleT = t.TypeVar('TupleT', bound=tuple)
T = t.TypeVar('T', bound=FromData)
U = t.TypeVar('U', bound=FromData)
K = t.TypeVar('K', bound=FromData)
V = t.TypeVar('V', bound=FromData)

DataTypes = (str, int, bool, float, complex, type(None), t.Mapping, t.Sequence)


class ParseInterrupt(Exception):
    ...


class ConvertError(Exception):
    def __init__(self, tree: ErrorNode):
        self.tree: ErrorNode = tree

    def __repr__(self) -> str:
        return f"ConvertError({self.tree!r})"

    def __str__(self) -> str:
        return str(self.tree)


class ErrorNode(abc.ABC):
    @abc.abstractmethod
    def print_error(self, indent="", inside_sum=False, file=sys.stdout):
        ...

    def __str__(self) -> str:
        buf = StringIO()
        self.print_error(file=buf)
        return buf.getvalue().rstrip('\n')


@dataclasses.dataclass
class WrongTypeError(ErrorNode):
    expected: str
    actual: t.Any
    cause: t.Optional[traceback.TracebackException] = None

    def print_error(self, indent="", inside_sum=False, file=sys.stdout):
        if inside_sum:
            print(f"{self.expected}", file=file)
        else:
            print(f"Expected {self.expected}, instead got `{self.actual}` of type `{type(self.actual).__name__}`", file=file)
        if self.cause is not None:
            s = f"{indent}\n".join(self.cause.format())
            print(f"Caused by exception:\n{indent}{s}", file=file)


@dataclasses.dataclass
class DuplicateKeyError(ErrorNode):
    key: str
    aliases: t.Sequence[str]

    def print_error(self, indent="", inside_sum=False, file=sys.stdout):
        assert not inside_sum
        print(f"Duplicate key {self.key} (same as {'/'.join(self.aliases)})", file=file)


@dataclasses.dataclass
class ProductErrorNode(ErrorNode):
    expected: str
    children: t.Dict[t.Union[int, str], ErrorNode]
    actual: t.Any
    missing: t.AbstractSet[t.Union[t.Sequence[str], str]] = dataclasses.field(default_factory=set)
    extra: t.AbstractSet[str] = dataclasses.field(default_factory=set)

    def print_error(self, indent="", inside_sum=False, file=sys.stdout):
        # fuse together non-branching productnodes
        while len(self.children) == 1 and not len(self.missing) and not len(self.extra):
            field, child = next(iter(self.children.items()))
            if not isinstance(child, ProductErrorNode):
                break
            children: t.Dict[t.Union[str, int], ErrorNode] = {f"{field}.{k}": v for (k, v) in child.children.items()}
            missing = set(f"{field}.{f}" for f in child.missing)
            extra = set(f"{field}.{f}" for f in child.extra)
            self = ProductErrorNode(self.expected, children, self.actual, missing, extra)

        print(f"{'' if inside_sum else 'Expected '}{self.expected}", file=file)
        for (field, child) in self.children.items():
            print(f"{indent}While parsing field '{field}':\n{indent}  ", end="", file=file)
            child.print_error(f"{indent}  ", file=file)

        for field in self.missing:
            if not isinstance(field, str):
                field = '/'.join(field)
            print(f"{indent}  Missing required field '{field}'", file=file)

        for field in self.extra:
            print(f"{indent}  Unexpected field '{field}'", file=file)


@dataclasses.dataclass
class SumErrorNode(ErrorNode):
    # sumnodes shouldn't contain sumnodes
    children: t.List[t.Union[ProductErrorNode, WrongTypeError]]

    def print_error(self, indent="", inside_sum=False, file=sys.stdout):
        print(f"Expected one of:", file=file)
        assert len(self.children)
        actual = None
        for child in self.children:
            print(f"{indent}- ", end="", file=file)
            child.print_error(f"{indent}  ", inside_sum=True, file=file)
            actual = child.actual
        print(f"{indent}Instead got `{actual}` of type `{type(actual).__name__}`", file=file)


class Converter(abc.ABC, t.Generic[T_co]):
    def convert(self, val: t.Any) -> T_co:
        try:
            return self.try_convert(val)
        except ParseInterrupt:
            pass
        node = self.collect_errors(val)
        if node is None:
            raise RuntimeError("convert() raised but ``collect_errors`` returned ``None``."
                               " This is a bug of the ``Converter`` implementation.")
        raise ConvertError(node)

    @abc.abstractmethod
    def expected(self, plural: bool = False) -> str:
        ...

    @abc.abstractmethod
    def try_convert(self, val: t.Any) -> T_co:
        """
        Attempt to convert ``val`` to ``T``.
        Should raise ``ParseInterrupt`` when
        a given parsing path reaches a dead end.
        """
        ...

    @abc.abstractmethod
    def collect_errors(self, val: t.Any) -> t.Optional[ErrorNode]:
        """
        Return an error tree caused by converting ``val`` to ``T``.
        ``collect_errors`` should return ``None`` iff ``convert`` doesn't raise.
        """
        ...


@dataclasses.dataclass
class AnyConverter(Converter[t.Any]):
    def try_convert(self, val: t.Any) -> t.Any:
        return val

    def expected(self, plural: bool = False) -> str:
        return "any value" + ("s" if plural else "")

    def collect_errors(self, val: t.Any) -> None:
        return None


@dataclasses.dataclass
class ScalarConverter(Converter[T]):
    ty: t.Type[T]
    allowed: t.Union[type, t.Tuple[type, ...]]
    expect: t.Optional[str] = None
    expect_plural: t.Optional[str] = None

    def __post_init__(self):
        self.expect = self.expect or self.ty.__name__
        self.expect_plural = self.expect_plural or self.expect

    def expected(self, plural: bool = False) -> str:
        return t.cast(str, self.expect_plural if plural else self.expect)

    def try_convert(self, val) -> T:
        if isinstance(val, self.allowed):
            return self.ty(val)  # type: ignore
        raise ParseInterrupt()

    def collect_errors(self, actual) -> t.Optional[WrongTypeError]:
        if isinstance(actual, self.allowed):
            return None
        return WrongTypeError(f'{self.expected()}', actual)


@dataclasses.dataclass
class NoneConverter(Converter[None]):
    def try_convert(self, val: t.Any) -> None:
        if val is None:
            return val
        raise ParseInterrupt()

    def expected(self, plural: bool = False) -> str:
        return "null value" + ("s" if plural else "")

    def collect_errors(self, val) -> t.Optional[WrongTypeError]:
        if val is None:
            return None
        return WrongTypeError(self.expected(), val)


@dataclasses.dataclass
class LiteralConverter(Converter):
    vals: t.Sequence[t.Any]

    def try_convert(self, val: t.Any) -> t.Any:
        if val in self.vals:
            return val
        raise ParseInterrupt()

    def expected(self, plural: bool = False) -> str:
        l = list_phrase(tuple(map(repr, self.vals)))
        return f"({l})" if plural else l

    def collect_errors(self, val) -> t.Optional[WrongTypeError]:
        if val in self.vals:
            return None
        return WrongTypeError(self.expected(), val)


@dataclasses.dataclass(init=False)
class UnionConverter(Converter[t.Any]):
    types: t.Tuple[t.Any, ...]
    converters: t.Tuple[Converter, ...]

    def __init__(self, types: t.Sequence[t.Any]):
        def _flatten_unions(ty: t.Any) -> t.Sequence[t.Any]:
            if t.get_origin(ty) is t.Union:
                return t.get_args(ty)
            return (ty,)

        types = tuple(chain.from_iterable(map(_flatten_unions, types)))
        self.types = types
        self.converters = tuple(map(make_converter, types))

    def expected(self, plural: bool = False) -> str:
        return list_phrase(tuple(conv.expected(plural) for conv in self.converters))

    def try_convert(self, val) -> t.Any:
        for conv in self.converters:
            try:
                return conv.try_convert(val)
            except ParseInterrupt:
                pass
        raise ParseInterrupt

    def collect_errors(self, actual) -> t.Optional[SumErrorNode]:
        failed_children = []
        for (ty, conv) in zip(self.types, self.converters):
            node = conv.collect_errors(actual)
            # if one branch is successful, the whole type is successful
            if node is None:
                return None
            failed_children.append(node)
        return SumErrorNode(failed_children)


@dataclasses.dataclass
class StructConverter(Converter[T]):
    ty: t.Type[T]
    fields: t.Dict[str, t.Any]
    name: t.Optional[str] = None
    opt_fields: t.Set[str] = dataclasses.field(default_factory=set, kw_only=True)
    field_converters: t.Dict[str, Converter] = dataclasses.field(init=False)

    def __post_init__(self):
        self.field_converters = {k: make_converter(v) for (k, v) in self.fields.items()}

    def expected(self, plural: bool = False) -> str:
        p = "s" if plural else ""
        name = f" {self.name}" if self.name is not None else ""
        return f"struct{p}{name}"

    def try_convert(self, val) -> T:
        if not isinstance(val, (dict, t.Mapping)):
            raise ParseInterrupt()
        d = {}
        for (k, v) in val.items():
            if k not in self.fields:
                raise ParseInterrupt()  # unknown field
            d[k] = self.field_converters[k].try_convert(v)
        missing = set(self.fields.keys()) - set(val.keys()) - self.opt_fields
        if len(missing):
            raise ParseInterrupt()
        return self.ty(d)  # type: ignore

    def collect_errors(self, actual) -> t.Union[WrongTypeError, ProductErrorNode, None]:
        if not isinstance(actual, (dict, t.Mapping)):
            return WrongTypeError(self.expected(), actual)

        children = {}
        extra = set()
        for (k, v) in actual.items():
            if k not in self.fields:
                extra.add(k)
                continue
            if (node := self.field_converters[k].collect_errors(v)) is not None:
                children[k] = node
        missing = set(self.fields.keys()) - set(actual.keys()) - self.opt_fields
        if len(children) or len(missing) or len(extra):
            return ProductErrorNode(self.expected(), children, actual, missing, extra)
        return None


@dataclasses.dataclass(init=False)
class TupleConverter(t.Generic[TupleT], Converter[TupleT]):
    ty: t.Type[TupleT]
    converters: t.Tuple[Converter, ...]

    def __init__(self, ty: t.Type[TupleT], types: t.Sequence[t.Type]):
        self.ty = ty
        self.converters = tuple(map(make_converter, types))

    def try_convert(self, val: t.Any) -> TupleT:
        if not isinstance(val, t.Sequence):
            raise ParseInterrupt
        if len(val) != len(self.converters):
            raise ParseInterrupt

        return self.ty(conv.try_convert(v) for (conv, v) in zip(self.converters, val))

    def expected(self, plural: bool = False) -> str:
        s = "s" if plural else ""
        return f"tuple{s} of length {len(self.converters)}"

    def collect_errors(self, val: t.Any) -> t.Union[None, ProductErrorNode, WrongTypeError]:
        if not isinstance(val, t.Sequence) or len(val) != len(self.converters):
            return WrongTypeError(self.expected(), val)
        children = {}
        for (i, (conv, v)) in enumerate(zip(self.converters, val)):
            node = conv.collect_errors(v)
            if node is not None:
                children[i] = node
        if len(children) == 0:
            return None
        return ProductErrorNode(self.expected(), children, val)


@dataclasses.dataclass(init=False)
class DictConverter(t.Generic[K, V], Converter[t.Dict[K, V]]):
    ty: t.Type[t.Dict]
    k_conv: Converter
    v_conv: Converter

    def __init__(self, ty: t.Type[t.Dict], k: t.Type[K] = t.Any, v: t.Type[V] = t.Any):
        self.ty: t.Type[t.Dict] = ty
        self.k_conv = make_converter(k)
        self.v_conv = make_converter(v)

    def expected(self, plural: bool = False) -> str:
        s = "s" if plural else ""
        return f"mapping{s} of {self.k_conv.expected(True)} => {self.v_conv.expected(True)}"

    def try_convert(self, val: t.Any) -> t.Dict[K, V]:
        if not isinstance(val, t.Mapping):
            raise ParseInterrupt()
        return {self.k_conv.try_convert(k): self.v_conv.try_convert(v) for (k, v) in val.items()}

    def collect_errors(self, val: t.Any) -> t.Union[None, WrongTypeError, ProductErrorNode]:
        if not isinstance(val, t.Mapping):
            return WrongTypeError(self.expected(), val)
        nodes = {}
        for (k, v) in val.items():
            if (node := self.k_conv.collect_errors(k)) is not None:
                nodes[k] = node  # TODO split bad fields from bad values
            if (node := self.v_conv.collect_errors(v)) is not None:
                nodes[k] = node
        if len(nodes):
            return ProductErrorNode(self.expected(), nodes, val)


@dataclasses.dataclass(init=False)
class SequenceConverter(t.Generic[T], Converter[t.Sequence[T]]):
    ty: t.Type[t.Sequence]
    v_conv: Converter[T]

    def __init__(self, ty: t.Type[t.Sequence], v: t.Type[T] = t.Any):
        self.ty = ty
        self.v_conv = make_converter(v)

    def expected(self, plural: bool = False) -> str:
        s = "s" if plural else ""
        return f"sequence{s} of {self.v_conv.expected(True)}"

    def try_convert(self, val: t.Any) -> t.Sequence[T]:
        if not isinstance(val, t.Sequence) or isinstance(val, str):
            raise ParseInterrupt
        return self.ty(self.v_conv.try_convert(v) for v in val)  # type: ignore

    def collect_errors(self, val: t.Any) -> t.Union[None, WrongTypeError, ProductErrorNode]:
        if not isinstance(val, t.Sequence) or isinstance(val, str):
            return WrongTypeError("sequence", val)
        nodes = {}
        for (i, v) in enumerate(val):
            if (node := self.v_conv.collect_errors(v)) is not None:
                nodes[i] = node
        if len(nodes):
            return ProductErrorNode("sequence", nodes, val)


@dataclasses.dataclass
class DelegateConverter(t.Generic[T, U], Converter[T]):
    from_type: t.Type[U]
    constructor: t.Callable[[U], T]
    expect: t.Optional[str] = None
    expect_plural: t.Optional[str] = None
    inner: Converter[U] = dataclasses.field(init=False)

    def __post_init__(self):
        self.inner = make_converter(self.from_type)
        self.expect = self.expect or self.from_type.__name__
        self.expect_plural = self.expect_plural or self.expect

    def expected(self, plural: bool = False) -> str:
        return t.cast(str, self.expect_plural if plural else self.expect)

    def try_convert(self, val: t.Any) -> T:
        val = self.inner.try_convert(val)
        try:
            return self.constructor(val)
        except Exception:
            raise ParseInterrupt from None

    def collect_errors(self, val: t.Any) -> t.Optional[ErrorNode]:
        try:
            val = self.inner.try_convert(val)
        except ParseInterrupt:
            return self.inner.collect_errors(val)
        try:
            self.constructor(val)
        except Exception as e:
            tb = e.__traceback__.tb_next  # type: ignore
            tb = traceback.TracebackException(type(e), e, tb)
            return WrongTypeError(self.expected(), val, tb)


_BASIC_CONVERTERS = {
    str: ScalarConverter(str, str, 'a string', 'strings'),
    int: ScalarConverter(int, int, 'an int', 'ints'),
    float: ScalarConverter(float, (int, float), 'a float', 'floats'),
    complex: ScalarConverter(complex, (int, float, complex), 'a complex float', 'complex floats'),
    type(None): NoneConverter(),
}


def make_converter(ty: t.Type[T]) -> Converter[T]:
    if ty is t.Any:
        return AnyConverter()
    if isinstance(ty, t.TypeVar):
        warnings.warn(f"Unbound TypeVar '{ty}'. Will be interpreted as Any.")
        return AnyConverter()
    if isinstance(ty, (dict, t.Dict)):
        return StructConverter(type(ty), ty)
    if isinstance(ty, (tuple, t.Tuple)):
        return TupleConverter(type(ty), ty)
    if isinstance(ty, t.ForwardRef) or isinstance(ty, str):
        raise TypeError(f"Unresolved forward reference '{ty}'")

    base = t.get_origin(ty) or ty
    args = t.get_args(ty)

    # TODO eat annotation

    # special types
    if base is t.Union:
        return UnionConverter(args)
    if base is t.Literal:
        return LiteralConverter(args)

    if not isinstance(base, type):
        raise TypeError(f"Unsupported special type '{base}'")

    if issubclass(base, HasFromData):
        return base._converter(*args, annotations=None)

    if ty in _BASIC_CONVERTERS:
        return _BASIC_CONVERTERS[ty]

    if issubclass(base, (tuple, t.Tuple)) and len(args) > 0 and args[-1] != Ellipsis:
        return TupleConverter(base, args)
    if issubclass(base, (list, t.Sequence)):
        return SequenceConverter(base, args[0] if len(args) > 0 else t.Any)  # type: ignore
    if issubclass(base, (dict, t.Mapping)):
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
    if isinstance(val, DataTypes):
        return val
    if isinstance(val, IntoData):
        return val.into_data()

    raise TypeError(f"Can't convert type '{type(val)}' into data.")


def from_data(val: DataType, ty: t.Type[T]) -> T:
    if not isinstance(val, DataTypes):
        raise TypeError(f"Type {type(val)} is not a valid data interchange type.")

    converter = make_converter(ty)
    return converter.convert(val)


def convert(val: IntoData, ty: t.Type[T]) -> T:
    data = into_data(val)
    return from_data(data, ty)

__all__ = [
    'FromData', 'IntoData', 'DataType', 'ConvertError',
    'from_data', 'into_data', 'make_converter', 'convert',
]
