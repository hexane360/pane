from __future__ import annotations

import abc
import sys
from io import StringIO
import dataclasses
from itertools import chain

import typing as t


DataType = t.Union[str, int, bool, float, complex, None, t.Mapping, t.Sequence]
DataTypes = (str, int, bool, float, complex, type(None), t.Mapping, t.Sequence)
T_co = t.TypeVar('T_co', covariant=True)
FromDataT = t.TypeVar('FromDataT', bound='FromData')
Convertible = t.Union[DataType, 'FromData']
T = t.TypeVar('T', bound=Convertible)
TupleT = t.TypeVar('TupleT', bound=tuple)
K = t.TypeVar('K', bound=Convertible)
V = t.TypeVar('V', bound=Convertible)


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
        return buf.getvalue()


@dataclasses.dataclass
class SimpleErrorNode(ErrorNode):
    expected: str
    actual: t.Any

    def print_error(self, indent="", inside_sum=False, file=sys.stdout):
        if inside_sum:
            print(f"{self.expected}", file=file)
        else:
            print(f"Expected {self.expected}, instead got `{self.actual}` of type `{type(self.actual).__name__}`", file=file)


@dataclasses.dataclass
class ProductErrorNode:
    expected: str
    children: t.Dict[str, ErrorNode]
    missing: t.Set[str]
    actual: t.Any

    def print_error(self, indent="", inside_sum=False, file=sys.stdout):
        # fuse together consecutive productnodes
        while len(self.children) == 1 and len(self.missing) == 0:
            field, child = next(iter(self.children.items()))
            if not isinstance(child, ProductErrorNode):
                break
            self = ProductErrorNode(self.expected, {f"{field}.{k}": v for (k, v) in child.children.items()}, set(), self.actual)

        print(f"{'' if inside_sum else 'Expected '}{self.expected}", file=file)
        for (field, child) in self.children.items():
            print(f"{indent}While parsing field '{field}':\n{indent}  ", end="", file=file)
            child.print_error(f"{indent}  ", file=file)

        for field in self.missing:
            print(f"{indent}  Missing required field '{field}'", file=file)


@dataclasses.dataclass
class SumErrorNode:
    # sumnodes shouldn't contain sumnodes
    children: t.List[t.Union[ProductErrorNode, SimpleErrorNode]]

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


@t.runtime_checkable
class FromData(t.Protocol):
    @classmethod
    def _converter(cls: t.Type[FromDataT], *args: t.Type[Convertible],
                   annotations: t.Optional[t.Tuple[t.Any, ...]] = None) -> Converter[FromDataT]:
        ...


@t.runtime_checkable
class IntoData(t.Protocol):
    @classmethod
    def into_data(cls) -> DataType:
        ...


@dataclasses.dataclass
class AnyConverter(Converter[t.Any]):
    def try_convert(self, val: t.Any) -> t.Any:
        return val

    def collect_errors(self, val: t.Any) -> None:
        return None


@dataclasses.dataclass
class ScalarConverter(Converter[T]):
    ty: t.Type[T]
    allowed: t.Union[type, t.Tuple[type, ...]]
    expected: t.Optional[str] = None

    def __post_init__(self):
        self.expected = self.expected or self.ty.__name__

    def try_convert(self, val) -> T:
        if isinstance(val, self.allowed):
            return self.ty(val)  # type: ignore
        raise ParseInterrupt()

    def collect_errors(self, actual) -> t.Optional[SimpleErrorNode]:
        if isinstance(actual, self.allowed):
            return None
        return SimpleErrorNode(f'{self.expected}', actual)


@dataclasses.dataclass
class NoneConverter(Converter[None]):
    def try_convert(self, val: t.Any) -> None:
        if val is None:
            return val
        raise ParseInterrupt()

    def collect_errors(self, val) -> t.Optional[SimpleErrorNode]:
        if val is None:
            return None
        return SimpleErrorNode("null value", val)


@dataclasses.dataclass
class LiteralConverter(Converter):
    vals: t.Sequence[t.Any]

    def try_convert(self, val: t.Any) -> t.Any:
        if val in self.vals:
            return val
        raise ParseInterrupt()

    def collect_errors(self, val) -> t.Optional[SimpleErrorNode]:
        if val in self.vals:
            return None
        return SimpleErrorNode(', '.join(map(repr, self.vals)), val)


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
class StructConverter(Converter[dict]):
    name: str
    fields: t.Dict[str, t.Any]
    field_converters: t.Dict[str, Converter] = dataclasses.field(init=False)

    def __post_init__(self):
        self.field_converters = {k: make_converter(v) for (k, v) in self.fields.items()}

    def try_convert(self, val) -> t.Any:
        if not isinstance(val, dict):
            raise ParseInterrupt()
        try:
            return {
                k: conv.try_convert(val[k]) for (k, conv) in self.field_converters.items()
            }
        except KeyError:
            raise ParseInterrupt() from None

    def collect_errors(self, actual) -> t.Optional[ProductErrorNode]:
        failed_children = {}
        missing = set()
        for (k, conv) in self.field_converters.items():
            if k not in actual:
                missing.add(k)
                continue
            node = conv.collect_errors(actual[k])
            if node is not None:
                failed_children[k] = node
        if not len(failed_children) and not len(missing):
            return None
        return ProductErrorNode(self.name, failed_children, missing, actual)


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

    def expected(self) -> str:
        return f"tuple of length {len(self.converters)}"

    def collect_errors(self, val: t.Any) -> t.Union[None, ProductErrorNode, SimpleErrorNode]:
        if not isinstance(val, t.Sequence) or len(val) != len(self.converters):
            return SimpleErrorNode(self.expected(), val)
        children = {}
        for (i, (conv, v)) in enumerate(zip(self.converters, val)):
            node = conv.collect_errors(v)
            if node is not None:
                children[i] = node
        if len(children) == 0:
            return None
        return ProductErrorNode(self.expected(), children, set(), val)


@dataclasses.dataclass(init=False)
class DictConverter(t.Generic[K, V], Converter[t.Dict[K, V]]):
    ty: t.Type[t.Dict]
    k_conv: Converter
    v_conv: Converter

    def __init__(self, ty: t.Type[t.Dict], k: t.Type[K] = t.Any, v: t.Type[V] = t.Any):
        self.ty: t.Type[t.Dict] = ty
        self.k_conv = make_converter(k)
        self.v_conv = make_converter(v)

    def try_convert(self, val: t.Any) -> t.Dict[K, V]:
        if not isinstance(val, t.Mapping):
            raise ParseInterrupt()
        return {self.k_conv.try_convert(k): self.v_conv.try_convert(v) for (k, v) in val.items()}

    def collect_errors(self, val: t.Any) -> t.Union[None, SimpleErrorNode, ProductErrorNode]:
        if not isinstance(val, t.Mapping):
            return SimpleErrorNode("mapping", val)
        nodes = {}
        for (k, v) in val.items():
            if (node := self.k_conv.collect_errors(k)) is not None:
                nodes[k] = node  # TODO split bad fields from bad values
            if (node := self.v_conv.collect_errors(v)) is not None:
                nodes[k] = node
        if len(nodes):
            return ProductErrorNode('mapping', nodes, set(), val)


@dataclasses.dataclass(init=False)
class SequenceConverter(t.Generic[T], Converter[t.Sequence[T]]):
    ty: t.Type[t.Sequence]
    v_conv: Converter[T]

    def __init__(self, ty: t.Type[t.Sequence], v: t.Type[T] = t.Any):
        self.ty = ty
        self.v_conv = make_converter(v)

    def try_convert(self, val: t.Any) -> t.Sequence[T]:
        if not isinstance(val, t.Sequence):
            raise ParseInterrupt
        return self.ty(self.v_conv.try_convert(v) for v in val)  # type: ignore

    def collect_errors(self, val: t.Any) -> t.Union[None, SimpleErrorNode, ProductErrorNode]:
        if not isinstance(val, t.Sequence):
            return SimpleErrorNode("sequence", val)
        nodes = {}
        for (i, v) in enumerate(val):
            if (node := self.v_conv.collect_errors(v)) is not None:
                nodes[i] = node
        if len(nodes):
            return ProductErrorNode("sequence", nodes, set(), val)


_BASIC_CONVERTERS = {
    str: ScalarConverter(str, str, 'a str'),
    int: ScalarConverter(int, int, 'an int'),
    float: ScalarConverter(float, (int, float), 'an int'),
    complex: ScalarConverter(complex, (int, float, complex), 'a complex float'),
}


def make_converter(ty: t.Type[T]) -> Converter[T]:
    if ty is t.Any:
        return AnyConverter()

    if isinstance(ty, (dict, t.Dict)):
        return StructConverter('dict', ty)  # type: ignore

    if isinstance(ty, (tuple, t.Tuple)):
        return TupleConverter(type(ty), ty)

    base = t.get_origin(ty) or ty
    args = t.get_args(ty)

    # TODO forward ref check
    # TODO eat annotation

    # special types
    if base is t.Union:
        return UnionConverter(args)
    if base is t.Literal:
        return LiteralConverter(args)
    if not isinstance(base, type):
        raise TypeError(f"Unsupported special type '{base}'")

    if issubclass(base, FromData):
        return base._converter(*args, annotations=None)

    if ty in _BASIC_CONVERTERS:
        return _BASIC_CONVERTERS[ty]
    if base is None:
        return NoneConverter()  # type: ignore

    if issubclass(base, (tuple, t.Tuple)) and len(args) > 0 and args[-1] != Ellipsis:
        return TupleConverter(base, args)
    if issubclass(base, (list, t.Sequence)):
        return SequenceConverter(base, args[0] if len(args) > 0 else t.Any)  # type: ignore
    if issubclass(base, (dict, t.Mapping)):
        args = t.get_args(ty)
        return DictConverter(base,  # type: ignore
                             args[0] if len(args) > 0 else t.Any,
                             args[1] if len(args) > 1 else t.Any)  # type: ignore

    raise TypeError(f"Can't convert data into type '{ty}'")