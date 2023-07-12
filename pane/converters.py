"""
Converter types, which do the hard work of recursive validation.
"""

import abc
import dataclasses
import traceback
from itertools import chain
import typing as t

from pane.errors import ErrorNode

from .convert import FromData, IntoConverter, make_converter
from .util import list_phrase, pluralize
from .errors import ConvertError, ParseInterrupt, WrongTypeError, ConditionFailedError
from .errors import ErrorNode, SumErrorNode, ProductErrorNode


T_co = t.TypeVar('T_co', covariant=True)
T = t.TypeVar('T')
U = t.TypeVar('U', bound=FromData)
FromDataT = t.TypeVar('FromDataT', bound=FromData)
FromDataK = t.TypeVar('FromDataK', bound=FromData)
FromDataV = t.TypeVar('FromDataV', bound=FromData)


class Converter(abc.ABC, t.Generic[T_co]):
    """
    Base class for a converter to a given type ``T_co``.
    """

    def convert(self, val: t.Any) -> T_co:
        """Convert ``val`` to ``T_co``. Raises a ``ConvertError`` on failure."""
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
        """Return a descriptive string indicating the value(s) expected."""
        ...

    @abc.abstractmethod
    def try_convert(self, val: t.Any) -> T_co:
        """
        Attempt to convert ``val`` to ``T``.
        Should raise ``ParseInterrupt`` (and only ``ParseInterrupt``)
        when a given parsing path reaches a dead end.
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
    """Converter for ``t.Any``."""
    def try_convert(self, val: t.Any) -> t.Any:
        return val

    def expected(self, plural: bool = False) -> str:
        return pluralize("any value", plural)

    def collect_errors(self, val: t.Any) -> None:
        return None


@dataclasses.dataclass
class ScalarConverter(Converter[T]):
    """
    Converter for a simple scalar type,
    constructible from a list of allowed types.
    """

    ty: t.Type[T]
    """Type to convert into."""
    allowed: t.Union[type, t.Tuple[type, ...]]
    """Type or list of allowed types."""
    expect: t.Optional[str] = None
    """Singular form of expected value."""
    expect_plural: t.Optional[str] = None
    """Plural form of expected value."""

    def __post_init__(self):
        self.expect = self.expect or self.ty.__name__
        self.expect_plural = self.expect_plural or self.expect

    def expected(self, plural: bool = False) -> str:
        return t.cast(str, self.expect_plural if plural else self.expect)

    def try_convert(self, val: t.Any) -> T:
        if isinstance(val, self.allowed):
            return self.ty(val)  # type: ignore
        raise ParseInterrupt()

    def collect_errors(self, val: t.Any) -> t.Optional[WrongTypeError]:
        if isinstance(val, self.allowed):
            return None
        return WrongTypeError(f'{self.expected()}', val)


@dataclasses.dataclass
class NoneConverter(Converter[None]):
    """
    Converter which accepts only ``None``.
    """

    def try_convert(self, val: t.Any) -> None:
        if val is None:
            return val
        raise ParseInterrupt()

    def expected(self, plural: bool = False) -> str:
        return pluralize("null value", plural)

    def collect_errors(self, val: t.Any) -> t.Optional[WrongTypeError]:
        if val is None:
            return None
        return WrongTypeError(self.expected(), val)


@dataclasses.dataclass
class LiteralConverter(Converter[T_co]):
    """
    Converter which accepts any of a list of literal values.
    """

    vals: t.Sequence[T_co]

    def try_convert(self, val: t.Any) -> T_co:
        if val in self.vals:
            return val
        raise ParseInterrupt()

    def expected(self, plural: bool = False) -> str:
        l = list_phrase(tuple(map(repr, self.vals)))
        return f"({l})" if plural else l

    def collect_errors(self, val: t.Any) -> t.Optional[WrongTypeError]:
        if val in self.vals:
            return None
        return WrongTypeError(self.expected(), val)


@dataclasses.dataclass(init=False)
class UnionConverter(Converter[t.Any]):
    """
    Converter which accepts one of a union of subtypes.
    Unions are always evaluated left-to-right.
    """
    types: t.Tuple[IntoConverter, ...]
    """List of potential types"""
    converters: t.Tuple[Converter[t.Any], ...]
    """List of type converters"""

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

    def try_convert(self, val: t.Any) -> t.Any:
        for conv in self.converters:
            try:
                return conv.try_convert(val)
            except ParseInterrupt:
                pass
        raise ParseInterrupt

    def collect_errors(self, val: t.Any) -> t.Optional[SumErrorNode]:
        failed_children: t.List[t.Union[ProductErrorNode, WrongTypeError]] = []
        for (_, conv) in zip(self.types, self.converters):
            node = conv.collect_errors(val)
            # if one branch is successful, the whole type is successful
            if node is None:
                return None
            failed_children.append(t.cast(t.Union[ProductErrorNode, WrongTypeError], node))
        return SumErrorNode(failed_children)


@dataclasses.dataclass
class StructConverter(Converter[T]):
    """
    Converter for a simple, hetereogeneous struct-like type, constructible from a dict.
    """

    ty: t.Type[T]
    """Type to convert into. Must be constructible from a dict/mapping."""
    fields: t.Mapping[str, IntoConverter]
    """List of fields and their types"""
    name: t.Optional[str] = None
    """Optional name of struct"""
    opt_fields: t.Set[str] = dataclasses.field(default_factory=set, kw_only=True)
    """Set of fields which are optional"""
    field_converters: t.Dict[str, Converter[t.Any]] = dataclasses.field(init=False)
    """Dict of sub-converters for each field"""

    def __post_init__(self):
        self.field_converters = {k: make_converter(v) for (k, v) in self.fields.items()}

    def expected(self, plural: bool = False) -> str:
        name = f" {self.name}" if self.name is not None else ""
        return f"{pluralize('struct', plural)}{name}"

    def try_convert(self, val: t.Any) -> T:
        if not isinstance(val, (dict, t.Mapping)):
            raise ParseInterrupt()
        val = t.cast(t.Dict[str, t.Any], val)
        d: t.Dict[str, t.Any] = {}
        for (k, v) in val.items():
            if k not in self.fields:
                raise ParseInterrupt()  # unknown field
            d[k] = self.field_converters[k].try_convert(v)
        missing = set(self.fields.keys()) - set(val.keys()) - self.opt_fields
        if len(missing):
            raise ParseInterrupt()
        return self.ty(d)  # type: ignore

    def collect_errors(self, val: t.Any) -> t.Union[WrongTypeError, ProductErrorNode, None]:
        if not isinstance(val, (dict, t.Mapping)):
            return WrongTypeError(self.expected(), val)
        val = t.cast(t.Dict[str, t.Any], val)

        children: t.Dict[t.Union[str, int], t.Any] = {}
        extra: t.Set[str] = set()
        for (k, v) in val.items():
            if k not in self.fields:
                extra.add(k)
                continue
            if (node := self.field_converters[k].collect_errors(v)) is not None:
                children[k] = node
        missing = set(self.fields.keys()) - set(val.keys()) - self.opt_fields
        if len(children) or len(missing) or len(extra):
            return ProductErrorNode(self.expected(), children, val, missing, extra)
        return None


@dataclasses.dataclass(init=False)
class TupleConverter(t.Generic[T], Converter[T]):
    """Converter for a simple, heterogeneous tuple-like type"""
    ty: t.Type[T]
    """Type to convert into. Must be constructible from a sequence/tuple"""
    converters: t.Tuple[Converter[t.Any], ...]
    """List of sub-converters for each field"""

    def __init__(self, ty: t.Type[T], types: t.Sequence[IntoConverter]):
        self.ty = ty
        self.converters = tuple(map(make_converter, types))

    def try_convert(self, val: t.Any) -> T:
        if not isinstance(val, t.Sequence):
            raise ParseInterrupt
        val = t.cast(t.Sequence[t.Any], val)
        if len(val) != len(self.converters):
            raise ParseInterrupt

        return self.ty(conv.try_convert(v) for (conv, v) in zip(self.converters, val))

    def expected(self, plural: bool = False) -> str:
        return f"{pluralize('tuple', plural)} of length {len(self.converters)}"

    def collect_errors(self, val: t.Any) -> t.Union[None, ProductErrorNode, WrongTypeError]:
        if not isinstance(val, t.Sequence) or len(val) != len(self.converters):  # type: ignore
            return WrongTypeError(self.expected(), val)
        val = t.cast(t.Sequence[t.Any], val)
        children = {}
        for (i, (conv, v)) in enumerate(zip(self.converters, val)):
            node = conv.collect_errors(v)
            if node is not None:
                children[i] = node
        if len(children) == 0:
            return None
        return ProductErrorNode(self.expected(), children, val)


@dataclasses.dataclass(init=False)
class DictConverter(t.Generic[FromDataK, FromDataV], Converter[t.Mapping[FromDataK, FromDataV]]):
    """Converter for a homogenous dict-like type."""
    ty: t.Type[t.Mapping[FromDataK, FromDataV]]
    """Type to convert into. Must be constructible from a dict"""
    k_conv: Converter[FromDataK]
    """Sub-converter for keys"""
    v_conv: Converter[FromDataV]
    """Sub-converter for values"""

    def __init__(self, ty: t.Type[t.Dict[t.Any, t.Any]], k: t.Type[FromDataK] = t.Any, v: t.Type[FromDataV] = t.Any):
        self.ty = ty
        self.k_conv = make_converter(k)
        self.v_conv = make_converter(v)

    def expected(self, plural: bool = False) -> str:
        return f"{pluralize('mapping', plural)} of {self.k_conv.expected(True)} => {self.v_conv.expected(True)}"

    def try_convert(self, val: t.Any) -> t.Mapping[FromDataK, FromDataV]:
        if not isinstance(val, t.Mapping):
            raise ParseInterrupt()
        val = t.cast(t.Mapping[t.Any, t.Any], val)
        d = {self.k_conv.try_convert(k): self.v_conv.try_convert(v) for (k, v) in val.items()}
        return self.ty(d)  # type: ignore

    def collect_errors(self, val: t.Any) -> t.Union[None, WrongTypeError, ProductErrorNode]:
        if not isinstance(val, t.Mapping):
            return WrongTypeError(self.expected(), val)
        val = t.cast(t.Mapping[t.Any, t.Any], val)
        nodes: t.Dict[t.Union[str, int], ErrorNode] = {}
        for (k, v) in val.items():
            if (node := self.k_conv.collect_errors(k)) is not None:
                nodes[str(k)] = node  # TODO split bad fields from bad values
            if (node := self.v_conv.collect_errors(v)) is not None:
                nodes[str(k)] = node
        if len(nodes):
            return ProductErrorNode(self.expected(), nodes, val)


@dataclasses.dataclass(init=False)
class SequenceConverter(t.Generic[FromDataT], Converter[t.Sequence[FromDataT]]):
    """Converter for a homogenous sequence-like type"""
    ty: type
    """Type to convert into. Must be constructible from a tuple/sequence."""
    v_conv: Converter[FromDataT]
    """Sub-converter for values"""

    def __init__(self, ty: t.Type[t.Sequence[t.Any]], v: t.Type[FromDataT] = t.Any):
        self.ty = ty
        self.v_conv = make_converter(v)

    def expected(self, plural: bool = False) -> str:
        return f"{pluralize('sequence', plural)} of {self.v_conv.expected(True)}"

    def try_convert(self, val: t.Any) -> t.Sequence[FromDataT]:
        if not isinstance(val, t.Sequence) or isinstance(val, str):
            raise ParseInterrupt
        return self.ty(self.v_conv.try_convert(v) for v in val)  # type: ignore

    def collect_errors(self, val: t.Any) -> t.Union[None, WrongTypeError, ProductErrorNode]:
        if not isinstance(val, t.Sequence) or isinstance(val, str):
            return WrongTypeError("sequence", val)
        val = t.cast(t.Sequence[t.Any], val)
        nodes = {}
        for (i, v) in enumerate(val):
            if (node := self.v_conv.collect_errors(v)) is not None:
                nodes[i] = node
        if len(nodes):
            return ProductErrorNode("sequence", nodes, val)


@dataclasses.dataclass
class ConditionalConverter(t.Generic[FromDataT], Converter[FromDataT]):
    """
    Converter which applies an arbitrary pre-condition to the converted value.
    """
    inner_type: t.Union[t.Type[FromDataT], Converter[FromDataT]]
    """Inner type to apply condition to"""
    condition: t.Callable[[FromDataT], bool]
    """Function to evaluate condition"""
    condition_name: str
    make_expected: t.Callable[[str, bool], str]
    """Function which takes ``(expected, plural)`` and makes a compound ``expected``."""
    inner: Converter[FromDataT] = dataclasses.field(init=False)
    """Inner sub-converter"""

    def __post_init__(self):
        if isinstance(self.inner_type, Converter):
            self.inner = self.inner_type
        else:
            self.inner = make_converter(t.cast(t.Type[FromDataT], self.inner_type))

    def expected(self, plural: bool = False) -> str:
        return self.make_expected(self.inner.expected(plural), plural)

    def try_convert(self, val: t.Any) -> FromDataT:
        val = self.inner.try_convert(val)
        try:
            if self.condition(val):
                return val
        except Exception:
            pass
        raise ParseInterrupt()

    def collect_errors(self, val: t.Any) -> t.Optional[ErrorNode]:
        try:
            conv_val = self.inner.try_convert(val)
        except ParseInterrupt:
            # TODO with_expected() here
            return self.inner.collect_errors(val)
        try:
            # condition failed
            if not self.condition(conv_val):
                return ConditionFailedError(self.expected(), val, self.condition_name)
        except Exception as e:
            tb = e.__traceback__.tb_next  # type: ignore
            tb = traceback.TracebackException(type(e), e, tb)
            return ConditionFailedError(self.expected(), val, self.condition_name, tb)
        return None


@dataclasses.dataclass
class DelegateConverter(t.Generic[T, U], Converter[T]):
    """
    Converter which delegates to a sub-converter, and then attempts
    to construct a different type
    """
    from_type: t.Type[U]
    """Inner type to convert to"""
    constructor: t.Callable[[U], T]
    """Constructor for outer type"""
    expect: t.Optional[str] = None
    """Expected value. Defaults to inner expected value."""
    expect_plural: t.Optional[str] = None
    """Plural expected value. Defaults to inner expected value."""
    inner: Converter[U] = dataclasses.field(init=False)
    """Inner sub-converter"""

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
            conv_val = self.inner.try_convert(val)
        except ParseInterrupt:
            # TODO with_expected() here
            return self.inner.collect_errors(val)
        try:
            self.constructor(conv_val)
        except Exception as e:
            tb = e.__traceback__.tb_next  # type: ignore
            tb = traceback.TracebackException(type(e), e, tb)
            return WrongTypeError(self.expected(), val, tb)


# converters for scalar types
_BASIC_CONVERTERS = {
    str: ScalarConverter(str, str, 'a string', 'strings'),
    int: ScalarConverter(int, int, 'an int', 'ints'),
    float: ScalarConverter(float, (int, float), 'a float', 'floats'),
    complex: ScalarConverter(complex, (int, float, complex), 'a complex float', 'complex floats'),
    type(None): NoneConverter(),
}
"""Built-in scalar converters for some basic types"""

__all__ = [
    'Converter', 'AnyConverter', 'ScalarConverter', 'NoneConverter', 'LiteralConverter',
    'UnionConverter', 'StructConverter', 'TupleConverter', 'DictConverter', 'SequenceConverter',
    'DelegateConverter', 'ConditionalConverter',
]
