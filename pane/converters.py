"""
Converter types, which do the hard work of recursive validation.
"""

import abc
import dataclasses
import traceback
import typing as t

from pane.errors import ErrorNode

from .convert import DataType, Convertible, IntoConverter, make_converter, into_data
from .util import list_phrase, pluralize, flatten_union_args, KW_ONLY
from .errors import ConvertError, ParseInterrupt, WrongTypeError, ConditionFailedError
from .errors import ErrorNode, SumErrorNode, ProductErrorNode


T_co = t.TypeVar('T_co', covariant=True)
T = t.TypeVar('T')
U = t.TypeVar('U', bound=Convertible)
FromDataT = t.TypeVar('FromDataT', bound=Convertible)
FromDataK = t.TypeVar('FromDataK', bound=Convertible)
FromDataV = t.TypeVar('FromDataV', bound=Convertible)
NestedSequence = t.Union[T, t.Sequence['NestedSequence[T]']]


def data_is_sequence(val: t.Any) -> t.TypeGuard[t.Sequence[t.Any]]:
    """Return whether `val` is a sequence-like data type."""
    return isinstance(val, t.Sequence) and not isinstance(val, (str, bytes, bytearray))


def data_is_mapping(val: t.Any) -> t.TypeGuard[t.Mapping[t.Any, t.Any]]:
    """Return whether `val` is a mapping-like data type."""
    return isinstance(val, (dict, t.Mapping))


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

    def into_data(self, val: t.Any) -> DataType:
        """
        Convert ``val`` into a data interchange format.

        ``val`` *should* be of a type returned by this converter,
        but don't count on it.
        """
        return into_data(val, None)

    @abc.abstractmethod
    def expected(self, plural: bool = False) -> str:
        """
        Return a descriptive string indicating the value(s) expected.

        Parameters:
          plural: Whether to pluralize the descriptive string
        """
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
        """See [`Converter.try_convert`][pane.converters.Converter.try_convert]"""
        return val

    def expected(self, plural: bool = False) -> str:
        """See [`Converter.expected`][pane.converters.Converter.expected]"""
        return pluralize("any value", plural)

    def collect_errors(self, val: t.Any) -> None:
        """See [`Converter.collect_errors`][pane.converters.Converter.collect_errors]"""
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
        """See [`Converter.expected`][pane.converters.Converter.expected]"""
        return t.cast(str, self.expect_plural if plural else self.expect)

    def try_convert(self, val: t.Any) -> T:
        """See [`Converter.try_convert`][pane.converters.Converter.try_convert]"""
        if isinstance(val, self.allowed):
            return self.ty(val)  # type: ignore
        raise ParseInterrupt()

    def collect_errors(self, val: t.Any) -> t.Optional[WrongTypeError]:
        """See [`Converter.collect_errors`][pane.converters.Converter.collect_errors]"""
        if isinstance(val, self.allowed):
            return None
        return WrongTypeError(f'{self.expected()}', val)


@dataclasses.dataclass
class NoneConverter(Converter[None]):
    """
    Converter which accepts only ``None``.
    """

    def try_convert(self, val: t.Any) -> None:
        """See [`Converter.try_convert`][pane.converters.Converter.try_convert]"""
        if val is None:
            return val
        raise ParseInterrupt()

    def expected(self, plural: bool = False) -> str:
        """See [`Converter.expected`][pane.converters.Converter.expected]"""
        return pluralize("null value", plural)

    def collect_errors(self, val: t.Any) -> t.Optional[WrongTypeError]:
        """See [`Converter.collect_errors`][pane.converters.Converter.collect_errors]"""
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
        """See [`Converter.try_convert`][pane.converters.Converter.try_convert]"""
        if val in self.vals:
            return val
        raise ParseInterrupt()

    def expected(self, plural: bool = False) -> str:
        """See [`Converter.expected`][pane.converters.Converter.expected]"""
        l = list_phrase(tuple(map(repr, self.vals)))
        return f"({l})" if plural else l

    def collect_errors(self, val: t.Any) -> t.Optional[WrongTypeError]:
        """See [`Converter.collect_errors`][pane.converters.Converter.collect_errors]"""
        if val in self.vals:
            return None
        return WrongTypeError(self.expected(), val)


@dataclasses.dataclass(init=False)
class UnionConverter(Converter[t.Any]):
    """
    Converter for an untagged union of subtypes.
    Unions are always evaluated left-to-right.
    """
    types: t.Tuple[IntoConverter, ...]
    """List of potential types"""
    converters: t.Tuple[Converter[t.Any], ...]
    """List of type converters"""
    constructor: t.Optional[t.Callable[[t.Any, int], t.Any]]
    """
    Constructor to call with parsed value.
    Called with ``(val, index of type in union)``
    """

    def __init__(self, types: t.Sequence[IntoConverter],
                 constructor: t.Optional[t.Callable[[t.Any, int], t.Any]] = None):
        self.types = tuple(flatten_union_args(types))
        self.converters = tuple(map(make_converter, types))
        self.constructor = constructor

    def expected(self, plural: bool = False) -> str:
        """See [`Converter.expected`][pane.converters.Converter.expected]"""
        return list_phrase(tuple(conv.expected(plural) for conv in self.converters))

    def into_data(self, val: t.Any) -> DataType:
        """See [`Converter.into_data`][pane.converters.Converter.into_data]"""
        # this is tricky, because we have no type information about which variant ``val`` is.
        # so we basically try_convert each until we find a match
        # this works because try_convert should be idempotent
        for conv in self.converters:
            try:
                conv.try_convert(val)
            except ParseInterrupt:
                pass
            else:
                return conv.into_data(val)
        # default to regular conversion
        return into_data(val)

    def construct(self, val: t.Any, i: int) -> t.Any:
        if self.constructor is None:
            return val
        return self.constructor(val, i)

    def try_convert(self, val: t.Any) -> t.Any:
        """See [`Converter.try_convert`][pane.converters.Converter.try_convert]"""
        for (i, conv) in enumerate(self.converters):
            try:
                val = conv.try_convert(val)
                try:
                    return self.construct(val, i)
                except Exception:
                    pass
            except ParseInterrupt:
                pass
        raise ParseInterrupt

    def collect_errors(self, val: t.Any) -> t.Optional[ErrorNode]:
        """See [`Converter.collect_errors`][pane.converters.Converter.collect_errors]"""
        failed_children: t.List[ErrorNode] = []
        for (i, conv) in enumerate(self.converters):
            # if one branch is successful, the whole type is successful
            try:
                conv_val = conv.try_convert(val)
            except ParseInterrupt:
                failed_children.append(t.cast(t.Union[ProductErrorNode, WrongTypeError], conv.collect_errors(val)))
                continue
            try:
                self.construct(conv_val, i)
                return None
            except Exception as e:
                tb = e.__traceback__.tb_next  # type: ignore
                tb = traceback.TracebackException(type(e), e, tb)
                failed_children.append(WrongTypeError(self.expected(), val, tb))
        return SumErrorNode(failed_children)


@dataclasses.dataclass(init=False)
class TaggedUnionConverter(UnionConverter):
    """
    Converter for a tagged union of subtypes.

    Tagged unions may be parsed in three ways. The default
    is an 'internally tagged' union, where the tag is found
    by looking for a given key in the given object. This is the
    default.
    An 'externally tagged' union is stored as a dict with a key
    and a single value ``{tag: content}``. This may be specified
    using ``external=True``.
    Finally, a 'adjacently tagged' union may be specified using
    two keys ``external=(t, c)``. The union is stored as
    ``{t: tag, c: content}``
    """
    tag: str
    tag_map: t.Dict[t.Any, int]
    """Map from tags to indices into self.types/self.converters"""
    external: t.Union[bool, t.Tuple[str, str]] = False
    """
    Tagged union representation.
    False: internal representation
    True: external representation
    (t, c): adjacent representation
    """

    def __init__(self, types: t.Sequence[t.Any], tag: str,
                 external: t.Union[bool, t.Tuple[str, str]] = False):
        super().__init__(types)
        self.tag = tag
        self.external = external if isinstance(external, t.Sequence) else bool(external)

        # look for tag in each of self.types
        self.tag_map = {}
        for (i, ty) in enumerate(self.types):
            try:
                # TODO error if used on non-literal
                val = getattr(ty, self.tag)
                if val in self.tag_map:
                    raise TypeError(f"Tag value '{val}' matches multiple types")
                self.tag_map[val] = i
            except AttributeError:
                raise AttributeError(f"Tag '{self.tag}' not found inside type '{ty}'")

    def tag_expected(self) -> str:
        """Return a string list of the expected/supported tags"""
        return list_phrase(tuple(map(repr, self.tag_map.keys())))

    def obj_expected(self, plural: bool = False) -> str:
        """Return a string list of the supported objects"""
        return list_phrase(tuple(conv.expected(plural) for conv in self.converters))

    def expected(self, plural: bool = False) -> str:
        """See [`Converter.expected`][pane.converters.Converter.expected]"""
        if self.external is False:
            # internally tagged
            return self.obj_expected(plural)
        obj = self.obj_expected(False)
        mapping = pluralize('mapping', plural, article='a')
        tag = list_phrase(tuple(map(repr, self.tag_map.keys())))
        if self.external is True:
            # externally tagged
            return f"{mapping} '{tag}' => {obj}"
        else:
            # adjacently tagged
            (t, c) = self.external 
            return f"{mapping} {repr(t)} => {tag}, {repr(c)} => {obj}"

    def into_data(self, val: t.Any) -> DataType:
        """See [`Converter.into_data`][pane.converters.Converter.into_data]"""
        tag = getattr(val, self.tag)
        inner_conv = self.converters[self.tag_map[tag]]
        if self.external is False:
            # internally tagged
            return inner_conv.into_data(val)
        if self.external is True:
            # externally tagged
            return {tag: inner_conv.into_data(val)}
        # adjacently tagged
        (t_r, c_r) = self.external
        return {t_r: tag, c_r: inner_conv.into_data(val)}

    def try_convert(self, val: t.Any) -> t.Any:
        """See [`Converter.try_convert`][pane.converters.Converter.try_convert]"""
        if not data_is_mapping(val):
            raise ParseInterrupt()
        val = t.cast(t.Dict[str, t.Any], val)
        tag: t.Any

        if self.external is False:
            try:
                tag = val[self.tag]
            except KeyError:
                raise ParseInterrupt()
        elif self.external is True:
            if len(val) != 1:
                raise ParseInterrupt()
            (tag, val) = next(iter(val.items()))
        else:
            (t_r, c_r) = self.external
            try:
                if len(val) != 2:
                    raise ParseInterrupt()
                tag, val = val[t_r], val[c_r]
            except KeyError:
                raise ParseInterrupt()
        try:
            i = self.tag_map[tag]
        except KeyError:
            raise ParseInterrupt()
        return self.converters[i].try_convert(val)

    def collect_errors(self, val: t.Any) -> t.Optional[ErrorNode]:
        """See [`Converter.collect_errors`][pane.converters.Converter.collect_errors]"""
        if not data_is_mapping(val):
            return WrongTypeError(self.expected(), val)
        val = t.cast(t.Dict[str, t.Any], val)
        tag: t.Any

        if self.external is False:
            try:
                tag = val[self.tag]
            except KeyError:
                return WrongTypeError(f"mapping with key '{self.tag}' => {self.tag_expected()}", val)
        elif self.external is True:
            if len(val) != 1:
                return WrongTypeError(self.expected(), val)
            (tag, val) = next(iter(val.items()))
        else:
            (t_r, c_r) = self.external
            try:
                if len(val) != 2:
                    raise KeyError()
                tag, val = val[t_r], val[c_r]
            except KeyError:
                return WrongTypeError(f"mapping with keys '{t_r}' and '{c_r}'", val)
        try:
            i = self.tag_map[tag]
        except KeyError:
            return WrongTypeError(f"tag '{self.tag}' one of {self.tag_expected()}", tag)
        return self.converters[i].collect_errors(val)


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

    _: KW_ONLY = dataclasses.field(init=False, repr=False, compare=False)
    opt_fields: t.Set[str] = dataclasses.field(default_factory=set)
    """Set of fields which are optional"""
    field_converters: t.Dict[str, Converter[t.Any]] = dataclasses.field(init=False)
    """Dict of sub-converters for each field"""

    def __post_init__(self):
        self.field_converters = {k: make_converter(v) for (k, v) in self.fields.items()}

    def expected(self, plural: bool = False) -> str:
        """See [`Converter.expected`][pane.converters.Converter.expected]"""
        name = f" {self.name}" if self.name is not None else ""
        return f"{pluralize('struct', plural)}{name}"

    def into_data(self, val: t.Any) -> DataType:
        """See [`Converter.into_data`][pane.converters.Converter.into_data]"""
        assert data_is_mapping(val)
        d: t.Dict[DataType, DataType] = {}
        for (k, v) in t.cast(t.Mapping[str, t.Any], val).items():
            if (conv := self.field_converters.get(k)) is not None:
                d[k] = conv.into_data(v)
            else:
                d[k] = into_data(v)
        return d

    def try_convert(self, val: t.Any) -> T:
        """See [`Converter.try_convert`][pane.converters.Converter.try_convert]"""
        if not data_is_mapping(val):
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
        """See [`Converter.collect_errors`][pane.converters.Converter.collect_errors]"""
        if not data_is_mapping(val):
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

    def into_data(self, val: t.Any) -> DataType:
        """See [`Converter.into_data`][pane.converters.Converter.into_data]"""
        return tuple(
            conv.into_data(v)
            for (v, conv) in zip(t.cast(t.Sequence[t.Any], val), self.converters)
        )

    def expected(self, plural: bool = False) -> str:
        """See [`Converter.expected`][pane.converters.Converter.expected]"""
        return f"{pluralize('tuple', plural)} of length {len(self.converters)}"

    def try_convert(self, val: t.Any) -> T:
        """See [`Converter.try_convert`][pane.converters.Converter.try_convert]"""
        if not data_is_sequence(val):
            raise ParseInterrupt
        if len(val) != len(self.converters):
            raise ParseInterrupt

        return self.ty(conv.try_convert(v) for (conv, v) in zip(self.converters, val))

    def collect_errors(self, val: t.Any) -> t.Union[None, ProductErrorNode, WrongTypeError]:
        """See [`Converter.collect_errors`][pane.converters.Converter.collect_errors]"""
        if not data_is_sequence(val) or len(val) != len(self.converters):
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

    def into_data(self, val: t.Any) -> DataType:
        """See [`Converter.into_data`][pane.converters.Converter.into_data]"""
        return {
            self.k_conv.into_data(k): self.v_conv.into_data(v)
            for (k, v) in t.cast(t.Mapping[FromDataK, FromDataV], val).items()
        }

    def expected(self, plural: bool = False) -> str:
        """See [`Converter.expected`][pane.converters.Converter.expected]"""
        return f"{pluralize('mapping', plural)} of {self.k_conv.expected(True)} => {self.v_conv.expected(True)}"

    def try_convert(self, val: t.Any) -> t.Mapping[FromDataK, FromDataV]:
        """See [`Converter.try_convert`][pane.converters.Converter.try_convert]"""
        if not data_is_mapping(val):
            raise ParseInterrupt()

        d = {self.k_conv.try_convert(k): self.v_conv.try_convert(v) for (k, v) in val.items()}
        return self.ty(d)  # type: ignore

    def collect_errors(self, val: t.Any) -> t.Union[None, WrongTypeError, ProductErrorNode]:
        """See [`Converter.collect_errors`][pane.converters.Converter.collect_errors]"""
        if not data_is_mapping(val):
            return WrongTypeError(self.expected(), val)

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

    def into_data(self, val: t.Any) -> DataType:
        """See [`Converter.into_data`][pane.converters.Converter.into_data]"""
        return [
            self.v_conv.into_data(v)
            for v in t.cast(t.Sequence[FromDataT], val)
        ]

    def expected(self, plural: bool = False) -> str:
        """See [`Converter.expected`][pane.converters.Converter.expected]"""
        return f"{pluralize('sequence', plural)} of {self.v_conv.expected(True)}"

    def try_convert(self, val: t.Any) -> t.Sequence[FromDataT]:
        """See [`Converter.try_convert`][pane.converters.Converter.try_convert]"""
        if not data_is_sequence(val):
            raise ParseInterrupt
        try:
            return self.ty(self.v_conv.try_convert(v) for v in val)  # type: ignore
        except Exception:
            raise ParseInterrupt()

    def collect_errors(self, val: t.Any) -> t.Union[None, WrongTypeError, ProductErrorNode]:
        """See [`Converter.collect_errors`][pane.converters.Converter.collect_errors]"""
        if not data_is_sequence(val):
            return WrongTypeError(self.expected(), val)

        nodes = {}
        vals: t.List[FromDataT] = []
        for (i, v) in enumerate(val):
            try:
                vals.append(self.v_conv.convert(v))
            except ConvertError as e:
                nodes[i] = e.tree

        if len(nodes):
            return ProductErrorNode(self.expected(), nodes, val)
        # try to construct val
        try:
            self.ty(iter(vals))
            return None
        except Exception as e:
            tb = e.__traceback__.tb_next  # type: ignore
            tb = traceback.TracebackException(type(e), e, tb)
            return WrongTypeError(self.expected(), val, tb)


@dataclasses.dataclass
class NestedSequenceConverter(t.Generic[T, U], Converter[T]):
    """
    Converter which delegates to a sub-converter, and then attempts
    to construct a different type
    """
    val_type: t.Type[U]
    """Inner type to convert to"""
    constructor: t.Callable[[NestedSequence[U]], T]
    """Constructor to call."""
    ragged: bool = False
    """Whether to accept ragged arrays."""
    val_conv: Converter[U] = dataclasses.field(init=False)
    """[`Converter`][pane.converters.Converter] for value type"""

    def __post_init__(self):
        self.val_conv = make_converter(self.val_type)

    def expected(self, plural: bool = False) -> str:
        """See [`Converter.expected`][pane.converters.Converter.expected]"""
        word = 'nested sequence' if self.ragged else 'n-d array'
        return f"{pluralize(word, plural, article='a')} of {self.val_conv.expected(True)}"

    @staticmethod
    def _check_shape(val: NestedSequence[t.Any], dim: int = 0) -> t.Tuple[int, ...]:
        if not data_is_sequence(val):
            # single value
            return ()
        shapes = [NestedSequenceConverter._check_shape(v, dim+1) for v in val]
        if len(shapes) == 0:
            return (0,)
        shape = shapes[0]
        if not all(s == shape for s in shapes):
            raise ValueError(f"shape mismatch at dim {dim}. Sub-shapes: {shapes}")
        new_shape = (len(shapes), *shape)
        return new_shape

    def try_convert(self, val: t.Any) -> T:
        """See [`Converter.try_convert`][pane.converters.Converter.try_convert]"""
        result = self._try_convert(val)
        if not self.ragged:
            try:
                self._check_shape(result)
            except ValueError:
                raise ParseInterrupt()
        return self.constructor(result)

    def _try_convert(self, val: t.Any) -> NestedSequence[U]:
        if not data_is_sequence(val):
            # single value
            return self.val_conv.try_convert(val)
        vals = list(map(self._try_convert, val))
        return t.cast(NestedSequence[U], vals)

    def collect_errors(self, val: t.Any) -> t.Optional[ErrorNode]:
        """See [`Converter.collect_errors`][pane.converters.Converter.collect_errors]"""
        if (node := self._collect_errors(val)) is not None:
            return node
        val = self._try_convert(val)
        if not self.ragged:
            try:
                self._check_shape(val)
            except ValueError as e:
                return WrongTypeError(self.expected(), val, info=e.args[0])
        try:
            self.constructor(val)
        except Exception as e:
            tb = e.__traceback__.tb_next  # type: ignore
            tb = traceback.TracebackException(type(e), e, tb)
            return WrongTypeError(self.expected(), val, tb)

    def _collect_errors(self, val: t.Any) -> t.Optional[ErrorNode]:
        if not data_is_sequence(val):
            return self.val_conv.collect_errors(val)
        nodes = {}
        for (i, v) in enumerate(val):
            if (node := self._collect_errors(v)) is not None:
                nodes[i] = node
        if len(nodes):
            return ProductErrorNode(self.expected(), nodes, val)


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
    """Human-readable name of condition"""
    make_expected: t.Callable[[str, bool], str]
    """Function which takes ``(expected, plural)`` and makes a compound ``expected``."""
    inner: Converter[FromDataT] = dataclasses.field(init=False)
    """Inner sub-converter"""

    def __post_init__(self):
        if isinstance(self.inner_type, Converter):
            self.inner = self.inner_type
        else:
            self.inner = make_converter(t.cast(t.Type[FromDataT], self.inner_type))

    def into_data(self, val: t.Any) -> DataType:
        """See [`Converter.into_data`][pane.converters.Converter.into_data]"""
        return self.inner.into_data(val)

    def expected(self, plural: bool = False) -> str:
        """See [`Converter.expected`][pane.converters.Converter.expected]"""
        return self.make_expected(self.inner.expected(plural), plural)

    def try_convert(self, val: t.Any) -> FromDataT:
        """See [`Converter.try_convert`][pane.converters.Converter.try_convert]"""
        val = self.inner.try_convert(val)
        try:
            if self.condition(val):
                return val
        except Exception:
            pass
        raise ParseInterrupt()

    def collect_errors(self, val: t.Any) -> t.Optional[ErrorNode]:
        """See [`Converter.collect_errors`][pane.converters.Converter.collect_errors]"""
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

    def into_data(self, val: t.Any) -> DataType:
        """See [`Converter.into_data`][pane.converters.Converter.into_data]"""
        # TODO: this is a hack, because we can't easily convert T back to U
        try:
            return self.inner.into_data(val)
        except Exception:
            pass
        return into_data(val)

    def expected(self, plural: bool = False) -> str:
        """See [`Converter.expected`][pane.converters.Converter.expected]"""
        return t.cast(str, self.expect_plural if plural else self.expect)

    def try_convert(self, val: t.Any) -> T:
        """See [`Converter.try_convert`][pane.converters.Converter.try_convert]"""
        val = self.inner.try_convert(val)
        try:
            return self.constructor(val)
        except Exception:
            raise ParseInterrupt from None

    def collect_errors(self, val: t.Any) -> t.Optional[ErrorNode]:
        """See [`Converter.collect_errors`][pane.converters.Converter.collect_errors]"""
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
_BASIC_CONVERTERS: t.Dict[type, Converter[t.Any]] = {
    complex: ScalarConverter(complex, (int, float, complex), 'a complex float', 'complex floats'),
    float: ScalarConverter(float, (int, float), 'a float', 'floats'),
    int: ScalarConverter(int, int, 'an int', 'ints'),
    str: ScalarConverter(str, str, 'a string', 'strings'),
    bytes: ScalarConverter(bytes, (bytes, bytearray), 'a bytestring', 'bytestrings'),
    bytearray: ScalarConverter(bytearray, (bytes, bytearray), 'a bytearray', 'bytearrays'),
    type(None): NoneConverter(),
}
"""Built-in scalar converters for some basic types"""

__all__ = [
    'Converter', 'AnyConverter', 'ScalarConverter', 'NoneConverter', 'LiteralConverter',
    'UnionConverter', 'StructConverter', 'TupleConverter', 'DictConverter', 'SequenceConverter',
    'DelegateConverter', 'ConditionalConverter',
]
