"""
Pane dataclasses.
"""

from __future__ import annotations

import dataclasses
from dataclasses import FrozenInstanceError
import functools
from inspect import Signature, Parameter
import traceback
import typing as t

from typing_extensions import dataclass_transform, ParamSpec, Self, TypeAlias

from .convert import DataType, Convertible, from_data, into_data, convert
from .convert import ConverterHandler, ConverterHandlers, IntoConverterHandlers
from .converters import Converter, make_converter
from .errors import ConvertError, ParseInterrupt, ErrorNode
from .errors import WrongTypeError, WrongLenError, ProductErrorNode, DuplicateKeyError
from .field import Field, FieldSpec, field, RenameStyle, rename_field, _MISSING
from .util import get_type_hints, list_phrase, KW_ONLY
from . import io


T = t.TypeVar('T')
PaneBaseT = t.TypeVar('PaneBaseT', bound='PaneBase')


@dataclass_transform(
    eq_default=True,
    order_default=True,
    frozen_default=True,
    kw_only_default=False,
    field_specifiers=(FieldSpec, field),
)
class PaneBase:
    """
    Base class for all `pane` dataclasses
    """
    __slots__ = ('__pane_set__',)

    __pane_info__: PaneInfo
    """Dunder attribute holding [`PaneInfo`][pane.classes.PaneInfo]"""
    __pane_set__: t.Set[str]
    """Dunder attribute holding a set of fields which have been set/modified"""

    def __init_subclass__(
        cls,
        *args: t.Any,
        name: t.Optional[str] = None,
        out_format: t.Optional[ClassLayout] = None,
        in_format: t.Optional[t.Sequence[ClassLayout]] = None,
        eq: t.Optional[bool] = None,
        order: t.Optional[bool] = None,
        frozen: t.Optional[bool] = None,
        init: t.Optional[bool] = None,
        kw_only: t.Optional[bool] = None,
        rename: t.Optional[RenameStyle] = None,
        in_rename: t.Optional[t.Union[RenameStyle, t.Sequence[RenameStyle]]] = None,
        out_rename: t.Optional[RenameStyle] = None,
        allow_extra: t.Optional[bool] = None,
        custom: t.Optional[IntoConverterHandlers] = None,
        **kwargs: t.Any,
    ):
        old_params = getattr(cls, '__parameters__', ())
        super().__init_subclass__(*args, **kwargs)
        setattr(cls, '__parameters__', old_params + getattr(cls, '__parameters__', ()))

        if rename is not None:
            if in_rename is not None or out_rename is not None:
                raise ValueError("'rename' cannot be specified with 'in_rename' or 'out_rename'")
            in_rename = t.cast(t.Tuple[RenameStyle, ...], (rename,))
            out_rename = rename
        elif in_rename is not None and isinstance(in_rename, str):
            in_rename = t.cast(t.Tuple[RenameStyle, ...], (in_rename,))

        # handle option inheritance
        opts: PaneOptions = getattr(cls, PANE_INFO).opts if hasattr(cls, PANE_INFO) else PaneOptions()
        opts = opts.replace(
            name=name, out_format=out_format, in_format=in_format,
            eq=eq, order=order, frozen=frozen, init=init, allow_extra=allow_extra,
            kw_only=kw_only, in_rename=in_rename, out_rename=out_rename,
            class_handlers=ConverterHandlers._process(custom),
        )

        _process(cls, opts)

    def __class_getitem__(cls, params: t.Union[type, t.Tuple[type, ...]]):
        if not isinstance(params, tuple):
            params = (params,)
        return _make_subclass(cls, params)

    def __repr__(self) -> str:
        inside = ", ".join(f"{field.name}={getattr(self, field.name)!r}" for field in self.__pane_info__.fields)
        return f"{self.__class__.__name__}({inside})"

    def __setattr__(self, name: str, value: t.Any) -> None:
        opts = self.__pane_info__.opts
        if opts.frozen:
            raise FrozenInstanceError(f"cannot assign to field {name!r}")
        super().__setattr__(name, value)
        set_fields: t.Set[str] = getattr(self, PANE_SET_FIELDS)
        set_fields.add(name)

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"cannot delete field {name!r}")

    def __copy__(self):
        return self.from_dict_unchecked(
            {field.name: getattr(self, field.name) for field in self.__pane_info__.fields}
        )

    def __deepcopy__(self, memo: t.Any):
        from copy import deepcopy
        return self.from_dict_unchecked(
            {field.name: deepcopy(getattr(self, field.name), memo) for field in self.__pane_info__.fields}
        )

    def __replace__(self, /, **changes: t.Any) -> Self:
        d = {field.name: getattr(self, field.name) for field in self.__pane_info__.fields}
        d.update(**changes)
        return self.__class__(**d)

    @classmethod
    def _converter(cls: t.Type[PaneBaseT], *args: t.Type[Convertible],
                   handlers: ConverterHandlers) -> Converter[PaneBaseT]:
        if len(args) > 0:
            cls = t.cast(t.Type[PaneBaseT], cls[tuple(args)])  # type: ignore
        return PaneConverter(cls, handlers=handlers)

    @classmethod
    def make_unchecked(cls, *args: t.Any, **kwargs: t.Any) -> Self:
        ...

    @classmethod
    def from_dict_unchecked(cls, d: t.Dict[str, t.Any]) -> Self:
        ...

    @classmethod
    def from_obj(cls, obj: Convertible, *,
                 custom: t.Optional[IntoConverterHandlers] = None) -> Self:
        """
        Convert `obj` into `cls`. Equivalent to `convert(obj, cls)`

        Parameters:
          obj: Object to convert. Must be convertible.
        """
        return convert(obj, cls, custom=custom)

    @classmethod
    def from_data(cls, data: DataType, *,
                  custom: t.Optional[IntoConverterHandlers] = None) -> Self:
        """
        Convert `data` into `cls`. Equivalent to `from_data(data, cls)`

        Parameters:
          data: Data to convert. Must be a data interchange type.
        """
        return from_data(data, cls, custom=custom)

    def into_data(self, *, custom: t.Optional[IntoConverterHandlers] = None) -> DataType:
        """Convert `self` into interchange data"""
        return into_data(self, self.__class__, custom=custom)

    def dict(self, *, set_only: bool = False, rename: t.Optional[RenameStyle] = None) -> t.Dict[str, t.Any]:
        """
        Return a dict of the fields in `self`

        Parameters:
          set_only: If `True`, return only the fields which have been set
          rename: Rename fields to match the given style
        """
        if set_only:
            return {
                rename_field(k, rename): getattr(self, k) for k in getattr(self, PANE_SET_FIELDS)
            }
        return {
            rename_field(field.name, rename): getattr(self, field.name) for field in self.__pane_info__.fields
        }

    @classmethod
    def from_json(cls, f: io.FileOrPath, *,
                  custom: t.Optional[IntoConverterHandlers] = None) -> Self:
        """
        Load `cls` from a JSON file `f`

        Parameters:
          f: File-like or path-like to load from
          custom: Custom converters to use
        """
        return io.from_json(f, cls, custom=custom)

    @classmethod
    def from_yaml(cls, f: io.FileOrPath, *,
                  custom: t.Optional[IntoConverterHandlers] = None) -> Self:
        """
        Load `cls` from a YAML file `f`

        Parameters:
          f: File-like or path-like to load from
          custom: Custom converters to use
        """
        return io.from_yaml(f, cls, custom=custom)

    @classmethod
    def from_yaml_all(cls, f: io.FileOrPath, *,
                  custom: t.Optional[IntoConverterHandlers] = None) -> t.List[Self]:
        """
        Load a list of `cls` from a YAML file `f`

        Parameters:
          f: File-like or path-like to load from
          custom: Custom converters to use
        """
        return io.from_yaml_all(f, cls, custom=custom)

    @classmethod
    def from_yamls(cls, s: str, *,
                   custom: t.Optional[IntoConverterHandlers] = None) -> Self:
        """
        Load `cls` from a YAML string `s`

        Parameters:
          s: YAML string to load from
          custom: Custom converters to use
        """
        from io import StringIO
        return io.from_yaml(StringIO(s), cls, custom=custom)

    @classmethod
    def from_jsons(cls, s: str, *,
                   custom: t.Optional[IntoConverterHandlers] = None) -> Self:
        """
        Load `cls` from a JSON string `s`

        Parameters:
          s: JSON string to load from
          custom: Custom converters to use
        """
        from io import StringIO
        return io.from_json(StringIO(s), cls, custom=custom)

    def write_json(self, f: io.FileOrPath, *,
                   indent: t.Union[str, int, None] = None,
                   sort_keys: bool = False,
                   custom: t.Optional[IntoConverterHandlers] = None):
        """
        Write data to a JSON file `f`

        Parameters:
          f: File-like or path-like to write to
          indent: Indent to format JSON with. Defaults to None (no indentation)
          sort_keys: Whether to sort keys prior to serialization.
          custom: Custom converters to use
        """
        io.write_json(
            self, f, ty=self.__class__,
            indent=indent, sort_keys=sort_keys, custom=custom
        )

    def write_yaml(self, f: io.FileOrPath, *,
                   indent: t.Optional[int] = None, width: t.Optional[int] = None,
                   allow_unicode: bool = True,
                   explicit_start: bool = True, explicit_end: bool = False,
                   default_style: t.Optional[t.Literal['"', '|', '>']] = None,
                   default_flow_style: t.Optional[bool] = None,
                   sort_keys: bool = False,
                   custom: t.Optional[IntoConverterHandlers] = None):
        """
        Write data to a YAML file `f`

        Parameters:
          f: File-like or path-like to write to
          indent: Number of spaces to indent blocks with
          width: Maximum width of file created
          allow_unicode: Whether to output unicode characters or escape them
          explicit_start: Whether to include a YAML document start "---"
          explicit_end: Whether to include a YAML document end "..."
          default_style: Default style to use for scalar nodes.
              See YAML documentation for more information.
          default_flow_style: Whether to default to flow style or block style for collections.
              See YAML documentation for more information.
          sort_keys: Whether to sort keys prior to serialization.
          custom: Custom converters to use
        """
        io.write_yaml(
            self, f, ty=self.__class__,
            indent=indent, width=width,
            allow_unicode=allow_unicode,
            explicit_start=explicit_start, explicit_end=explicit_end,
            default_style=default_style, default_flow_style=default_flow_style,
            sort_keys=sort_keys, custom=custom
        )

    def into_json(self, *,
                  indent: t.Union[str, int, None] = None,
                  sort_keys: bool = False,
                  custom: t.Optional[IntoConverterHandlers] = None) -> str:
        """
        Write data to a JSON string.

        Parameters:
          indent: Indent to format JSON with. Defaults to None (no indentation)
          sort_keys: Whether to sort keys prior to serialization.
          custom: Custom converters to use
        """
        from io import StringIO

        buf = StringIO()
        io.write_json(
            self, buf, ty=self.__class__,
            indent=indent, sort_keys=sort_keys, custom=custom
        )
        return buf.getvalue()

    def into_yaml(self, *,
                  indent: t.Optional[int] = None, width: t.Optional[int] = None,
                  allow_unicode: bool = True,
                  explicit_start: bool = True, explicit_end: bool = False,
                  default_style: t.Optional[t.Literal['"', '|', '>']] = None,
                  default_flow_style: t.Optional[bool] = None,
                  sort_keys: bool = False,
                  custom: t.Optional[IntoConverterHandlers] = None) -> str:
        """
        Write data to a YAML string.

        Parameters:
          indent: Number of spaces to indent blocks with
          width: Maximum width of file created
          allow_unicode: Whether to output unicode characters or escape them
          explicit_start: Whether to include a YAML document start "---"
          explicit_end: Whether to include a YAML document end "..."
          default_style: Default style to use for scalar nodes.
              See YAML documentation for more information.
          default_flow_style: Whether to default to flow style or block style for collections.
              See YAML documentation for more information.
          sort_keys: Whether to sort keys prior to serialization.
          custom: Custom converters to use
        """
        from io import StringIO

        buf = StringIO()
        io.write_yaml(
            self, buf, ty=self.__class__,
            indent=indent, width=width, allow_unicode=allow_unicode,
            explicit_start=explicit_start, explicit_end=explicit_end,
            default_style=default_style, default_flow_style=default_flow_style,
            sort_keys=sort_keys, custom=custom
        )
        return buf.getvalue()


@dataclasses.dataclass
class PaneInfo:
    """Structure holding internal information about a `pane` dataclass"""
    opts: PaneOptions
    """Dataclass options"""
    specs: t.Dict[str, FieldSpec]
    """
    Dict of raw field specifications

    This is used by subclasses to build [`Field`][pane.field.Field]s
    """
    fields: t.Tuple[Field, ...]
    """
    Tuple of processed [`Field`][pane.field.Field]s
    """
    pos_args: t.Tuple[int, int]
    """
    Range of allowed positional argument numbers, `[min, max]` inclusive
    """


@dataclasses.dataclass(frozen=True)
class PaneOptions:
    name: t.Optional[str] = None
    """Dataclass name"""
    _: KW_ONLY = dataclasses.field(init=False, repr=False, compare=False)
    eq: bool = True
    """Whether to generate `__eq__`/`__ne__` methods"""
    order: bool = True
    """Whether to generate `__gt__`/`__ge__`/`__lt__`/`__le__` methods"""
    frozen: bool = True
    """Whether dataclass fields are frozen"""
    init: bool = True
    """Whether to generate `__init__` method"""
    kw_only: bool = False
    """Whether all fields should be keyword-only"""
    out_format: ClassLayout = 'struct'
    """Data format to convert class into"""
    in_format: t.Sequence[ClassLayout] = ('struct',)
    """Set of data formats class is convertible from"""
    in_rename: t.Optional[t.Sequence[RenameStyle]] = None
    """Set of rename styles class is convertible from"""
    out_rename: t.Optional[RenameStyle] = None
    """Rename style to convert class into"""
    allow_extra: bool = False
    """Whether extra fields are allowed in conversion"""
    class_handlers: t.Tuple[ConverterHandler, ...] = ()
    """Custom converters to use for field datatypes"""

    def replace(self, **changes: t.Any):
        """Return `self` with the given changes applied"""
        changes['name'] = changes.get('name', None)
        return dataclasses.replace(self, **{k: v for (k, v) in changes.items() if v is not None})


@functools.lru_cache(maxsize=256)
def _make_subclass(cls: t.Any, params: t.Tuple[t.Any, ...]) -> type:
    sup: t.Any = super(PaneBase, cls)
    if not hasattr(sup, '__class_getitem__'):
        raise TypeError(f"type '{cls}' is not subscriptable")
    alias: t.Type[PaneBase] = sup.__class_getitem__(params)  # type: ignore
    typevars: t.Tuple[Parameter, ...] = getattr(cls, '__parameters__', ())

    # return subclass with bound type variables
    bound_vars = dict(zip(typevars, params))
    return type(cls.__name__, (cls,), {
        PANE_BOUNDVARS: bound_vars,
        '__origin__': cls,
        '__parameters__': getattr(alias, '__parameters__'),
    })


def _make_init(cls: t.Type[PaneBase], fields: t.Sequence[Field]):
    params: t.List[Parameter] = []
    for f in fields:
        if f.default is not _MISSING:
            default = f.default
        elif f.default_factory is not None:
            default = f.default_factory()
        else:
            default = Parameter.empty
        kind = Parameter.KEYWORD_ONLY if f.kw_only else Parameter.POSITIONAL_OR_KEYWORD
        annotation = f.type if f.type is not _MISSING else Parameter.empty
        params.append(Parameter(f.name, kind, default=default, annotation=annotation))

    sig = Signature(params, return_annotation=None)

    def __init__(self: PaneBase, *args: t.Any, **kwargs: t.Any):
        from_dict = kwargs.pop('_pane_from_dict', None)
        if from_dict is not None:
            for (k, v) in from_dict.items():
                object.__setattr__(self, k, v)
            object.__setattr__(self, PANE_SET_FIELDS, set(from_dict.keys()))
            if hasattr(self, POST_INIT):
                getattr(self, POST_INIT)()
            return

        checked = kwargs.pop('_pane_checked', True)
        try:
            bound_args = sig.bind(*args, **kwargs).arguments
        except TypeError as e:
            raise TypeError(*e.args) from None

        set_fields: t.Set[str] = set()

        for f in self.__pane_info__.fields:
            if f.name in bound_args:
                val = bound_args[f.name]
                if checked:
                    val = convert(val, f.type)
                set_fields.add(f.name)
            elif f.default is not _MISSING:
                val = f.default
            elif f.default_factory is not None:
                val = f.default_factory()
            else:
                raise RuntimeError("Mismatch between fields and signature. This shouldn't happen")
            object.__setattr__(self, f.name, val)

        object.__setattr__(self, PANE_SET_FIELDS, set_fields)

        if hasattr(self, POST_INIT):
            getattr(self, POST_INIT)()

    setattr(__init__, '__signature__', sig)
    setattr(cls, '__init__', __init__)
    setattr(cls, '__signature__', sig)

    @classmethod
    def make_unchecked(cls, *args, **kwargs):  # type: ignore
        return cls(*args, **kwargs, _pane_checked=False)  # type: ignore

    sig2 = Signature([Parameter('cls', Parameter.POSITIONAL_OR_KEYWORD), *params], return_annotation=cls)
    make_unchecked.__func__.__signature__ = sig2  # type: ignore
    setattr(cls, 'make_unchecked', make_unchecked)

    @classmethod
    def from_dict_unchecked(cls, d):  # type: ignore
        return cls(_pane_from_dict=d)  # type: ignore

    sig2 = Signature([Parameter('cls', Parameter.POSITIONAL_OR_KEYWORD), Parameter('d', Parameter.POSITIONAL_OR_KEYWORD, annotation=t.Dict[str, t.Any])], return_annotation=cls)
    from_dict_unchecked.__func__.__signature__ = sig2  # type: ignore
    setattr(cls, 'from_dict_unchecked', from_dict_unchecked)


def _make_eq(cls: t.Type[PaneBase], fields: t.Sequence[Field]):
    #eq_fields = list(filter(lambda f: f.eq, fields))
    def __eq__(self: PaneBase, other: t.Any) -> bool:
        # check if classes are the same (modulo type variables)
        if self.__class__.__dict__.get('__origin__', self.__class__) != other.__class__.__dict__.get('__origin__', other.__class__):
            return False
        return all(
            getattr(self, field.name) == getattr(other, field.name)
            for field in fields
        )

    setattr(cls, '__eq__', __eq__)


def _make_ord(cls: t.Type[PaneBase], fields: t.Sequence[Field]):
    #ord_fields = list(filter(lambda f: f.ord, fields))
    def _pane_ord(self: PaneBase, other: t.Any) -> t.Literal[-1, 0, 1]:
        if self.__class__ != other.__class__:
            return NotImplemented  # type: ignore
        for f in fields:
            if getattr(self, f.name) == getattr(other, f.name):
                continue
            return 1 if getattr(self, f.name) > getattr(other, f.name) else -1
        return 0

    def __lt__(self: PaneBase, other: t.Any) -> bool:
        return NotImplemented if (o := _pane_ord(self, other)) is NotImplemented else t.cast(int, o) < 0

    def __le__(self: PaneBase, other: t.Any) -> bool:
        return NotImplemented if (o := _pane_ord(self, other)) is NotImplemented else t.cast(int, o) <= 0

    def __gt__(self: PaneBase, other: t.Any) -> bool:
        return NotImplemented if (o := _pane_ord(self, other)) is NotImplemented else t.cast(int, o) > 0

    def __ge__(self: PaneBase, other: t.Any) -> bool:
        return NotImplemented if (o := _pane_ord(self, other)) is NotImplemented else t.cast(int, o) >= 0

    setattr(cls, '_pane_ord', _pane_ord)
    setattr(cls, '__lt__', __lt__)
    setattr(cls, '__le__', __le__)
    setattr(cls, '__gt__', __gt__)
    setattr(cls, '__ge__', __ge__)


def _process(cls: t.Type[PaneBase], opts: PaneOptions):
    fields: t.List[Field] = []

    specs: t.Dict[str, FieldSpec] = {}

    # collect FieldSpecs from base classes
    for base in reversed(cls.__mro__[1:]):
        if not hasattr(base, PANE_INFO):
            continue  # not a pane dataclass
        cls_specs = getattr(base, PANE_INFO).specs

        # apply typevar replacements
        bound_vars = t.cast(t.Mapping[t.Union[t.TypeVar, ParamSpec], type], getattr(base, PANE_BOUNDVARS, {}))
        specs.update(cls_specs)
        specs = {k: spec.replace_typevars(bound_vars) for (k, spec) in specs.items()}

    annotations = get_type_hints(cls)
    kw_only = opts.kw_only  # current kw_only state
    cls_specs: t.Dict[str, FieldSpec] = {}

    for name, ty in annotations.items():
        if ty is KW_ONLY:
            # all further params are kw_only
            kw_only = True
            continue

        if isinstance(getattr(cls, name, None), FieldSpec):
            # process existing FieldSpec
            spec: FieldSpec = getattr(cls, name)
            spec.ty = ty
        else:
            # make new spec
            spec = FieldSpec(ty=ty, default=getattr(cls, name, _MISSING))

        spec.kw_only |= kw_only
        cls_specs[name] = spec

    # apply typevar replacements
    bound_vars = getattr(cls, PANE_BOUNDVARS, {})
    specs.update(cls_specs)
    specs = {k: spec.replace_typevars(bound_vars) for (k, spec) in specs.items()}

    # bake FieldSpecs into Fields
    fields = [spec.make_field(name, opts.in_rename, opts.out_rename) for (name, spec) in specs.items()]
    # reorder kw-only fields to end
    fields = [*filter(lambda f: not f.kw_only, fields), *filter(lambda f: f.kw_only, fields)]

    # positional argument lengths
    min_len, max_len = (0, 0)
    seen_opt = False
    for f in fields:
        if f.kw_only:
            if not f.has_default() and 'tuple' in opts.in_format:
                raise TypeError(f"Field '{f.name}' is kw_only but mandatory. This is incompatible with the 'tuple' in_format.")
            continue
        max_len += 1
        if f.has_default():
            seen_opt = True
        else:
            if seen_opt:
                raise TypeError(f"Mandatory field '{f.name}' follows optional field")
            min_len = max_len  # expand min length

    cls.__pane_info__ = PaneInfo(
        opts=opts, specs=cls_specs, fields=tuple(fields), pos_args=(min_len, max_len)
    )

    for f in fields:
        name = str(f.name)
        # remove Fields from class, and set defaults
        if f.default is _MISSING:
            if isinstance(getattr(cls, name, None), Field):
                delattr(cls, name)
        else:
            setattr(cls, name, f.default)

    if opts.init:
        _make_init(cls, fields)
    if opts.eq:
        _make_eq(cls, fields)
    if opts.order:
        _make_ord(cls, fields)

    return cls


class PaneConverter(Converter[PaneBaseT]):
    """
    [`Converter`][pane.converters.Converter] for `pane` dataclasses
    """
    def __init__(self, cls: t.Type[PaneBaseT], *,
                 handlers: ConverterHandlers):
        super().__init__()

        self.cls = cls
        self.name = self.cls.__name__
        self.cls_info: PaneInfo = getattr(self.cls, PANE_INFO)
        self.opts: PaneOptions = self.cls_info.opts
        self.fields: t.Sequence[Field] = self.cls_info.fields

        # prioritize:
        # - field converter
        # - custom passed to make_converter (handlers.globals)
        # - custom passed to this class (which inherits from superclasses) (self.opts.class_handlers)
        # - custom passed to parent (by composition) class (handlers.opts.class_handlers)
        handlers = ConverterHandlers(handlers.globals, (*self.opts.class_handlers, *handlers.class_local))

        self.field_converters: t.Sequence[Converter[t.Any]] = [
            field.converter if field.converter is not None else make_converter(field.type, handlers)
            for field in self.fields
        ]
        self.field_map: t.Dict[str, int] = {}

        for (i, f) in enumerate(self.fields):
            self.field_map[f.name] = i
            for alias in f.in_names:
                self.field_map[alias] = i

    def into_data(self, val: t.Any) -> DataType:
        """Convert dataclass `val` into data interchange, using the correct 'out_format'"""
        assert isinstance(val, PaneBase)
        if self.opts.out_format == 'tuple':
            return tuple(
                conv.into_data(getattr(val, field.name))
                for (field, conv) in zip(self.fields, self.field_converters)
            )
        elif self.opts.out_format == 'struct':
            return {
                field.out_name: conv.into_data(getattr(val, field.name))
                for (field, conv) in zip(self.fields, self.field_converters)
            }
        raise ValueError(f"Unknown 'out_format' '{self.opts.out_format}'")

    def expected(self, plural: bool = False) -> str:
        """Expected value for this converter"""
        return f"{list_phrase(self.opts.in_format)} {self.name}"

    def try_convert(self, val: t.Any) -> PaneBaseT:
        """
        See [`Converter.try_convert`][pane.converters.Converter.try_convert]

        Dispatches to [`try_convert_tuple`][pane.classes.PaneConverter.try_convert_tuple]
        and [`try_convert_struct`][pane.classes.PaneConverter.try_convert_struct]
        """
        # based on type, try to delegate to try_convert_tuple or try_convert_struct
        if isinstance(val, (list, tuple, t.Sequence)):
            val = t.cast(t.Sequence[t.Any], val)
            if 'tuple' not in self.opts.in_format:
                raise ParseInterrupt()

            return self.try_convert_tuple(t.cast(t.Sequence[t.Any], val))

        elif isinstance(val, (dict, t.Mapping)):
            if 'struct' not in self.opts.in_format:
                raise ParseInterrupt()

            return self.try_convert_struct(t.cast(t.Mapping[str, t.Any], val))

        raise ParseInterrupt()

    def collect_errors(self, val: t.Any) -> t.Union[WrongTypeError, WrongLenError, ProductErrorNode, None]:
        """
        See [`Converter.collect_errors`][pane.converters.Converter.collect_errors]

        Dispatches to [`collect_errors_tuple`][pane.classes.PaneConverter.collect_errors_tuple]
        and [`collect_errors_struct`][pane.classes.PaneConverter.collect_errors_struct]
        """
        # based on type, try to delegate to collect_errors_tuple or collect_errors_struct
        if isinstance(val, (list, tuple, t.Sequence)):
            if 'tuple' not in self.opts.in_format:
                return WrongTypeError(self.expected_struct(), val)

            return self.collect_errors_tuple(t.cast(t.Sequence[t.Any], val))

        elif isinstance(val, (dict, t.Mapping)):
            val = t.cast(t.Mapping[str, t.Any], val)
            if 'struct' not in self.opts.in_format:
                return WrongTypeError(f'tuple {self.name}', val)
            
            return self.collect_errors_struct(t.cast(t.Mapping[str, t.Any], val))

        return WrongTypeError(self.name, val)

    def expected_struct(self, plural: bool = False) -> str:
        """Expected value for the 'struct' data format"""
        return f"struct {self.name}"

    def try_convert_struct(self, val: t.Mapping[str, t.Any]) -> PaneBaseT:
        """[`Converter.try_convert`][pane.converters.Converter.try_convert] for the 'struct' data format"""
        # loop through values, and handle accordingly
        values: t.Dict[str, t.Any] = {}
        for (k, v) in t.cast(t.Dict[str, t.Any], val).items():
            if k not in self.field_map:
                if not self.opts.allow_extra:
                    raise ParseInterrupt()  # extra key
                continue
            field = self.fields[self.field_map[k]]
            conv = self.field_converters[self.field_map[k]]

            if field.name in values:
                raise ParseInterrupt()  # multiple values for key
            values[field.name] = conv.try_convert(v)

        for field in filter(lambda field: field.name not in values, self.fields):
            if field.default is not _MISSING:
                values[field.name] = field.default
            elif field.default_factory is not None:
                values[field.name] = field.default_factory
            else:
                raise ParseInterrupt()  # missing field

        try:
            return self.cls.from_dict_unchecked(values)
        except Exception:  # error in __post_init__
            raise ParseInterrupt()

    def collect_errors_struct(self, val: t.Mapping[str, t.Any]) -> t.Union[WrongTypeError, ProductErrorNode, None]:
        """[`Converter.collect_errors`][pane.converters.Converter.collect_errors] for the 'struct' data format"""
        values: t.Dict[str, t.Any] = {}  # converted field values. Required to check for __post_init__ errors
        children: t.Dict[t.Union[str, int], ErrorNode] = {}  # errors in converting fields
        extra: t.Set[str] = set()  # extra fields found
        seen: t.Set[str] = set()   # fields seen already (used to find dupes)
        for (k, v) in val.items():
            if k not in self.field_map:
                if not self.opts.allow_extra:
                    extra.add(k)  # unknown key
                continue

            field = self.fields[self.field_map[k]]
            conv = self.field_converters[self.field_map[k]]
            if field.name in seen:
                children[k] = DuplicateKeyError(k, field.in_names)
                continue
            seen.add(field.name)

            # this is a little tricky. we need to call convert() rather
            # than collect_errors to grab a successful value
            try:
                values[field.name] = conv.convert(v)
            except ConvertError as e:
                # then we can collect errors if that fails.
                children[k] = e.tree

        missing: t.Set[str] = set()
        for field in self.fields:
            if field.name not in seen and not field.has_default():
                missing.add(field.name)

        if len(missing) or len(children) or len(extra):
            # return field errors
            return ProductErrorNode(self.expected_struct(), children, val, missing, extra)
        try:
            self.cls.make_unchecked(**values)
            return None
        except Exception as e:  # error in __post_init__
            tb = e.__traceback__.tb_next  # type: ignore
            tb = traceback.TracebackException(type(e), e, tb)
            return WrongTypeError(f'struct {self.name}', val, tb)

    def expected_tuple(self, plural: bool = False) -> str:
        """Expected value for the 'tuple' data format"""
        return f"tuple {self.name}"

    def try_convert_tuple(self, val: t.Sequence[t.Any]) -> PaneBaseT:
        """[`Converter.try_convert`][pane.converters.Converter.try_convert] for the 'tuple' data format"""
        (min_len, max_len) = self.cls_info.pos_args
        if min_len < len(val) > max_len:
            raise ParseInterrupt()

        vals: t.List[t.Any] = []
        for (conv, v) in zip(self.field_converters, val):
            vals.append(conv.try_convert(v))

        try:
            return self.cls.make_unchecked(*vals)
        except Exception:  # error in __post_init__
            raise ParseInterrupt()

    def collect_errors_tuple(self, val: t.Sequence[t.Any]) -> t.Union[WrongTypeError, ProductErrorNode, WrongLenError, None]:
        """[`Converter.collect_errors`][pane.converters.Converter.collect_errors] for the 'tuple' data format"""
        (min_len, max_len) = self.cls_info.pos_args
        if min_len < len(val) > max_len:
            return WrongLenError(f'tuple {self.name}', (min_len, max_len), val, len(val))

        vals: t.List[t.Any] = []
        children: t.Dict[t.Union[str, int], ErrorNode] = {}
        for (i, (conv, v)) in enumerate(zip(self.field_converters, val)):
            # this is a little tricky. we need to call convert() rather
            # than collect_errors to grab a successful value
            try:
                vals.append(conv.convert(v))
            except ConvertError as e:
                # then we can collect errors if that fails.
                children[i] = e.tree

        if len(children):
            # return field errors
            return ProductErrorNode(self.expected_tuple(), children, val)

        try:
            self.cls.make_unchecked(*vals)
            return None
        except Exception as e:  # error in __post_init__
            tb = e.__traceback__.tb_next  # type: ignore
            tb = traceback.TracebackException(type(e), e, tb)
            return WrongTypeError(f'tuple {self.name}', val, tb)


ClassLayout: TypeAlias = t.Literal['tuple', 'struct']
"""Set of known class layouts for 'in_formats' and 'out_format'."""
PANE_INFO = '__pane_info__'  # class information
"""Name of dunder attribute holding [`PaneInfo`][pane.classes.PaneInfo]"""
PANE_SET_FIELDS = '__pane_set__'  # fields currently set
"""Name of dunder attribute holding a set of fields which have been set/modified"""
PANE_BOUNDVARS = '__pane_boundvars__'  # bound variables, for subtypes of generics
"""Name of dunder attribute holding a dictionary of bound type variables (for generic subclasses only)."""
POST_INIT = '__post_init__'  # post-init function
"""Name of post-init method"""


__all__ = [
    'PaneBase', 'PaneOptions', 'field', 'Field', 'KW_ONLY',
]
