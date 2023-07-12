"""
Pane dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, KW_ONLY, replace, FrozenInstanceError
from inspect import Signature, Parameter
from types import NotImplementedType
import typing as t
from typing_extensions import dataclass_transform, Self

from .convert import DataType, FromData, from_data, into_data, convert
from .converters import Converter, make_converter
from .errors import ParseInterrupt, ErrorNode, WrongTypeError, ProductErrorNode, DuplicateKeyError
from .field import Field, FieldSpec, field, RenameStyle, _MISSING
from .util import FileOrPath, open_file, get_type_hints, list_phrase


ClassLayout = t.Literal['tuple', 'struct']
T = t.TypeVar('T')


PANE_FIELDS = '__pane_fields__'
PANE_SET_FIELDS = '__pane_set__'
PANE_SPECS = '__pane_specs__'
PANE_BOUNDVARS = '__pane_boundvars__'
PANE_OPTS = '__pane_opts__'
POST_INIT = '__post_init__'


@dataclass
class PaneOptions:
    name: t.Optional[str] = None
    _: KW_ONLY
    eq: bool = True
    order: bool = True
    frozen: bool = True
    init: bool = True
    kw_only: bool = False
    out_format: ClassLayout = 'struct'
    in_format: t.Sequence[ClassLayout] = ('struct',)
    in_rename: t.Optional[t.Sequence[RenameStyle]] = None
    out_rename: t.Optional[RenameStyle] = None
    allow_extra: bool = False

    def replace(self, **changes: t.Any):
        changes['name'] = changes.get('name', None)
        return replace(self, **{k: v for (k, v) in changes.items() if v is not None})


@dataclass_transform(
    eq_default=True,
    order_default=True,
    frozen_default=True,
    kw_only_default=False,
    field_specifiers=(FieldSpec, field),
)
class PaneBase:
    __pane_opts__: PaneOptions
    __pane_fields__: t.Sequence[Field]
    __pane_specs__: t.Dict[str, FieldSpec]
    __pane_set__: t.Set[str]

    def __init_subclass__(
        cls,
        *args: t.Any,
        name: t.Optional[str] = None,
        ser_format: t.Optional[ClassLayout] = None,
        de_format: t.Optional[t.Sequence[ClassLayout]] = None,
        eq: t.Optional[bool] = None,
        order: t.Optional[bool] = None,
        frozen: t.Optional[bool] = None,
        init: t.Optional[bool] = None,
        kw_only: t.Optional[bool] = None,
        rename: t.Optional[RenameStyle] = None,
        in_rename: t.Optional[t.Union[RenameStyle, t.Sequence[RenameStyle]]] = None,
        out_rename: t.Optional[RenameStyle] = None,
        allow_extra: t.Optional[bool] = None,
        **kwargs: t.Any,
    ):
        old_params = getattr(cls, '__parameters__', ())
        super().__init_subclass__(*args, **kwargs)
        setattr(cls, '__parameters__', old_params + getattr(cls, '__parameters__', ()))

        if rename is not None:
            if in_rename is not None or out_rename is not None:
                print("'rename' cannot be specified with 'in_rename' or 'out_rename'")
            in_rename = t.cast(t.Tuple[RenameStyle, ...], (rename,))
            out_rename = rename
        elif in_rename is not None and isinstance(in_rename, str):
            in_rename = t.cast(t.Tuple[RenameStyle, ...], (in_rename,))

        # handle option inheritance
        opts = getattr(cls, '__pane_opts__', PaneOptions())
        opts = opts.replace(
            name=name, ser_format=ser_format, de_format=de_format,
            eq=eq, order=order, frozen=frozen, init=init, allow_extra=allow_extra,
            kw_only=kw_only, in_rename=in_rename, out_rename=out_rename,
        )
        setattr(cls, PANE_OPTS, opts)

        _process(cls, opts)

    def __class_getitem__(cls, params: t.Union[type, t.Tuple[type, ...]]):
        typevars = getattr(cls, '__parameters__', ())
        if not isinstance(params, tuple):
            params = (params,)

        if not hasattr(super(), '__class_getitem__'):
            raise TypeError(f"type '{cls}' is not subscriptable")

        alias: t.Type[PaneBase] = super().__class_getitem__(params)  # type: ignore

        # return subclass with bound type variables
        bound_vars = dict(zip(typevars, params))
        bound = type(cls.__name__, (cls,), {
            PANE_BOUNDVARS: bound_vars,
            '__parameters__': getattr(alias, '__parameters__'),  # type: ignore
        })
        return bound

    def __repr__(self) -> str:
        inside = ", ".join(f"{field.name}={getattr(self, field.name)!r}" for field in self.__pane_fields__)
        return f"{self.__class__.__name__}({inside})"

    def __setattr__(self, name: str, value: t.Any) -> None:
        opts: PaneOptions = getattr(self, PANE_OPTS)
        if opts.frozen:
            raise FrozenInstanceError(f"cannot assign to field {name!r}")
        super().__setattr__(name, value)
        set_fields: t.Set[str] = getattr(self, PANE_SET_FIELDS)
        set_fields.add(name)

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"cannot delete field {name!r}")

    @classmethod
    def _converter(cls: t.Type[T], *args: t.Type[FromData],
                   annotations: t.Optional[t.Tuple[t.Any, ...]] = None) -> Converter[T]:
        return t.cast(Converter[T], PaneConverter(cls, annotations))

    @classmethod
    def from_data(cls, data: t.Any) -> Self:
        return from_data(data, cls)

    def into_data(self) -> DataType:
        opts: PaneOptions = getattr(self, PANE_OPTS)
        if opts.out_format == 'tuple':
            return tuple(into_data(getattr(self, field.name)) for field in getattr(self, PANE_FIELDS))
        elif opts.out_format == 'struct':
            return { field.out_name: into_data(getattr(self, field.name)) for field in getattr(self, PANE_FIELDS) }
        raise ValueError(f"Unknown 'out_format' '{opts.out_format}'")

    def dict(self, set_only: bool = False) -> t.Dict[str, t.Any]:
        """Return a dict of the fields in `self`."""
        if set_only:
            return {
                k : getattr(self, k) for k in getattr(self, PANE_SET_FIELDS)
            }
        return {
            field.name: getattr(self, field.name) for field in getattr(self, PANE_FIELDS)
        }

    @classmethod
    def from_json(cls, f: FileOrPath) -> Self:
        import json
        with open_file(f) as f:
            obj = json.load(f)
        return cls.from_data(obj)

    @classmethod
    def from_yaml(cls, f: FileOrPath) -> Self:
        import yaml
        try:
            from yaml import CLoader as Loader
        except ImportError:
            from yaml import Loader

        with open_file(f) as f:
            obj = yaml.load(f, Loader)
        return cls.from_data(obj)

    @classmethod
    def make_unchecked(cls, *args: t.Any, **kwargs: t.Any) -> Self:
        ...


def _make_init(cls: t.Type[PaneBase], fields: t.Sequence[Field]):
    params: t.List[Parameter] = []
    for field in fields:
        if field.default is not _MISSING:
            default = field.default
        elif field.default_factory is not None:
            default = field.default_factory()
        else:
            default = Parameter.empty
        kind = Parameter.KEYWORD_ONLY if field.kw_only else Parameter.POSITIONAL_OR_KEYWORD
        annotation = field.type if field.type is not _MISSING else Parameter.empty
        params.append(Parameter(field.name, kind, default=default, annotation=annotation))

    sig = Signature(params, return_annotation=None)

    def __init__(self: PaneBase, *args: t.Any, **kwargs: t.Any):
        checked = kwargs.pop('_pane_checked', True)
        try:
            bound_args = sig.bind(*args, **kwargs).arguments
        except TypeError as e:
            raise TypeError(*e.args) from None

        set_fields: t.Set[str] = set()

        for field in t.cast(t.Sequence[Field], getattr(self, PANE_FIELDS)):
            if field.name in bound_args:
                val = bound_args[field.name]
                if checked:
                    val = convert(val, field.type)  # type: ignore
                set_fields.add(field.name)
            elif field.default is not _MISSING:
                val = field.default
            elif field.default_factory is not None:
                val = field.default_factory()
            else:
                raise RuntimeError("Mismatch between fields and signature. This shouldn't happen")
            object.__setattr__(self, field.name, val)

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


def _make_eq(cls: t.Type[PaneBase], fields: t.Sequence[Field]):
    #eq_fields = list(filter(lambda f: f.eq, fields))
    def __eq__(self: PaneBase, other: t.Any) -> bool:
        if self.__class__ != other.__class__:
            return False
        return all(
            getattr(self, field.name) == getattr(other, field.name)
            for field in fields
        )

    setattr(cls, '__eq__', __eq__)


def _make_ord(cls: t.Type[PaneBase], fields: t.Sequence[Field]):
    #ord_fields = list(filter(lambda f: f.ord, fields))
    def _pane_ord(self: PaneBase, other: t.Any) -> t.Union[NotImplementedType, t.Literal[-1, 0, 1]]:
        if self.__class__ != other.__class__:
            return NotImplemented
        for field in fields:
            if getattr(self, field.name) == getattr(other, field.name):
                continue
            return 1 if getattr(self, field.name) > getattr(other, field.name) else -1
        return 0

    def __lt__(self: PaneBase, other: t.Any) -> t.Union[bool, NotImplementedType]:
        return NotImplemented if (o := _pane_ord(self, other)) is NotImplemented else t.cast(int, o) < 0

    def __le__(self: PaneBase, other: t.Any) -> t.Union[bool, NotImplementedType]:
        return NotImplemented if (o := _pane_ord(self, other)) is NotImplemented else t.cast(int, o) <= 0

    def __gt__(self: PaneBase, other: t.Any) -> t.Union[bool, NotImplementedType]:
        return NotImplemented if (o := _pane_ord(self, other)) is NotImplemented else t.cast(int, o) > 0

    def __ge__(self: PaneBase, other: t.Any) -> t.Union[bool, NotImplementedType]:
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
        if not hasattr(base, PANE_SPECS):
            continue  # not a pane dataclass
        cls_specs = getattr(base, PANE_SPECS)

        # apply typevar replacements
        bound_vars = t.cast(t.Mapping[t.Union[t.TypeVar, t.ParamSpec], type], getattr(base, PANE_BOUNDVARS, {}))
        specs.update(cls_specs)
        specs = {k: spec.replace_typevars(bound_vars) for (k, spec) in specs.items()}

    annotations = get_type_hints(cls)
    kw_only = opts.kw_only  # current kw_only state
    cls_specs = {}

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

    # save specs for subclasses
    setattr(cls, PANE_SPECS, cls_specs)
    specs.update(cls_specs)

    # apply typevar replacements
    bound_vars = getattr(cls, PANE_BOUNDVARS, {})
    specs = {k: spec.replace_typevars(bound_vars) for (k, spec) in specs.items()}

    # bake FieldSpecs into Fields
    fields = [spec.make_field(name, opts.in_rename, opts.out_rename) for (name, spec) in specs.items()]
    # reorder kw-only fields to end
    fields = [*filter(lambda f: not f.kw_only, fields), *filter(lambda f: f.kw_only, fields)]
    # TODO error on default positional arg before non-default arg

    setattr(cls, PANE_FIELDS, fields)
    setattr(cls, '_converter', classmethod(PaneConverter))

    for field in fields:
        name = str(field.name)
        # remove Fields from class, and set defaults
        if field.default is _MISSING:
            if isinstance(getattr(cls, name, None), Field):
                delattr(cls, name)
        else:
            setattr(cls, name, field.default)

    if opts.init:
        _make_init(cls, fields)
    if opts.eq:
        _make_eq(cls, fields)
    if opts.order:
        _make_ord(cls, fields)

    return cls


class PaneConverter(Converter[PaneBase]):
    def __init__(self, cls: t.Type[PaneBase],
                 annotations: t.Optional[t.Tuple[t.Any, ...]] = None):
        self.cls = cls
        self.name = self.cls.__name__
        self.opts: PaneOptions = getattr(self.cls, PANE_OPTS)
        self.fields: t.Sequence[Field] = getattr(self.cls, PANE_FIELDS)
        self.field_converters: t.Sequence[Converter[t.Any]] = [make_converter(field.type) for field in self.fields]
        self.field_map: t.Dict[str, int] = {}

        for (i, field) in enumerate(self.fields):
            self.field_map[field.name] = i
            for alias in field.in_names:
                self.field_map[alias] = i

    def expected(self, plural: bool = False) -> str:
        return f"{list_phrase(self.opts.in_format)} {self.name}"

    def try_convert(self, val: t.Any) -> PaneBase:
        if isinstance(val, (list, tuple, t.Sequence)):
            if 'tuple' not in self.opts.in_format:
                raise ParseInterrupt()
            raise NotImplementedError()

        elif isinstance(val, (dict, t.Mapping)):
            if 'struct' not in self.opts.in_format:
                raise ParseInterrupt()

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

            for field in self.fields:
                if field.name not in values and not field.is_optional():
                    raise ParseInterrupt()  # missing field

            return self.cls.make_unchecked(**values)

        raise ParseInterrupt()

    def collect_errors(self, val: t.Any) -> t.Union[WrongTypeError, ProductErrorNode, None]:
        if isinstance(val, (list, tuple, t.Sequence)):
            val = t.cast(t.Sequence[t.Any], val)
            if 'tuple' not in self.opts.in_format:
                return WrongTypeError(f'struct {self.name}', val)
            return WrongTypeError(f'tuple {self.name}', "Not implemented")
        elif isinstance(val, (dict, t.Mapping)):
            val = t.cast(t.Mapping[str, t.Any], val)
            if 'struct' not in self.opts.out_format:
                return WrongTypeError(f'tuple {self.name}', val)

            children: t.Dict[t.Union[str, int], ErrorNode] = {}
            extra: t.Set[str] = set()
            seen: t.Set[str] = set()
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

                if (node := conv.collect_errors(v)) is not None:
                    children[k] = node

            missing: t.Set[str] = set()
            for field in self.fields:
                if field.name not in seen and not field.is_optional():
                    missing.add(field.name)

            if len(missing) or len(children) or len(extra):
                return ProductErrorNode(self.name, children, val, missing, extra)
            return None
        return WrongTypeError(self.name, val)


__all__ = [
    'PaneBase', 'PaneOptions', 'field', 'Field', 'KW_ONLY',
]
