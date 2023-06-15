from __future__ import annotations

import sys
from dataclasses import dataclass, KW_ONLY
from inspect import Signature, Parameter
from io import StringIO
import typing as t

from .convert import DataType, IntoData, FromData, from_data, into_data, convert
from .convert import Converter, make_converter, ConvertError
from .convert import ParseInterrupt, WrongTypeError, ProductErrorNode, DuplicateKeyError
from .field import Field, FieldSpec, field, RenameStyle, _MISSING
from .util import FileOrPath, open_file


ClassLayout = t.Literal['tuple', 'struct']
T = t.TypeVar('T')


PANE_FIELDS = '__pane_fields__'
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
    ser_format: ClassLayout = 'struct'
    de_format: t.Optional[t.Sequence[ClassLayout]] = None
    rename: t.Optional[RenameStyle] = None


@t.dataclass_transform(
    eq_default=True,
    order_default=True,
    frozen_default=True,
    kw_only_default=False,
    field_specifiers=(FieldSpec, field),
)
class PaneBase:
    __pane_opts__: PaneOptions
    __pane_fields__: t.Sequence[Field]

    def __init_subclass__(
        cls,
        *,
        name: t.Optional[str] = None,
        ser_format: ClassLayout = 'struct',
        de_format: t.Optional[t.Sequence[ClassLayout]] = None,
        eq: bool = True,
        order: bool = True,
        frozen: bool = False,
        init: bool = True,
        kw_only: bool = False,
        rename: t.Optional[RenameStyle] = None
    ):
        opts = PaneOptions(
            name=name, eq=eq, order=order, frozen=frozen,
            init=init, kw_only=kw_only, rename=rename,
            ser_format=ser_format, de_format=de_format,
        )
        setattr(cls, PANE_OPTS, opts)

        _process(cls, opts)

    def __repr__(self) -> str:
        inside = ", ".join(f"{field.py_name}={getattr(self, field.py_name)!r}" for field in self.__pane_fields__)
        return f"{self.__class__.__name__}({inside})"

    @classmethod
    def make(cls, obj) -> t.Self:
        conv: Converter = getattr(cls, '_converter')()
        return conv.convert(obj)

    @classmethod
    def from_json(cls, f: FileOrPath) -> t.Self:
        import json
        with open_file(f) as f:
            obj = json.load(f)
        return cls.make(obj)

    @classmethod
    def from_yaml(cls, f: FileOrPath) -> t.Self:
        import yaml
        try:
            from yaml import CLoader as Loader
        except ImportError:
            from yaml import Loader

        with open_file(f) as f:
            obj = yaml.load(f, Loader)
        return cls.make(obj)

    # TODO can we give this proper types?
    @classmethod
    def make_unchecked(cls, *args, **kwargs) -> t.Self:
        ...


def _make_init(cls, opts: PaneOptions, fields: t.Sequence[Field]):
    params = []
    for field in fields:
        kind = Parameter.KEYWORD_ONLY if field.kw_only else Parameter.POSITIONAL_OR_KEYWORD
        default = field.default if field.default is not _MISSING else Parameter.empty
        annotation = field.type if field.type is not _MISSING else Parameter.empty
        params.append(Parameter(field.py_name, kind, default=default, annotation=annotation))

    sig = Signature(params, return_annotation=None)

    def __init__(self, *args, **kwargs):
        checked = kwargs.pop('_pane_checked', True)
        try:
            args = sig.bind(*args, **kwargs).arguments
        except TypeError as e:
            raise TypeError(*e.args) from None

        for field in t.cast(t.Sequence[Field], getattr(self, PANE_FIELDS)):
            if field.py_name in args:
                val = args[field.py_name]
                if checked:
                    val = convert(val, field.type)  # type: ignore
            elif field.default is not _MISSING:
                val = field.default
            elif field.default_factory is not None:
                val = field.default_factory()
            else:
                raise RuntimeError()
            object.__setattr__(self, field.py_name, val)

        if hasattr(self, POST_INIT):
            getattr(self, POST_INIT)()

    __init__.__signature__ = sig
    setattr(cls, '__init__', __init__)
    setattr(cls, '__signature__', sig)

    @classmethod
    def make_unchecked(cls, *args, **kwargs):
        return cls(*args, **kwargs, _pane_checked=False)

    sig2 = Signature([Parameter('cls', Parameter.POSITIONAL_OR_KEYWORD), *params], return_annotation=cls)
    make_unchecked.__func__.__signature__ = sig2
    setattr(cls, 'make_unchecked', make_unchecked)


def _make_eq(cls, fields: t.Sequence[Field]):
    def __eq__(self, other: t.Any) -> bool:
        if self.__class__ != other.__class__:
            return False
        return all(
            getattr(self, field.py_name) == getattr(other, field.py_name)
            for field in fields
        )

    setattr(cls, '__eq__', __eq__)


def _process(cls, opts: PaneOptions):
    fields = {}

    globals = sys.modules[cls.__module__].__dict__ if cls.__module__ in sys.modules else {}

    # todo work with mro/subclassing
    for base in cls.__mro__[-1:0:-1]:
        base_fields = getattr(base, "__ser_fields__", None)
        if base_fields is not None:
            for f in base_fields.values():
                fields[f.name] = f

    annotations = t.get_type_hints(cls, localns={cls.__name__: cls}, include_extras=True)

    cls_fields = []
    kw_only_fields = []
    kw_only = opts.kw_only

    for name, ty in annotations.items():
        if name in (PANE_FIELDS, PANE_OPTS):
            continue
        if ty == KW_ONLY:
            kw_only = True
            cls_fields.extend(kw_only_fields)
            continue

        if isinstance(getattr(cls, name, None), FieldSpec):
            # process existing Field
            spec: FieldSpec = getattr(cls, name)
            field = spec.make_field(name, ty)
            # delete Field and set default value
            if field.default is _MISSING:
                delattr(cls, name)
            else:
                setattr(cls, name, field.default)
        else:
            # otherwise make new Field
            field = Field(name=name, py_name=name, type=ty)
            if hasattr(cls, name):
                field.default = getattr(cls, name)

        if field.kw_only and not kw_only:
            # delay keyword-only fields
            kw_only_fields.append(field)
            continue

        field.kw_only |= kw_only
        cls_fields.append(field)

    # put keyword-only fields at end
    if not kw_only:
        cls_fields.extend(kw_only_fields)

    setattr(cls, PANE_FIELDS, cls_fields)
    setattr(cls, '_converter', classmethod(PaneConverter))

    for field in cls_fields:
        name = str(field.name)
        field.default = getattr(cls, name, _MISSING)
        # remove Fields from class
        if field.default is _MISSING:
            if isinstance(getattr(cls, name, None), Field):
                delattr(cls, name)
        else:
            setattr(cls, name, field.default)

    if opts.init:
        _make_init(cls, opts, cls_fields)

    if opts.eq:
        _make_eq(cls, cls_fields)

    return cls


class PaneConverter(Converter[PaneBase]):
    def __init__(self, cls: t.Type[PaneBase], *args, annotations: t.Optional[t.Tuple[t.Any, ...]] = None):
        if len(args):
            cls = cls[*args]  # type: ignore
        self.cls: t.Type[PaneBase] = cls
        self.name = self.cls.__name__
        self.opts: PaneOptions = getattr(self.cls, PANE_OPTS)
        self.fields: t.Sequence[Field] = getattr(self.cls, PANE_FIELDS)
        self.field_converters: t.Sequence[Converter] = [make_converter(field.type) for field in self.fields]
        self.field_map: t.Dict[str, int] = {}

        for (i, field) in enumerate(self.fields):
            self.field_map[field.name] = i
            for alias in field.aliases or ():
                self.field_map[alias] = i

    @property
    def de_format(self) -> t.Sequence[ClassLayout]:
        return self.opts.de_format or ('struct', 'tuple')

    def try_convert(self, val: t.Any) -> PaneBase:
        if isinstance(val, (list, tuple, t.Sequence)):
            if 'tuple' not in self.de_format:
                raise ParseInterrupt()
            raise NotImplementedError()

        elif isinstance(val, (dict, t.Mapping)):
            if 'struct' not in self.de_format:
                raise ParseInterrupt()

            values: t.Dict[str, t.Any] = {}
            for (k, v) in val.items():
                if k not in self.field_map:
                    raise ParseInterrupt()  # unknown key
                field = self.fields[self.field_map[k]]
                conv = self.field_converters[self.field_map[k]]

                if field.py_name in values:
                    raise ParseInterrupt()  # multiple values for key
                values[field.py_name] = conv.try_convert(v)

            for field in self.fields:
                if field.py_name not in values and not field.is_optional:
                    raise ParseInterrupt()

            return self.cls.make_unchecked(**values)

        raise ParseInterrupt()

    def collect_errors(self, val: t.Any) -> t.Union[WrongTypeError, ProductErrorNode, None]:
        if isinstance(val, (list, tuple, t.Sequence)):
            if 'tuple' not in self.de_format:
                return WrongTypeError(f'struct {self.name}', val)
            return None
        elif isinstance(val, (dict, t.Mapping)):
            if 'struct' not in self.de_format:
                return WrongTypeError(f'tuple {self.name}', val)

            children = {}
            extra = set()
            seen = set()
            for (k, v) in val.items():
                if k not in self.field_map:
                    extra.add(k)  # unknown key
                    continue

                field = self.fields[self.field_map[k]]
                conv = self.field_converters[self.field_map[k]]
                if field.py_name in seen:
                    children[k] = DuplicateKeyError(k, field.aliases or field.name)
                    continue
                seen.add(field.py_name)

                if (node := conv.collect_errors(v)) is not None:
                    children[k] = node

            missing = set()
            for field in self.fields:
                if field.py_name not in seen and not field.is_optional:
                    missing.add(field.name)

            if len(missing) or len(children) or len(extra):
                return ProductErrorNode(self.name, children, val, missing, extra)
            return None
        return WrongTypeError(self.name, val)


__ALL__ = [
    DataType, IntoData, FromData, from_data, into_data, convert,
    Field, PaneOptions, PaneBase, ConvertError
]
