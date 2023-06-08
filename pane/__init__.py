
import sys
from dataclasses import dataclass
import typing as t

from .convert import DataType, IntoData, FromData, from_data, into_data, convert
from .field import Field


ClassLayout = t.Literal['tuple', 'struct']
T = t.TypeVar('T')


@dataclass(init=False)
class PaneOptions:
    name: t.Optional[str]
    ser_name: t.Optional[str]
    de_name: t.Optional[str]
    ser_format: ClassLayout
    de_format: t.Optional[t.Sequence[ClassLayout]]
    closed: bool

    def __init__(self, name: t.Optional[str] = None, *,
                 ser_name: t.Optional[str] = None,
                 de_name: t.Optional[str] = None,
                 ser_format: ClassLayout = 'struct',
                 de_format: t.Optional[t.Sequence[ClassLayout]] = None,
                 closed: bool = False):
        if name is not None:
            if ser_name is not None or de_name is not None:
                raise ValueError("`name` overrides `ser_name` and `de_name`.")
            self.ser_name = name
            self.de_name = name
        else:
            self.name = name

        self.ser_format = ser_format
        self.de_format = de_format

        self.closed = closed


def pane(cls=None, /,
         name: t.Optional[str] = None, *,
         ser_name: t.Optional[str] = None,
         de_name: t.Optional[str] = None,
         closed: bool = False):

    opts = PaneOptions(name, ser_name=ser_name, de_name=de_name, closed=closed)

    def wrap(cls):
        return _process(cls, opts)

    if cls is None:
        return wrap

    return wrap(cls)


def _process(cls, opts: PaneOptions):
    fields = {}

    globals = sys.modules[cls.__module__].__dict__ if cls.__module__ in sys.modules else {}

    # todo work with mro/subclassing
    for base in cls.__mro__[-1:0:-1]:
        base_fields = getattr(base, "__ser_fields__", None)
        if base_fields is not None:
            for f in base_fields.values():
                fields[f.name] = f

    annotations = cls.__dict__.get('__annotations__', {})

    cls_fields = [Field.with_name(name, ty)
                  for name, ty in annotations.items()]


__ALL__ = [DataType, IntoData, FromData, from_data, into_data, convert, Field, pane]
