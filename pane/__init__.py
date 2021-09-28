
import sys
from dataclasses import dataclass
from typing import Optional, Callable, Union, Sequence, Literal, Any
from typing import Dict, Tuple, List

from .field import Field


Layout = Union[Literal['int'], Literal['float'], Literal['complex'], Literal['bool'], Literal['none'],
               Literal['str'], Literal['tuple'], Literal['struct'], Literal['map'], Literal['seq']]

ClassLayout = Union[Literal['tuple'], Literal['struct']]


@dataclass(init=False)
class PaneOptions:
	name: Optional[str]
	ser_name: Optional[str]
	de_name: Optional[str]
	ser_format: ClassLayout
	de_format: Optional[Sequence[ClassLayout]]
	closed: bool

	def __init__(self, name: Optional[str] = None, *,
	             ser_name: Optional[str] = None,
	             de_name: Optional[str] = None,
	             ser_format: ClassLayout = 'struct',
	             de_format: Optional[Sequence[ClassLayout]] = None,
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
         name: str = None, *,
         ser_name: str = None,
         de_name: str = None,
         closed: bool = False):

	opts = PaneOptions(name, ser_name, de_name, closed)

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


__ALL__ = [Field, pane]
