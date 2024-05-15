# Dataclasses

`pane` dataclasses work similarly to many other libraries.
They aim to have a superset of the features of the standard library dataclasses.
A dataclass can be made by subclassing `pane.PaneBase`:

```python
import pane
from typing import t

class MyDataclass(pane.PaneBase):
    x: int = 5  # basic field w/ default
    y: t.Optional[float] = None
    # remaining fields should be keyword only
    _: pane.KW_ONLY  
    # advanced field specification
    z: t.List[float] = pane.field(aliases=('w',), default_factory=list)
    # apply custom logic to a field
    w: str = pane.field(converter=CustomConverter())
```

## Constructors

`pane` automatically makes constructors for you:

```python
>>> import inspect; inspect.signature(MyDataclass.__init__)
<Signature (x: int = 5, y: Optional[float] = None, *, z: list[float] = []) -> None>
```

`MyDataclass.__init__` performs conversion on arguments:

```python
>>> MyDataclass()
MyDataclass(x=5, y=None, z=[])

>>> MyDataclass(z=[5.0, 10.0, 's'])
Traceback (most recent call last):
...
pane.errors.ConvertError: Expected sequence of floats
While parsing field '2':
  Expected a float, instead got `s` of type `str`
```

To bypass conversion, use `Mydataclass.make_unchecked` instead:

```python
>>> MyDataclass.make_unchecked(z=[5.0, 10.0, 's'])
MyDataclass(x=5, y=None, z=[5.0, 10.0, 's'])
```

Like regular dataclasses, `__post_init__` is called on construction, allowing extra validation logic and initializing `init=False` fields.

`MyDataclass.from_obj`, `MyDataclass.from_data`, `MyDataclass.from_json`, and `MyDataclass.from_yaml` perform conversion from an object:

```python
# note the use of an alias for 'z'
>>> MyDataclass.from_data({'x': 10, 'y': 10., 'w': [10.]})
MyDataclass(x=10, y=10.0, z=[10.0])

>>> from io import StringIO
>>> MyDataclass.from_json(StringIO('{"x": 10, "y": 10.0, "w": [10.0]}'))
MyDataclass(x=10, y=10.0, z=[10.0])
```

(`from_json` and `from_yaml` take a file-like object or filename. `from_jsons` and `from_yamls` take a string).

## Member methods

`MyDataclass.into_data` performs the reverse operation, converting a dataclass to a data interchange type (usually a dict):

```python
>>> MyDataclass(x=10, y=10.0, z=[10.0]).into_data()
{'x': 10, 'y': 10.0, 'z': [10.0]}
```

`MyDataclass.dict` returns a dictionary containing the fields, using their Python names. `set_only` may be specified
to return only explicitly set fields, and `rename` may be specified to return using differently-styled names:

## Index of methods

| Method name | Method type | Description                     |
| :---------- | :---------: | :------------------------------ |
| `__init__` | classmethod | Instantiate a dataclass from data |
| [`make_unchecked`][pane.classes.PaneBase.make_unchecked] | classmethod | Instantiate a dataclass from data |
| [`from_obj`][pane.classes.PaneBase.from_obj] | classmethod | Instantiate a dataclass from a convertible object |
| [`from_data`][pane.classes.PaneBase.from_data] | classmethod | Instantiate a dataclass from interchange data |
| [`from_json`][pane.classes.PaneBase.from_json] | classmethod | Instantiate a dataclass from a JSON file |
| [`from_yaml`][pane.classes.PaneBase.from_yaml] | classmethod | Instantiate a dataclass from a YAML file |
| [`from_jsons`][pane.classes.PaneBase.from_jsons] | classmethod | Instantiate a dataclass from a JSON string |
| [`from_yamls`][pane.classes.PaneBase.from_yamls] | classmethod | Instantiate a dataclass from a YAML string |
| [`into_data`][pane.classes.PaneBase.into_data] | instance method | Convert dataclass into interchange data |
| [`dict`][pane.classes.PaneBase.dict] | instance method | Return dataclass fields as a dict (optionally, return only set fields) |

## Class arguments

Dataclass behavior can be customized using arguments passed to the class constructor. The following arguments are supported:

Args:

   - name (`str`): Name of the class, used in error messages.
   - `out_format` ([`ClassLayout`][pane.classes.ClassLayout]): Format to serialize class in (default: `struct`)
   - `in_format` (`t.Sequence[ClassLayout]`): Formats class may be serialized from (default: `['struct', 'tuple']` if no keyword arguments, else `['struct']`)
   - `eq` (`bool`): Whether to add `__eq__` method
   - `order` (`bool`): Whether to add `__ord__` method
   - `frozen` (`bool`): Whether to freeze dataclass
   - `init` (`bool`): Whether to add `__init__` method
   - `kw_only` (`bool`): If true, make all parameters keyword-only
   - `rename` ([`RenameStyle`][pane.field.RenameStyle]): How to rename parameters upon conversion to data. Cannot be specified with `in_rename` or `out_rename`
   - `in_rename` (`RenameStyle` or `t.Sequence[RenameStyle]`): Allow inputs renamed from these formats
   - `out_rename` (`RenameStyle`): Output renamed to this format
   - `allow_extra` (bool): Whether to allow (and ignore) extra parameters
   - `custom` ([`IntoConverterHandlers`][pane.convert.IntoConverterHandlers]): Custom handlers to apply to members of this class. Can be a handler, a list of handlers, or a dict mapping types to converters.

In general, these arguments are inherited by subclasses, unless overridden.

## Generic classes

Generic classes are supported natively, with proper inheritance.