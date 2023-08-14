# Using pane

## [`convert`][pane.convert.convert], [`from_data`][pane.convert.from_data] and [`into_data`][pane.convert.into_data]

The core of `pane` is its data conversion routines. The simplest to use is `convert`:

```python
>>> import typing as t
>>> import pane
>>> pane.convert(-5., float)  # simple scalar type
-5.0
>>> pane.convert([1., 'mystring', 5], (float, str, int))  # tuple-like type
(1.0, 'mystring', 5)
>>> pane.convert({'x': 5.0, 'y': 'mystring'}, {'x': float, 'y': str})  # struct-like type
{'x': 5.0, 'y': 'mystring'}
>>> pane.convert(5, t.Union[str, int, None])  # untagged union
5
```

Under the hood, [`convert`][pane.convert.convert] first calls [`into_data`][pane.convert.into_data] on a value. This attempts to convert the value into a "data interchange type", which is a dialect of types supported by all converters.

After [`into_data`][pane.convert.into_data], [`convert`][pane.convert.convert] calls [`from_data`][pane.convert.from_data], which attempts to convert the value into the desired type.
This function recursively parses the type to find a [`Converter`][pane.converters.Converter] implementation to call, then calls [`Converter.convert`][pane.converters.Converter.convert].

Currently, the data interchange types are:
 - Scalar types: [str][], [bytes][], [bool][], [int][], [float][], [complex][], and [None][]
 - Sequences: [list][], [tuple][] and [`t.Sequence`][typing.Sequence]
 - Mappings: [dict][] and [`t.Mapping`][typing.Mapping]

All implementations of [`Converter.into_data`][pane.converters.Converter.into_data] must output a data interchange type, and all implementations of [`Converter.convert`][pane.converters.Converter.convert] must handle any data interchange type (even if 'handle' just means raising an error message).
Data interchange types may be added in major releases, so [`Converter`][pane.converters.Converter] implementations should be made robust to new types.

## `pane` dataclasses

`pane` dataclasses work similarly to in many other libraries. A dataclass can be made by subclassing `pane.PaneBase`:

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
```

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

`MyDataclass.from_data`, `MyDataclass.from_json`, and `MyDataclass.from_yaml` perform conversion from an object:

```python
# note the use of an alias for 'z'
>>> MyDataclass.from_data({'x': 10, 'y': 10., 'w': [10.]})
MyDataclass(x=10, y=10.0, z=[10.0])

>>> from io import StringIO
>>> MyDataclass.from_json(StringIO('{"x": 10, "y": 10.0, "w": [10.0]}'))
MyDataclass(x=10, y=10.0, z=[10.0])
```

(`from_json` and `from_yaml` take a file-like object or filename)

`MyDataclass.into_data` performs the reverse operation:

```python
>>> MyDataclass(x=10, y=10.0, z=[10.0]).into_data()
{'x': 10, 'y': 10.0, 'z': [10.0]}
```

## [`Condition`][pane.annotations.Condition]: Adding extra restrictions to types


[`Condition`][pane.annotations.Condition] is `pane`'s solution to field validation.
In `pane`, features are built into the type system whenever possible, increasing composibility and flexibility.
So to with value restrictions.
These are implemented using [`t.Annotated`][typing.Annotated].

We'll start with some examples:

```python
>>> import typing as t
>>> from pane import convert, val_range, Condition, Positive

# built-in conditions
>>> convert(5.0, t.Annotated[float, Positive])
5.0

>>> convert(-1.0, t.Annotated[float, Positive])
Traceback (most recent call last):
...
pane.errors.ConvertError: Expected a positive float, instead got `-1.0` (failed condition 'positive')

>>> convert(6.0, t.Annotated[float, val_range(max=5.0)])
Traceback (most recent call last):
...
pane.errors.ConvertError: Expected a float satisfying v <= 5.0, instead got `6.0` (failed condition 'v <= 5.0')

# custom conditions
>>> convert([0, 1, 2, 4], t.List[t.Annotated[int, Condition(lambda v: v % 2 == 0, name='even')]])
Traceback (most recent call last):
...
pane.errors.ConvertError: Expected sequence of ints satisfying even
While parsing field '1':
  Expected an int satisfying even, instead got `1` (failed condition 'even')
```

These conditions can be applied at any nesting level, not limited to top-level fields. Conditions also support the `&` and `|` bitwise operators (interpreted as boolean operators).

## [`Tagged`][pane.annotations.Tagged]: Tagged unions

Documentation coming soon!

## Custom converters

Out-of-the-box, `pane` supports `numpy` arrays and datatypes, as well as types which follow
the [`t.Sequence`][typing.Sequence]/[`t.Mapping`][typing.Mapping] protocol.

However, `pane` can easily be extended to support additional types.
The first step is to create a [`Converter`][pane.converters.Converter] which handles the type.
The [`Converter`][pane.converters.Converter] interface is quite simple. Three functions are required: [`expected`][pane.converters.Converter.expected], [`try_convert`][pane.converters.Converter.try_convert], and [`collect_errors`][pane.converters.Converter.collect_errors].

Say we have a type `CountryCode`, which contains a standard country code. `CountryCodeConverter` should accept a string-like type and convert it to a `CountryCode`, making
sure that the string really is a country code. (In reality, this type could be implemented as `t.Literal['gb', 'cn', ...]`)

An example implementation of `CountryCodeConverter` is shown below:

```python
import typing as t
from pane.errors import WrongTypeError, ErrorNode

class CountryCodeConverter:
    countries = {'gb', 'us', 'cn', 'uk'}
    def __init__(self, ty: t.Type[CountryCode]):
        # type of CountrySet (could be a subclass)
        self.ty = ty

    def expected(self, plural: bool = False):
        """Return the value we expected (pluralized if `plural`)."""
        return "country codes" if plural else "a country code"

    # attempt to convert `val`.
    # in this function, we only raise ParseInterrupt, never
    # constructing an error
    # this is to save time in case another conversion branch succeeds
    def try_convert(self, val: t.Any) -> CountryCode:
        # the only data interchange type we support is `str`. Everything
        # else should error
        if not isinstance(val, str):
            raise ParseInterrupt()

        # check that `val` is a valid country code
        if val not in self.countries:
            raise ParseInterrupt()

        return CountryCode(val)

    # after try_convert fails, collect_errors is called
    # to make full error messages.
    # collect_errors should return an error iff try_convert raises ParseInterrupt
    def collect_errors(self, val: t.Any) -> t.Optional[ErrorNode]:
        if not isinstance(val, str):
            # every ParseInterrupt() in try_convert corresponds
            # to an error in collect_errors
            return WrongTypeError(self.expected(), val)

        if val not in self.countries:
            return WrongTypeError(self.expected(), val, info=f"Unknown country code '{val}'")

        return None
```

There are a couple ways to inform `pane` of the presence of `CountryCodeConverter`. The simplest is through the [`HasConverter`][pane.convert.HasConverter] protocol. Just add a class method to `CountryCode`:

```python
class CountryCode:
    ...

    @classmethod
    def _converter(cls: t.Type[T], *args: type) -> CountryCodeConverter:
        if len(args):
            raise TypeError("'CountryCode' doesn't support type arguments")
        return CountryCodeConverter(cls)
```

Now, `convert()`, `from_data()` and dataclasses will work seamlessly with `CountryCode`.

### Supporting third-party datatypes

Sometimes you don't have access to a type to add a method to it.
In these instances, you may instead add a custom handler to [`make_converter`][pane.convert.make_converter] using [`register_converter_handler`][pane.convert.register_converter_handler].
Say there's a type `Foo` that we'd like to support.
First, we need to make a `FooConverter` (see [Custom converters](#custom-converters) above).
Next, we make a function called `foo_converter_handler`, and register it:

```python
from pane.convert import register_converter_handler

# called with type to make a converter for
# and any type arguments
def foo_converter_handler(ty: t.Any, args: t.Tuple[t.Any, ...]) -> FooConverter:
    if not issubclass(ty, Foo):
        return NotImplemented  # not a foo type, can't handle it
    return FooConverter(ty, args)

register_converter_handler(foo_converter_handler)
```

Converter handlers are applied after basic type handlers and the [`HasConverter`][pane.convert.HasConverter] protocol (to increase performance), but before handlers for subclasses, `tuple`s, or `dict`s.