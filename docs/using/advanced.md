# Under the hood: Advanced usage

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

`expected` returns a brief string description of what values were expected by the converter. `try_convert` and `collect_errors` work together to perform parsing. When called with the same value, whenever `try_convert` succeeds, `collect_errors` should return `None`. Conversely, whenever `try_convert` raises a [`ParseInterrupt`][pane.errors.ParseInterrupt], `collect_errors` should return an [`ErrorNode`][pane.errors.ErrorNode]. This means much of the same control flow should be present in both functions. However, `try_convert` is on the fast path; it should do as little work as possible, including avoiding constructing errors.

With that said, There are a couple ways to inform `pane` of the presence of `CountryCodeConverter`. The simplest is through the [`HasConverter`][pane.convert.HasConverter] protocol. Just add a class method to `CountryCode`:

```python
class CountryCode:
    ...

    @classmethod
    def _converter(cls: t.Type[T], *args: type,
                   handlers: ConverterHandlers) -> CountryCodeConverter:
        if len(args):
            raise TypeError("'CountryCode' doesn't support type arguments")
        return CountryCodeConverter(cls)
```

In this protocol, any type arguments are passed to `args`. `handlers` contains
a invocation-specific set of custom handlers. If you call `make_converter` inside
of your `Converter`, you must pass `handlers` through to it.

With that defined, `convert()`, `from_data()` and dataclasses will work seamlessly with `CountryCode`.

## Supporting third-party datatypes

Sometimes you don't have access to a type to add a method to it.
In these instances, you may instead add a global custom handler to [`make_converter`][pane.convert.make_converter] using [`register_converter_handler`][pane.convert.register_converter_handler].
Say there's a type `Foo` that we'd like to support.
First, we need to make a `FooConverter` (see [Custom converters](#custom-converters) above).
Next, we make a function called `foo_converter_handler`, and register it:

```python
from pane.convert import register_converter_handler

# called with the type to make a converter for, and any type arguments
def foo_converter_handler(ty: t.Any, args: t.Tuple[t.Any, ...], /, *
                          handlers: ConverterHandlers) -> FooConverter:
    if not issubclass(ty, Foo):
        return NotImplemented  # not a foo type, can't handle it
    return FooConverter(ty, args)

register_converter_handler(foo_converter_handler)
```

Converter handlers can also be passed to [`convert`][pane.convert.convert], [`into_data`][pane.convert.into_data], and [`from_data`][pane.convert.from_data] using the `custom` option:

```python
foo = from_data({'foo': 'bar'}, Foo, custom=foo_converter_handler)
```

`custom` may be a handler, a sequence of handlers (called in order), or a dict mapping types to `Converter`s.

Local converter handlers are applied after special forms (e.g. `t.Union`), but before anything else.
Global converter handlers are applied after basic type handlers and the [`HasConverter`][pane.convert.HasConverter] protocol (to increase performance), but before handlers for subclasses, `tuple`s, or `dict`s.

## Custom annotations

Custom annotations are also supported. To create a custom annotation, subclass [`ConvertAnnotation`][pane.annotations.ConvertAnnotation]. `_converter` will be called to construct a converter, with `inner_type` containing the type inside the annotation (or a `Converter` in the case of nested annotations). Raise a `TypeError` if `inner_type` isn't supported or understood by the annotation. `handlers` is a set of local converter handlers, which again must be passed through to any calls to `make_converter`.