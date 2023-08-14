# Basic type conversion: [`convert`][pane.convert.convert] & friends

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

[`convert`][pane.convert.convert] actually performs two separate actions. First, it calls [`into_data`][pane.convert.into_data] on a value. This attempts to convert the value into a "data interchange type", which is a dialect of types supported by all converters.

After [`into_data`][pane.convert.into_data], [`convert`][pane.convert.convert] calls [`from_data`][pane.convert.from_data], which attempts to convert the value into the desired type.
<!-- This function recursively parses the type to find a [`Converter`][pane.converters.Converter] implementation to call, then calls [`Converter.convert`][pane.converters.Converter.convert]. -->

Currently, the data interchange types are:
 - Scalar types: [str][], [bytes][], [bool][], [int][], [float][], [complex][], and [None][]
 - Sequences: [list][], [tuple][] and [`t.Sequence`][typing.Sequence]
 - Mappings: [dict][] and [`t.Mapping`][typing.Mapping]

All implementations of [`Converter.into_data`][pane.converters.Converter.into_data] must output a data interchange type, and all implementations of [`Converter.convert`][pane.converters.Converter.convert] must handle any data interchange type (even if 'handle' just means raising an error message).
Data interchange types may be added in major releases, so [`Converter`][pane.converters.Converter] implementations should be made robust to new types.