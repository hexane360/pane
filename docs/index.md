# py-pane

`pane` is a modern data conversion and dataclass library for Python.

There are many existing dataclass libraries for Python. `pane` gains in composibility,
flexibility, and simplicity because it treats validating a dataclass as a special case
of general type conversion. This base layer is built with robust support for product types,
tagged & untagged unions, and custom predicates.

## Features

- Conversion between arbitrary types
- Helpful, detailed error messages
- Conversion to and from dataclasses
  - Optional fields
  - Field renaming
  - Conversion from tuples & dicts
  - Dataclass inheritance
  - Generic dataclasses
  - Condition (field validators)
- First class typing support
- Tagged & untagged unions
- Composable conversion
- Custom converters and hooks for extension

## Installation

`pane` is available from PyPI. To install:

```sh
pip install py-pane
```

`pane` deliberately has very few depedencies, and has no binary dependencies.

## Supported datatypes

`pane` aims to support a broad range of standard library and third-party datatypes.
Currently, the following datatypes are supported:

### Standard library

Sequence/collection types:
- `list`/`typing.List`
- `tuple`/`typing.Tuple`/`typing.Sequence`/`collections.abc.Sequence`
- `set`/`typing.Set`/`collections.abc.Set`
- `frozenset`/`typing.FrozenSet`/`collections.abc.FrozenSet`
- `collections.deque`/`typing.Deque`

Tuple types:
- Heterogeneous: `tuple[int, str]`/`t.Tuple[int, str]`
- Homogeneous: `tuple[int, ...]`/`t.Tuple[int, ...]`
- Empty: `tuple[()]`/`t.Tuple[()]`

Mapping types:
- `dict`/`typing.Dict`/`typing.Mapping`/`typing.MutableMapping`/`collections.abc.Mapping`/`collections.abc.MutableMapping`
- `collections.defaultdict`/`typing.DefaultDict`
- `collections.OrderedDict`/`typing.OrderedDict`
- `collections.Counter`

String/bytes types:
- `str`
- `bytes`
- `bytearray`

Numeric types:
- `int`
- `float`
- `complex`
- `decimal.Decimal`
- `fraction.Fraction`

Datetime types:
- `datetime.datetime`
- `datetime.date`
- `datetime.time`

Other scalar types:
- `bool`
- `None`
- `re.Pattern`/`typing.Pattern`

### Third-party datatypes

- `numpy.ndarray`/`NDArray[]`

## Comparison to other libraries

Coming soon