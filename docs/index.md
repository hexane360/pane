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

## Comparison to other libraries

Coming soon