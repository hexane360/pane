# pane

`pane` is a modern Python library for dataclasses and data conversion, aiming
for speed and expressiveness.

[![][ci-badge]][ci-url] [![][commit-badge]][commit-url] [![][pypi-badge]][pypi-url] [![][docs-stable-badge]][docs-stable-url] [![][docs-dev-badge]][docs-dev-url]

`pane` draws heavy inspiration from [Pydantic][pydantic] (among others), but its goals
are quite different. For example, `pane` has first-class conversion to and from
all JSON datatypes, not just mappings (and no '__root__' hacks necessary).
In this sense `pane` is a library for general data conversion & validation,
while Pydantic is a dataclass-first library. In addition, `pane` is stricter
than Pydantic in several cases. For instance, `pane` [will not attempt to coerce
`3.14` or `"3"` to an integer](https://github.com/pydantic/pydantic/issues/578).

`pane` is designed to be used to create complex declarative configuration languages
in formats like JSON, TOML, and YAML. This requires full support for complex, nested
types like `t.Union[int, t.Tuple[int, str], list[int]]`. It also requires
useful error messages:

```python
>>> pane.convert('fail', t.Union[int, t.Tuple[int, str], list[int]])
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "pane/convert.py", line 489, in convert
    return from_data(data, ty)
           ^^^^^^^^^^^^^^^^^^^
  File "pane/convert.py", line 484, in from_data
    return converter.convert(val)
           ^^^^^^^^^^^^^^^^^^^^^^
  File "pane/convert.py", line 128, in convert
    raise ConvertError(node)
pane.convert.ConvertError: Expected one of:
- an int
- tuple of length 2
- sequence
Instead got `fail` of type `str`
```

## Features

`pane` is a work in progress. The following is a roadmap of features:

| Feature                      | State   |
| :--------------------------- | :----   |
| Basic type conversions       | Done    |
| Numeric array types (numpy)  | Done    |
| Sum & product type support   | Done    |
| Tagged unions                | Partial |
| 'Flattened' fields           | Planned |
| Basic dataclasses            | Done    |
| Dataclass helpers            | Partial |
| Generic & inherited dataclasses | Done |
| Parameter aliases & renaming | Done    |
| Arbitrary validation logic   | Done    |
| Schema import/export         | Not planned |

[pydantic]: https://github.com/pydantic/pydantic
[ci-badge]: https://github.com/hexane360/pane/workflows/Tests/badge.svg
[ci-url]: https://github.com/hexane360/pane/actions?query=workflow%3ATests
[docs-dev-badge]: https://img.shields.io/badge/docs-dev-blue
[docs-dev-url]: https://hexane360.github.io/pane/dev/
[docs-stable-badge]: https://img.shields.io/badge/docs-stable-blue
[docs-stable-url]: https://hexane360.github.io/pane/latest/
[commit-badge]: https://img.shields.io/github/last-commit/hexane360/pane
[commit-url]: https://github.com/hexane360/pane/commits
[pypi-badge]: https://badge.fury.io/py/py-pane.svg
[pypi-url]: https://pypi.org/project/py-pane/
