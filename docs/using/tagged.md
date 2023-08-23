# Tagged unions

`pane` supports parsing [tagged unions](https://en.wikipedia.org/wiki/Tagged_union), which are invaluable
in representing complex data types. Unlike a untagged union (represented by [`t.Union`][typing.Union]),
tagged unions use a discriminating value to separate variants unambiguously.

Tagged unions are specified with the [`Tagged`][pane.annotations.Tagged] annotation wrapping a [`t.Union`][typing.Union]:

```python
import pane
from pane.annotations import Tagged

class Variant1(pane.PaneBase):
    x: t.Literal['variant1'] = 'variant1'
    y: int = 6

class Variant2(pane.PaneBase):
    x: t.Literal['variant2'] = 'variant2'
    y: str = 'mystring'

class Variant3(pane.PaneBase):
    x: t.Literal['variant3'] = 'variant3'
    y: int = 6
    z: int = 7

TaggedUnion = t.Annotated[t.Union[Variant1, Variant2, Variant3], Tagged('x')]
```

This specifies a tagged union with a tag (in Python) of `'x'`. Attribute `'x'`
is examined for each variant type, so that every possible value is uniquely associated
with a type.

When converting a value, the tag is matched first, and then the variant corresponding
to that tag:

```python
>>> pane.convert({'x': 'variant3'}, TaggedUnion)
Variant3(x='variant3', y=6, z=7)

>>> pane.convert({'x': 'variant2', 'y': 'str'}, TaggedUnion)
Variant2(x='variant2', y='str')

>>> pane.convert({'x': 'unknown'}, TaggedUnion)
Traceback (most recent call last):
...
pane.errors.ConvertError: Expected tag 'x' one of 'variant1', 'variant2', or 'variant3', instead got `unknown` of type `str`
```

Note that if we had used an untagged union instead, we would have no way to distinguish
between `Variant1` and `Variant3` in general.

## Tagged Union Layouts

By default, tagged unions are stored in the 'internally tagged' format, where tags are stored alongside
the variant's values.

Two other layouts are possible. First, the 'externally tagged' layout:

```python
>>> ExtTagged = t.Annotated[t.Union[Variant1, Variant2], Tagged('x', external=True)]
>>> pane.convert({'variant2': {'y': 'str'}}, ExtTagged)
Variant2(x='variant2', y='str')
```

In this format, the tag is stored as the sole key in a mapping enclosing the variant object.
The externally tagged format is often used by functional, type-safe languages such as Rust.

The final layout is the 'adjacently tagged' layout:

```python
>>> AdjTagged = t.Annotated[t.Union[Variant1, Variant2], Tagged('x', external=('t', 'c'))]
>>> pane.convert({'t': 'variant1', 'c': {'y': 8}}, AdjTagged)
Variant1(x='variant1', y=8)
```

In this format, the tag and content are stored alongside each other in a mapping. The
tuple `('t', 'c')` specifies the keys identifying the tag and content respectively. This
format is often used in Haskell.

These tagged union layouts (along with untagged unions) are modeled after the
[enum representations](https://serde.rs/enum-representations.html) in Rust's `serde` library.