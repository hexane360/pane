# [`Condition`][pane.annotations.Condition]: Adding extra restrictions to types

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