"""
Annotations supported by ``pane.convert()`` and dataclasses.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
import math
import typing as t

from .util import flatten_union_args, is_broadcastable
from .util import list_phrase, pluralize, remove_article

if t.TYPE_CHECKING:
    from .convert import IntoConverter, ConverterHandlers
    from .converters import Converter, ConditionalConverter


class ConvertAnnotation(abc.ABC, t.Hashable):
    """
    Abstract annotation supported by `pane`.
    """
    @abc.abstractmethod
    def _converter(self, inner_type: t.Union[Converter[t.Any], IntoConverter], *,
                   handlers: ConverterHandlers) -> Converter[t.Any]:
        ...


@dataclass(frozen=True)
class Tagged(ConvertAnnotation):
    tag: str
    """Name of tag. This name will be searched in every Union member"""
    external: t.Union[bool, t.Tuple[str, str]] = False
    """
    Tagged unions can be stored three ways:
     - Internally tagged (`external=False`, default). In this format, the tags are stored inside of each object: `{tag_name: tag_value, **obj}`
     - Externally tagged (`external=True`). In this format, the tag is stored as a key outside the rest of the object: `{tag_value: obj}`
     - Adjacently tagged (`external=(tag_key, value_key)`). In this format, the tag and value are stored under separate items: `{tag_key: tag_value, value_key: obj}`

    This specification affects conversion into and out symmetrically.
    """

    # for some reason this isn't auto-generated on python 3.9
    def __hash__(self):
        return hash((self.__class__.__name__, self.tag, self.external))

    def _converter(self, inner_type: t.Union[Converter[t.Any], IntoConverter], *,
                   handlers: ConverterHandlers) -> Converter[t.Any]:

        from .converters import TaggedUnionConverter
        origin = t.get_origin(inner_type)
        if origin is not t.Union:
            raise TypeError("'Tagged' must surround a 'Union' type.")
        types = tuple(flatten_union_args(t.get_args(inner_type)))
        return TaggedUnionConverter(types, tag=self.tag, external=self.external, handlers=handlers)


@dataclass(frozen=True)
class Condition(ConvertAnnotation):
    f: t.Callable[[t.Any], bool]
    """
    Condition/predicate function.

    This is called with a parsed value, and should return `True` if it passes the condition.
    """
    name: t.Optional[str] = None
    """Human-readable name of this condition"""
    make_expected: t.Optional[t.Callable[[str, bool], str]] = None
    """
    Given an inner `expected` string, and a boolean indicating plurality, this should return a
    formatted `expected` string including the condition.

    This is a low-level function that can be overrided for better error messages.
    """

    # for some reason this isn't auto-generated on python 3.9
    def __hash__(self):
        return hash((
            self.__class__.__name__, self.f, self.name,
        ))

    def __and__(self, other: Condition) -> Condition:
        return Condition.all(self, other)

    def __or__(self, other: Condition) -> Condition:
        return Condition.any(self, other)

    def __invert__(self) -> Condition:
        return Condition(
            lambda val: not self.f(val),
            f"not {self.cond_name()}"
        )

    @staticmethod
    def all(*conditions: Condition, make_expected: t.Optional[t.Callable[[str, bool], str]] = None) -> Condition:
        """
        Create a condition by `and`ing together multiple conditions.

        Parameters:
          conditions: Conditions to combine
          make_expected: If specified, override `make_expected` on the result `Condition`.
        """
        return Condition(
            lambda val: all(cond.f(val) for cond in conditions),
            list_phrase(tuple(cond.cond_name() for cond in conditions), 'and'),
            make_expected
        )

    @staticmethod
    def any(*conditions: Condition, make_expected: t.Optional[t.Callable[[str, bool], str]] = None) -> Condition:
        """
        Create a condition by `or`ing together multiple conditions.

        Parameters:
          conditions: Conditions to combine
          make_expected: If specified, override `make_expected` on the result `Condition`.
        """
        return Condition(
            lambda val: any(cond.f(val) for cond in conditions),
            list_phrase(tuple(cond.cond_name() for cond in conditions), 'or'),
            make_expected
        )

    def cond_name(self) -> str:
        """Get the name of this condition"""
        return self.name or self.f.__name__

    def _converter(self, inner_type: t.Union[Converter[t.Any], IntoConverter], *,
                   handlers: ConverterHandlers) -> ConditionalConverter[t.Any]:
        from .converters import ConditionalConverter
        return ConditionalConverter(
            inner_type, self.f, self.cond_name(),
            self.make_expected or (lambda conv, plural: f"{conv} satisfying {self.cond_name()}"),
            handlers=handlers,
        )


def val_range(*, min: t.Union[int, float, None] = None, max: t.Union[int, float, None] = None) -> Condition:
    """`Condition` indicating that a value must be between `min` and `max` (inclusive)."""
    conds: t.List[Condition] = []
    if min is not None:
        conds.append(Condition(lambda v: v >= min, f"v >= {min}"))
    if max is not None:
        conds.append(Condition(lambda v: v <= max, f"v <= {max}"))
    return Condition.all(*conds)


def len_range(*, min: t.Optional[int] = None, max: t.Optional[int] = None) -> Condition:
    """`Condition` indicating that a value must have between `min` and `max` elements (inclusive)."""
    conds: t.List[Condition] = []
    if min is not None:
        conds.append(Condition(lambda v: len(v) >= min, f"at least {min} {pluralize('elem', min)}"))
    if max is not None:
        conds.append(Condition(lambda v: len(v) <= max, f"at most {max} {pluralize('elem', max)}"))
    cond = Condition.all(*conds, make_expected=lambda exp, plural: f"{exp} with {cond.cond_name()}")
    return cond


def shape(shape: t.Sequence[int]) -> Condition:
    """
    `Condition` indicating that a value must have a shape `shape`.

    Fails on objects that don't have a `shape` attribute.
    """
    name = f"shape {tuple(shape)}"
    return Condition(
        lambda v: v.shape == shape, name,
        lambda exp, plural: f"{exp} with {name}"
    )


def broadcastable(shape: t.Sequence[int]) -> Condition:
    """
    `Condition` indicating that a value must be broadcastable to shape `shape`.

    Fails on objects that don't have a `shape` attribute.
    """
    name = f"broadcastable to {tuple(shape)}"
    return Condition(
        lambda v: is_broadcastable(v.shape, shape), name,
        lambda exp, plural: f"{exp} {name}"
    )


def adjective_condition(f: t.Callable[[t.Any], bool], adjective: str, article: str = 'a') -> Condition:
    """
    Make a condition that can be expressed as a simple adjective (e.g. 'empty' or 'non-empty').

    Parameters:
      f: Condition/predicate function
      adjective: Adjective corresponding to `f` (e.g. 'empty' or 'non-empty').
      article: Article to put in front of `adjective` (only when not pluralized).
    """
    return Condition(
        f, adjective,
        # e.g. 'positive ints' if plural else 'a positive int'
        lambda exp, plural: f"{adjective} {exp}" if plural else f"{article} {adjective} {remove_article(exp)}"
    )


Positive = adjective_condition(lambda v: v > 0, 'positive')
"""`Condition` indicating value must be positive"""
Negative = adjective_condition(lambda v: v < 0, 'negative')
"""`Condition` indicating value must be negative"""
NonPositive = adjective_condition(lambda v: v <= 0, 'non-positive')
"""`Condition` indicating value must be non-positive"""
NonNegative = adjective_condition(lambda v: v >= 0, 'non-negative')
"""`Condition` indicating value must be non-negative"""
Finite = adjective_condition(math.isfinite, 'finite')
"""`Condition` indicating value must be finite"""
Empty = adjective_condition(lambda v: len(v) == 0, 'empty')
"""`Condition` indicating value must be empty (have no elements)"""
NonEmpty = adjective_condition(lambda v: len(v) != 0, 'non-empty')
"""`Condition` indicating value must not be empty"""


__all__ = [
    'ConvertAnnotation', 'Tagged', 'Condition', 'val_range', 'len_range',
    'Positive', 'Negative', 'NonPositive', 'NonNegative',
    'Finite', 'Empty', 'NonEmpty',
]
