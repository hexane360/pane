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
    from .convert import IntoConverter
    from .converters import Converter, ConditionalConverter


class ConvertAnnotation(abc.ABC, t.Hashable):
    @abc.abstractmethod
    def _converter(self, inner_type: t.Union[Converter[t.Any], IntoConverter]) -> Converter[t.Any]:
        ...


@dataclass(frozen=True)
class Tagged(ConvertAnnotation):
    tag: str
    external: t.Union[bool, t.Tuple[str, str]] = False

    # for some reason this isn't auto-generated on python 3.9
    def __hash__(self):
        return hash((self.__class__.__name__, self.tag, self.external))

    def _converter(self, inner_type: t.Union[Converter[t.Any], IntoConverter]) -> Converter[t.Any]:
        from .converters import TaggedUnionConverter
        origin = t.get_origin(inner_type)
        if origin is not t.Union:
            raise TypeError(f"'Tagged' must surround a 'Union' type.")
        types = tuple(flatten_union_args(t.get_args(inner_type)))
        return TaggedUnionConverter(types, tag=self.tag, external=self.external)


@dataclass(frozen=True)
class Condition(ConvertAnnotation):
    f: t.Callable[[t.Any], bool]
    name: t.Optional[str] = None
    make_expected: t.Optional[t.Callable[[str, bool], str]] = None

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
        return Condition(
            lambda val: all(cond.f(val) for cond in conditions),
            list_phrase(tuple(cond.cond_name() for cond in conditions), 'and'),
            make_expected
        )

    @staticmethod
    def any(*conditions: Condition, make_expected: t.Optional[t.Callable[[str, bool], str]] = None) -> Condition:
        return Condition(
            lambda val: any(cond.f(val) for cond in conditions),
            list_phrase(tuple(cond.cond_name() for cond in conditions), 'or'),
            make_expected
        )

    def cond_name(self) -> str:
        return self.name or self.f.__name__

    def _converter(self, inner_type: t.Union[Converter[t.Any], IntoConverter]) -> ConditionalConverter[t.Any]:
        from .converters import ConditionalConverter
        return ConditionalConverter(
            inner_type, self.f, self.cond_name(),
            self.make_expected or (lambda conv, plural: f"{conv} satisfying {self.cond_name()}"),
        )


def val_range(*, min: t.Union[int, float, None] = None, max: t.Union[int, float, None] = None) -> Condition:
    conds: t.List[Condition] = []
    if min is not None:
        conds.append(Condition(lambda v: v >= min, f"v >= {min}"))
    if max is not None:
        conds.append(Condition(lambda v: v <= max, f"v <= {max}"))
    return Condition.all(*conds)


def len_range(*, min: t.Optional[int] = None, max: t.Optional[int] = None) -> Condition:
    conds: t.List[Condition] = []
    if min is not None:
        conds.append(Condition(lambda v: len(v) >= min, f"at least {min} {pluralize('elem', min)}"))
    if max is not None:
        conds.append(Condition(lambda v: len(v) <= max, f"at most {max} {pluralize('elem', max)}"))
    cond = Condition.all(*conds, make_expected=lambda exp, plural: f"{exp} with {cond.cond_name()}")
    return cond


def shape(shape: t.Sequence[int]) -> Condition:
    name = f"shape {tuple(shape)}"
    return Condition(
        lambda v: v.shape == shape, name,
        lambda exp, plural: f"{exp} with {name}"
    )


def broadcastable(shape: t.Sequence[int]) -> Condition:
    name = f"broadcastable to {tuple(shape)}"
    return Condition(
        lambda v: is_broadcastable(v.shape, shape), name,
        lambda exp, plural: f"{exp} {name}"
    )


def adjective_condition(f: t.Callable[[t.Any], bool], adjective: str, article: str = 'a') -> Condition:
    """Make a condition that can be expressed as a simple adjective (e.g. 'empty' or 'non-empty')."""
    return Condition(
        f, adjective,
        # e.g. 'positive ints' if plural else 'a positive int'
        lambda exp, plural: f"{adjective} {exp}" if plural else f"{article} {adjective} {remove_article(exp)}"
    )


Positive = adjective_condition(lambda v: v > 0, 'positive')
Negative = adjective_condition(lambda v: v < 0, 'negative')
NonPositive = adjective_condition(lambda v: v <= 0, 'non-positive')
NonNegative = adjective_condition(lambda v: v >= 0, 'non-negative')
Finite = adjective_condition(math.isfinite, 'finite')
Empty = adjective_condition(lambda v: len(v) == 0, 'empty')
NonEmpty = adjective_condition(lambda v: len(v) != 0, 'non-empty')


__all__ = [
    'ConvertAnnotation', 'Tagged', 'Condition', 'val_range', 'len_range',
    'Positive', 'Negative', 'NonPositive', 'NonNegative',
    'Finite', 'Empty', 'NonEmpty',
]
