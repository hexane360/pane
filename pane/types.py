"""Helper types for use with ``pane.convert`` and dataclasses."""

import typing as t

from pane.annotations import *


PositiveInt = t.Annotated[int, Positive]
NonNegativeInt = t.Annotated[int, NonNegative]
NegativeInt = t.Annotated[int, Negative]
NonPositiveInt = t.Annotated[int, NonPositive]

PositiveFloat = t.Annotated[float, Positive]
NonNegativeFloat = t.Annotated[float, NonNegative]
NegativeFloat = t.Annotated[float, Negative]
NonPositiveFloat = t.Annotated[float, NonPositive]
FiniteFloat = t.Annotated[float, Finite]


__all__ = [
    'PositiveInt', 'NonNegativeInt', 'NegativeInt', 'NonPositiveInt',
    'PositiveFloat', 'NonNegativeFloat', 'NegativeFloat', 'NonPositiveFloat', 'FiniteFloat',
]