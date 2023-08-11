"""Error types for ``pane`` library."""

from __future__ import annotations

import abc
from io import StringIO
import sys
import traceback
import dataclasses
import typing as t


class ParseInterrupt(Exception):
    ...


class UnsupportedAnnotation(Exception):
    def __init__(self, obj: t.Any):
        self.obj: t.Any = obj

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.obj!r})"

    def __str__(self) -> str:
        return f"Unsupported annotation: {self.obj!r}"


class ConvertError(Exception):
    def __init__(self, tree: ErrorNode):
        self.tree: ErrorNode = tree

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.tree!r})"

    def __str__(self) -> str:
        return str(self.tree)


class ErrorNode(abc.ABC):
    @abc.abstractmethod
    def print_error(self, indent: str = "", inside_sum: bool = False, file: t.TextIO = sys.stdout):
        ...

    def __str__(self) -> str:
        buf = StringIO()
        self.print_error(file=buf)
        return buf.getvalue().rstrip('\n')


@dataclasses.dataclass
class WrongTypeError(ErrorNode):
    expected: str
    actual: t.Any
    cause: t.Optional[traceback.TracebackException] = None
    info: t.Optional[str] = None

    def print_error(self, indent: str = "", inside_sum: bool = False, file: t.TextIO = sys.stdout):
        if inside_sum:
            print(f"{self.expected}", file=file)
        else:
            print(f"Expected {self.expected}, instead got `{self.actual}` of type `{type(self.actual).__name__}`", file=file)
        if self.info is not None:
            print(f"{indent}{self.info}", file=file)
        if self.cause is not None:
            s = f"{indent}\n".join(self.cause.format())
            print(f"Caused by exception:\n{indent}{s}", file=file)

    def _get_cause(self) -> str:
        if self.cause is None:
            return 'None'
        if isinstance(self.cause, traceback.TracebackException):
            return "\n".join(self.cause.format_exception_only())
        return "\n".join(traceback.format_exception(type(self.cause), self.cause, None))

    def __repr__(self) -> str:
        return f"WrongTypeError(expected={self.expected!r}, actual={self.actual!r}, cause={self._get_cause()!r}, info={self.info!r})"

    def __eq__(self, other: t.Any) -> bool:
        if not self.__class__ == other.__class__:
            return False

        return (
            self.expected == other.expected and
            self.actual == other.actual and
            self.info == other.info and
            self._get_cause() == other._get_cause()
        )


@dataclasses.dataclass
class ConditionFailedError(ErrorNode):
    expected: str
    actual: t.Any
    condition: str
    cause: t.Optional[traceback.TracebackException] = None

    def print_error(self, indent: str = "", inside_sum: bool = False, file: t.TextIO = sys.stdout):
        if inside_sum:
            print(self.expected, end="", file=file)
        else:
            print(f"Expected {self.expected}, instead got `{self.actual}`", end="", file=file)
        if self.cause is not None:
            s = f"{indent}\n".join(self.cause.format())
            print(f"\nFailed to call condition '{self.condition}':\n{indent}{s}", file=file)
        else:
            print(f" (failed condition '{self.condition}')", file=file)


@dataclasses.dataclass
class DuplicateKeyError(ErrorNode):
    key: str
    aliases: t.Sequence[str]

    def print_error(self, indent: str = "", inside_sum: bool = False, file: t.TextIO = sys.stdout):
        assert not inside_sum
        print(f"Duplicate key {self.key} (same as {'/'.join(self.aliases)})", file=file)


@dataclasses.dataclass
class ProductErrorNode(ErrorNode):
    expected: str
    children: t.Dict[t.Union[int, str], ErrorNode]
    actual: t.Any
    missing: t.AbstractSet[t.Union[t.Sequence[str], str]] = dataclasses.field(default_factory=set)
    extra: t.AbstractSet[str] = dataclasses.field(default_factory=set)

    def print_error(self, indent: str = "", inside_sum: bool = False, file: t.TextIO = sys.stdout):
        # fuse together non-branching productnodes
        while len(self.children) == 1 and not len(self.missing) and not len(self.extra):
            field, child = next(iter(self.children.items()))
            if not isinstance(child, ProductErrorNode):
                break
            children: t.Dict[t.Union[str, int], ErrorNode] = {f"{field}.{k}": v for (k, v) in child.children.items()}
            missing = set(f"{field}.{f}" for f in child.missing)
            extra = set(f"{field}.{f}" for f in child.extra)
            self = ProductErrorNode(self.expected, children, self.actual, missing, extra)

        print(f"{'' if inside_sum else 'Expected '}{self.expected}", file=file)
        for (field, child) in self.children.items():
            print(f"{indent}While parsing field '{field}':\n{indent}  ", end="", file=file)
            child.print_error(f"{indent}  ", file=file)

        for field in self.missing:
            if not isinstance(field, str):
                field = '/'.join(field)
            print(f"{indent}  Missing required field '{field}'", file=file)

        for field in self.extra:
            print(f"{indent}  Unexpected field '{field}'", file=file)


@dataclasses.dataclass
class SumErrorNode(ErrorNode):
    children: t.List[ErrorNode]

    def print_error(self, indent: str = "", inside_sum: bool = False, file: t.TextIO = sys.stdout):
        def _flatten_sum(children: t.Iterable[ErrorNode]) -> t.Iterator[ErrorNode]:
            for child in children:
                if isinstance(child, SumErrorNode):
                    yield from child.children
                else:
                    yield child

        print(f"Expected one of:", file=file)
        actual = None
        for child in _flatten_sum(self.children):
            print(f"{indent}- ", end="", file=file)
            child.print_error(f"{indent}  ", inside_sum=True, file=file)
            actual = getattr(child, 'actual', actual)
        print(f"{indent}Instead got `{actual}` of type `{type(actual).__name__}`", file=file)


__all__ = [
    'ErrorNode', 'ProductErrorNode', 'SumErrorNode',
    'DuplicateKeyError', 'WrongTypeError', 'ConvertError',
    'ParseInterrupt',
]
