"""Error types for ``pane`` library."""

from __future__ import annotations

import abc
from io import StringIO
import sys
import traceback
import dataclasses
import typing as t


class ParseInterrupt(Exception):
    """
    Raised by [`Converter`][pane.converters.Converter]s to indicate that a given parsing path has failed
    (without materializing a detailed error message).
    """
    ...


class UnsupportedAnnotation(Exception):
    """
    Raised when a given [`t.Annotated`][typing.Annotated] isn't understood by `pane`.
    """
    def __init__(self, obj: t.Any):
        self.obj: t.Any = obj

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.obj!r})"

    def __str__(self) -> str:
        return f"Unsupported annotation: {self.obj!r}"


class ConvertError(Exception):
    """
    `pane` conversion error.

    `self.tree` contains a detailed error tree, and `str(self)`
    is a human-friendly representation of the same.
    """
    def __init__(self, tree: ErrorNode):
        self.tree: ErrorNode = tree

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.tree!r})"

    def __str__(self) -> str:
        return str(self.tree)


class ErrorNode(abc.ABC):
    """
    Abstract node in a conversion error tree
    """

    @abc.abstractmethod
    def print_error(self, indent: str = "", inside_sum: bool = False, file: t.TextIO = sys.stdout):
        """
        Print a description of this error to `file`.

        Parameters:
          indent: String to indent all extra lines we print
          inside_sum: Whether we are printing inside a [`SumErrorNode`][pane.errors.SumErrorNode] 
                      and so should omit printing the actual value we got)
          file: File-like object to print text to
        """
        ...

    def __str__(self) -> str:
        buf = StringIO()
        self.print_error(file=buf)
        return buf.getvalue().rstrip('\n')


@dataclasses.dataclass
class WrongTypeError(ErrorNode):
    expected: str
    """Short description of expected value type"""
    actual: t.Any
    """Actual value received"""
    cause: t.Optional[traceback.TracebackException] = None
    """If this was caused by an error, contains a traceback to that error"""
    info: t.Optional[str] = None
    """Additional information to supply on an new line"""

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
        """Format `cause` as a human-readable string"""
        if self.cause is None:
            return 'None'
        if isinstance(self.cause, traceback.TracebackException):
            return "\n".join(self.cause.format_exception_only())
        return "\n".join(traceback.format_exception(type(self.cause), self.cause, None))

    def __repr__(self) -> str:
        return f"WrongTypeError(expected={self.expected!r}, actual={self.actual!r}, cause={self._get_cause()!r}, info={self.info!r})"

    def __eq__(self, other: t.Any) -> bool:
        # mostly useful for testing
        if not self.__class__ == other.__class__:
            return False

        return (
            self.expected == other.expected and
            self.actual == other.actual and
            self.info == other.info and
            self._get_cause() == other._get_cause()
        )


@dataclasses.dataclass
class WrongLenError(ErrorNode):
    expected: str
    """Short description of expected value type"""
    expected_len: t.Tuple[int, int]
    """(min, max) expected value length"""
    actual: t.Any
    """Actual value received"""
    actual_len: int
    """Actual length received"""

    def print_error(self, indent: str = "", inside_sum: bool = False, file: t.TextIO = sys.stdout):
        len_range = '-'.join(map(str, self.expected_len))
        if inside_sum:
            print(f"{self.expected} (length {len_range})", file=file)
        else:
            print(f"Expected {self.expected} of length {len_range}, instead got `{self.actual}` of length {self.actual_len}", file=file)


@dataclasses.dataclass
class ConditionFailedError(ErrorNode):
    expected: str
    """Short description of expected value type"""
    actual: t.Any
    """Actual value received"""
    condition: str
    """Name of condition which failed"""
    cause: t.Optional[traceback.TracebackException] = None
    """If this was caused by an error, contains a traceback to that error"""

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
    """Offending key"""
    aliases: t.Sequence[str]
    """List of keys semantically identical to `key`"""

    def print_error(self, indent: str = "", inside_sum: bool = False, file: t.TextIO = sys.stdout):
        assert not inside_sum
        print(f"Duplicate key {self.key} (same as {'/'.join(self.aliases)})", file=file)


@dataclasses.dataclass
class ProductErrorNode(ErrorNode):
    expected: str
    """Short description of expected value type"""
    children: t.Dict[t.Union[int, str], ErrorNode]
    """Map containing errors parsing subfields, if any"""
    actual: t.Any
    """Actual value received"""
    missing: t.AbstractSet[t.Union[t.Sequence[str], str]] = dataclasses.field(default_factory=set[t.Union[t.Sequence[str], str]])
    """List of missing fields/equivalent aliases to fields"""
    extra: t.AbstractSet[str] = dataclasses.field(default_factory=set[str])
    """List of extra, unexpected fields"""

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
    """Map containing the errors while parsing as each variant"""

    def print_error(self, indent: str = "", inside_sum: bool = False, file: t.TextIO = sys.stdout):
        def _flatten_sum(children: t.Iterable[ErrorNode]) -> t.Iterator[ErrorNode]:
            for child in children:
                if isinstance(child, SumErrorNode):
                    yield from child.children
                else:
                    yield child

        print("Expected one of:", file=file)
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
