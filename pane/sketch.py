from __future__ import annotations

import abc
import dataclasses
import typing as t

T = t.TypeVar('T')

missing = object()

# conversion is a DFS over possibilities. if it fails, we rerun and collect
# the failures in a tree of possibilities

@dataclasses.dataclass
class SimpleNode:
    expected: str
    actual: t.Any

@dataclasses.dataclass
class ProductNode:
    name: str
    children: t.Dict[str, t.Union[ProductNode, SumNode, SimpleNode]]
    missing: t.Set[str]
    actual: t.Any

@dataclasses.dataclass
class SumNode:
    # sumnodes shouldn't contain sumnodes
    children: t.List[t.Union[ProductNode, SimpleNode]]


def print_error(node: t.Union[ProductNode, SumNode, SimpleNode], indent="", inside_sum=False):
    if isinstance(node, SimpleNode):
        if inside_sum:
            print(f"{node.expected}")
        else:
            print(f"Expected {node.expected}, instead got `{node.actual}` of type `{type(node.actual).__name__}`")
    elif isinstance(node, SumNode):
        print(f"Expected one of:")
        assert len(node.children)
        actual = None
        for child in node.children:
            print(f"{indent}- ", end="")
            print_error(child, f"{indent}  ", inside_sum=True)
            actual = child.actual
        print(f"{indent}Instead got `{actual}` of type `{type(actual).__name__}`")
    elif isinstance(node, ProductNode):
        # fuse together consecutive productnodes
        while len(node.children) == 1 and len(node.missing) == 0:
            field, child = next(iter(node.children.items()))
            if not isinstance(child, ProductNode):
                break
            node = ProductNode(node.name, {f"{field}.{k}": v for (k, v) in child.children.items()}, set(), node.actual)

        print(f"{'' if inside_sum else 'Expected '}'{node.name}'")
        for (field, child) in node.children.items():
            print(f"{indent}While parsing field '{field}':\n{indent}  ", end="")
            print_error(child, f"{indent}  ")

        for field in node.missing:
            print(f"{indent}  Missing required field '{field}'")


def make_converter(ty):
    if isinstance(ty, t.Dict):
        return StructConverter('struct', ty)
    if isinstance(ty, t.Sequence):
        return StructConverter('tuple', {str(i): v for (i, v) in enumerate(ty)})
    if t.get_origin(ty) == t.Union:
        return UnionConverter(t.get_args(ty))
    return SimpleConverter(ty)


class ParseInterrupt(Exception):
    ...


class Converter(abc.ABC, t.Generic[T]):
    @abc.abstractmethod
    def convert(self, val: t.Any) -> T:
        """
        Attempt to convert ``val`` to ``T``.
        Should raise ``ParseInterrupt`` when
        a given parsing path reaches a dead end.
        """
        ...

    @abc.abstractmethod
    def collect_errors(self, val: t.Any) -> t.Union[None, ProductNode, SimpleNode, SumNode]:
        """
        Return an error tree caused by converting ``val`` to ``T``.
        ``collect_errors`` should return ``None`` iff ``convert`` doesn't raise.
        """
        ...



class StructConverter(Converter[dict]):
    def __init__(self, name: str, fields: t.Dict[str, t.Any]):
        self.name = name
        self.fields = fields
        self.field_converters = {k: make_converter(v) for (k, v) in self.fields.items()}

    def convert(self, val) -> t.Any:
        if not isinstance(val, dict):
            raise ParseInterrupt()
        try:
            return {
                k: conv.convert(val[k]) for (k, conv) in self.field_converters.items()
            }
        except KeyError:
            raise ParseInterrupt() from None

    def collect_errors(self, actual) -> t.Optional[ProductNode]:
        failed_children = {}
        missing = set()
        for (k, conv) in self.field_converters.items():
            if k not in actual:
                missing.add(k)
                continue
            node = conv.collect_errors(actual[k])
            if node is not None:
                failed_children[k] = node
        if not len(failed_children) and not len(missing):
            return None
        return ProductNode(self.name, failed_children, missing, actual)


class UnionConverter(Converter[t.Any]):
    def __init__(self, types: t.Sequence[t.Any]):
        self.types = types
        self.converters = list(map(make_converter, types))

    def convert(self, val) -> t.Any:
        for conv in self.converters:
            try:
                return conv.convert(val)
            except Exception:
                pass
        raise ValueError("Can't parse")

    def collect_errors(self, actual) -> t.Optional[SumNode]:
        failed_children = []
        for (ty, conv) in zip(self.types, self.converters):
            node = conv.collect_errors(actual)
            # if one branch is successful, the whole type is successful
            if node is None:
                return None
            failed_children.append(node)
        return SumNode(failed_children)


class SimpleConverter:
    def __init__(self, ty: t.Any, expected: t.Optional[str] = None):
        self.expected = expected or ty.__name__
        self.ty = ty

    def convert(self, val) -> t.Any:
        if isinstance(val, self.ty):
            return val
        raise ParseInterrupt()

    def collect_errors(self, actual) -> t.Optional[SimpleNode]:
        if isinstance(actual, self.ty):
            return None
        return SimpleNode(f'{self.expected}', actual)


# now we just have to build this tree
tree = ProductNode('struct', {
    'n': ProductNode('struct2', {'m':
        SumNode([
        SimpleNode('a string', 3.),
        SimpleNode('an int', 3.),
        ProductNode('tuple', {
            'x': SimpleNode('a string', 3.),
        }, set(), {'x': 3.})
    ]),
}, set(), {})}, set(), {})

print_error(tree)

conv = make_converter({
    'x': int,
    'y': float,
    'z': str,
})

print(conv.convert({'x': 5, 'y': 2., 'z': 's'}))
node = conv.collect_errors({'x': 5, 'y': 2., 'z': 's'})
print(node)
if node is not None:
    print_error(node)

node = conv.collect_errors({'x': 5, 'y': 's'})
print(node)
if node is not None:
    print_error(node)

# example
"""
struct:
  n: t.Tuple[int, t.Union[str, int]]

{"n": (1, 3.)}

Result:
While parsing 'n[0]':
Expected one of:
 - a string
 - an int
Instead got `3.` of type `float`

struct:
  n: t.Tuple[int, t.Union[str, int]]
or
struct:
  m: t.Tuple[int, t.Union[str, int]]

Result:
While parsing 'struct':                # field name
Expected one of:                       # sumnode
 - struct_n                              # type name
   While parsing 'n[0]':                 # field name
   Expected one of:
    - a string
    - an int
    Instead got `3.` of type `float`
 - struct_m (Missing field 'm')
 Instead got `{}`

"""