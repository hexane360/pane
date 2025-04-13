from contextlib import AbstractContextManager, nullcontext
from io import TextIOBase, IOBase, TextIOWrapper, BufferedIOBase
from pathlib import Path
import typing as t

from typing_extensions import TypeAlias

from .convert import (
    from_data, into_data, IntoConverterHandlers,
    Convertible, IntoConverter
)


T = t.TypeVar('T', bound='Convertible')
FileOrPath: TypeAlias = t.Union[str, Path, TextIOBase, t.TextIO]


def from_json(f: FileOrPath, ty: t.Type[T], *,
              custom: t.Optional[IntoConverterHandlers] = None) -> T:
    """
    Load an object of type `ty` from a JSON file `f`

    Parameters:
        f: File-like or path-like to load from
        custom: Custom converters to use
    """
    import json
    with open_file(f) as f:
        obj = json.load(f)
    return from_data(obj, ty, custom=custom)


def from_yaml(f: FileOrPath, ty: t.Type[T], *,
              custom: t.Optional[IntoConverterHandlers] = None) -> T:
    """
    Load an object of type `ty` from a YAML file `f`

    Parameters:
        f: File-like or path-like to load from
        custom: Custom converters to use
    """
    import yaml
    try:
        from yaml import CSafeLoader as Loader
    except ImportError:
        from yaml import SafeLoader as Loader

    with open_file(f) as f:
        obj = t.cast(t.Any, yaml.load(f, Loader))  # type: ignore

    return from_data(obj, ty, custom=custom)


def from_yaml_all(f: FileOrPath, ty: t.Type[T], *,
                  custom: t.Optional[IntoConverterHandlers] = None) -> t.List[T]:
    """
    Load an object of type `ty` from a YAML file `f`

    Parameters:
        f: File-like or path-like to load from
        custom: Custom converters to use
    """
    import yaml
    try:
        from yaml import CSafeLoader as Loader
    except ImportError:
        from yaml import SafeLoader as Loader

    with open_file(f) as f:
        obj = t.cast(t.List[t.Any], list(yaml.load_all(f, Loader)))  # type: ignore

    return from_data(obj, t.List[ty], custom=custom)


def write_json(obj: Convertible, f: FileOrPath, *,
               ty: t.Optional[IntoConverter] = None,
               indent: t.Union[str, int, None] = None,
               sort_keys: bool = False,
               custom: t.Optional[IntoConverterHandlers] = None):
    """
    Write data to a JSON file `f`

    Parameters:
      obj: Object to write
      ty: Type of object
      f: File-like or path-like to write to
      indent: Indent to format JSON with. Defaults to None (no indentation)
      sort_keys: Whether to sort keys prior to serialization.
      custom: Custom converters to use
    """
    import json

    with open_file(f, 'w') as f:
        json.dump(
            into_data(obj, ty, custom=custom),
            f, indent=indent, sort_keys=sort_keys
        )


def write_yaml(obj: Convertible, f: FileOrPath, *,
               ty: t.Optional[IntoConverter] = None,
               indent: t.Optional[int] = None,
               width: t.Optional[int] = None,
               allow_unicode: bool = True,
               explicit_start: bool = True, explicit_end: bool = False,
               default_style: t.Optional[t.Literal['"', '|', '>']] = None,
               default_flow_style: t.Optional[bool] = None,
               sort_keys: bool = False,
               custom: t.Optional[IntoConverterHandlers] = None):
    """
    Write data to a YAML file `f`

    Parameters:
      obj: Object to write
      ty: Type of object
      f: File-like or path-like to write to
      indent: Number of spaces to indent blocks with
      width: Maximum width of file created
      allow_unicode: Whether to output unicode characters or escape them
      explicit_start: Whether to include a YAML document start "---"
      explicit_end: Whether to include a YAML document end "..."
      default_style: Default style to use for scalar nodes.
          See YAML documentation for more information.
      default_flow_style: Whether to default to flow style or block style for collections.
          See YAML documentation for more information.
      sort_keys: Whether to sort keys prior to serialization.
      custom: Custom converters to use
    """
    import yaml
    try:
        from yaml import CSafeDumper as Dumper
    except ImportError:
        from yaml import SafeDumper as Dumper

    with open_file(f, 'w') as f:
        yaml.dump(  # type: ignore
            into_data(obj, ty, custom=custom), f, Dumper=Dumper,
            indent=indent, width=width, allow_unicode=allow_unicode,
            explicit_start=explicit_start, explicit_end=explicit_end,
            default_style=default_style, default_flow_style=default_flow_style,
            sort_keys=sort_keys
        )


def _validate_file(f: t.Union[t.IO[t.AnyStr], IOBase], mode: t.Union[t.Literal['r'], t.Literal['w']]):
    if f.closed:
        raise IOError("Error: Provided file is closed.")

    if mode == 'r':
        if not f.readable():
            raise IOError("Error: Provided file not readable.")
    elif mode == 'w':
        if not f.writable():
            raise IOError("Error: Provided file not writable.")


def open_file(f: FileOrPath,
              mode: t.Literal['r', 'w'] = 'r',
              newline: t.Optional[str] = None,
              encoding: t.Optional[str] = 'utf-8') -> AbstractContextManager[TextIOBase]:
    """
    Open the given file for text I/O.

    If given a path-like, opens it with the specified settings.
    Otherwise, make an effort to reconfigure the encoding, and
    check that it is readable/writable as specified.

    Parameters:
      f: File to open/reconfigure
      mode: Mode file should be opened in
      newline: Newline mode file should be opened in
      encoding: Encoding file should be opened in
    """
    if not isinstance(f, (IOBase, t.BinaryIO, t.TextIO)):
        return open(f, mode, newline=newline, encoding=encoding)

    if isinstance(f, TextIOWrapper):
        f.reconfigure(newline=newline, encoding=encoding)
    elif isinstance(f, t.TextIO):
        f = TextIOWrapper(f.buffer, newline=newline, encoding=encoding)
    elif isinstance(f, (BufferedIOBase, t.BinaryIO)):
        f = TextIOWrapper(t.cast(t.BinaryIO, f), newline=newline, encoding=encoding)

    _validate_file(t.cast(TextIOBase, f), mode)
    return nullcontext(t.cast(TextIOBase, f))  # don't close a f we didn't open


__all__ = [
    'from_json', 'from_yaml', 'from_yaml_all',
    'write_json', 'write_yaml', 'open_file',
]