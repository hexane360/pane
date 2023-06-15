
from pathlib import Path
from io import TextIOBase, IOBase, TextIOWrapper, BufferedIOBase
from contextlib import AbstractContextManager, nullcontext
import typing as t


FileOrPath = t.Union[str, Path, TextIOBase, t.TextIO]


def _validate_file(f: t.Union[t.IO, IOBase], mode: t.Union[t.Literal['r'], t.Literal['w']]):
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
    """
    if not isinstance(f, (IOBase, t.BinaryIO, t.TextIO)):
        return open(f, mode, newline=newline, encoding=encoding)

    if isinstance(f, TextIOWrapper):
        f.reconfigure(newline=newline, encoding=encoding)
    elif isinstance(f, t.TextIO):
        f = TextIOWrapper(f.buffer, newline=newline, encoding=encoding)
    elif isinstance(f, (BufferedIOBase, t.BinaryIO)):
        f = TextIOWrapper(t.cast(t.BinaryIO, f), newline=newline, encoding=encoding)

    _validate_file(f, mode)
    return nullcontext(f)  # don't close a f we didn't open