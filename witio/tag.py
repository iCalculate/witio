"""Low-level reader for the WIT-tag binary format used by WITec .wip/.wid files.

Format (always little-endian; see the original wit_io project's
"README on WIT-tag format.txt" for the authoritative reference):

    MAGIC (8 bytes)                     'WIT_PRCT'/'WIT_DATA' (v0-v5) or 'WIT_PR06'/'WIT_DA06' (v6-v7)
    then recursively, for every tag:
        NameLength (uint32)
        Name       (NameLength bytes, windows-1252)
        Type       (uint32)             0=tree, 2=double, 3=float, 4=int64, 5=int32,
                                         6=datetime(uint16 x7), 7=uint8, 8=bool, 9=string
        Start      (uint64)             file offset of payload start
        End        (uint64)             file offset of payload end (exclusive)
        Data       (End-Start bytes)    children (if Type==0) or typed payload
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import BinaryIO

import numpy as np

TYPE_TREE = 0
TYPE_DOUBLE = 2
TYPE_FLOAT = 3
TYPE_INT64 = 4
TYPE_INT32 = 5
TYPE_DATETIME = 6
TYPE_UINT8 = 7
TYPE_BOOL = 8
TYPE_STRING = 9

_DTYPE_BY_TYPE = {
    TYPE_DOUBLE: "<f8",
    TYPE_FLOAT: "<f4",
    TYPE_INT64: "<i8",
    TYPE_INT32: "<i4",
    TYPE_DATETIME: "<u2",
    TYPE_UINT8: "u1",
}

MAGIC_STRINGS = (b"WIT_PRCT", b"WIT_DATA", b"WIT_PR06", b"WIT_DA06")

_ENCODING = "windows-1252"


class WitFormatError(ValueError):
    """Raised when a file does not conform to the expected WIT-tag layout."""


@dataclass
class WitTag:
    """A single node of the WIT-tag tree.

    Leaf tags (``type != TYPE_TREE``) carry their payload in ``data`` (a numpy
    array, a decoded string, or a list of strings for Type==9 with several
    entries). Tree tags (``type == TYPE_TREE``) carry ``children`` instead.
    """

    name: str
    type: int
    data: object = None
    children: list["WitTag"] = field(default_factory=list)
    parent: "WitTag" = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        for child in self.children:
            child.parent = self

    def find(self, name: str) -> "WitTag | None":
        """Return the first direct child named ``name``, or None."""
        for child in self.children:
            if child.name == name:
                return child
        return None

    def findall(self, name: str) -> list["WitTag"]:
        """Return every direct child named ``name``."""
        return [child for child in self.children if child.name == name]

    def path(self, *names: str) -> "WitTag | None":
        """Walk a chain of direct-child lookups, e.g. tag.path('GraphData', 'Data')."""
        node = self
        for name in names:
            if node is None:
                return None
            node = node.find(name)
        return node

    def __getitem__(self, name: str) -> "WitTag":
        node = self.find(name)
        if node is None:
            raise KeyError(name)
        return node

    def __contains__(self, name: str) -> bool:
        return self.find(name) is not None

    def scalar(self):
        """Return a single Python/numpy scalar for a 1-element leaf tag."""
        if isinstance(self.data, np.ndarray):
            if self.data.size != 1:
                raise ValueError(f"Tag {self.name!r} does not hold a scalar (size={self.data.size})")
            return self.data.item()
        return self.data

    def as_datetimes(self) -> list[datetime]:
        """Interpret a Type==6 payload as a list of datetimes (year/month/day/h/m/s/ms)."""
        if self.type != TYPE_DATETIME:
            raise ValueError(f"Tag {self.name!r} is not a datetime tag (Type={self.type})")
        values = np.asarray(self.data).reshape(-1, 7)
        out = []
        for year, month, day, hour, minute, second, millisecond in values:
            if year == 0:
                out.append(None)
                continue
            out.append(datetime(int(year), int(month), int(day), int(hour), int(minute), int(second),
                                 int(millisecond) * 1000))
        return out

    def dump(self, max_depth: int | None = None, _depth: int = 0) -> str:
        """Pretty-print the (sub)tree, mainly for interactive exploration."""
        lines = []
        indent = "  " * _depth
        if self.type == TYPE_TREE:
            lines.append(f"{indent}{self.name} ({len(self.children)} children)")
            if max_depth is None or _depth < max_depth:
                for child in self.children:
                    lines.append(child.dump(max_depth, _depth + 1))
        else:
            preview = self.data
            if isinstance(preview, np.ndarray) and preview.size > 6:
                preview = f"{preview[:6]}... ({preview.size} values, {preview.dtype})"
            lines.append(f"{indent}{self.name} = {preview!r}")
        return "\n".join(lines)


def _read_string_array(raw: bytes) -> str | list[str]:
    strings = []
    offset = 0
    n = len(raw)
    while offset < n:
        (length,) = struct.unpack_from("<I", raw, offset)
        offset += 4
        strings.append(raw[offset:offset + length].decode(_ENCODING))
        offset += length
    if len(strings) == 1:
        return strings[0]
    return strings


def _read_payload(f: BinaryIO, tag_type: int, start: int, end: int):
    f.seek(start)
    length = end - start
    raw = f.read(length)
    if len(raw) != length:
        raise WitFormatError(f"Unexpected end of file while reading {length} bytes of tag data")
    if tag_type == TYPE_STRING:
        return _read_string_array(raw)
    if tag_type == TYPE_BOOL:
        return np.frombuffer(raw, dtype="u1").astype(bool)
    dtype = _DTYPE_BY_TYPE.get(tag_type)
    if dtype is None:
        raise WitFormatError(f"Unsupported tag Type ({tag_type})")
    return np.frombuffer(raw, dtype=dtype).copy()


def _read_tag(f: BinaryIO) -> WitTag | None:
    header = f.read(4)
    if len(header) < 4:
        return None
    (name_length,) = struct.unpack("<I", header)
    name = f.read(name_length).decode(_ENCODING)
    tag_type, start, end = struct.unpack("<IQQ", f.read(20))

    if tag_type == TYPE_TREE:
        children = []
        while f.tell() < end:
            child = _read_tag(f)
            if child is None:
                break
            children.append(child)
        return WitTag(name=name, type=tag_type, children=children)

    data = _read_payload(f, tag_type, start, end)
    return WitTag(name=name, type=tag_type, data=data)


def read_bytes(buffer: bytes) -> tuple[bytes, WitTag]:
    """Parse an in-memory WIT-tag buffer. Returns (magic, root_tag)."""
    import io
    f = io.BytesIO(buffer)
    magic = f.read(8)
    if magic not in MAGIC_STRINGS:
        raise WitFormatError(f"Unrecognized magic string {magic!r}; not a WIT-tag file")
    root = _read_tag(f)
    if root is None:
        raise WitFormatError("File ended before any tag could be read")
    return magic, root


def read_file(path: str | Path) -> tuple[bytes, WitTag]:
    """Parse a .wip/.wid file. Returns (magic, root_tag)."""
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic not in MAGIC_STRINGS:
            raise WitFormatError(f"Unrecognized magic string {magic!r}; not a WIT-tag file: {path}")
        root = _read_tag(f)
        if root is None:
            raise WitFormatError(f"File ended before any tag could be read: {path}")
        return magic, root


def _write_string_array(value) -> bytes:
    if isinstance(value, str):
        value = [value]
    out = bytearray()
    for s in value:
        encoded = s.encode(_ENCODING)
        out += struct.pack("<I", len(encoded))
        out += encoded
    return bytes(out)


def _write_payload(tag: WitTag) -> bytes:
    if tag.type == TYPE_STRING:
        return _write_string_array(tag.data)
    if tag.type == TYPE_BOOL:
        return np.asarray(tag.data, dtype="u1").tobytes()
    dtype = _DTYPE_BY_TYPE.get(tag.type)
    if dtype is None:
        raise WitFormatError(f"Unsupported tag Type ({tag.type})")
    return np.asarray(tag.data, dtype=dtype).tobytes()


def _write_tag(tag: WitTag, out: bytearray) -> None:
    name_bytes = tag.name.encode(_ENCODING)
    out += struct.pack("<I", len(name_bytes))
    out += name_bytes
    out += struct.pack("<I", tag.type)
    start_pos = len(out)
    out += b"\x00" * 16  # placeholder for Start/End, patched below

    payload_start = len(out)
    if tag.type == TYPE_TREE:
        for child in tag.children:
            _write_tag(child, out)
    else:
        out += _write_payload(tag)
    payload_end = len(out)

    struct.pack_into("<QQ", out, start_pos, payload_start, payload_end)


def write_bytes(magic: bytes, root: WitTag) -> bytes:
    """Serialize a tag tree back to the WIT-tag binary format.

    This exists mainly to round-trip-test the reader; it does not attempt to
    reproduce every bookkeeping tag (NumberOfData, IDs, ...) that a real
    WITec-written project/data file would need to stay valid for WITec software.
    """
    if magic not in MAGIC_STRINGS:
        raise WitFormatError(f"Unrecognized magic string {magic!r}")
    out = bytearray(magic)
    _write_tag(root, out)
    return bytes(out)
