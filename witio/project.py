"""Top-level reader for WITec Project (.wip) and WITec Data (.wid) files.

Ported from wit_io's WITio.obj.wip (see find_Data.m for the ID-based
cross-referencing this module's `find_by_id` mirrors).
"""
from __future__ import annotations

from pathlib import Path

from . import metadata as _metadata
from . import tag as _tag
from .data import WitData
from .metadata import TraceRecord
from .tag import WitTag


class WitProject:
    """A parsed .wip/.wid file: its Data entries plus ID-based lookup for
    resolving cross-references (e.g. a TDGraph's XTransformationID)."""

    def __init__(self, magic: bytes, root: WitTag, file_path: str | None = None):
        self.magic = magic
        self.root = root
        self.file_path = file_path
        self._data: list[WitData] | None = None
        self._by_id: dict[int, WitData] | None = None
        self._system_metadata: dict[str, str] | None = None
        self._trace_lookup: dict[str, TraceRecord] | None = None

    @property
    def version(self) -> int | None:
        tag = self.root.find("Version")
        return int(tag.scalar()) if tag is not None else None

    @property
    def data(self) -> list[WitData]:
        if self._data is None:
            self._data = list(self._iter_data())
        return self._data

    def _iter_data(self):
        data_root = self.root.find("Data")
        if data_root is None:
            return
        n_tag = data_root.find("NumberOfData")
        n = int(n_tag.scalar()) if n_tag is not None else None
        i = 0
        while n is None or i < n:
            class_name_tag = data_root.find(f"DataClassName {i}")
            data_tag = data_root.find(f"Data {i}")
            if class_name_tag is None or data_tag is None:
                break  # indices are contiguous from 0 per the WIT-tag format spec
            yield WitData(tag=data_tag, class_name=class_name_tag.scalar(), project=self)
            i += 1

    def find_by_id(self, data_id) -> WitData | None:
        if data_id is None:
            return None
        if self._by_id is None:
            self._by_id = {d.id: d for d in self.data if d.id is not None}
        return self._by_id.get(int(data_id))

    def find(self, class_name: str | None = None, caption: str | None = None) -> list[WitData]:
        """Filter Data entries by class name (e.g. 'TDGraph') and/or exact caption."""
        out = self.data
        if class_name is not None:
            out = [d for d in out if d.class_name == class_name]
        if caption is not None:
            out = [d for d in out if d.caption == caption]
        return out

    @property
    def system_metadata(self) -> dict[str, str]:
        """Project-level metadata from 'root/SystemInformation' (system_id,
        application_version, service_id, license_id -- only fields present are returned)."""
        if self._system_metadata is None:
            self._system_metadata = _metadata.parse_system_metadata(self.root)
        return self._system_metadata

    def build_trace_metadata_lookup(self) -> dict[str, TraceRecord]:
        """Parse 'root/Trace' and return {normalize_guid(DataGuid): TraceRecord},
        associating each Data entry's TData/GUID with the acquisition metadata
        (ParamSets) of the Trace record that produced it. Cached after first call."""
        if self._trace_lookup is None:
            self._trace_lookup = _metadata.build_trace_lookup(self.root)
        return self._trace_lookup

    @property
    def trace_lookup(self) -> dict[str, TraceRecord]:
        """{normalize_guid(DataGuid): TraceRecord} -- see `build_trace_metadata_lookup`."""
        return self.build_trace_metadata_lookup()

    @property
    def trace_records(self) -> list[TraceRecord]:
        """Every TraceRecord in this project, one per Trace output DataGuid."""
        return list(self.trace_lookup.values())

    def __repr__(self):
        return f"<WitProject file={self.file_path!r} version={self.version} n_data={len(self.data)}>"


def read(path: str | Path) -> WitProject:
    """Read a WITec .wip (project) or .wid (data) file."""
    magic, root = _tag.read_file(path)
    return WitProject(magic, root, file_path=str(path))


def read_bytes(buffer: bytes) -> WitProject:
    """Read a WITec .wip/.wid file already loaded into memory."""
    magic, root = _tag.read_bytes(buffer)
    return WitProject(magic, root, file_path=None)
