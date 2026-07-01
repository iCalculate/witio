"""Interpretation of WITec "Data <N>" tag-tree entries (TData/TDGraph/TDImage/TDBitmap/TDText)
into numpy arrays and calibrated axes.

Ported from wit_io's WITio.obj.wid (wid_Data_get_Graph.m, wid_Data_get_Image.m,
wid_Data_get_Bitmap.m, wid_Data_get_Text.m and friends). GUI/plotting/fitting/
filtering/write-back features of the original toolbox are intentionally not
ported -- see the project README for the full scope.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from . import transform, units
from .tag import WitTag

_DATATYPE_DTYPE = {
    1: "<i8", 2: "<i4", 3: "<i2", 4: "i1",
    5: "<u4", 6: "<u2", 7: "u1",
    9: "<f4", 10: "<f8",
}

_INTERP_CLASS_TO_KIND = {
    "TDSpaceInterpretation": "space",
    "TDSpectralInterpretation": "spectral",
    "TDTimeInterpretation": "time",
    "TDZInterpretation": "z",
    "TDFrequencyInterpretation": "frequency",
    "TDInverseSpaceInterpretation": "inverse_space",
    "TDPhaseInterpretation": "phase",
}

_TRANSFORM_KIND_FALLBACK = {
    "TDSpectralTransformation": "spectral",
}

_RTF_HEX = re.compile(r"\\'([0-9a-fA-F]{2})")
_RTF_COMMAND = re.compile(r"\\[a-zA-Z]+(-?[0-9]+)? ?")


class DataFormatError(ValueError):
    pass


def _decode_datatype(raw: np.ndarray, data_type: int) -> np.ndarray:
    """Reinterpret a raw uint8 byte buffer according to the WITec DataType code
    (1=int64, 2=int32, 3=int16, 4=int8, 5=uint32, 6=uint16, 7=uint8, 8=bool,
    9=float32, 10=float64)."""
    data_type = int(data_type)
    if data_type == 8:
        return np.asarray(raw, dtype="u1").astype(bool)
    dtype = _DATATYPE_DTYPE.get(data_type)
    if dtype is None:
        raise DataFormatError(f"Unsupported DataType ({data_type})")
    raw_bytes = np.asarray(raw, dtype="u1").tobytes()
    return np.frombuffer(raw_bytes, dtype=dtype)


def _apply_line_valid(arr: np.ndarray, line_valid: np.ndarray) -> np.ndarray:
    """Mirror wit_io's wip_AutoNanInvalid default: mask out incomplete scan lines (axis 1) with NaN."""
    line_valid = np.asarray(line_valid, dtype=bool)
    if line_valid.all():
        return arr
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype("f8" if arr.dtype.itemsize > 4 else "f4")
    else:
        arr = arr.copy()
    arr[:, ~line_valid, ...] = np.nan
    return arr


def _rtf_to_text(rtf: str) -> str:
    """Best-effort RTF-to-plaintext conversion (mirrors wit_io's own reverse-engineered
    regex approach; font/formatting information is discarded either way)."""
    match = re.match(r"^[^{]*\{(.*)\}[^}]*$", rtf, flags=re.DOTALL)
    body = match.group(1) if match else rtf
    body = _RTF_HEX.sub(lambda m: chr(int(m.group(1), 16)), body)
    body = body.replace("\\~", " ").replace("\\_", "-")
    body = body.replace("\\-", "").replace("\\*", "")
    body = _RTF_COMMAND.sub(lambda m: "\n" if m.group(0).startswith("\\par") else
                             ("\t" if m.group(0).startswith("\\tab") else ""), body)
    body = re.sub(r"[\a\b\f\r\v]", "", body)
    body = body.replace("{", "").replace("}", "")
    return body.strip("\n")


@dataclass
class HistoryEntry:
    date: datetime | None
    text: str
    type: int | None


class WitData:
    """One entry of a WITec project's/data file's Data-array (e.g. a TDGraph holding
    a hyperspectral map, or a TDImage/TDBitmap/TDText)."""

    def __init__(self, tag: WitTag, class_name: str, project: "WitProject"):
        self.tag = tag
        self.class_name = class_name
        self.project = project

    def __repr__(self):
        return f"<WitData id={self.id} class={self.class_name!r} caption={self.caption!r}>"

    # -- TData metadata -----------------------------------------------------
    @property
    def tdata(self) -> WitTag | None:
        return self.tag.find("TData")

    @property
    def id(self) -> int | None:
        tdata = self.tdata
        if tdata is None:
            return None
        tag = tdata.find("ID")
        return int(tag.scalar()) if tag is not None else None

    @property
    def caption(self) -> str | None:
        tdata = self.tdata
        if tdata is None:
            return None
        tag = tdata.find("Caption")
        return tag.scalar() if tag is not None else None

    @property
    def metadata(self) -> WitTag | None:
        """Raw 'MetaData' subtree (WITec Suite v7 only); structure not further interpreted here."""
        tdata = self.tdata
        return tdata.find("MetaData") if tdata is not None else None

    @property
    def history(self) -> list[HistoryEntry]:
        tdata = self.tdata
        if tdata is None:
            return []
        history_list = tdata.find("HistoryList")
        if history_list is None:
            return []
        dates_tag = history_list.find("Dates")
        texts_tag = history_list.find("Histories")
        types_tag = history_list.find("Types")
        dates = dates_tag.as_datetimes() if dates_tag is not None else []
        texts = texts_tag.data if texts_tag is not None else []
        if isinstance(texts, str):
            texts = [texts]
        types = np.asarray(types_tag.data).ravel() if types_tag is not None else []
        out = []
        for i, date in enumerate(dates):
            out.append(HistoryEntry(
                date=date,
                text=texts[i] if i < len(texts) else "",
                type=int(types[i]) if i < len(types) else None,
            ))
        return out

    # -- Payload dispatch -----------------------------------------------------
    @property
    def payload(self) -> WitTag | None:
        """The class-specific tag (e.g. 'TDGraph') that is a direct sibling of 'TData'."""
        return self.tag.find(self.class_name)

    def array(self) -> np.ndarray:
        """Return the numpy array held by this Data entry (shape depends on class_name)."""
        if self.class_name == "TDGraph":
            return self._graph_array()
        if self.class_name == "TDImage":
            return self._image_array()
        if self.class_name == "TDBitmap":
            return self._bitmap_array()
        raise NotImplementedError(f"array() is not implemented for class {self.class_name!r}")

    def text(self) -> str:
        if self.class_name != "TDText":
            raise NotImplementedError(f"text() is not defined for class {self.class_name!r}")
        stream = self.tag.find("TDStream")
        if stream is None:
            raise NotImplementedError(
                "This TDText entry has no legacy TDStream/StreamData child (likely a WITec "
                "Suite SIX v7 file). That layout was never fully reverse-engineered upstream "
                "either -- inspect `.tag.dump()` to explore the raw structure."
            )
        raw = np.asarray(stream["StreamData"].data, dtype="u1").tobytes()
        rtf = raw.decode("windows-1252")
        if not rtf.startswith("{\\rtf1"):
            raise DataFormatError("Unsupported TDText format (missing RTF header)")
        return _rtf_to_text(rtf)

    def raw_bitmap_bytes(self) -> bytes:
        """Raw embedded BMP file bytes for a legacy (v0-v5) TDBitmap entry."""
        stream = self.tag.find("TDStream")
        if stream is None:
            raise ValueError("No embedded TDStream found (not a legacy-format TDBitmap)")
        return np.asarray(stream["StreamData"].data, dtype="u1").tobytes()

    def _graph_array(self, auto_nan_invalid: bool = True) -> np.ndarray:
        payload = self.payload
        if payload is None:
            raise DataFormatError("TDGraph payload tag not found")
        data_leaf = payload.path("GraphData", "Data")
        data_type = payload.path("GraphData", "DataType")
        if data_leaf is None or data_type is None:
            raise DataFormatError("TDGraph<GraphData<Data/DataType not found")
        flat = _decode_datatype(data_leaf.data, data_type.scalar())

        size_x = int(payload["SizeX"].scalar())
        size_y = int(payload["SizeY"].scalar())
        size_graph = int(payload["SizeGraph"].scalar())
        inverted_tag = payload.find("DataFieldInverted")
        inverted = bool(inverted_tag.scalar()) if inverted_tag is not None else False

        if inverted:
            arr = flat.reshape((size_graph, size_x, size_y), order="F").transpose(1, 2, 0)
        else:
            arr = flat.reshape((size_graph, size_y, size_x), order="F").transpose(2, 1, 0)

        if auto_nan_invalid:
            line_valid = payload.find("LineValid")
            if line_valid is not None:
                arr = _apply_line_valid(arr, line_valid.data)
        return arr  # shape (size_x, size_y, size_graph)

    def _image_array(self, auto_nan_invalid: bool = True) -> np.ndarray:
        payload = self.payload
        if payload is None:
            raise DataFormatError("TDImage payload tag not found")
        data_leaf = payload.path("ImageData", "Data")
        data_type = payload.path("ImageData", "DataType")
        if data_leaf is None or data_type is None:
            raise DataFormatError("TDImage<ImageData<Data/DataType not found")
        flat = _decode_datatype(data_leaf.data, data_type.scalar())

        size_x = int(payload["SizeX"].scalar())
        size_y = int(payload["SizeY"].scalar())
        inverted_tag = payload.find("ImageDataIsInverted")
        inverted = bool(inverted_tag.scalar()) if inverted_tag is not None else False

        if inverted:
            arr = flat.reshape((size_x, size_y), order="F")
        else:
            arr = flat.reshape((size_y, size_x), order="F").transpose(1, 0)

        if auto_nan_invalid:
            line_valid = payload.find("LineValid")
            if line_valid is not None:
                arr = _apply_line_valid(arr, line_valid.data)
        return arr  # shape (size_x, size_y)

    def _bitmap_array(self) -> np.ndarray:
        payload = self.payload
        if payload is None:
            raise DataFormatError("TDBitmap payload tag not found")
        version = self.project.version
        if version is not None and version <= 5:
            return _decode_bmp_bytes(self.raw_bitmap_bytes())

        size_x = int(payload["SizeX"].scalar())
        size_y = int(payload["SizeY"].scalar())
        data_leaf = payload.path("BitmapData", "Data")
        data_type = payload.path("BitmapData", "DataType")
        if data_leaf is None or data_type is None:
            raise DataFormatError("TDBitmap<BitmapData<Data/DataType not found")
        flat_i32 = _decode_datatype(data_leaf.data, data_type.scalar())
        raw_u8 = np.asarray(flat_i32, dtype="<i4").tobytes()
        arr = np.frombuffer(raw_u8, dtype="u1").reshape((4, size_x, size_y), order="F").transpose(1, 2, 0)
        return np.ascontiguousarray(arr[:, :, :3])  # RGB, alpha channel dropped

    # -- Calibrated axes ------------------------------------------------------
    def x_axis(self, unit: str | int | None = None):
        """Return (values, unit_name) for a TDGraph's spectral/frequency/time axis."""
        if self.class_name != "TDGraph":
            raise NotImplementedError("x_axis() is only defined for TDGraph data")
        payload = self.payload
        size_graph = int(payload["SizeGraph"].scalar())
        index = np.arange(size_graph, dtype=float)

        xt_tag = payload.find("XTransformationID")
        transform_data = self.project.find_by_id(xt_tag.scalar()) if xt_tag is not None else None
        if transform_data is None:
            return index, units.ARBITRARY_UNIT
        values, value_unit = transform_data.apply_transform(index)

        xi_tag = payload.find("XInterpretationID")
        interp_data = self.project.find_by_id(xi_tag.scalar()) if xi_tag is not None else None
        return _apply_interpretation(values, value_unit, interp_data, transform_data, unit)

    def apply_transform(self, index: np.ndarray):
        """Evaluate this Data entry (assumed to be a TD*Transformation) at 0-based pixel
        `index`. Returns (values, unit_name) where unit_name is the transform's default unit."""
        payload = self.payload
        common = self.tag.find("TDTransformation")
        standard_unit = common["StandardUnit"].scalar() if common is not None else None

        if self.class_name == "TDLinearTransformation":
            model_origin = _first_scalar(payload, "ModelOrigin_D", "ModelOrigin")
            world_origin = _first_scalar(payload, "WorldOrigin_D", "WorldOrigin")
            scale = _first_scalar(payload, "Scale_D", "Scale")
            values = transform.linear_transform(index, model_origin, world_origin, scale)
            return values, standard_unit or units.ARBITRARY_UNIT

        if self.class_name == "TDSpectralTransformation":
            stt = int(payload["SpectralTransformationType"].scalar())
            if stt == 0:
                values = transform.spectral_transform_polynomial(index, payload["Polynom"].data)
            elif stt == 1:
                values = transform.spectral_transform_grating(
                    index,
                    nC=float(payload["nC"].scalar()), LambdaC=float(payload["LambdaC"].scalar()),
                    Gamma=float(payload["Gamma"].scalar()), Delta=float(payload["Delta"].scalar()),
                    m=float(payload["m"].scalar()), d=float(payload["d"].scalar()),
                    x=float(payload["x"].scalar()), f=float(payload["f"].scalar()),
                )
            elif stt == 2:
                values = transform.spectral_transform_free_polynomial(
                    index, payload["FreePolynom"].data, int(payload["FreePolynomOrder"].scalar()),
                    float(payload["FreePolynomStartBin"].scalar()), float(payload["FreePolynomStopBin"].scalar()),
                )
            else:
                raise NotImplementedError(f"Unimplemented SpectralTransformationType ({stt})")
            return values, "nm"

        if self.class_name == "TDLUTTransformation":
            lut_size = int(payload["LUTSize"].scalar())
            lut = np.asarray(payload["LUT"].data)[:lut_size]
            values = transform.lut_transform(index, lut)
            return values, standard_unit or units.ARBITRARY_UNIT

        raise NotImplementedError(f"{self.class_name} is not a 1D transformation type")

    def apply_space_transform(self, pixel_xyz: np.ndarray):
        """Evaluate a TDSpaceTransformation at 0-based (x, y, z) pixel coordinates
        (array shaped (..., 3)). Returns (values, unit_name)."""
        if self.class_name != "TDSpaceTransformation":
            raise NotImplementedError("apply_space_transform() requires a TDSpaceTransformation entry")
        payload = self.payload
        viewport = payload["ViewPort3D"]
        model_origin = np.asarray(viewport["ModelOrigin"].data, dtype=float)
        world_origin = np.asarray(viewport["WorldOrigin"].data, dtype=float)
        scale = np.asarray(viewport["Scale"].data, dtype=float).reshape(3, 3, order="F")
        rotation = np.asarray(viewport["Rotation"].data, dtype=float).reshape(3, 3, order="F")
        values = transform.space_transform(pixel_xyz, model_origin, world_origin, scale, rotation)

        common = self.tag.find("TDTransformation")
        unit_kind = int(common["UnitKind"].scalar()) if common is not None else 1
        default_unit = "1/um" if unit_kind == 6 else "um"
        return values, default_unit

    def position_grid(self, unit: str | int | None = None):
        """For a TDGraph/TDImage/TDBitmap, return (X, Y) arrays of physical pixel-center
        coordinates shaped (SizeX, SizeY). Returns raw 0-based pixel indices if no
        space transformation is linked."""
        payload = self.payload
        size_x = int(payload["SizeX"].scalar())
        size_y = int(payload["SizeY"].scalar())
        id_field = "PositionTransformationID" if self.class_name == "TDImage" else "SpaceTransformationID"
        st_tag = payload.find(id_field)
        space_data = self.project.find_by_id(st_tag.scalar()) if st_tag is not None else None

        xi, yi = np.meshgrid(np.arange(size_x, dtype=float), np.arange(size_y, dtype=float), indexing="ij")
        if space_data is None:
            return xi, yi
        pixel_xyz = np.stack([xi, yi, np.zeros_like(xi)], axis=-1)
        values, value_unit = space_data.apply_space_transform(pixel_xyz)
        if unit is not None and unit != value_unit:
            kind = "inverse_space" if value_unit == "1/um" else "space"
            flat = values.reshape(-1, 3)
            converted = np.empty_like(flat)
            for k in range(3):
                converted[:, k], value_unit = units.convert(kind, flat[:, k], value_unit, unit)
            values = converted.reshape(values.shape)
        return values[..., 0], values[..., 1]


def _first_scalar(tag: WitTag, *names: str) -> float:
    for name in names:
        if name in tag:
            return float(tag[name].scalar())
    raise DataFormatError(f"None of {names} found under {tag.name!r}")


def _apply_interpretation(values, value_unit, interp_data, transform_data, unit_override):
    kind = None
    unit_index = None
    excitation = None
    if interp_data is not None:
        kind = _INTERP_CLASS_TO_KIND.get(interp_data.class_name)
        if kind is not None:
            td_interp = interp_data.tag.find("TDInterpretation")
            if td_interp is not None and "UnitIndex" in td_interp:
                unit_index = int(td_interp["UnitIndex"].scalar())
            if kind == "spectral":
                excitation = float(interp_data.payload["ExcitationWaveLength"].scalar())
    if kind is None and transform_data is not None:
        kind = _TRANSFORM_KIND_FALLBACK.get(transform_data.class_name)
    if kind is None or kind == "z":
        return values, value_unit  # arbitrary/uninterpreted unit: pass through unchanged
    target = unit_override if unit_override is not None else (unit_index if unit_index is not None else None)
    if target is None:
        target = units.DEFAULT_UNIT.get(kind, value_unit)
    return units.convert(kind, values, value_unit, target, excitation_wavelength_nm=excitation)


def _decode_bmp_bytes(raw: bytes) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Reading a legacy (v0-v5) TDBitmap requires Pillow (`pip install pillow`); "
            "the raw embedded BMP bytes are available via WitData.raw_bitmap_bytes()."
        ) from exc
    with Image.open(io.BytesIO(raw)) as img:
        return np.array(img.convert("RGB"))
