"""Structured interpretation of the WITec Suite v7/v8 project-level metadata that
lives outside the Data-array tree: `SystemInformation` and `Trace`.

`Trace/Element*` records the acquisition history of a project: one element per
software operation (measurement, video snapshot, ...), each carrying the
operation's `ParamSets` (instrument/laser/camera/spectrograph/scan settings)
and an `Outputs` list of `DataGuid`s identifying which `Data <N>` entries it
produced. `Data <N>/TData/GUID` is the other end of that link -- see
`build_trace_lookup`.

`ParamSets` entries are keyed by opaque `ParamGuid`/`EnumValueGuid` values
that carry no semantic meaning on their own; `TRACE_PARAM_GUID_FIELD_MAP` and
`ENUM_VALUE_LABELS` translate the ones that have been identified by inspecting
real WITec Suite v7/v8 projects into the field names documented in the
project README. Unrecognized GUIDs are never dropped -- they stay reachable
through `TraceRecord.raw_params`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .tag import WitTag

# -- ParamGuid -> semantic field name -----------------------------------------
# Confirmed against a real WITec Suite v7/v8 project (point spectra, line scans,
# and area scans). GUIDs not listed here are preserved verbatim in
# `TraceRecord.raw_params` rather than raising or being discarded.
TRACE_PARAM_GUID_FIELD_MAP: dict[str, str] = {
    "{50786913-FDEE-4C98-A3D2-69CEAD859087}": "system_id",
    "{1AFAD5D3-7519-4017-9402-169E14483ACB}": "configuration_name",
    "{3BD5B7C3-02B3-4226-8541-B4FA098F9FAA}": "duration_s",
    "{C59F0865-BCA6-42CC-970B-E78863813092}": "laser_wavelength_nm",
    "{E1353512-9D5D-4B42-85B0-563E1D2E0922}": "laser_power_in_fiber_mw",
    "{DE90D5DF-6355-4003-8C30-69F4ECA394D6}": "laser_power_mw",
    "{C7BF2E9E-4588-4FE7-A661-8BC5F9ABBEA9}": "integration_time_s",
    "{09449FFD-F53F-48FE-A687-D17A866CDB64}": "accumulations",
    "{16EDE678-FAF2-4B8D-B8E4-D7F8805438A0}": "objective_name",
    "{B4B5BA2D-BCF0-4DF3-AFE8-09808536C16B}": "objective_magnification",
    "{B9B7E353-ACAD-434A-B02E-65260A793A76}": "is_lambda_4_coupled",
    "{0894921D-94F8-4B6A-92E0-B1F8F9C875E7}": "sample_position_x_um",
    "{459B4B02-6301-4309-AB36-122A31ACF47B}": "sample_position_y_um",
    "{FD47BC57-3E4E-48B3-BF05-CD87CA53540E}": "sample_position_z_um",
    "{8DF0A965-92FB-478B-BA56-9F4BCFAB9EFE}": "spectrograph_name",
    "{D4FB7097-4A2B-48FB-8619-27FA848640C1}": "spectrograph_serial_number",
    "{70C15292-3849-485A-A19C-1C63BA2E737A}": "grating",
    "{4A1C85B9-F3E1-4D07-936D-D80C3E68F86F}": "center_wavelength_nm",
    "{75B3D31D-985E-4622-84E7-D07229299AA3}": "camera_name",
    "{41B6F20B-F543-4221-BCE0-FF70518D61A3}": "camera_serial_number",
    "{144EA40E-6F56-4321-9575-E08F11CD9DF8}": "camera_exposure_time_s",
    "{D3C1554D-27F0-414C-9E26-0A58EA8C3DB7}": "camera_cycle_time_s",
    "{5751D616-EEB4-4778-9BFC-21F7E98A916C}": "camera_readout_mode_guid",
    "{4BF5EC32-10B7-484F-8E8E-D1B6AF14182D}": "camera_single_track_range",
    "{F555CD8E-1549-4A19-9215-DFE1FAE4543A}": "camera_track_height_px",
    "{57946D40-7278-42C9-9F0F-96E3F2F04BA0}": "camera_vertical_shift_speed_us",
    "{979250FF-18FC-4E38-99AE-ED8B40F16A93}": "camera_horizontal_shift_speed_mhz",
    "{577AED5A-C75E-4DF0-A472-A8A9B4C00152}": "camera_pre_amplifier_gain",
    "{499DAAD1-793E-45DC-AE0E-5010C2839F66}": "camera_sensor_temperature_c",

    "{D5B07CC0-BD16-4827-9ABB-EC6D9CE841FF}": "line_start_x_um",
    "{3E30A781-526F-4B8F-ACCA-BEA2A3C47386}": "line_start_y_um",
    "{F98A0D8C-3666-4AD4-8B54-E679D570C83F}": "line_end_x_um",
    "{0A2B614F-E455-45D8-A27F-020F27D5435B}": "line_end_y_um",

    "{E0920DB3-7DC4-49B6-8D24-52913DA42DCF}": "scan_size_x_px",
    "{FCD082AF-F137-4D56-A3A5-E339ABDB2961}": "scan_size_y_px",
    "{9AB8AC43-E3DE-4494-9A7E-65AE9B5A185D}": "scan_span_x_um",
    "{55DDF72F-12EF-4AE2-9093-F92E4CDCA753}": "scan_span_y_um",
    "{E33819CD-8D98-453B-B7A0-4E2854AE3727}": "scan_center_x_um",
    "{BFC9F215-E47E-443A-92C8-FCFE847FECED}": "scan_center_y_um",
    "{57B1AA04-C0FC-45BB-8E23-738653046C53}": "scan_center_z_um",
    "{50613848-5B02-471C-BF75-96C4F03FFB3E}": "scan_rotation_deg",
    "{757E7375-9D48-4446-B1CA-567894E63996}": "scan_cycle_time_s",
    "{75F48F26-DEDB-482E-88B5-4A94438E95D6}": "scan_tilt_x_deg",
    "{3537D883-F44E-41CE-9E65-5A2D7A17AC9D}": "scan_tilt_y_deg",

    "{A1B7BA27-73D1-4BA9-9C4F-E8B48EE0A9BD}": "scan_pattern_guid",
    "{97AC98CA-00A2-4DB1-BE2A-E7542CB2438E}": "line_sampling_mode_guid",
    "{BFDB7906-8B4F-4165-ACC0-E3F02F5CCBB5}": "line_scan_pattern_guid",
}

# Per-field EnumValueGuid -> readable label tables. Keyed by the "*_guid" field
# name above; the resolved label is exposed under the same name with that
# suffix stripped (e.g. "camera_readout_mode_guid" -> "camera_readout_mode").
ENUM_VALUE_LABELS: dict[str, dict[str, str]] = {
    "camera_readout_mode_guid": {
        "{A9769D28-E279-400B-9129-9473586C1A9F}": "Single Track",
    },
}

def normalize_guid(value: str | None) -> str | None:
    """Case/whitespace-normalize a `{...}`-style GUID string for dict lookups."""
    if value is None:
        return None
    return value.strip().upper()


@dataclass
class TraceRecord:
    """One `Trace/Element*/Outputs/Element*` entry: the acquisition metadata
    behind a single `Data <N>` entry (matched via `data_guid`)."""

    data_guid: str
    trace_guid: str | None
    trace_source_guid: str | None
    trace_source_version: int | None
    creation_time_utc: datetime | None
    creation_time_local: datetime | None
    user_name: str | None
    param_set_count: int
    fields: dict[str, object] = field(default_factory=dict)
    raw_params: dict[str, object] = field(default_factory=dict)

    def to_metadata_dict(self) -> dict[str, object]:
        """The flat dict returned by `WitData.measurement_metadata`: every
        recognized semantic field, plus the raw trace bookkeeping fields."""
        out = dict(self.fields)
        out["trace_guid"] = self.trace_guid
        out["trace_source_guid"] = self.trace_source_guid
        out["trace_source_version"] = self.trace_source_version
        out["trace_creation_time_utc"] = self.creation_time_utc
        out["trace_creation_time_local"] = self.creation_time_local
        out["trace_user_name"] = self.user_name
        out["trace_param_set_count"] = self.param_set_count
        return out


def _iter_indexed_children(tag: WitTag, prefix: str):
    """Yield `tag`'s `{prefix}0`, `{prefix}1`, ... children until one is missing
    (mirrors the contiguous-index convention used throughout the WIT-tag format)."""
    i = 0
    while True:
        child = tag.find(f"{prefix}{i}")
        if child is None:
            return
        yield child
        i += 1


def _extract_param_value(elem: WitTag):
    """Read a `ParamSets/*/Params/Element*` payload from whichever value tag is
    present (StringValue/DoubleValue/IntValue/BoolValue/EnumValueGuid/Start+Stop)."""
    if (t := elem.find("StringValue")) is not None:
        return t.scalar()
    if (t := elem.find("DoubleValue")) is not None:
        return float(t.scalar())
    if (t := elem.find("IntValue")) is not None:
        return int(t.scalar())
    if (t := elem.find("BoolValue")) is not None:
        return bool(t.scalar())
    if (t := elem.find("EnumValueGuid")) is not None:
        return t.scalar()
    start, stop = elem.find("Start"), elem.find("Stop")
    if start is not None and stop is not None:
        return (int(start.scalar()), int(stop.scalar()))
    return None


def _parse_params(params_tag: WitTag) -> dict[str, object]:
    out = {}
    for elem in _iter_indexed_children(params_tag, "Element"):
        guid_tag = elem.find("ParamGuid")
        if guid_tag is None:
            continue
        out[guid_tag.scalar()] = _extract_param_value(elem)
    return out


def _parse_param_sets(param_sets_tag: WitTag) -> tuple[dict[str, object], int]:
    """Flatten every `ParamSets/Element*/Params` subtree (the general settings
    plus every device-linked settings group) into one ParamGuid -> value dict."""
    merged: dict[str, object] = {}
    count = 0
    for elem in _iter_indexed_children(param_sets_tag, "Element"):
        count += 1
        params_tag = elem.find("Params")
        if params_tag is not None:
            merged.update(_parse_params(params_tag))
    return merged, count


def _build_fields(raw_params: dict[str, object]) -> dict[str, object]:
    fields: dict[str, object] = {}
    for guid, value in raw_params.items():
        name = TRACE_PARAM_GUID_FIELD_MAP.get(normalize_guid(guid))
        if name is None:
            continue
        fields[name] = value
        if name.endswith("_guid") and isinstance(value, str):
            label = ENUM_VALUE_LABELS.get(name, {}).get(normalize_guid(value))
            if label is not None:
                fields[name[:-len("_guid")]] = label
    return fields


def _parse_utc_time(text: str | None) -> datetime | None:
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _parse_local_time(text: str | None) -> datetime | None:
    if not text:
        return None
    try:
        return datetime.strptime(text, "%d.%m.%Y %H:%M:%S")
    except ValueError:
        return None


def _single_child_name(tag: WitTag | None) -> str | None:
    """WITec stores some scalar settings (SystemID, ApplicationVersions, ...)
    as a single childless child tag whose *name* is the value itself."""
    if tag is None or not tag.children:
        return None
    return tag.children[0].name


def parse_system_metadata(root: WitTag) -> dict[str, str]:
    """Parse `root/SystemInformation` into system_id/application_version/
    service_id/license_id (only the fields actually present are returned)."""
    system_information = root.find("SystemInformation")
    if system_information is None:
        return {}
    out = {}
    for field_name, tag_name in (
        ("system_id", "SystemID"),
        ("application_version", "ApplicationVersions"),
        ("service_id", "ServiceID"),
        ("license_id", "LicenseID"),
    ):
        value = _single_child_name(system_information.find(tag_name))
        if value is not None:
            out[field_name] = value
    return out


def build_trace_lookup(root: WitTag) -> dict[str, TraceRecord]:
    """Parse `root/Trace` and return `{normalize_guid(DataGuid): TraceRecord}`,
    resolving every `Outputs/Element*/DataGuid` produced by every `Trace/Element*`
    acquisition record. Returns an empty dict if the project has no Trace subtree
    (e.g. legacy WITec software versions)."""
    trace_root = root.find("Trace")
    if trace_root is None:
        return {}

    lookup: dict[str, TraceRecord] = {}
    for elem in _iter_indexed_children(trace_root, "Element"):
        trace_guid = _scalar_or_none(elem.find("TraceGuid"))
        trace_source_guid = _scalar_or_none(elem.find("TraceSourceGuid"))
        version_tag = elem.find("TraceSourceVersion")
        trace_source_version = int(version_tag.scalar()) if version_tag is not None else None
        creation_utc = _parse_utc_time(_scalar_or_none(elem.find("CreationUTCTime")))
        creation_local = _parse_local_time(_scalar_or_none(elem.find("CreationLocalTime")))
        user_name = _scalar_or_none(elem.find("UserName"))

        raw_params: dict[str, object] = {}
        param_set_count = 0
        param_sets_tag = elem.find("ParamSets")
        if param_sets_tag is not None:
            raw_params, param_set_count = _parse_param_sets(param_sets_tag)
        fields = _build_fields(raw_params)

        outputs = elem.find("Outputs")
        if outputs is None:
            continue
        for out_elem in _iter_indexed_children(outputs, "Element"):
            data_guid_tag = out_elem.find("DataGuid")
            if data_guid_tag is None:
                continue
            data_guid = data_guid_tag.scalar()
            lookup[normalize_guid(data_guid)] = TraceRecord(
                data_guid=data_guid,
                trace_guid=trace_guid,
                trace_source_guid=trace_source_guid,
                trace_source_version=trace_source_version,
                creation_time_utc=creation_utc,
                creation_time_local=creation_local,
                user_name=user_name,
                param_set_count=param_set_count,
                fields=fields,
                raw_params=raw_params,
            )
    return lookup


def _scalar_or_none(tag: WitTag | None):
    return tag.scalar() if tag is not None else None
