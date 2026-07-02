"""Tests for the Trace/ParamSets/SystemInformation metadata layer (witio.metadata),
covering: SystemInformation parsing, Trace <-> Data linking via DataGuid/TData GUID,
semantic ParamGuid field resolution, unknown-GUID passthrough, and enum label
resolution. The synthetic tree mirrors the structure confirmed against a real
WITec Suite v7/v8 project (point spectra, line scans, area scans, and video-image
TDBitmap traces)."""
from datetime import datetime, timezone

import numpy as np
import pytest

from witio.metadata import ENUM_VALUE_LABELS, TRACE_PARAM_GUID_FIELD_MAP
from witio.project import WitProject
from witio.tag import WitTag, TYPE_TREE, TYPE_INT32, TYPE_DOUBLE, TYPE_BOOL, TYPE_STRING

_GUID_BY_FIELD = {v: k for k, v in TRACE_PARAM_GUID_FIELD_MAP.items()}
_READOUT_MODE_KNOWN_GUID = next(iter(ENUM_VALUE_LABELS["camera_readout_mode_guid"]))
_UNKNOWN_PARAM_GUID = "{FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF}"


def _i32(name, value):
    return WitTag(name=name, type=TYPE_INT32, data=np.asarray(value, dtype="<i4").ravel())


def _f64(name, value):
    return WitTag(name=name, type=TYPE_DOUBLE, data=np.asarray(value, dtype="<f8").ravel())


def _bool(name, value):
    return WitTag(name=name, type=TYPE_BOOL, data=np.asarray(value, dtype=bool).ravel())


def _str(name, value):
    return WitTag(name=name, type=TYPE_STRING, data=value)


def _tree(name, children):
    return WitTag(name=name, type=TYPE_TREE, children=children)


def _named_value(value_as_name):
    """WITec's 'value embedded in a childless child's Name' encoding, used by
    SystemInformation/SystemID etc."""
    return _tree(value_as_name, [])


def _param(guid, value_tag):
    return _tree("_", [_str("ParamGuid", guid), _i32("ContentKind", 0), value_tag])


def _string_param(guid, value):
    return _param(guid, _str("StringValue", value))


def _double_param(guid, value):
    return _param(guid, _f64("DoubleValue", value))


def _bool_param(guid, value):
    return _param(guid, _bool("BoolValue", value))


def _enum_param(guid, enum_value_guid):
    return _param(guid, _str("EnumValueGuid", enum_value_guid))


def _range_param(guid, start, stop):
    return _tree("_", [_str("ParamGuid", guid), _i32("ContentKind", 4), _i32("Start", start), _i32("Stop", stop)])


def _indexed(prefix, items):
    """Build {prefix}0, {prefix}1, ... children plus a NumElements sibling."""
    children = [_i32("NumElements", len(items))]
    for i, item in enumerate(items):
        item.name = f"{prefix}{i}"
        children.append(item)
    return children


def _params_tag(param_tags):
    return _tree("Params", _indexed("Element", param_tags))


def _param_sets_tag(*param_set_groups):
    """Each group is a list of param tags for one ParamSets/Element*/Params subtree."""
    elements = [_tree("_", [_params_tag(list(group))]) for group in param_set_groups]
    return _tree("ParamSets", _indexed("Element", elements))


def _trace_element(trace_guid, source_guid, data_guid, param_set_groups,
                    utc_time="2026-01-26T16:49:43.751Z", local_time="26.01.2026 16:49:43",
                    user_name="TESTHOST\\Witec"):
    outputs_element = _tree("_", [_str("DataGuid", data_guid)])
    return _tree("_", [
        _str("TraceGuid", trace_guid),
        _str("TraceSourceGuid", source_guid),
        _i32("TraceSourceVersion", 1),
        _str("CreationUTCTime", utc_time),
        _str("CreationLocalTime", local_time),
        _str("UserName", user_name),
        _tree("Inputs", [_i32("NumElements", 0)]),
        _tree("Outputs", _indexed("Element", [outputs_element])),
        _param_sets_tag(*param_set_groups),
    ])


def _tdata(guid, caption):
    return _tree("TData", [_i32("Version", 0), _i32("ID", 0), _str("GUID", guid), _str("Caption", caption)])


def build_project_with_metadata():
    system_information = _tree("SystemInformation", [
        _tree("SystemID", [_named_value("TESTSYS-1")]),
        _tree("ApplicationVersions", [_named_value("WITec Control 9.9.9 (Test)")]),
        _tree("ServiceID", [_named_value("111,111,111")]),
        _tree("LicenseID", [_named_value("222,222,222")]),
    ])

    point_guid = "{AAAAAAAA-0000-0000-0000-000000000001}"
    area_guid = "{AAAAAAAA-0000-0000-0000-000000000002}"
    untraced_guid = "{AAAAAAAA-0000-0000-0000-000000000003}"

    data_root = _tree("Data", [
        _str("DataClassName 0", "TDGraph"), _tree("Data 0", [_tdata(point_guid, "Point Spectrum")]),
        _str("DataClassName 1", "TDGraph"), _tree("Data 1", [_tdata(area_guid, "Area Scan")]),
        _str("DataClassName 2", "TDGraph"), _tree("Data 2", [_tdata(untraced_guid, "No Trace Record")]),
        _i32("NumberOfData", 3),
    ])

    point_trace = _trace_element(
        trace_guid="{TRACE-0001}", source_guid="{SOURCE-POINT}", data_guid=point_guid,
        param_set_groups=[
            [
                _string_param(_GUID_BY_FIELD["system_id"], "TESTSYS-1"),
                _string_param(_GUID_BY_FIELD["configuration_name"], "Test Config"),
                _double_param(_GUID_BY_FIELD["duration_s"], 12.5),
                _bool_param(_GUID_BY_FIELD["is_lambda_4_coupled"], False),
                _enum_param(_GUID_BY_FIELD["camera_readout_mode_guid"], _READOUT_MODE_KNOWN_GUID),
                _string_param(_UNKNOWN_PARAM_GUID, "mystery"),
            ],
            [
                _string_param(_GUID_BY_FIELD["grating"], "G3: 600 g/mm BLZ 500.00 nm"),
                _range_param(_GUID_BY_FIELD["camera_single_track_range"], 1, 20),
            ],
        ],
    )
    area_trace = _trace_element(
        trace_guid="{TRACE-0002}", source_guid="{SOURCE-AREA}", data_guid=area_guid,
        param_set_groups=[[
            _string_param(_GUID_BY_FIELD["grating"], "G1: 300 g/mm BLZ 500.00 nm"),
            _i32("_int_placeholder", 0),  # ignored: no ParamGuid child
        ]],
    )

    trace_root = _tree("Trace", _indexed("Element", [point_trace, area_trace]))

    root = _tree("WITec Project", [_i32("Version", 8), system_information, data_root, trace_root])
    return WitProject(magic=b"WIT_PR06", root=root, file_path=None)


def test_system_metadata():
    project = build_project_with_metadata()
    assert project.system_metadata == {
        "system_id": "TESTSYS-1",
        "application_version": "WITec Control 9.9.9 (Test)",
        "service_id": "111,111,111",
        "license_id": "222,222,222",
    }


def test_system_metadata_missing_section_returns_empty_dict():
    root = _tree("WITec Project", [_i32("Version", 8)])
    project = WitProject(magic=b"WIT_PR06", root=root, file_path=None)
    assert project.system_metadata == {}


def test_trace_records_and_lookup_cover_every_traced_output():
    project = build_project_with_metadata()
    assert len(project.trace_records) == 2
    assert len(project.trace_lookup) == 2


def test_measurement_metadata_resolves_known_fields_across_param_sets():
    project = build_project_with_metadata()
    point = project.find(caption="Point Spectrum")[0]
    meta = point.measurement_metadata

    assert meta["system_id"] == "TESTSYS-1"
    assert meta["configuration_name"] == "Test Config"
    assert meta["duration_s"] == pytest.approx(12.5)
    assert meta["is_lambda_4_coupled"] is False
    assert meta["grating"] == "G3: 600 g/mm BLZ 500.00 nm"
    assert meta["camera_single_track_range"] == (1, 20)


def test_measurement_metadata_includes_trace_bookkeeping_fields():
    project = build_project_with_metadata()
    point = project.find(caption="Point Spectrum")[0]
    meta = point.measurement_metadata

    assert meta["trace_guid"] == "{TRACE-0001}"
    assert meta["trace_source_guid"] == "{SOURCE-POINT}"
    assert meta["trace_source_version"] == 1
    assert meta["trace_creation_time_utc"] == datetime(2026, 1, 26, 16, 49, 43, 751000, tzinfo=timezone.utc)
    assert meta["trace_creation_time_local"] == datetime(2026, 1, 26, 16, 49, 43)
    assert meta["trace_user_name"] == "TESTHOST\\Witec"
    assert meta["trace_param_set_count"] == 2


def test_enum_value_guid_resolved_to_readable_label():
    project = build_project_with_metadata()
    point = project.find(caption="Point Spectrum")[0]
    meta = point.measurement_metadata

    assert meta["camera_readout_mode_guid"] == _READOUT_MODE_KNOWN_GUID
    assert meta["camera_readout_mode"] == "Single Track"


def test_unknown_param_guid_is_preserved_raw_not_dropped_or_erroring():
    project = build_project_with_metadata()
    point = project.find(caption="Point Spectrum")[0]
    record = point.trace_record

    assert record.raw_params[_UNKNOWN_PARAM_GUID] == "mystery"
    assert _UNKNOWN_PARAM_GUID not in point.measurement_metadata


def test_data_entry_without_trace_record_returns_empty_metadata():
    project = build_project_with_metadata()
    untraced = project.find(caption="No Trace Record")[0]

    assert untraced.trace_record is None
    assert untraced.measurement_metadata == {}


def test_grating_field_is_resolved_like_any_other_string_param():
    project = build_project_with_metadata()
    point = project.find(caption="Point Spectrum")[0]
    area = project.find(caption="Area Scan")[0]

    assert point.measurement_metadata["grating"] == "G3: 600 g/mm BLZ 500.00 nm"
    assert area.measurement_metadata["grating"] == "G1: 300 g/mm BLZ 500.00 nm"
