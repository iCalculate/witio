"""End-to-end test with a hand-built synthetic .wip-like tree (no real WITec sample
file is available), covering: Data-array enumeration, ID-based cross-referencing,
TDGraph array reshaping, and spectral axis calibration via a linear polynomial
TDSpectralTransformation + TDSpectralInterpretation pair."""
import numpy as np

from witio.project import WitProject
from witio.tag import WitTag, TYPE_TREE, TYPE_INT32, TYPE_DOUBLE, TYPE_UINT8, TYPE_BOOL, TYPE_STRING


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


def _tdata(id_, caption):
    return _tree("TData", [_i32("Version", 0), _i32("ID", id_), _i32("ImageIndex", 0), _str("Caption", caption)])


def build_synthetic_project():
    size_x, size_y, size_graph = 2, 3, 4

    # Raw spectral data, DataType=10 (double), non-inverted layout: fastest axis
    # in the raw stream is SizeGraph, then SizeY, then SizeX (column-major).
    flat = np.arange(size_x * size_y * size_graph, dtype="<f8")
    graph_data = WitTag(name="Data", type=TYPE_UINT8, data=flat.view("u1").copy())

    graph_data_node = _tree("GraphData", [
        _i32("Dimension", 1),
        _i32("DataType", 10),
        _i32("Ranges", flat.size),
        graph_data,
    ])

    tdgraph = _tree("TDGraph", [
        _i32("Version", 1),
        _i32("SizeX", size_x),
        _i32("SizeY", size_y),
        _i32("SizeGraph", size_graph),
        _i32("SpaceTransformationID", -1),
        _i32("XTransformationID", 100),
        _i32("XInterpretationID", 101),
        _i32("ZInterpretationID", -1),
        _bool("DataFieldInverted", False),
        graph_data_node,
        _bool("LineChanged", False),
        _bool("LineValid", [True] * size_y),
    ])
    data0 = _tree("Data 0", [_tdata(0, "Spectrum Map"), tdgraph])

    # Linear spectral polynomial transformation: wavelength(nm) = 500 + 2*i
    tdtransformation = _tree("TDTransformation", [
        _i32("Version", 0), _str("StandardUnit", "nm"), _i32("UnitKind", 2),
        _i32("InterpretationID", 101), _bool("IsCalibrated", True),
    ])
    tdspectraltransformation = _tree("TDSpectralTransformation", [
        _i32("Version", 0), _i32("SpectralTransformationType", 0),
        _f64("Polynom", [500.0, 2.0]),
        _f64("nC", 0.0), _f64("LambdaC", 0.0), _f64("Gamma", 0.0), _f64("Delta", 0.0),
        _f64("m", 1.0), _f64("d", 1.0), _f64("x", 1.0), _f64("f", 1.0),
    ])
    data100 = _tree("Data 1", [
        _tdata(100, "X Transformation"), tdtransformation, tdspectraltransformation,
    ])

    # Spectral interpretation requesting display in Raman shift (rel. 1/cm), excitation 500nm
    tdinterpretation = _tree("TDInterpretation", [_i32("Version", 0), _i32("UnitIndex", 3)])  # 3 = rel. 1/cm
    tdspectralinterpretation = _tree("TDSpectralInterpretation", [
        _i32("Version", 0), _f64("ExcitationWaveLength", 500.0),
    ])
    data101 = _tree("Data 2", [_tdata(101, "X Interpretation"), tdinterpretation, tdspectralinterpretation])

    data_root = _tree("Data", [
        _str("DataClassName 0", "TDGraph"), data0,
        _str("DataClassName 1", "TDSpectralTransformation"), data100,
        _str("DataClassName 2", "TDSpectralInterpretation"), data101,
        _i32("NumberOfData", 3),
    ])

    root = _tree("WITec Project", [_i32("Version", 7), data_root])
    return WitProject(magic=b"WIT_PR06", root=root, file_path=None)


def test_data_enumeration_and_ids():
    project = build_synthetic_project()
    assert project.version == 7
    assert len(project.data) == 3
    classes = [d.class_name for d in project.data]
    assert classes == ["TDGraph", "TDSpectralTransformation", "TDSpectralInterpretation"]
    assert project.find_by_id(100).class_name == "TDSpectralTransformation"
    assert project.find_by_id(999) is None


def test_graph_array_reshape_matches_expected_layout():
    project = build_synthetic_project()
    graph = project.find(class_name="TDGraph")[0]
    assert graph.caption == "Spectrum Map"
    arr = graph.array()
    assert arr.shape == (2, 3, 4)

    size_x, size_y, size_graph = 2, 3, 4
    flat = np.arange(size_x * size_y * size_graph, dtype="<f8")
    expected = flat.reshape((size_graph, size_y, size_x), order="F").transpose(2, 1, 0)
    np.testing.assert_array_equal(arr, expected)


def test_spectral_axis_default_unit_from_stored_interpretation():
    project = build_synthetic_project()
    graph = project.find(class_name="TDGraph")[0]
    # No explicit unit requested: should honor the stored UnitIndex (3 == 'rel. 1/cm')
    values, unit = graph.x_axis()
    assert unit == "rel. 1/cm"
    # wavelength(i) = 500 + 2*i; at i=0, wavelength==excitation==500nm -> shift == 0
    expected_nm = 500.0 + 2.0 * np.arange(4)
    expected_shift = 1e7 * (1 / 500.0 - 1 / expected_nm)
    np.testing.assert_allclose(values, expected_shift)


def test_spectral_axis_explicit_unit_override():
    project = build_synthetic_project()
    graph = project.find(class_name="TDGraph")[0]
    values, unit = graph.x_axis(unit="nm")
    assert unit == "nm"
    np.testing.assert_allclose(values, 500.0 + 2.0 * np.arange(4))
