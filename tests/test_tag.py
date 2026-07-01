import numpy as np
import pytest

from witio.tag import WitTag, read_bytes, write_bytes, WitFormatError, TYPE_TREE, TYPE_INT32, TYPE_DOUBLE, TYPE_STRING


def _leaf(name, type_, data):
    return WitTag(name=name, type=type_, data=data)


def test_round_trip_simple_tree():
    root = WitTag(name="Root", type=TYPE_TREE, children=[
        _leaf("Answer", TYPE_INT32, np.array([42], dtype="<i4")),
        _leaf("Pi", TYPE_DOUBLE, np.array([3.14159], dtype="<f8")),
        _leaf("Label", TYPE_STRING, "hello"),
    ])
    buffer = write_bytes(b"WIT_PR06", root)
    magic, parsed = read_bytes(buffer)

    assert magic == b"WIT_PR06"
    assert parsed.name == "Root"
    assert parsed["Answer"].scalar() == 42
    assert parsed["Pi"].scalar() == pytest.approx(3.14159)
    assert parsed["Label"].scalar() == "hello"


def test_round_trip_nested_children_and_multiple_strings():
    root = WitTag(name="Root", type=TYPE_TREE, children=[
        WitTag(name="Group", type=TYPE_TREE, children=[
            _leaf("X", TYPE_INT32, np.array([1, 2, 3], dtype="<i4")),
        ]),
        _leaf("Names", TYPE_STRING, ["alpha", "beta", "gamma"]),
    ])
    buffer = write_bytes(b"WIT_DATA", root)
    magic, parsed = read_bytes(buffer)

    assert magic == b"WIT_DATA"
    np.testing.assert_array_equal(parsed.path("Group", "X").data, [1, 2, 3])
    assert parsed["Names"].data == ["alpha", "beta", "gamma"]


def test_rejects_bad_magic():
    with pytest.raises(WitFormatError):
        read_bytes(b"NOT_A_WIT" + b"\x00" * 20)


def test_find_and_findall():
    root = WitTag(name="Root", type=TYPE_TREE, children=[
        _leaf("Data 0", TYPE_INT32, np.array([1], dtype="<i4")),
        _leaf("Data 1", TYPE_INT32, np.array([2], dtype="<i4")),
    ])
    assert len(root.findall("Data 0")) == 1
    assert root.find("Data 2") is None
    with pytest.raises(KeyError):
        root["Missing"]
