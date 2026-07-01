import numpy as np
import pytest

from witio import transform, units


def test_space_unit_default_is_identity():
    value, unit = units.convert("space", [1.0, 2.0], unit_from="um")
    np.testing.assert_allclose(value, [1.0, 2.0])
    assert unit == "um"


def test_space_unit_um_to_nm():
    value, unit = units.convert("space", [1.0], unit_from="um", unit_to="nm")
    np.testing.assert_allclose(value, [1000.0])
    assert unit == "nm"


def test_spectral_nm_to_wavenumber():
    # 500 nm -> 1e7/500 = 20000 1/cm
    value, unit = units.convert("spectral", [500.0], unit_from="nm", unit_to="1/cm")
    np.testing.assert_allclose(value, [20000.0])
    assert unit == "1/cm"


def test_spectral_rel_wavenumber_needs_excitation():
    with pytest.raises(units.UnitError):
        units.convert("spectral", [500.0], unit_from="nm", unit_to="rel. 1/cm")


def test_spectral_raman_shift_of_excitation_line_is_zero():
    # At the excitation wavelength itself, Raman shift must be 0 cm^-1.
    value, unit = units.convert(
        "spectral", [532.0], unit_from="nm", unit_to="rel. 1/cm", excitation_wavelength_nm=532.0
    )
    np.testing.assert_allclose(value, [0.0], atol=1e-9)
    assert unit == "rel. 1/cm"


def test_linear_transform_matches_hand_calc():
    # value = scale * (i - model_origin) + world_origin
    index = np.array([0, 1, 2], dtype=float)
    values = transform.linear_transform(index, model_origin=0.0, world_origin=10.0, scale=2.0)
    np.testing.assert_allclose(values, [10.0, 12.0, 14.0])


def test_lut_transform_clamps_out_of_range():
    lut = np.array([0.0, 10.0, 20.0])
    values = transform.lut_transform(np.array([-1.0, 0.0, 0.5, 2.0, 5.0]), lut)
    np.testing.assert_allclose(values, [0.0, 0.0, 5.0, 20.0, 20.0])


def test_spectral_polynomial_matches_hand_calc():
    # 500 + 0.5*i (linear polynomial)
    index = np.array([0.0, 1.0, 2.0])
    values = transform.spectral_transform_polynomial(index, [500.0, 0.5])
    np.testing.assert_allclose(values, [500.0, 500.5, 501.0])


def test_space_transform_identity_when_no_rotation_and_unit_scale():
    pixel_xyz = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0]])
    identity = np.eye(3)
    values = transform.space_transform(
        pixel_xyz, model_origin=[0, 0, 0], world_origin=[5, 5, 0], scale=identity, rotation=identity
    )
    np.testing.assert_allclose(values, [[5.0, 5.0, 0.0], [6.0, 7.0, 0.0]])
