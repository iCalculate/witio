"""Unit interpretation tables for WITec TDxxxInterpretation tags.

Ported from wit_io's WITio.obj.wip.interpret.m. Each *kind* of interpretation
has an internal "default" unit that WITio.obj.transform functions (see
transform.py) produce values in, plus a table of named units where each row is
(name, from_default_fn, to_default_fn): from_default_fn maps a value in the
default unit to this unit, to_default_fn maps a value in this unit back to the
default unit. UnitIndex numbering matches the tables documented in "README on
WIT-tag format.txt" (0-based), so a tag's raw UnitIndex integer can be used
directly as an index into UNIT_TABLES[kind].

Kinds: 'space' (default unit: um), 'spectral' (default: nm), 'time' (default: s),
'frequency' (default: Hz), 'inverse_space' (default: 1/um), 'phase' (default: rad).
'z' is handled separately since it carries an arbitrary, non-convertible unit name.
"""
from __future__ import annotations

import numpy as np

ARBITRARY_UNIT = "a.u."


class UnitError(ValueError):
    pass


def _table(rows):
    # rows: list of (name, to_default, from_default)
    return rows


UNIT_TABLES = {
    "space": _table([
        ("m", lambda x: 1e-6 * x, lambda y: 1e6 * y),
        ("mm", lambda x: 1e-3 * x, lambda y: 1e3 * y),
        ("um", lambda x: x, lambda y: y),  # default
        ("nm", lambda x: 1e3 * x, lambda y: 1e-3 * y),
        ("A", lambda x: 1e4 * x, lambda y: 1e-4 * y),
        ("pm", lambda x: 1e6 * x, lambda y: 1e-6 * y),
    ]),
    "time": _table([
        ("h", lambda x: x / 3600, lambda y: 3600 * y),
        ("min", lambda x: x / 60, lambda y: 60 * y),
        ("s", lambda x: x, lambda y: y),  # default
        ("ms", lambda x: 1e3 * x, lambda y: 1e-3 * y),
        ("us", lambda x: 1e6 * x, lambda y: 1e-6 * y),
        ("ns", lambda x: 1e9 * x, lambda y: 1e-9 * y),
        ("ps", lambda x: 1e12 * x, lambda y: 1e-12 * y),
        ("fs", lambda x: 1e15 * x, lambda y: 1e-15 * y),
    ]),
    "frequency": _table([
        ("uHz", lambda x: 1e6 * x, lambda y: 1e-6 * y),
        ("mHz", lambda x: 1e3 * x, lambda y: 1e-3 * y),
        ("Hz", lambda x: x, lambda y: y),  # default
        ("kHz", lambda x: 1e-3 * x, lambda y: 1e3 * y),
        ("MHz", lambda x: 1e-6 * x, lambda y: 1e6 * y),
        ("GHz", lambda x: 1e-9 * x, lambda y: 1e9 * y),
        ("THz", lambda x: 1e-12 * x, lambda y: 1e12 * y),
    ]),
    "inverse_space": _table([
        ("1/m", lambda x: 1e6 * x, lambda y: 1e-6 * y),
        ("1/mm", lambda x: 1e3 * x, lambda y: 1e-3 * y),
        ("1/um", lambda x: x, lambda y: y),  # default
        ("1/nm", lambda x: 1e-3 * x, lambda y: 1e3 * y),
        ("1/A", lambda x: 1e-4 * x, lambda y: 1e4 * y),
        ("1/pm", lambda x: 1e-6 * x, lambda y: 1e6 * y),
    ]),
    "phase": _table([
        ("rad", lambda x: x, lambda y: y),  # default
        ("mrad", lambda x: 1e3 * x, lambda y: 1e-3 * y),
        ("deg", lambda x: 180 / np.pi * x, lambda y: np.pi / 180 * y),
        ("grad", lambda x: 200 / np.pi * x, lambda y: np.pi / 200 * y),
        ("mgrad", lambda x: 2e5 / np.pi * x, lambda y: np.pi / 2e5 * y),
    ]),
}

DEFAULT_UNIT = {
    "space": "um",
    "time": "s",
    "frequency": "Hz",
    "inverse_space": "1/um",
    "phase": "rad",
    "spectral": "nm",
}


def _spectral_table(excitation_wavelength_nm):
    x0 = excitation_wavelength_nm
    rows = [
        ("nm", lambda x: x, lambda y: y),  # default
        ("um", lambda x: 1e-3 * x, lambda y: 1e3 * y),
        ("1/cm", lambda x: 1e7 / x, lambda y: 1e7 / y),
        ("rel. 1/cm", lambda x: 1e7 * (1 / x0 - 1 / x), lambda y: 1 / (1 / x0 - 1e-7 * y)),
        ("eV", lambda x: 1.23984193e3 / x, lambda y: 1.23984193e3 / y),
        ("meV", lambda x: 1.23984193e6 / x, lambda y: 1.23984193e6 / y),
        ("rel. eV", lambda x: -1.23984193e3 * (1 / x0 - 1 / x), lambda y: 1 / (1 / x0 + y / 1.23984193e3)),
        ("rel. meV", lambda x: -1.23984193e6 * (1 / x0 - 1 / x), lambda y: 1 / (1 / x0 + y / 1.23984193e6)),
    ]
    return rows


def _find_unit_index(rows, unit) -> int:
    if isinstance(unit, (int, np.integer)):
        if 0 <= unit < len(rows):
            return int(unit)
        raise UnitError(f"Unit index {unit} out of range [0, {len(rows) - 1}]")
    names = [row[0] for row in rows]
    if unit in names:
        return names.index(unit)
    matches = [i for i, name in enumerate(names) if unit.lower() in name.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise UnitError(f"Ambiguous unit {unit!r} matches: {[names[i] for i in matches]}")
    raise UnitError(f"Unknown unit {unit!r}; expected one of {names}")


def convert(kind: str, value, unit_from, unit_to=None, *, excitation_wavelength_nm: float | None = None):
    """Convert `value` from `unit_from` into `unit_to` (defaults to the kind's default unit).

    Both `unit_from`/`unit_to` may be the WITec UnitIndex (int, matching the
    README's UnitIndex tables) or a unit name string such as 'nm', 'rel. 1/cm', 'um'.
    `excitation_wavelength_nm` is only required when `kind == 'spectral'` and either
    unit involved is one of the 'rel. ...' (Raman-shift-style) units.
    """
    if kind == "spectral":
        rows = _spectral_table(excitation_wavelength_nm if excitation_wavelength_nm is not None else float("nan"))
    elif kind in UNIT_TABLES:
        rows = UNIT_TABLES[kind]
    else:
        raise UnitError(f"Unknown interpretation kind {kind!r}")

    value = np.asarray(value, dtype=float)
    from_idx = _find_unit_index(rows, unit_from)
    if unit_to is None:
        unit_to = DEFAULT_UNIT.get(kind, rows[0][0])
    to_idx = _find_unit_index(rows, unit_to)

    if kind == "spectral" and excitation_wavelength_nm is None:
        if rows[from_idx][0].startswith("rel.") or rows[to_idx][0].startswith("rel."):
            raise UnitError("excitation_wavelength_nm is required for 'rel.' spectral units")

    default_value = rows[from_idx][2](value)  # this unit -> default
    return rows[to_idx][1](default_value), rows[to_idx][0]  # default -> this unit


def unit_names(kind: str, excitation_wavelength_nm: float | None = None) -> list[str]:
    if kind == "spectral":
        return [row[0] for row in _spectral_table(excitation_wavelength_nm or 1.0)]
    return [row[0] for row in UNIT_TABLES[kind]]
