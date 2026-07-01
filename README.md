<div align="center">

# witio

**Read WITec Project (`.wip`) and Data (`.wid`) files directly in Python — spectra, hyperspectral maps, images, and calibrated axes as NumPy arrays.**

[![CI](https://github.com/iCalculate/witio/actions/workflows/ci.yml/badge.svg)](https://github.com/iCalculate/witio/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/witio.svg)](https://pypi.org/project/witio/)
[![Python versions](https://img.shields.io/pypi/pyversions/witio.svg)](https://pypi.org/project/witio/)
[![License: MIT-0](https://img.shields.io/badge/license-MIT--0-blue.svg)](LICENSE)

</div>

---

`witio` is a pure-Python port of the **read/import/extract** core of
[**wit_io (WITio)**](https://gitlab.com/jtholmi/wit_io), the reference MATLAB
toolbox for WITec's binary file formats. No MATLAB required, no proprietary
license — just `pip install` and go.

```python
import witio

project = witio.read("sample.wip")
for entry in project.find(class_name="TDGraph"):
    spectra = entry.array()                       # (SizeX, SizeY, SizeGraph)
    raman_shift, unit = entry.x_axis("rel. 1/cm")  # calibrated axis, unit == 'rel. 1/cm'
```

## Contents

- [Why this exists](#why-this-exists)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Supported data](#supported-data)
- [Scope & limitations](#scope--limitations)
- [Validation](#validation)
- [Development](#development)
- [License & credits](#license--credits)

## Why this exists

There was no Python package that reads WITec's native `.wip`/`.wid` binary
format directly under a permissive license:

| Package | Reads `.wip`/`.wid` directly? | License |
|---|---|---|
| [wit_io / WITio](https://gitlab.com/jtholmi/wit_io) (MATLAB) | ✅ Yes (reference implementation) | MIT-0, but MATLAB-only |
| [RamanSPy](https://ramanspy.readthedocs.io/) | ❌ Only pre-exported `.mat` files | Open source |
| `wip_loader` (photonicdata-files-wip) | ✅ Yes | ⚠️ Proprietary, non-redistributable |
| [py-wdf-reader](https://github.com/alchem0x2A/py-wdf-reader) | ❌ Different format (Renishaw `.wdf`) | Open source |
| **witio** | ✅ Yes | **MIT-0** |

`witio` follows the WIT-tag binary format spec that wit_io documents and
reverse-engineers, reimplemented from scratch in Python — no MATLAB source was
copied. See [License & credits](#license--credits).

## Installation

```bash
pip install witio
```

Legacy (WITec software v0-v5) bitmap images are embedded as raw BMP files and
need [Pillow](https://pypi.org/project/Pillow/) to decode:

```bash
pip install "witio[bitmap]"
```

## Quick start

```python
import witio

project = witio.read("sample.wip")
print(project)  # <WitProject file='sample.wip' version=7 n_data=42>

# List everything in the file
for entry in project.data:
    print(entry.id, entry.class_name, entry.caption)

# Hyperspectral / line / point Raman data
for graph in project.find(class_name="TDGraph"):
    spectra = graph.array()                        # shape (SizeX, SizeY, SizeGraph)
    raman_shift, unit = graph.x_axis("rel. 1/cm")   # or "nm", "1/cm", "eV", ...
    x_um, y_um = graph.position_grid("um")          # physical stage coordinates

# Video / camera images
for image in project.find(class_name="TDBitmap"):
    rgb = image.array()                             # shape (SizeX, SizeY, 3)

# Per-entry metadata
entry = project.find(class_name="TDGraph")[0]
print(entry.caption, entry.id, entry.history)
```

A runnable version of this lives in [`examples/read_example.py`](examples/read_example.py).

## Supported data

| WITec class | Extraction | Notes |
|---|---|---|
| `TDGraph` | `.array()` → `(SizeX, SizeY, SizeGraph)` | Spectra, line scans, hyperspectral area scans. Incomplete scan lines auto-masked to `NaN` (matches wit_io's `wip_AutoNanInvalid`). |
| `TDImage` | `.array()` → `(SizeX, SizeY)` | |
| `TDBitmap` | `.array()` → `(SizeX, SizeY, 3)` RGB | Raw RGBA (v6/v7+) or embedded BMP (legacy, needs Pillow). |
| `TDText` | `.text()` | Best-effort RTF → plaintext (legacy `TDStream` layout only). |
| Any `TDGraph`/`TDImage`/`TDBitmap` | `.x_axis(unit)` / `.position_grid(unit)` | Resolves linked `TD*Transformation`/`TD*Interpretation` entries by ID and converts units. |
| `TData` (all entries) | `.id`, `.caption`, `.history`, `.metadata` | |

Spectral axis units: `nm`, `um`, `1/cm`, `rel. 1/cm` (Raman shift), `eV`, `meV`,
`rel. eV`, `rel. meV`. Spatial units: `m`, `mm`, `um`, `nm`, `A`, `pm`.

## Scope & limitations

This is a **read/extract** library, not a full wit_io replacement. Intentionally
not ported (GUI/MATLAB-only concerns that don't matter once your data is in
NumPy):

- Writing/saving `.wip`/`.wid` project files (the low-level tag tree can
  round-trip for testing, but doesn't regenerate the ID/`NumberOfData`
  bookkeeping a real project file needs).
- Curve fitting, image filtering, plotting, the interactive mask editor, and
  `Viewer` tags — all GUI-only in the original toolbox.
- Live instrument control (that's what WITec's own `WITecSDK` is for).
- `TDText` on WITec Suite SIX (v7): that layout was never fully
  reverse-engineered upstream either.
- Format versions v0-v1 and v3-v4 were only partially analyzed even by wit_io
  itself ("file donations welcome" — same applies here).

## Validation

No real `.wip` file was available while writing the initial port, so the test
suite validates the reader against hand-built synthetic WIT-tag binaries: a
tag-tree round-trip test, unit/transform math checked against hand
calculations, and a full synthetic project exercising Data enumeration, ID
cross-referencing, and spectral calibration end-to-end.

It has since been run against a real 679 MB WITec Suite v8 project (126 `Data`
entries: point spectra, line scans, several hyperspectral area scans including
one incomplete scan, and video-image bitmaps) with **zero errors** — spectra
reshaped correctly, incomplete-scan lines came out as `NaN`, bitmap frames
decoded to sane-looking RGB images, and axis/position calibration correctly
cross-referenced different `TD*Interpretation` entries (some in eV, some in
Raman shift) per spectrum.

If you hit a file this doesn't handle, please [open an issue](https://github.com/iCalculate/witio/issues)
with as much detail as you can share (WITec software version, what part failed).

## Development

```bash
git clone https://github.com/iCalculate/witio.git
cd witio
pip install -e ".[test,bitmap]"
pytest
```

## License & credits

MIT-0 (MIT No Attribution) — see [LICENSE](LICENSE).

`witio` is an independent Python reimplementation of the file-format knowledge
documented by [**wit_io (WITio)**](https://gitlab.com/jtholmi/wit_io) by
Joonas T. Holmi, also MIT-0 licensed. No MATLAB source was copied; its binary
format specification and reverse-engineered calibration formulas (spectral
grating equation, space transformations, unit conversions) were used as a
reference and are re-derived in [`witio/transform.py`](witio/transform.py) and
[`witio/units.py`](witio/units.py). If you use this in published research,
please cite the original wit_io paper:

> J. T. Holmi, "WITio: A MATLAB data evaluation toolbox to script broader
> insights into big data from WITec microscopes," *SoftwareX*, 2022.
