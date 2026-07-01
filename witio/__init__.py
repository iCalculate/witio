"""witio: a Python port of the read/import/extract core of the WITio (wit_io) MATLAB
toolbox (https://gitlab.com/jtholmi/wit_io) for WITec Project (.wip) and Data (.wid) files.

Quick start:

    import witio
    project = witio.read("sample.wip")
    for entry in project.find(class_name="TDGraph"):
        spectra = entry.array()                # shape (SizeX, SizeY, SizeGraph)
        wavenumber, unit = entry.x_axis("rel. 1/cm")

See README.md for the supported feature set and its limitations relative to
the original MATLAB toolbox.
"""
from .project import WitProject, read, read_bytes
from .data import WitData, HistoryEntry, DataFormatError
from .tag import WitTag, WitFormatError

__version__ = "0.1.0"

__all__ = [
    "read", "read_bytes", "WitProject", "WitData", "HistoryEntry",
    "WitTag", "WitFormatError", "DataFormatError", "__version__",
]
