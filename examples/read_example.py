"""Example: extract Raman spectra + axes from a WITec .wip/.wid file.

Usage: python examples/read_example.py path/to/file.wip
"""
import sys

import witio


def main(path: str) -> None:
    project = witio.read(path)
    print(project)

    for entry in project.data:
        print(f"- id={entry.id} class={entry.class_name!r} caption={entry.caption!r}")

    for graph in project.find(class_name="TDGraph"):
        spectra = graph.array()  # shape (SizeX, SizeY, SizeGraph)
        wavenumber, unit = graph.x_axis("rel. 1/cm")
        print(f"\n{graph.caption}: spectra shape {spectra.shape}, x-axis in {unit}")
        print(f"  first spectrum (x=0,y=0): {spectra[0, 0, :5]} ...")
        print(f"  axis: {wavenumber[:5]} ...")

    for image in project.find(class_name="TDImage"):
        arr = image.array()
        print(f"\n{image.caption}: image shape {arr.shape}")

    print(f"\nsystem metadata: {project.system_metadata}")
    for graph in project.find(class_name="TDGraph"):
        meta = graph.measurement_metadata
        if not meta:
            continue
        print(f"\n{graph.caption}:")
        print(f"  laser={meta.get('laser_wavelength_nm')}nm  "
              f"integration_time={meta.get('integration_time_s')}s  "
              f"objective={meta.get('objective_name')!r}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(1)
    main(sys.argv[1])
