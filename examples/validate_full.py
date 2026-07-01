import sys
import numpy as np
import witio

path = sys.argv[1]
project = witio.read(path)
print(project)
print(f"version={project.version}, magic={project.magic!r}")

errors = []
for entry in project.data:
    try:
        if entry.class_name == "TDGraph":
            arr = entry.array()
            x, unit = entry.x_axis()
            if arr.shape[0] > 1 or arr.shape[1] > 1:
                px, py = entry.position_grid("um")
                print(f"{entry.caption!r}: shape={arr.shape} nan_frac={np.isnan(arr).mean():.3f} "
                      f"x_unit={unit} pos_x_range=({px.min():.2f},{px.max():.2f})um")
        elif entry.class_name == "TDBitmap":
            arr = entry.array()
            print(f"{entry.caption!r}: TDBitmap shape={arr.shape} dtype={arr.dtype} "
                  f"min={arr.min()} max={arr.max()}")
        elif entry.class_name == "TDImage":
            arr = entry.array()
            print(f"{entry.caption!r}: TDImage shape={arr.shape}")
        h = entry.history
    except Exception as e:
        errors.append((entry.id, entry.class_name, entry.caption, repr(e)))

print(f"\n{len(errors)} errors out of {len(project.data)} entries")
for e in errors:
    print(" ", e)
