"""Legacy binary VTK writer for viewing grids and properties in ParaView."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from resim.grid import Grid

# Hexahedron corner offsets in VTK order: bottom face then top face.
_CORNERS = np.array(
    [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
    dtype=float,
)


def write_vtk(path, grid: Grid, cell_data: dict[str, np.ndarray]) -> None:
    """Write active cells as hexahedra, with z flipped so depth points down.

    ``cell_data`` arrays may be global (nx*ny*nz) or per active cell.
    """
    n = grid.n_cells
    points = grid.cell_centroid[:, None, :] + 0.5 * grid.cell_dxyz[:, None, :] * _CORNERS[None]
    points[..., 2] *= -1.0
    cells = np.column_stack([np.full(n, 8), np.arange(8 * n).reshape(n, 8)])

    with open(Path(path), "wb") as fh:
        fh.write(b"# vtk DataFile Version 3.0\nresim grid\nBINARY\nDATASET UNSTRUCTURED_GRID\n")
        fh.write(f"POINTS {8 * n} float\n".encode())
        fh.write(points.reshape(-1, 3).astype(">f4").tobytes())
        fh.write(f"\nCELLS {n} {9 * n}\n".encode())
        fh.write(cells.astype(">i4").tobytes())
        fh.write(f"\nCELL_TYPES {n}\n".encode())
        fh.write(np.full(n, 12, dtype=">i4").tobytes())
        fh.write(f"\nCELL_DATA {n}\n".encode())
        for name, values in cell_data.items():
            values = np.asarray(values, dtype=float)
            if values.size == grid.n_global:
                values = values[grid.cell_global]
            if values.size != n:
                raise ValueError(f"{name}: {values.size} values, expected {n} or {grid.n_global}")
            fh.write(f"SCALARS {name} float 1\nLOOKUP_TABLE default\n".encode())
            fh.write(values.astype(">f4").tobytes())
            fh.write(b"\n")
