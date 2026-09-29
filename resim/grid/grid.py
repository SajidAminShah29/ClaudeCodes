"""Grid geometry shared by every grid builder.

A ``Grid`` is a general unstructured description (cells + faces + neighbours).
Cartesian and, later, corner-point builders both produce this same object, so
everything downstream is independent of how the grid was generated.

Conventions
-----------
* Global cell index follows Eclipse ordering: ``g = i + nx*(j + ny*k)``
  (i fastest, 0-based).
* ``z`` is depth, positive downwards.
* Only active cells (ACTNUM = 1) carry geometry. ``cell_global`` maps an active
  cell to its global index; ``global_to_active`` maps back (-1 when inactive).
* Internal faces: ``face_cells[f] = (a, b)`` with the normal pointing a -> b.
  Boundary faces: ``face_cells[f] = (a, -1)`` with an outward normal and a
  ``face_tag`` naming the side of the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BOUNDARY_TAGS = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
INTERNAL = -1


def global_to_ijk(g: np.ndarray, dims: tuple[int, int, int]) -> np.ndarray:
    """Convert global indices to 0-based (i, j, k), shape (n, 3)."""
    nx, ny, _ = dims
    g = np.asarray(g)
    return np.column_stack([g % nx, (g // nx) % ny, g // (nx * ny)])


@dataclass
class Grid:
    dims: tuple[int, int, int]

    cell_global: np.ndarray  # (n_cells,) global index of each active cell
    global_to_active: np.ndarray  # (nx*ny*nz,) active index, -1 if inactive

    cell_volume: np.ndarray  # (n_cells,) m³
    cell_centroid: np.ndarray  # (n_cells, 3) m
    cell_dxyz: np.ndarray  # (n_cells, 3) representative dx, dy, dz (wells)

    face_cells: np.ndarray  # (n_faces, 2)
    face_area: np.ndarray  # (n_faces,) m²
    face_normal: np.ndarray  # (n_faces, 3) unit normal
    face_centroid: np.ndarray  # (n_faces, 3) m
    face_dir: np.ndarray  # (n_faces,) 0=X, 1=Y, 2=Z
    face_tag: np.ndarray  # (n_faces,) INTERNAL or index into BOUNDARY_TAGS
    face_is_nnc: np.ndarray  # (n_faces,) bool

    @property
    def n_cells(self) -> int:
        return int(self.cell_global.size)

    @property
    def n_faces(self) -> int:
        return int(self.face_cells.shape[0])

    @property
    def n_global(self) -> int:
        nx, ny, nz = self.dims
        return nx * ny * nz

    @property
    def actnum(self) -> np.ndarray:
        """Global boolean activity mask."""
        return self.global_to_active >= 0

    @property
    def ijk(self) -> np.ndarray:
        """0-based (i, j, k) of each active cell, shape (n_cells, 3)."""
        return global_to_ijk(self.cell_global, self.dims)

    @property
    def internal_faces(self) -> np.ndarray:
        return self.face_tag == INTERNAL

    @property
    def boundary_faces(self) -> np.ndarray:
        return self.face_tag != INTERNAL

    def summary(self) -> str:
        nx, ny, nz = self.dims
        return (
            f"Grid {nx}x{ny}x{nz}: {self.n_cells} active of {self.n_global} cells, "
            f"{int(self.internal_faces.sum())} internal faces "
            f"({int(self.face_is_nnc.sum())} NNC), "
            f"{int(self.boundary_faces.sum())} boundary faces, "
            f"bulk volume {self.cell_volume.sum():.6g} m³"
        )
