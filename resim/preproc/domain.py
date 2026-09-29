"""Preprocessor: grid + properties -> Domain (the only thing the solver sees).

The Domain is a graph: cells with pore volumes and depths, connections with
transmissibilities. All quantities are SI.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from resim.grid import BOUNDARY_TAGS, Grid, global_to_ijk
from resim.preproc.transmissibility import half_transmissibility, harmonic
from resim.props import PropertySet
from resim.units import MILLIDARCY

_MULT = ("MULTX", "MULTY", "MULTZ")


@dataclass
class Domain:
    dims: tuple[int, int, int]
    cell_global: np.ndarray  # (n_cells,) global index of each simulation cell

    pore_volume: np.ndarray  # (n_cells,) m³
    bulk_volume: np.ndarray  # (n_cells,) m³
    depth: np.ndarray  # (n_cells,) m, cell centroid
    centroid: np.ndarray  # (n_cells, 3) m
    cell_dxyz: np.ndarray  # (n_cells, 3) m, for well indices
    porosity: np.ndarray  # (n_cells,)
    ntg: np.ndarray  # (n_cells,)
    perm: np.ndarray  # (n_cells, 3) m², (kx, ky, kz)

    conn: np.ndarray  # (n_conn, 2) connected cell pairs
    trans: np.ndarray  # (n_conn,) m³, geometric transmissibility
    conn_dir: np.ndarray  # (n_conn,) 0=X, 1=Y, 2=Z
    conn_is_nnc: np.ndarray  # (n_conn,) bool

    bnd_cell: np.ndarray  # (n_bnd,) cell of each boundary face
    bnd_trans: np.ndarray  # (n_bnd,) m³, cell-to-face half transmissibility
    bnd_depth: np.ndarray  # (n_bnd,) m, face centroid depth
    bnd_tag: np.ndarray  # (n_bnd,) index into BOUNDARY_TAGS

    regions: dict[str, np.ndarray]  # SATNUM, PVTNUM, EQLNUM, FIPNUM

    @property
    def n_cells(self) -> int:
        return int(self.cell_global.size)

    @property
    def n_conn(self) -> int:
        return int(self.conn.shape[0])

    @property
    def ijk(self) -> np.ndarray:
        return global_to_ijk(self.cell_global, self.dims)

    def cell_index(self, i: int, j: int, k: int) -> int:
        """Domain index of a cell given 1-based (i, j, k)."""
        nx, ny, nz = self.dims
        if not (1 <= i <= nx and 1 <= j <= ny and 1 <= k <= nz):
            raise ValueError(f"cell ({i},{j},{k}) is outside the grid {self.dims}")
        g = (i - 1) + nx * ((j - 1) + ny * (k - 1))
        hit = np.flatnonzero(self.cell_global == g)
        if hit.size == 0:
            raise ValueError(f"cell ({i},{j},{k}) is not an active simulation cell")
        return int(hit[0])

    def boundary(self, tag: str) -> np.ndarray:
        """Indices into the ``bnd_*`` arrays for one side of the model."""
        return np.flatnonzero(self.bnd_tag == BOUNDARY_TAGS.index(tag))

    def summary(self) -> str:
        return (
            f"Domain: {self.n_cells} cells, {self.n_conn} connections "
            f"({int(self.conn_is_nnc.sum())} NNC), {self.bnd_cell.size} boundary faces\n"
            f"  pore volume {self.pore_volume.sum():.6g} m³ "
            f"(bulk {self.bulk_volume.sum():.6g} m³), depth {self.depth.min():.2f}-{self.depth.max():.2f} m\n"
            f"  transmissibility {self.trans.min() if self.n_conn else 0:.4g}-"
            f"{self.trans.max() if self.n_conn else 0:.4g} m³"
        )


def build_domain(grid: Grid, props: PropertySet, minpv: float = 0.0) -> Domain:
    """Compute pore volumes and transmissibilities.

    Cells with pore volume <= ``minpv`` (m³) are removed, with their
    connections; zero pore volume cells are always removed.
    """
    active = props.active
    poro, ntg = active("PORO"), active("NTG")
    pv_all = grid.cell_volume * poro * ntg * active("MULTPV")
    keep = pv_all > max(minpv, 0.0)
    if not keep.any():
        raise ValueError("no cells with positive pore volume")

    perm = np.column_stack([active("PERMX"), active("PERMY"), active("PERMZ")]) * MILLIDARCY
    perm_flow = perm.copy()
    perm_flow[:, :2] *= ntg[:, None]  # net-to-gross reduces horizontal flow only

    fc = grid.face_cells
    internal = grid.internal_faces & keep[fc[:, 0]] & np.where(fc[:, 1] >= 0, keep[fc[:, 1]], False)
    boundary = grid.boundary_faces & keep[fc[:, 0]]

    def half(faces, side):
        cells = fc[faces, side]
        return half_transmissibility(
            grid.face_area[faces],
            grid.face_normal[faces],
            grid.face_centroid[faces],
            grid.cell_centroid[cells],
            perm_flow[cells],
        )

    # Eclipse convention: MULTX of cell (i,j,k) scales its +X face.
    a = fc[internal, 0]
    mult = np.stack([active(m) for m in _MULT], axis=1)[a, grid.face_dir[internal]]
    trans = mult * harmonic(half(internal, 0), half(internal, 1))

    new_index = np.full(grid.n_cells, -1, dtype=np.int64)
    new_index[keep] = np.arange(int(keep.sum()))

    return Domain(
        dims=grid.dims,
        cell_global=grid.cell_global[keep],
        pore_volume=pv_all[keep],
        bulk_volume=grid.cell_volume[keep],
        depth=grid.cell_centroid[keep, 2],
        centroid=grid.cell_centroid[keep],
        cell_dxyz=grid.cell_dxyz[keep],
        porosity=poro[keep],
        ntg=ntg[keep],
        perm=perm[keep],
        conn=new_index[fc[internal]],
        trans=trans,
        conn_dir=grid.face_dir[internal],
        conn_is_nnc=grid.face_is_nnc[internal],
        bnd_cell=new_index[fc[boundary, 0]],
        bnd_trans=half(boundary, 0),
        bnd_depth=grid.face_centroid[boundary, 2],
        bnd_tag=grid.face_tag[boundary],
        regions={r: active(r)[keep] for r in ("SATNUM", "PVTNUM", "EQLNUM", "FIPNUM")},
    )
