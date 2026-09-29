"""Cartesian (block-centred) grid builder: DX / DY / DZ / TOPS.

Structure and dip are represented through TOPS, which may be a scalar, a 2D
map (nx*ny values, top of layer 1) or a full array (top of every cell).
"""

from __future__ import annotations

import numpy as np

from resim.grid.grid import BOUNDARY_TAGS, INTERNAL, Grid

_AXES = "xyz"


def expand_array(value, dims: tuple[int, int, int], name: str, axis: int | None = None) -> np.ndarray:
    """Broadcast a scalar, per-axis list or full array to shape (nx, ny, nz).

    Full arrays are given in Eclipse order (i fastest).
    """
    arr = np.asarray(value, dtype=float)
    n = int(np.prod(dims))
    if arr.ndim == 0:
        return np.full(dims, float(arr))
    arr = arr.ravel()
    if arr.size == n:
        return arr.reshape(dims, order="F").copy()
    if axis is not None and arr.size == dims[axis]:
        shape = [1, 1, 1]
        shape[axis] = dims[axis]
        return np.broadcast_to(arr.reshape(shape), dims).copy()
    expected = f"1, {dims[axis]} or {n}" if axis is not None else f"1 or {n}"
    raise ValueError(f"{name}: got {arr.size} values, expected {expected}")


def _expand_tops(tops, dz: np.ndarray, dims: tuple[int, int, int]) -> np.ndarray:
    """Return the depth of the top of every cell, shape (nx, ny, nz)."""
    nx, ny, nz = dims
    arr = np.asarray(tops, dtype=float)
    if arr.ndim == 0 or arr.size == nx * ny:
        top_map = np.broadcast_to(arr.reshape(nx, ny, order="F") if arr.ndim else arr, (nx, ny))
        return top_map[:, :, None] + np.cumsum(dz, axis=2) - dz
    if arr.size == nx * ny * nz:
        return arr.reshape(dims, order="F").copy()
    raise ValueError(f"tops: got {arr.size} values, expected 1, {nx * ny} or {nx * ny * nz}")


def build_cartesian(
    dims,
    dx,
    dy,
    dz,
    tops=0.0,
    actnum=None,
    origin=(0.0, 0.0),
) -> Grid:
    """Build a Cartesian grid.

    Parameters
    ----------
    dims : (nx, ny, nz)
    dx, dy, dz : scalar, per-axis list (length nx / ny / nz) or full array, m
    tops : scalar, nx*ny map or full array, depth of cell tops, m
    actnum : optional global array (nonzero = active)
    origin : (x0, y0) of the model corner, m
    """
    dims = tuple(int(d) for d in dims)
    if len(dims) != 3 or min(dims) < 1:
        raise ValueError(f"dims must be three positive integers, got {dims}")
    nx, ny, nz = dims
    n = nx * ny * nz

    DX = expand_array(dx, dims, "dx", axis=0)
    DY = expand_array(dy, dims, "dy", axis=1)
    DZ = expand_array(dz, dims, "dz", axis=2)
    for name, arr in (("dx", DX), ("dy", DY), ("dz", DZ)):
        if not np.all(arr > 0):
            raise ValueError(f"{name} must be strictly positive")
    ZT = _expand_tops(tops, DZ, dims)
    XL = origin[0] + np.cumsum(DX, axis=0) - DX
    YL = origin[1] + np.cumsum(DY, axis=1) - DY

    lower = (XL, YL, ZT)
    size = (DX, DY, DZ)
    face_area_of_cell = (DY * DZ, DX * DZ, DX * DY)

    def flat(a):
        return a.ravel(order="F")

    G = np.arange(n).reshape(dims, order="F")
    centroid = np.column_stack([flat(lo + s / 2) for lo, s in zip(lower, size)])

    cells, areas, normals, centroids, dirs, tags = [], [], [], [], [], []

    # Internal faces: pair each cell with its +1 neighbour along each axis.
    for axis in range(3):
        if dims[axis] < 2:
            continue
        sa, sb = [slice(None)] * 3, [slice(None)] * 3
        sa[axis], sb[axis] = slice(0, -1), slice(1, None)
        sa, sb = tuple(sa), tuple(sb)
        a, b = flat(G[sa]), flat(G[sb])
        fc = 0.5 * (centroid[a] + centroid[b])
        fc[:, axis] = 0.5 * (flat((lower[axis] + size[axis])[sa]) + flat(lower[axis][sb]))
        normal = np.zeros((a.size, 3))
        normal[:, axis] = 1.0
        cells.append(np.column_stack([a, b]))
        areas.append(0.5 * (flat(face_area_of_cell[axis][sa]) + flat(face_area_of_cell[axis][sb])))
        normals.append(normal)
        centroids.append(fc)
        dirs.append(np.full(a.size, axis))
        tags.append(np.full(a.size, INTERNAL))

    # Boundary faces on the six sides of the model.
    for axis in range(3):
        for side in (0, 1):
            s = [slice(None)] * 3
            s[axis] = 0 if side == 0 else dims[axis] - 1
            s = tuple(s)
            a = flat(G[s])
            fc = centroid[a].copy()
            edge = lower[axis][s] if side == 0 else (lower[axis] + size[axis])[s]
            fc[:, axis] = flat(edge)
            normal = np.zeros((a.size, 3))
            normal[:, axis] = -1.0 if side == 0 else 1.0
            cells.append(np.column_stack([a, np.full(a.size, -1)]))
            areas.append(flat(face_area_of_cell[axis][s]))
            normals.append(normal)
            centroids.append(fc)
            dirs.append(np.full(a.size, axis))
            tags.append(np.full(a.size, BOUNDARY_TAGS.index(f"{_AXES[axis]}{'min' if side == 0 else 'max'}")))

    face_cells = np.concatenate(cells)
    face_tag = np.concatenate(tags)

    # Restrict to active cells; faces touching inactive cells carry no flow.
    active = np.ones(n, dtype=bool) if actnum is None else np.asarray(actnum).ravel() != 0
    if active.size != n:
        raise ValueError(f"actnum: got {active.size} values, expected {n}")
    g2a = np.full(n, -1, dtype=np.int64)
    g2a[active] = np.arange(int(active.sum()))
    internal = face_tag == INTERNAL
    keep = active[face_cells[:, 0]] & np.where(internal, active[face_cells[:, 1]], True)
    fcells = face_cells[keep]
    fcells = np.column_stack([g2a[fcells[:, 0]], np.where(fcells[:, 1] >= 0, g2a[fcells[:, 1]], -1)])

    return Grid(
        dims=dims,
        cell_global=np.flatnonzero(active),
        global_to_active=g2a,
        cell_volume=flat(DX * DY * DZ)[active],
        cell_centroid=centroid[active],
        cell_dxyz=np.column_stack([flat(DX), flat(DY), flat(DZ)])[active],
        face_cells=fcells,
        face_area=np.concatenate(areas)[keep],
        face_normal=np.concatenate(normals)[keep],
        face_centroid=np.concatenate(centroids)[keep],
        face_dir=np.concatenate(dirs)[keep],
        face_tag=face_tag[keep],
        face_is_nnc=np.zeros(int(keep.sum()), dtype=bool),
    )
