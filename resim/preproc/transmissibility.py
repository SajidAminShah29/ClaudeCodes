"""Two-point flux approximation (TPFA) transmissibilities.

Half-transmissibility of cell i through face f:

    t_i = A_f * |c · (K_i n_f)| / (c · c),    c = x_f - x_i

Face transmissibility is the series (harmonic) combination of both halves,
scaled by the direction multiplier:  T_f = MULT / (1/t_a + 1/t_b).
Units: m² (area) * m² (perm) / m = m³.
"""

from __future__ import annotations

import numpy as np


def half_transmissibility(
    face_area: np.ndarray,
    face_normal: np.ndarray,
    face_centroid: np.ndarray,
    cell_centroid: np.ndarray,
    cell_perm: np.ndarray,
) -> np.ndarray:
    """Half-transmissibilities for (face, cell) pairs given row-aligned arrays.

    ``cell_perm`` holds the diagonal permeability tensor (kx, ky, kz) in m².
    """
    c = face_centroid - cell_centroid
    kn = cell_perm * face_normal
    return face_area * np.abs(np.sum(c * kn, axis=1)) / np.sum(c * c, axis=1)


def harmonic(t_a: np.ndarray, t_b: np.ndarray) -> np.ndarray:
    """Series combination, returning 0 where either half is 0."""
    with np.errstate(divide="ignore"):
        denom = 1.0 / t_a + 1.0 / t_b
    return np.where(np.isfinite(denom), 1.0 / denom, 0.0)
