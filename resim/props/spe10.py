"""Loader for the SPE10 Model 2 dataset (60 x 220 x 85 cells).

Download ``spe_phi.dat`` and ``spe_perm.dat`` from the SPE Comparative
Solution Project site (https://www.spe.org/web/csp/datasets/set02.htm).

* ``spe_phi.dat``  : 1,122,000 porosity values
* ``spe_perm.dat`` : 3,366,000 values: all kx, then all ky, then all kz (mD)

Both are in natural order (i fastest, then j, then k).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SPE10_DIMS = (60, 220, 85)
SPE10_CELL_SIZE = (20 * 0.3048, 10 * 0.3048, 2 * 0.3048)  # 20 ft x 10 ft x 2 ft
_N = SPE10_DIMS[0] * SPE10_DIMS[1] * SPE10_DIMS[2]
_COMPONENTS = {"poro": None, "permx": 0, "permy": 1, "permz": 2}


def read_spe10_file(path) -> np.ndarray:
    data = np.fromfile(Path(path), sep=" ")
    if data.size not in (_N, 3 * _N):
        raise ValueError(f"{path}: got {data.size} values, expected {_N} (phi) or {3 * _N} (perm)")
    return data


def extract(data: np.ndarray, component: str, dims, origin=(1, 1, 1)) -> np.ndarray:
    """Cut a ``dims`` sub-box starting at 1-based ``origin`` out of an SPE10 array."""
    component = component.lower()
    if component not in _COMPONENTS:
        raise ValueError(f"unknown SPE10 component {component!r} (poro, permx, permy, permz)")
    block = _COMPONENTS[component]
    if block is None:
        if data.size != _N:
            raise ValueError("component 'poro' requires spe_phi.dat")
        arr = data
    else:
        if data.size != 3 * _N:
            raise ValueError(f"component {component!r} requires spe_perm.dat")
        arr = data[block * _N : (block + 1) * _N]
    cube = arr.reshape(SPE10_DIMS, order="F")
    lo = [o - 1 for o in origin]
    hi = [l + d for l, d in zip(lo, dims)]
    if min(lo) < 0 or any(h > s for h, s in zip(hi, SPE10_DIMS)):
        raise ValueError(f"box origin={tuple(origin)} dims={tuple(dims)} exceeds SPE10 extent {SPE10_DIMS}")
    return cube[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]].ravel(order="F")
