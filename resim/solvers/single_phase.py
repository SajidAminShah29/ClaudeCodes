"""Single-phase flow on a Domain: steady incompressible and transient
slightly-compressible (backward Euler). Both are linear in pressure.

Flux from cell a to cell b (m³/s):

    F_ab = (T_ab / mu) * (p_a - p_b - rho * g * (z_a - z_b))

Sources ``q`` are in m³/s, positive = injection. Boundary conditions are
fixed pressures on named model sides (``xmin`` ... ``zmax``), specified at the
depth of each boundary face; sides without a condition are no-flow.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from resim.preproc import Domain
from resim.units import GRAVITY


@dataclass
class Fluid:
    viscosity: float  # Pa·s
    density: float = 1000.0  # kg/m³
    compressibility: float = 0.0  # 1/Pa


@dataclass
class SteadyResult:
    pressure: np.ndarray  # Pa
    flux: np.ndarray  # (n_conn,) m³/s, from conn[:,0] to conn[:,1]
    boundary_flux: np.ndarray  # (n_bnd,) m³/s, out of the domain (only on BC faces)


@dataclass
class TransientResult:
    times: np.ndarray  # s, including t=0
    pressure: np.ndarray  # (n_steps+1, n_cells) Pa
    boundary_outflow: np.ndarray  # (n_steps,) m³ per step leaving through BCs


def _assemble(domain: Domain, fluid: Fluid, bc: dict[str, float] | None, g: float):
    """Matrix A and right-hand side b such that A p = b + q gives flux balance."""
    n = domain.n_cells
    a, b = domain.conn[:, 0], domain.conn[:, 1]
    lam = domain.trans / fluid.viscosity
    grav = lam * fluid.density * g * (domain.depth[a] - domain.depth[b])

    rows = np.concatenate([a, b, a, b])
    cols = np.concatenate([a, b, b, a])
    vals = np.concatenate([lam, lam, -lam, -lam])
    rhs = np.zeros(n)
    np.add.at(rhs, a, grav)
    np.add.at(rhs, b, -grav)

    bc_faces, bc_pressure = _boundary_faces(domain, bc)
    cells = domain.bnd_cell[bc_faces]
    lam_b = domain.bnd_trans[bc_faces] / fluid.viscosity
    rows = np.concatenate([rows, cells])
    cols = np.concatenate([cols, cells])
    vals = np.concatenate([vals, lam_b])
    np.add.at(rhs, cells, lam_b * (bc_pressure + fluid.density * g * (domain.depth[cells] - domain.bnd_depth[bc_faces])))

    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return A, rhs, (bc_faces, bc_pressure, lam_b)


def _boundary_faces(domain: Domain, bc: dict[str, float] | None):
    faces, pressures = [], []
    for tag, p in (bc or {}).items():
        idx = domain.boundary(tag)
        if idx.size == 0:
            raise ValueError(f"boundary {tag!r} has no active faces")
        faces.append(idx)
        pressures.append(np.full(idx.size, float(p)))
    if not faces:
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    return np.concatenate(faces), np.concatenate(pressures)


def _fluxes(domain, fluid, p, bc_info, g):
    a, b = domain.conn[:, 0], domain.conn[:, 1]
    flux = domain.trans / fluid.viscosity * (p[a] - p[b] - fluid.density * g * (domain.depth[a] - domain.depth[b]))
    bc_faces, bc_pressure, lam_b = bc_info
    cells = domain.bnd_cell[bc_faces]
    bflux = np.zeros(domain.bnd_cell.size)
    bflux[bc_faces] = lam_b * (
        p[cells] - bc_pressure - fluid.density * g * (domain.depth[cells] - domain.bnd_depth[bc_faces])
    )
    return flux, bflux


def solve_steady(
    domain: Domain,
    fluid: Fluid,
    bc: dict[str, float] | None = None,
    sources: np.ndarray | None = None,
    gravity: float = GRAVITY,
) -> SteadyResult:
    """Steady incompressible flow. Requires at least one pressure boundary."""
    if not bc:
        raise ValueError("steady incompressible flow needs at least one pressure boundary condition")
    A, rhs, bc_info = _assemble(domain, fluid, bc, gravity)
    q = np.zeros(domain.n_cells) if sources is None else np.asarray(sources, dtype=float)
    p = spsolve(A.tocsc(), rhs + q)
    flux, bflux = _fluxes(domain, fluid, p, bc_info, gravity)
    return SteadyResult(pressure=p, flux=flux, boundary_flux=bflux)


def simulate_transient(
    domain: Domain,
    fluid: Fluid,
    p0,
    dt: float,
    n_steps: int,
    bc: dict[str, float] | None = None,
    sources: np.ndarray | None = None,
    rock_compressibility: float = 0.0,
    gravity: float = GRAVITY,
) -> TransientResult:
    """Slightly compressible single-phase flow, linearized about p0.

    Accumulation: PV * c_t * (p^{n+1} - p^n) / dt with c_t = c_f + c_r.
    """
    ct = fluid.compressibility + rock_compressibility
    if ct <= 0:
        raise ValueError("transient flow needs positive total compressibility")
    A, rhs, bc_info = _assemble(domain, fluid, bc, gravity)
    acc = domain.pore_volume * ct / dt
    M = (A + sp.diags(acc)).tocsc()
    q = np.zeros(domain.n_cells) if sources is None else np.asarray(sources, dtype=float)

    p = np.broadcast_to(np.asarray(p0, dtype=float), (domain.n_cells,)).copy()
    history, outflow = [p.copy()], []
    for _ in range(n_steps):
        p = spsolve(M, rhs + q + acc * p)
        history.append(p.copy())
        _, bflux = _fluxes(domain, fluid, p, bc_info, gravity)
        outflow.append(bflux.sum() * dt)
    return TransientResult(
        times=np.arange(n_steps + 1) * dt,
        pressure=np.array(history),
        boundary_outflow=np.array(outflow),
    )
