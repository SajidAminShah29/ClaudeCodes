import numpy as np
import pytest

from resim.grid import build_cartesian
from resim.preproc import build_domain
from resim.props import build_properties
from resim.solvers import Fluid, simulate_transient, solve_steady
from resim.units import BAR, CENTIPOISE, DAY, GRAVITY, MILLIDARCY


def _domain(dims, dx, dy, dz, props, tops=0.0):
    g = build_cartesian(dims, dx, dy, dz, tops=tops)
    return build_domain(g, build_properties(g, props))


def test_linear_flow_matches_darcy():
    L, A, k, mu = 100.0, 50.0, 200.0, 2.0
    d = _domain((20, 1, 1), L / 20, 10.0, 5.0, {"PORO": 0.2, "PERMX": k})
    res = solve_steady(d, Fluid(mu * CENTIPOISE), bc={"xmin": 200 * BAR, "xmax": 100 * BAR}, gravity=0.0)
    q_exact = k * MILLIDARCY * A / (mu * CENTIPOISE) * 100 * BAR / L
    assert np.allclose(res.flux, q_exact)
    assert np.isclose(res.boundary_flux[d.boundary("xmax")].sum(), q_exact)
    x = d.centroid[:, 0]
    assert np.allclose(res.pressure, 200 * BAR - 100 * BAR * x / L)


def test_heterogeneous_series_flow():
    ks = np.array([10.0, 100.0, 1.0, 50.0])
    d = _domain((4, 1, 1), 10.0, 1.0, 1.0, {"PORO": 0.2, "PERMX": {"source": "array", "values": ks}})
    res = solve_steady(d, Fluid(CENTIPOISE), bc={"xmin": 10 * BAR, "xmax": 0.0}, gravity=0.0)
    k_eff = 40.0 / np.sum(10.0 / ks)  # harmonic average
    q_exact = k_eff * MILLIDARCY * 1.0 / CENTIPOISE * 10 * BAR / 40.0
    assert np.allclose(res.flux, q_exact)


def test_hydrostatic_column_has_no_flow():
    rho = 800.0
    d = _domain((1, 1, 10), 10.0, 10.0, 3.0, {"PORO": 0.2, "PERMX": 100}, tops=2000.0)
    res = solve_steady(d, Fluid(CENTIPOISE, density=rho), bc={"zmin": 200 * BAR})
    assert np.allclose(res.flux, 0.0, atol=1e-15)
    assert np.allclose(res.pressure, 200 * BAR + rho * GRAVITY * (d.depth - 2000.0))


def test_sources_balance_boundary_outflow():
    d = _domain((5, 5, 1), 10.0, 10.0, 5.0, {"PORO": 0.2, "PERMX": 50})
    q = np.zeros(d.n_cells)
    q[d.cell_index(3, 3, 1)] = 100.0 / DAY
    res = solve_steady(d, Fluid(CENTIPOISE), bc={"xmin": 100 * BAR, "xmax": 100 * BAR}, sources=q)
    assert np.isclose(res.boundary_flux.sum(), q.sum())
    assert res.pressure.argmax() == d.cell_index(3, 3, 1)


def test_steady_requires_boundary_condition():
    d = _domain((2, 1, 1), 1.0, 1.0, 1.0, {"PORO": 0.2, "PERMX": 1})
    with pytest.raises(ValueError):
        solve_steady(d, Fluid(CENTIPOISE))


def test_transient_closed_system_conserves_mass():
    g = build_cartesian((10, 10, 3), 20.0, 20.0, 5.0)
    props = build_properties(
        g,
        {"PORO": {"source": "gaussian_field", "mean": 0.2, "std": 0.03, "variogram": {"range": 60}, "seed": 1},
         "PERMX": {"source": "poro_perm", "from": "PORO", "a": 10, "b": 0}},
    )
    d = build_domain(g, props)
    fluid = Fluid(CENTIPOISE, compressibility=1e-4 / BAR)
    q = np.zeros(d.n_cells)
    q[d.cell_index(1, 1, 1)] = 50.0 / DAY
    q[d.cell_index(10, 10, 3)] = -20.0 / DAY
    res = simulate_transient(d, fluid, p0=200 * BAR, dt=1 * DAY, n_steps=10, sources=q, rock_compressibility=5e-5 / BAR)
    ct = 1.5e-4 / BAR
    stored = np.sum(d.pore_volume * ct * (res.pressure[-1] - res.pressure[0]))
    assert np.isclose(stored, q.sum() * 10 * DAY, rtol=1e-10)


def test_transient_approaches_steady_state():
    d = _domain((10, 1, 1), 10.0, 10.0, 10.0, {"PORO": 0.2, "PERMX": 100})
    fluid = Fluid(CENTIPOISE, compressibility=1e-5 / BAR)
    bc = {"xmin": 150 * BAR, "xmax": 100 * BAR}
    steady = solve_steady(d, fluid, bc=bc, gravity=0.0)
    res = simulate_transient(d, fluid, p0=100 * BAR, dt=0.1 * DAY, n_steps=50, bc=bc, gravity=0.0)
    assert np.allclose(res.pressure[-1], steady.pressure, rtol=1e-6)
