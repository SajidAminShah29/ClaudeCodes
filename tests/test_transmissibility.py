import numpy as np
import pytest

from resim.grid import build_cartesian
from resim.preproc import build_domain
from resim.props import build_properties
from resim.units import MILLIDARCY


def _domain(dims, dx, dy, dz, props, edits=None, **kw):
    g = build_cartesian(dims, dx, dy, dz)
    return build_domain(g, build_properties(g, props, edits), **kw)


def test_homogeneous_1d_transmissibility_is_kA_over_dx():
    d = _domain((5, 1, 1), 10.0, 4.0, 2.0, {"PORO": 0.2, "PERMX": 100})
    assert np.allclose(d.trans, 100 * MILLIDARCY * 8.0 / 10.0)
    # boundary half-transmissibility uses half the cell length
    assert np.allclose(d.bnd_trans[d.boundary("xmin")], 100 * MILLIDARCY * 8.0 / 5.0)


def test_two_cell_harmonic_average():
    d = _domain((2, 1, 1), [2.0, 6.0], 1.0, 1.0, {"PORO": 0.2, "PERMX": {"source": "array", "values": [10, 30]}})
    # series resistance: (dx1/2)/k1 + (dx2/2)/k2
    expected = 1.0 / (1.0 / (10 * MILLIDARCY) + 3.0 / (30 * MILLIDARCY))
    assert np.isclose(d.trans[0], expected)


def test_layered_vertical_transmissibility():
    d = _domain((1, 1, 2), 1.0, 1.0, [1.0, 3.0], {"PORO": 0.2, "PERMX": 100, "PERMZ": {"source": "layers", "values": [5, 1]}})
    expected = 1.0 / (0.5 / (5 * MILLIDARCY) + 1.5 / (1 * MILLIDARCY))
    assert np.isclose(d.trans[0], expected)


def test_pore_volume_ntg_and_multipliers():
    edits = [{"op": "equals", "field": "MULTX", "value": 0.5, "box": [1, 1, 1, 1, 1, 1]}]
    d = _domain((2, 1, 1), 10.0, 10.0, 10.0, {"PORO": 0.25, "PERMX": 100, "NTG": 0.8, "MULTPV": 2.0}, edits)
    assert np.allclose(d.pore_volume, 1000 * 0.25 * 0.8 * 2.0)
    # NTG scales horizontal perm; MULTX of cell 1 scales its +X face
    assert np.isclose(d.trans[0], 0.5 * 0.8 * 100 * MILLIDARCY * 100 / 10)


def test_ntg_does_not_affect_vertical_flow():
    d = _domain((1, 1, 2), 10.0, 10.0, 10.0, {"PORO": 0.25, "PERMX": 100, "NTG": 0.5})
    assert np.isclose(d.trans[0], 100 * MILLIDARCY * 100 / 10)


def test_minpv_and_zero_porosity_remove_cells():
    props = {"PORO": {"source": "array", "values": [0.2, 0.0, 0.2, 0.001]}, "PERMX": 10}
    d = _domain((4, 1, 1), 1.0, 1.0, 1.0, props)
    assert d.n_cells == 3 and d.n_conn == 1  # only cells 3-4 remain connected
    d = _domain((4, 1, 1), 1.0, 1.0, 1.0, props, minpv=0.01)
    assert d.n_cells == 2 and d.n_conn == 0
    assert d.cell_global.tolist() == [0, 2]


def test_zero_permeability_gives_zero_transmissibility():
    d = _domain((3, 1, 1), 1.0, 1.0, 1.0, {"PORO": 0.2, "PERMX": {"source": "array", "values": [10, 0, 10]}})
    assert np.allclose(d.trans, 0.0)


def test_cell_index_lookup():
    d = _domain((3, 2, 2), 1.0, 1.0, 1.0, {"PORO": 0.2, "PERMX": 1})
    assert tuple(d.ijk[d.cell_index(3, 2, 2)]) == (2, 1, 1)
    with pytest.raises(ValueError):
        d.cell_index(4, 1, 1)
