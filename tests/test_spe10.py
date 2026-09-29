import numpy as np
import pytest

from resim.grid import build_cartesian
from resim.props import build_properties
from resim.props import spe10


@pytest.fixture(scope="module")
def spe10_files(tmp_path_factory):
    """Synthetic files in SPE10 layout: value encodes (component, i, j, k)."""
    tmp_path = tmp_path_factory.mktemp("spe10")
    nx, ny, nz = spe10.SPE10_DIMS
    i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    code = (i + 100 * j + 100_000 * k).ravel(order="F").astype(float)
    np.savetxt(tmp_path / "spe_phi.dat", (code * 1e-9).reshape(-1, 6), fmt="%.10e")
    perm = np.concatenate([code + 1e8, code + 2e8, code + 3e8])
    np.savetxt(tmp_path / "spe_perm.dat", perm.reshape(-1, 6), fmt="%.10e")
    return tmp_path


def test_extract_subbox(spe10_files):
    phi = spe10.read_spe10_file(spe10_files / "spe_phi.dat")
    perm = spe10.read_spe10_file(spe10_files / "spe_perm.dat")
    box = spe10.extract(perm, "permy", (2, 3, 2), origin=(5, 10, 40))
    # first cell of the box is 0-based (4, 9, 39)
    assert np.isclose(box[0], 2e8 + 4 + 900 + 3_900_000)
    assert box.size == 12
    assert np.isclose(spe10.extract(phi, "poro", (1, 1, 1), origin=(60, 220, 85))[0], (59 + 21900 + 8_400_000) * 1e-9)
    with pytest.raises(ValueError):
        spe10.extract(perm, "permx", (2, 2, 2), origin=(60, 1, 1))
    with pytest.raises(ValueError):
        spe10.extract(phi, "permx", (1, 1, 1))


def test_spe10_property_source(spe10_files):
    g = build_cartesian((3, 2, 2), *spe10.SPE10_CELL_SIZE)
    spec = {
        "PORO": {"source": "spe10", "path": "spe_phi.dat", "component": "poro", "origin": [1, 1, 35]},
        "PERMX": {"source": "spe10", "path": "spe_perm.dat", "component": "permx", "origin": [1, 1, 35]},
    }
    p = build_properties(g, spec, base_dir=spe10_files)
    assert np.isclose(p["PERMX"][0], 1e8 + 3_400_000)
    assert np.isclose(p["PERMX"][-1], 1e8 + 2 + 100 + 3_500_000)
