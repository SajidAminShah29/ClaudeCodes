from pathlib import Path

import numpy as np
import pytest

from resim.cli import main
from resim.io import build_case, load_case, run_simulation, write_vtk

EXAMPLES = Path(__file__).parent.parent / "examples"


@pytest.mark.parametrize("name", ["layered_steady.yaml", "channelized_transient.yaml"])
def test_examples_run(name, capsys):
    assert main(["run", str(EXAMPLES / name)]) == 0
    assert "pressure:" in capsys.readouterr().out


def test_actnum_from_expression():
    case = build_case(
        {
            "grid": {"dims": [4, 4, 1], "dx": 10, "dy": 10, "dz": 5, "actnum": {"expr": "(I + J) > 2"}},
            "properties": {"PORO": 0.2, "PERMX": 10},
        }
    )
    assert case.grid.n_cells == 15 and case.domain.n_cells == 15


def test_steady_case_units():
    case = build_case(
        {
            "grid": {"dims": [10, 1, 1], "dx": 10, "dy": 10, "dz": 10},
            "properties": {"PORO": 0.2, "PERMX": 1000},
            "simulation": {"type": "steady", "fluid": {"viscosity": 1.0}, "boundary": {"xmin": 101, "xmax": 100}},
        }
    )
    res = run_simulation(case)
    q = 1000 * 9.869233e-16 * 100 / 1e-3 * 1e5 / 100  # m³/s
    assert np.allclose(res.flux, q)


def test_missing_sections_and_bad_grid(tmp_path):
    with pytest.raises(ValueError, match="grid"):
        build_case({"properties": {}})
    bad = tmp_path / "bad.yaml"
    bad.write_text("grid: {dims: [2, 2, 1], dx: 1, dy: 1}\nproperties: {PORO: 0.2, PERMX: 1}\n")
    assert main(["info", str(bad)]) == 1


def test_vtk_output(tmp_path):
    case = load_case(EXAMPLES / "layered_steady.yaml")
    out = tmp_path / "model.vtk"
    write_vtk(out, case.grid, {"PORO": case.props["PORO"]})
    data = out.read_bytes()
    assert data.startswith(b"# vtk DataFile") and f"CELL_DATA {case.grid.n_cells}".encode() in data
