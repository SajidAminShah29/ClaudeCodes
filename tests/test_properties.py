import numpy as np
import pytest

from resim.grid import build_cartesian
from resim.props import build_properties
from resim.props.expr import evaluate, expression_names
from resim.props.grdecl import parse_grdecl


@pytest.fixture
def grid():
    return build_cartesian((4, 3, 2), 10.0, 10.0, 1.0)


def test_constant_layers_and_defaults(grid):
    p = build_properties(grid, {"PORO": 0.25, "PERMX": {"source": "layers", "values": [100, 10]}})
    assert np.all(p["PORO"] == 0.25)
    assert np.all(p["PERMX"][:12] == 100) and np.all(p["PERMX"][12:] == 10)
    assert np.array_equal(p["PERMY"], p["PERMX"])
    assert np.all(p["NTG"] == 1.0) and np.all(p["MULTZ"] == 1.0)
    assert p["SATNUM"].dtype == np.int64 and np.all(p["SATNUM"] == 1)


def test_dependency_order_and_expressions(grid):
    spec = {
        "PERMZ": {"expr": "0.1 * PERMX"},  # defined before its dependency
        "PERMX": {"expr": "where(K == 1, 200, 20)"},
        "PORO": {"expr": "0.1 + 0.01 * I"},
    }
    p = build_properties(grid, spec)
    assert np.allclose(p["PERMZ"], 0.1 * p["PERMX"])
    assert np.isclose(p["PORO"][3], 0.14)  # I = 4


def test_circular_and_undefined_dependencies(grid):
    with pytest.raises(ValueError, match="circular"):
        build_properties(grid, {"PORO": {"expr": "PERMX"}, "PERMX": {"expr": "PORO"}})
    with pytest.raises(ValueError, match="undefined"):
        build_properties(grid, {"PORO": 0.2, "PERMX": {"copy": "FOO"}})


def test_box_edits(grid):
    edits = [
        {"op": "multiply", "field": "PERMX", "value": 0.01, "box": [1, 4, 1, 3, 2, 2]},
        {"op": "equals", "field": "MULTZ", "value": 0.0, "box": [2, 2, 1, 1, 1, 1]},
        {"op": "equals", "field": "SATNUM", "value": 2, "box": [1, 2, 1, 3, 1, 2]},
    ]
    p = build_properties(grid, {"PORO": 0.2, "PERMX": 100}, edits)
    assert np.all(p["PERMX"][:12] == 100) and np.allclose(p["PERMX"][12:], 1.0)
    # PERMY defaults to the edited PERMX (edits are applied before defaults)
    assert np.allclose(p["PERMY"], p["PERMX"])
    assert p["MULTZ"][1] == 0 and p["MULTZ"].sum() == grid.n_global - 1
    assert p["SATNUM"][0] == 2 and p["SATNUM"][2] == 1


def test_validation_reports_cells(grid):
    with pytest.raises(ValueError, match=r"PORO: 1 active cells.*\(2,1,1\)"):
        build_properties(grid, {"PORO": 0.2, "PERMX": 1}, [{"op": "equals", "field": "PORO", "value": 1.5, "box": [2, 2, 1, 1, 1, 1]}])
    with pytest.raises(ValueError, match="required"):
        build_properties(grid, {"PORO": 0.2})


def test_inactive_cells_are_not_validated():
    g = build_cartesian((2, 1, 1), 1.0, 1.0, 1.0, actnum=[1, 0])
    p = build_properties(g, {"PORO": {"source": "array", "values": [0.2, -9]}, "PERMX": 10})
    assert p.active("PORO").tolist() == [0.2]


def test_file_source_grdecl_and_npy(grid, tmp_path):
    (tmp_path / "poro.grdecl").write_text("-- porosity\nPORO\n 12*0.2 -- layer 1\n 12*0.3 /\n")
    np.save(tmp_path / "perm.npy", np.arange(24, dtype=float) + 1)
    p = build_properties(
        grid,
        {"PORO": {"source": "file", "path": "poro.grdecl"}, "PERMX": {"source": "file", "path": "perm.npy"}},
        base_dir=tmp_path,
    )
    assert np.allclose(p["PORO"][:12], 0.2) and np.allclose(p["PORO"][12:], 0.3)
    assert p["PERMX"][23] == 24


def test_grdecl_parser_handles_flags_strings_and_bare_numbers():
    rec = parse_grdecl("NOECHO\nSPECGRID\n 2 2 1 1 F /\nPERMX\n 3*1.5 2.0D0 /\n")
    assert rec["NOECHO"].size == 0
    assert rec["SPECGRID"] == ["2", "2", "1", "1", "F"]
    assert rec["PERMX"].tolist() == [1.5, 1.5, 1.5, 2.0]
    assert parse_grdecl("1 2\n3")[None].tolist() == [1, 2, 3]


def test_poro_perm_per_facies(grid):
    spec = {
        "FACIES": {"source": "layers", "values": [1, 2]},
        "PORO": 0.2,
        "PERMX": {"source": "poro_perm", "from": "PORO", "by": "FACIES", "params": {1: {"a": 10, "b": 0}, 2: {"a": 5, "b": 0}}},
    }
    p = build_properties(grid, spec)
    assert np.allclose(p["PERMX"][:12], 100.0) and np.allclose(p["PERMX"][12:], 10.0)


def test_expression_safety():
    assert expression_names("where(PORO > 0.1, PERMX, 0)") == {"PORO", "PERMX"}
    with pytest.raises(ValueError):
        evaluate("__import__('os')", {})
    with pytest.raises(ValueError):
        evaluate("PORO.real", {"PORO": np.ones(2)})
