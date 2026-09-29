import numpy as np
import pytest

from resim.grid import BOUNDARY_TAGS, build_cartesian


def test_volumes_and_face_counts():
    g = build_cartesian((3, 3, 3), 10.0, 20.0, 5.0)
    assert g.n_cells == 27
    assert np.isclose(g.cell_volume.sum(), 27 * 10 * 20 * 5)
    # 3 directions * (2 interfaces * 9 cells) internal, 6 sides * 9 boundary
    assert g.internal_faces.sum() == 54
    assert g.boundary_faces.sum() == 54


def test_single_cell_has_only_boundary_faces():
    g = build_cartesian((1, 1, 1), 1.0, 1.0, 1.0)
    assert g.internal_faces.sum() == 0
    assert sorted(g.face_tag.tolist()) == list(range(6))


def test_eclipse_ordering_and_centroids():
    g = build_cartesian((4, 3, 2), [1, 2, 3, 4], 5.0, [1.0, 2.0], tops=100.0)
    # Global index i + nx*(j + ny*k); cell (i=2, j=1, k=1) 0-based
    idx = 2 + 4 * (1 + 3 * 1)
    x, y, z = g.cell_centroid[idx]
    assert np.isclose(x, 1 + 2 + 1.5)
    assert np.isclose(y, 5 + 2.5)
    assert np.isclose(z, 100 + 1 + 1.0)
    assert tuple(g.ijk[idx]) == (2, 1, 1)


def test_depths_follow_tops_map():
    tops = np.array([1000.0, 1010.0, 1005.0, 1015.0])  # nx*ny, i fastest
    g = build_cartesian((2, 2, 3), 1.0, 1.0, 2.0, tops=tops)
    z = g.cell_centroid[:, 2].reshape((2, 2, 3), order="F")
    assert np.allclose(z[:, :, 0], [[1001, 1006], [1011, 1016]])
    assert np.all(np.diff(z, axis=2) > 0)


def test_internal_face_geometry():
    g = build_cartesian((2, 1, 1), [2.0, 4.0], 3.0, 5.0)
    f = np.flatnonzero(g.internal_faces)[0]
    assert tuple(g.face_cells[f]) == (0, 1)
    assert np.isclose(g.face_area[f], 15.0)
    assert np.allclose(g.face_centroid[f], [2.0, 1.5, 2.5])
    assert np.allclose(g.face_normal[f], [1, 0, 0])


def test_boundary_normals_point_outward():
    g = build_cartesian((2, 2, 2), 1.0, 1.0, 1.0)
    for f in np.flatnonzero(g.boundary_faces):
        c = g.face_cells[f, 0]
        outward = g.face_centroid[f] - g.cell_centroid[c]
        assert np.dot(outward, g.face_normal[f]) > 0
        assert BOUNDARY_TAGS[g.face_tag[f]]


def test_actnum_removes_cells_and_their_faces():
    actnum = np.ones(3)
    actnum[1] = 0
    g = build_cartesian((3, 1, 1), 1.0, 1.0, 1.0, actnum=actnum)
    assert g.n_cells == 2
    assert g.internal_faces.sum() == 0
    assert list(g.cell_global) == [0, 2]
    assert list(g.global_to_active) == [0, -1, 1]


def test_invalid_input():
    with pytest.raises(ValueError):
        build_cartesian((2, 2, 2), [1.0, 2.0, 3.0], 1.0, 1.0)
    with pytest.raises(ValueError):
        build_cartesian((2, 1, 1), 0.0, 1.0, 1.0)
