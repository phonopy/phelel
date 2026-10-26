"""Tests of the derivative of the local potential."""

from __future__ import annotations

import itertools
import pathlib

import numpy as np
import pytest
from phonopy.structure.symmetry import Symmetry

import phelel
from phelel.api_phelel import Phelel
from phelel.base.local_potential import (
    DeltaLocalPotential,
    DLocalPotential,
    LocalPotentialInterpolationNUFFT,
    get_grid_points,
)

cwd = pathlib.Path(__file__).parent

# Reciprocal lattice vectors (h, k, l) with 0 < h^2 + k^2 + l^2 <= 4.  The set
# is closed under the point group of a cubic lattice.
_HKL = np.array(
    [
        hkl
        for hkl in itertools.product(range(-2, 3), repeat=3)
        if 0 < sum(n * n for n in hkl) <= 4
    ]
)


def _periodic_function(frac: np.ndarray, i_dir: int) -> np.ndarray:
    """Return a function periodic in the supercell, different for each direction.

    ``frac`` are coordinates in the supercell basis vectors.

    """
    n = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])[i_dir]
    m = np.array([0, 1, 2])
    return np.cos(2 * np.pi * frac @ n) + 0.5 * np.sin(2 * np.pi * frac @ m)


def _read_locpot(filename: pathlib.Path) -> tuple[list[int], np.ndarray]:
    """Return the mesh and the values of a LOCPOT-like file.

    The mesh is the first line of three integers after the structure.  The
    values are read separated by white space, as VESTA reads them, and are
    returned with the shape (n_z, n_y, n_x).

    """
    lines = filename.read_text().splitlines()
    i_mesh = next(
        i
        for i, line in enumerate(lines)
        if len(line.split()) == 3 and all(v.isdigit() for v in line.split())
    )
    mesh = [int(v) for v in lines[i_mesh].split()]
    values = np.array(" ".join(lines[i_mesh + 1 :]).split(), dtype=float)
    return mesh, values.reshape(mesh[::-1])


def test_visualize(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, capsys):
    """dV/du on the grid points is written at the right positions.

    C-1x1x1 has a non-diagonal matrix from the primitive cell to the supercell,
    so the file is written for a multiple of the primitive cell that contains
    the supercell.  The value at each point of the file has to be the function
    given on the grid points, evaluated at the position of that point.  The
    second atom for the derivatives is the fifth atom of the supercell, which
    has to be the number in the file names and in the printed message.

    """
    monkeypatch.chdir(tmp_path)
    phe = phelel.load(cwd / ".." / "phelel_disp_C111.yaml")
    fft_mesh = np.array([4, 4, 4])
    atom_indices = phe.primitive.p2s_map
    dlp = DLocalPotential(
        fft_mesh, phe.p2s_matrix, phe.supercell, atom_indices=atom_indices
    )
    grid_points, _ = get_grid_points(fft_mesh, phe.p2s_matrix)
    dlp.grid_points = grid_points
    dVdu = np.zeros((1, len(atom_indices), 3, len(grid_points)), dtype="complex128")
    i_atom = 1
    for i_dir in range(3):
        dVdu[0, i_atom, i_dir] = _periodic_function(grid_points, i_dir)
    dlp.dVdu = dVdu

    dlp.visualize(phe.primitive, i_atom)

    atom_number = atom_indices[i_atom] + 1
    assert atom_number == 5
    assert f'"locpot_viz-{atom_number:03d}-{{x,y,z}}.dat"' in capsys.readouterr().out
    for i_dir, xyz in enumerate("xyz"):
        filename = tmp_path / f"locpot_viz-{atom_number:03d}-{xyz}.dat"
        mesh, values = _read_locpot(filename)
        # Positions of the points of the file in the supercell basis vectors.
        multiplicity = np.array(mesh) // fft_mesh
        lattice_viz = phe.primitive.cell.T @ np.diag(multiplicity)
        index = np.array(list(np.ndindex(*mesh[::-1])))[:, ::-1]
        frac = (np.linalg.inv(phe.supercell.cell.T) @ lattice_viz @ (index / mesh).T).T
        expected = _periodic_function(frac, i_dir).reshape(mesh[::-1])
        np.testing.assert_allclose(values, expected, atol=1e-10)


def test_get_grid_points():
    """Grid points are the FFT mesh of the primitive cell at each lattice point.

    The order of the points is that of phelel_params.hdf5: x runs fastest, then
    y and z, and the points of one lattice point come together, in the order of
    the lattice points.  C-1x1x1 has a non-diagonal matrix from the primitive
    cell to the supercell, and the mesh differs along the three axes.

    """
    phe = phelel.load(cwd / ".." / "phelel_disp_C111.yaml")
    p2s_matrix = phe.p2s_matrix
    fft_mesh = [2, 3, 4]
    grid_points, lattice_points = get_grid_points(fft_mesh, p2s_matrix)

    np.testing.assert_array_equal(
        lattice_points, [[0, 0, 0], [-1, 0, 0], [1, -1, 0], [0, -1, 0]]
    )
    n_mesh = np.prod(fft_mesh)
    assert grid_points.shape == (len(lattice_points) * n_mesh, 3)
    mesh_points = np.array(
        [
            [i / fft_mesh[0], j / fft_mesh[1], k / fft_mesh[2]]
            for k in range(fft_mesh[2])
            for j in range(fft_mesh[1])
            for i in range(fft_mesh[0])
        ]
    )
    # Grid points in the primitive cell basis vectors.
    points_prim = grid_points @ p2s_matrix.T
    for i, lattice_point in enumerate(lattice_points):
        np.testing.assert_allclose(
            points_prim[i * n_mesh : (i + 1) * n_mesh],
            mesh_points + lattice_point,
            atol=1e-12,
        )
    # No two grid points are the same point of the supercell.
    diff = grid_points[:, None, :] - grid_points[None, :, :]
    is_same = np.all(np.abs(diff - np.rint(diff)) < 1e-8, axis=-1)
    np.testing.assert_array_equal(is_same, np.eye(len(grid_points), dtype=bool))


def _mesh_fractions(mesh: tuple[int, int, int]) -> np.ndarray:
    """Return the points of a VASP mesh in the supercell basis vectors.

    ``mesh`` is (n_z, n_y, n_x), the shape of a local potential read from
    VASP.  The returned array has the shape (n_z, n_y, n_x, 3), and the point
    [k, j, i] is (i / n_x, j / n_y, k / n_z).

    """
    n_z, n_y, n_x = mesh
    k, j, i = np.meshgrid(range(n_z), range(n_y), range(n_x), indexing="ij")
    return np.stack([i / n_x, j / n_y, k / n_z], axis=-1)


def _anisotropic_function(frac: np.ndarray) -> np.ndarray:
    """Return a periodic function that differs along the three axes.

    It has sine terms, so it changes when the argument changes its sign.

    """
    x, y, z = frac[..., 0], frac[..., 1], frac[..., 2]
    return (
        np.cos(2 * np.pi * x)
        + 0.5 * np.sin(2 * np.pi * (2 * y - z))
        + 0.25 * np.sin(2 * np.pi * (x + 3 * z))
    )


def test_run_without_symmetry(capsys: pytest.CaptureFixture):
    """dV/du is the change of the potential divided by the displacement.

    With the identity as the only symmetry operation, the displacement of atom
    1 along x gives dV/du_x of atom 1, and dV/du_y and dV/du_z are zero.  The
    change of the potential is given on a VASP mesh with different numbers of
    points along the three axes, and dV/du_x is evaluated at the grid points by
    NUFFT.  The function differs along the axes and has sine terms, so mixing
    up the axes or the sign of the Fourier transform changes the result.  With
    verbose=False, nothing is printed.

    """
    phe = phelel.load(cwd / ".." / "phelel_disp_C111.yaml")
    supercell = phe.supercell
    u = 0.01
    mesh = (10, 8, 6)
    V_loc_per = np.zeros((1,) + mesh, dtype="complex128")
    V_loc_disp = (u * _anisotropic_function(_mesh_fractions(mesh)))[None].astype(
        "complex128"
    )
    dlp = DLocalPotential(
        [4, 4, 4],
        phe.p2s_matrix,
        supercell,
        symmetry=Symmetry(supercell, is_symmetry=False),
        atom_indices=np.array([0]),
        verbose=False,
    )
    dlp.run(V_loc_per, [V_loc_disp], [{"number": 0, "displacement": [u, 0, 0]}])

    assert dlp.dVdu is not None
    expected = _anisotropic_function(dlp.grid_points)
    atol = 1e-5 * np.abs(expected).max()
    np.testing.assert_allclose(dlp.dVdu[0, 0, 0], expected, atol=atol)
    np.testing.assert_allclose(dlp.dVdu[0, 0, 1:], 0, atol=atol)
    assert capsys.readouterr().out == ""


def _model_potential_gradient(
    frac: np.ndarray, position: np.ndarray, lattice: np.ndarray, i_dir: int
) -> np.ndarray:
    """Return -d phi(r - R) / d r_alpha of a model atomic potential phi.

    phi(r) = sum_G exp(-|G|^2 sigma^2 / 2) cos(G . r) over the reciprocal
    lattice vectors of _HKL, with sigma = 0.5 Angstrom.  The coefficient depends
    only on |G|, and the set of G is closed under the point group of the cubic
    supercell, so phi is unchanged by the rotations of the point group about
    the atom.  ``frac`` and ``position`` are in the supercell basis vectors, and
    ``lattice`` has the basis vectors as rows.

    """
    G = 2 * np.pi * _HKL @ np.linalg.inv(lattice).T
    coef = np.exp(-(G**2).sum(axis=1) * 0.5**2 / 2)
    phase = 2 * np.pi * (frac - position) @ _HKL.T
    return np.sin(phase) @ (coef * G[:, i_dir])


def test_run_with_symmetry():
    """dV/du of all atoms and directions are obtained from one displacement.

    The model potential of diamond is the sum of phi(r - R) of the atoms, with
    phi of _model_potential_gradient.  The displacement u of atom 1 along x
    changes it by -u d phi(r - R_1) / d x exactly, because the change is given
    linear in u.  Only the change of the potential is used, so the potential of
    the perfect supercell is given as zero and that of the displaced supercell
    as the change alone.  From this one displacement, the symmetrization has to
    give dV/du_alpha of every atom b as -d phi(r - R_b) / d r_alpha, for the
    three directions.  The space group of diamond has operations with fractional
    translations, and the site symmetry of atom 1 maps x to y and z.

    """
    phe, V_loc_per, V_loc_disp, displacement = _diamond_model_input()
    supercell = phe.supercell
    lattice = supercell.cell
    positions = supercell.scaled_positions
    dlp = DLocalPotential([6, 6, 6], phe.p2s_matrix, supercell, verbose=False)
    dlp.run(V_loc_per, [V_loc_disp], [displacement])

    assert dlp.dVdu is not None
    np.testing.assert_array_equal(dlp.atom_indices, np.arange(len(supercell)))
    expected = np.array(
        [
            [
                _model_potential_gradient(dlp.grid_points, position, lattice, i_dir)
                for i_dir in range(3)
            ]
            for position in positions
        ]
    )
    atol = 1e-5 * np.abs(expected).max()
    np.testing.assert_allclose(dlp.dVdu[0], expected, atol=atol)


def test_interpolation_with_atom_indices_in_any_order():
    """dV/du of each atom is that of the atom, in the order of atom_indices.

    The site-symmetry operations are collected for the equivalent atoms in
    ascending order, and they have to be matched to atom_indices given in
    another order.

    """
    phe, V_loc_per, V_loc_disp, displacement = _diamond_model_input()
    supercell = phe.supercell
    lpi = LocalPotentialInterpolationNUFFT(
        [6, 6, 6],
        phe.p2s_matrix,
        supercell,
        phe.symmetry,
        atom_indices=np.array([1, 0]),
        verbose=False,
    )
    lpi.delta_Vs = [DeltaLocalPotential(V_loc_per, V_loc_disp, displacement)]
    for _ in lpi:
        pass

    np.testing.assert_array_equal(lpi.atom_indices_returned, [1, 0])
    assert lpi.dVdu is not None
    for dVdu, i_atom in zip(lpi.dVdu[0], lpi.atom_indices_returned, strict=True):
        expected = np.array(
            [
                _model_potential_gradient(
                    lpi.grid_points,
                    supercell.scaled_positions[i_atom],
                    supercell.cell,
                    i_dir,
                )
                for i_dir in range(3)
            ]
        )
        atol = 1e-5 * np.abs(expected).max()
        np.testing.assert_allclose(dVdu, expected, atol=atol)


def test_run_with_undisplaced_atoms():
    """Atoms not equivalent to any displaced atom are an error.

    In NaCl, the displacement of a Na atom gives no dV/du of the Cl atoms,
    which would be left zero.

    """
    phe = phelel.load(cwd / ".." / "phelel_disp_NaCl111.yaml")
    supercell = phe.supercell
    assert supercell.symbols == ["Na"] * 4 + ["Cl"] * 4
    V_loc = np.zeros((1, 12, 12, 12), dtype="complex128")
    dlp = DLocalPotential([4, 4, 4], phe.p2s_matrix, supercell, verbose=False)
    with pytest.raises(ValueError, match="atoms 5, 6, 7, 8 of the supercell"):
        dlp.run(V_loc, [V_loc], [{"number": 0, "displacement": [0.01, 0, 0]}])


def _diamond_model_input() -> tuple[Phelel, np.ndarray, np.ndarray, dict]:
    """Return the input of the model potential of diamond.

    The potential of the perfect supercell is zero, and that of the supercell
    with atom 1 displaced along x by u is -u d phi(r - R_1) / d x, given on a
    VASP mesh.  See test_run_with_symmetry.

    """
    phe = phelel.load(cwd / ".." / "phelel_disp_C111.yaml")
    supercell = phe.supercell
    u = 0.01
    mesh = (18, 16, 14)
    frac = _mesh_fractions(mesh)
    position = supercell.scaled_positions[0]
    gradient = _model_potential_gradient(frac, position, supercell.cell, 0)
    V_loc_per = np.zeros((1,) + mesh, dtype="complex128")
    V_loc_disp = (u * gradient)[None].astype("complex128")
    return phe, V_loc_per, V_loc_disp, {"number": 0, "displacement": [u, 0, 0]}
