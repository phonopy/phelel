"""Tests of the derivative of the local potential."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

import phelel
from phelel.base.local_potential import DLocalPotential, get_grid_points

cwd = pathlib.Path(__file__).parent


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
    atom_indices = phe.atom_indices_in_derivatives
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
