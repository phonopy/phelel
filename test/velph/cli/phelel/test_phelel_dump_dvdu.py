"""Tests of velph phelel dump-dvdu."""

from __future__ import annotations

import pathlib
import shutil
import warnings

import numpy as np
import pytest
from click.testing import CliRunner

from phelel import Phelel
from phelel.base.local_potential import get_potential_and_magnetic_parts
from phelel.exception import MagneticMomentsWarning
from phelel.file_IO import read_phelel_params_hdf5
from phelel.interface.phelel_yaml import PhelelYaml
from phelel.interface.vasp.derivatives import create_derivatives
from phelel.velph.cli.velph_cmd_root import cmd_root

vasp_dir = pathlib.Path(__file__).parents[3] / "interface" / "vasp"


@pytest.fixture(scope="module")
def phelel_params_C111_ncl(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Return phelel_params.hdf5 of non-collinear C-1x1x1 with fft_mesh 4x4x4.

    MagneticMomentsWarning for the non-collinear data without magnetic moments
    is ignored.

    """
    phe_yml = PhelelYaml().read(vasp_dir / "phelel_disp_C111.yaml")
    phe = Phelel(
        phe_yml.unitcell,
        supercell_matrix=phe_yml.supercell_matrix,
        primitive_matrix=phe_yml.primitive_matrix,
        fft_mesh=[4, 4, 4],
        log_level=0,
    )
    phe.dataset = phe_yml.dataset
    dirs = [vasp_dir / f"C111-ncl_disp-{i:03d}" for i in range(2)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MagneticMomentsWarning)
        create_derivatives(phe, dirs, subtract_rfs=True)
    filename = tmp_path_factory.mktemp("C111-ncl") / "phelel_params.hdf5"
    phe.save_hdf5(filename=filename)
    return filename


def _read_locpot(filename: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """Return the basis vectors in rows and the values of a LOCPOT-like file."""
    lines = filename.read_text().splitlines()
    lattice = float(lines[1]) * np.array([line.split() for line in lines[2:5]], float)
    i_mesh = next(
        i
        for i, line in enumerate(lines)
        if len(line.split()) == 3 and all(v.isdigit() for v in line.split())
    )
    values = np.array(" ".join(lines[i_mesh + 1 :]).split(), dtype=float)
    return lattice, values


def test_dump_dvdu(
    phelel_params_C111_ncl: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Files of v, v_x, v_y, v_z of the given atom are written in phelel/.

    dV/du is given for the atoms 1 and 5 of the supercell.  The values of a
    file have to be those of its atom, direction and part in
    phelel_params.hdf5, which differ between them.  The cell of the file is a
    multiple of the primitive cell, whose basis vectors are p2s_matrix^-1
    times those of the supercell.

    """
    (tmp_path / "phelel").mkdir()
    shutil.copy(phelel_params_C111_ncl, tmp_path / "phelel" / "phelel_params.hdf5")
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cmd_root, ["phelel", "dump-dvdu", "--atom", "5"])
    assert result.exit_code == 0, result.output
    assert result.output == (
        'dV/du of atom 5 was written in 12 files "phelel/locpot_viz-005-*.dat".\n'
    )
    written = sorted(p.name for p in (tmp_path / "phelel").glob("locpot_viz-*"))
    assert written == sorted(
        f"locpot_viz-005-{xyz}{suffix}.dat"
        for xyz in "xyz"
        for suffix in ("", "-vx", "-vy", "-vz")
    )

    dVdu, _, _, _ = read_phelel_params_hdf5(phelel_params_C111_ncl)
    i_atom = list(dVdu.atom_indices).index(4)
    plat = np.linalg.inv(dVdu.p2s_matrix).T @ dVdu.supercell.cell
    for i_dir, xyz in enumerate("xyz"):
        parts = get_potential_and_magnetic_parts(dVdu.dVdu[:, i_atom, i_dir])
        for name, part in parts.items():
            suffix = "" if name == "v" else f"-{name}"
            lattice, values = _read_locpot(
                tmp_path / "phelel" / f"locpot_viz-005-{xyz}{suffix}.dat"
            )
            multiplicity = np.rint(
                np.linalg.norm(lattice, axis=1) / np.linalg.norm(plat, axis=1)
            )
            np.testing.assert_allclose(lattice, np.diag(multiplicity) @ plat)
            expected = np.array(["%17.11e" % v for v in part], dtype=float)
            np.testing.assert_array_equal(np.unique(values), np.unique(expected))


def test_dump_dvdu_all_atoms(
    phelel_params_C111_ncl: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Without --atom, the files of all the atoms of dV/du are written."""
    (tmp_path / "phelel").mkdir()
    shutil.copy(phelel_params_C111_ncl, tmp_path / "phelel" / "phelel_params.hdf5")
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cmd_root, ["phelel", "dump-dvdu"])
    assert result.exit_code == 0, result.output
    assert result.output == (
        'dV/du of atom 1 was written in 12 files "phelel/locpot_viz-001-*.dat".\n'
        'dV/du of atom 5 was written in 12 files "phelel/locpot_viz-005-*.dat".\n'
    )
    assert len(list((tmp_path / "phelel").glob("locpot_viz-*"))) == 24


def test_dump_dvdu_errors(
    phelel_params_C111_ncl: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A missing hdf5 file or an atom without dV/du gives exit code 1."""
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cmd_root, ["phelel", "dump-dvdu"])
    assert result.exit_code == 1
    assert 'Error: "phelel/phelel_params.hdf5" does not exist.' in result.output

    (tmp_path / "phelel").mkdir()
    shutil.copy(phelel_params_C111_ncl, tmp_path / "phelel" / "phelel_params.hdf5")
    result = CliRunner().invoke(cmd_root, ["phelel", "dump-dvdu", "--atom", "2"])
    assert result.exit_code == 1
    assert "Error: Atom 2 is not one of the atoms of dV/du (1, 5)." in result.output
    assert not list((tmp_path / "phelel").glob("locpot_viz-*"))
