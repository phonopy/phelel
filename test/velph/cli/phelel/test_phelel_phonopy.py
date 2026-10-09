"""Tests of velph phelel phonopy."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from phonopy.structure.atoms import PhonopyAtoms

import phelel.velph.cli.phelel.phonopy as phelel_phonopy
from phelel import Phelel
from phelel.velph.cli.phelel.phonopy import create_phonopy_yaml
from phelel.velph.cli.utils import get_num_digits


def _write_phelel_disp_yaml(path: pathlib.Path, with_phonon: bool = True) -> Phelel:
    """Write phelel_disp.yaml of NaCl, with a larger phonon supercell or not."""
    a = 5.69
    fcc = [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
    cell = PhonopyAtoms(
        cell=np.eye(3) * a,
        symbols=["Na"] * 4 + ["Cl"] * 4,
        scaled_positions=fcc + [[(x + 0.5) % 1 for x in p] for p in fcc],
    )
    phe = Phelel(
        cell,
        supercell_matrix=np.eye(3, dtype=int),
        phonon_supercell_matrix=np.diag([2, 2, 2]) if with_phonon else None,
        primitive_matrix="F",
    )
    phe.generate_displacements()
    if with_phonon:
        phe.generate_phonon_displacements()
    path.write_text(str(phe.to_phelel_yaml()))
    return phe


def test_create_phonopy_yaml_reads_phonon_supercells(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    """Forces are read from the phonon supercells, ph-disp-*, of dir_name."""
    phe = _write_phelel_disp_yaml(tmp_path / "phelel_disp.yaml")
    assert phe.phonon_supercell is not None
    assert phe.phonon_supercells_with_displacements is not None
    n_atoms = len(phe.phonon_supercell)
    filenames: list[pathlib.Path] = []

    def _read_forces(vasprun_filenames, supercell, subtract_rfs, log_level):
        filenames.extend(vasprun_filenames)
        assert len(supercell) == n_atoms
        return [np.zeros((n_atoms, 3))] * (len(vasprun_filenames) - 1)

    monkeypatch.setattr(phelel_phonopy, "read_forces_from_vasprunxmls", _read_forces)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "velph.toml").write_text("")
    create_phonopy_yaml(
        pathlib.Path("velph.toml"), pathlib.Path("phelel_disp.yaml"), "phelel"
    )

    disps = phe.phonon_supercells_with_displacements
    nd = get_num_digits(disps)
    assert [f.parent.name for f in filenames] == [
        f"ph-disp-{i:0{nd}d}" for i in range(len(disps) + 1)
    ]
    assert all(f.parent.parent == pathlib.Path("phelel") for f in filenames)
    assert (tmp_path / "phelel" / "phonopy_params.yaml").exists()


def test_create_phonopy_yaml_reads_phelel_supercells(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    """Without phonon supercell, forces are read from disp-* of dir_name."""
    phe = _write_phelel_disp_yaml(tmp_path / "phelel_disp.yaml", with_phonon=False)
    assert phe.supercells_with_displacements is not None
    n_atoms = len(phe.supercell)
    filenames: list[pathlib.Path] = []

    def _read_forces(vasprun_filenames, supercell, subtract_rfs, log_level):
        filenames.extend(vasprun_filenames)
        assert len(supercell) == n_atoms
        return [np.zeros((n_atoms, 3))] * (len(vasprun_filenames) - 1)

    monkeypatch.setattr(phelel_phonopy, "read_forces_from_vasprunxmls", _read_forces)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "velph.toml").write_text("")
    create_phonopy_yaml(
        pathlib.Path("velph.toml"), pathlib.Path("phelel_disp.yaml"), "phelel"
    )

    disps = phe.supercells_with_displacements
    nd = get_num_digits(disps)
    assert [str(f.parent) for f in filenames] == [
        f"phelel/disp-{i:0{nd}d}" for i in range(len(disps) + 1)
    ]
