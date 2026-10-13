"""Tests of velph phelel phonopy."""

from __future__ import annotations

import pathlib
from collections.abc import Callable

import click
import numpy as np
import phonopy
import pytest
import tomli_w
from phonopy.structure.atoms import PhonopyAtoms

import phelel.velph.cli.phelel.phonopy as phelel_phonopy
from phelel import Phelel
from phelel.velph.cli.phelel.init import run_init
from phelel.velph.cli.phelel.phonopy import create_phonopy_yaml
from phelel.velph.cli.utils import get_num_digits
from phelel.velph.config.schema import VelphConfig


def _write_phelel_disp_yaml(path: pathlib.Path, with_phonon: bool = True) -> Phelel:
    """Write phelel_disp.yaml of NaCl, with a larger phonon supercell or not.

    NAC params are written in it.

    """
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
    phe.nac_params = {
        "born": np.array([np.eye(3) * 1.1, -np.eye(3) * 1.1]),
        "dielectric": np.eye(3) * 2.4,
        "factor": 14.4,
    }
    path.write_text(str(phe.to_phelel_yaml()))
    return phe


def _write_force_files(
    directory: pathlib.Path,
    prefix: str,
    cells: list[PhonopyAtoms],
    vaspout_ids: tuple[int, ...] = (),
) -> None:
    """Write empty force files of the perfect supercell and cells.

    The files are written in <directory>/<prefix>-*, vaspout.h5 for vaspout_ids
    and vasprun.xml for the others.

    """
    nd = get_num_digits(cells)
    for i in range(len(cells) + 1):
        d = directory / f"{prefix}-{i:0{nd}d}"
        d.mkdir(parents=True)
        (d / ("vaspout.h5" if i in vaspout_ids else "vasprun.xml")).touch()


@pytest.mark.parametrize("with_vasp_phonopy", [True, False])
def test_create_phonopy_yaml_with_phonon_supercell(
    with_vasp_phonopy: bool,
    nacl_velph_toml_v014: Callable[[], dict],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    """With the phonon supercell, forces are read from disp-* of dir_name.

    phonopy_params.yaml is that of the supercells of phelel with their forces
    and the NAC params of phelel_disp.yaml.  With [vasp.phonopy], it is said
    that the phonopy calculation is in "phonopy".

    """
    phe = _write_phelel_disp_yaml(tmp_path / "phelel_disp.yaml")
    assert phe.phonon_supercell is not None
    assert phe.supercells_with_displacements is not None
    assert phe.nac_params is not None
    n_atoms = len(phe.supercell)
    assert len(phe.phonon_supercell) != n_atoms
    disps = phe.supercells_with_displacements
    _write_force_files(tmp_path / "phelel", "disp", disps)
    filenames: list[pathlib.Path] = []
    forces = np.random.default_rng(5).standard_normal((len(disps), n_atoms, 3))

    def _read_forces(vasp_filenames, supercell, perfect_filename=None):
        filenames.extend([perfect_filename, *vasp_filenames])
        assert len(supercell) == n_atoms
        return list(forces)

    monkeypatch.setattr(phelel_phonopy, "read_vasp_forces", _read_forces)
    monkeypatch.chdir(tmp_path)
    toml_dict = nacl_velph_toml_v014()
    if not with_vasp_phonopy:
        del toml_dict["vasp"]["phonopy"]
    (tmp_path / "velph.toml").write_text(tomli_w.dumps(toml_dict))
    create_phonopy_yaml(
        pathlib.Path("velph.toml"), pathlib.Path("phelel_disp.yaml"), "phelel"
    )

    nd = get_num_digits(disps)
    assert [str(f) for f in filenames] == [
        f"phelel/disp-{i:0{nd}d}/vasprun.xml" for i in range(len(disps) + 1)
    ]
    ph = phonopy.load(tmp_path / "phelel" / "phonopy_params.yaml", produce_fc=False)
    assert len(ph.supercell) == n_atoms
    np.testing.assert_allclose(ph.forces, forces)
    assert ph.nac_params is not None
    np.testing.assert_allclose(ph.nac_params["born"], phe.nac_params["born"])
    out = capsys.readouterr().out
    assert "made from the phelel supercells" in out
    assert ('calculation with the [phonopy] supercell is in "phonopy"' in out) is (
        with_vasp_phonopy
    )


def test_create_phonopy_yaml_without_force_files(
    nacl_velph_toml_v014: Callable[[], dict],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A directory without vaspout.h5 and vasprun.xml is click.ClickException."""
    phe = _write_phelel_disp_yaml(tmp_path / "phelel_disp.yaml", with_phonon=False)
    assert phe.supercells_with_displacements is not None
    _write_force_files(tmp_path / "phelel", "disp", phe.supercells_with_displacements)
    nd = get_num_digits(phe.supercells_with_displacements)
    (tmp_path / "phelel" / f"disp-{1:0{nd}d}" / "vasprun.xml").unlink()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "velph.toml").write_text(tomli_w.dumps(nacl_velph_toml_v014()))
    with pytest.raises(click.ClickException, match=f"disp-{1:0{nd}d}"):
        create_phonopy_yaml(
            pathlib.Path("velph.toml"), pathlib.Path("phelel_disp.yaml"), "phelel"
        )


def test_create_phonopy_yaml_reads_phelel_supercells(
    nacl_velph_toml_v014: Callable[[], dict],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Without phonon supercell, forces are read from disp-* of dir_name.

    vaspout.h5 is read in place of vasprun.xml where it is found.

    """
    phe = _write_phelel_disp_yaml(tmp_path / "phelel_disp.yaml", with_phonon=False)
    assert phe.supercells_with_displacements is not None
    n_atoms = len(phe.supercell)
    disps = phe.supercells_with_displacements
    _write_force_files(tmp_path / "phelel", "disp", disps, vaspout_ids=(1,))
    filenames: list[pathlib.Path] = []

    def _read_forces(vasp_filenames, supercell, perfect_filename=None):
        filenames.extend([perfect_filename, *vasp_filenames])
        assert len(supercell) == n_atoms
        return [np.zeros((n_atoms, 3))] * len(vasp_filenames)

    monkeypatch.setattr(phelel_phonopy, "read_vasp_forces", _read_forces)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "velph.toml").write_text(tomli_w.dumps(nacl_velph_toml_v014()))
    create_phonopy_yaml(
        pathlib.Path("velph.toml"), pathlib.Path("phelel_disp.yaml"), "phelel"
    )

    nd = get_num_digits(disps)
    assert [str(f) for f in filenames] == [
        f"phelel/disp-{i:0{nd}d}/" + ("vaspout.h5" if i == 1 else "vasprun.xml")
        for i in range(len(disps) + 1)
    ]


def test_create_phonopy_yaml_site_mixture_merge(
    site_mixture_velph_toml: Callable[[bool], dict],
    code_init_config: Callable[[dict], VelphConfig],
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """With the merge scheme, forces are on the atoms of the unmerged supercell."""
    phe = run_init(
        code_init_config(site_mixture_velph_toml(False)), current_directory=tmp_path
    )
    assert phe.unmerged_supercell is not None
    assert phe.supercells_with_displacements is not None
    n_atoms = len(phe.unmerged_supercell)
    (tmp_path / "phelel_disp.yaml").write_text(str(phe.to_phelel_yaml()))
    _write_force_files(tmp_path / "phelel", "disp", phe.supercells_with_displacements)
    rng = np.random.default_rng(3)

    def _read_forces(vasp_filenames, supercell, perfect_filename=None):
        assert len(supercell) == n_atoms
        return list(rng.standard_normal((len(vasp_filenames), n_atoms, 3)))

    monkeypatch.setattr(phelel_phonopy, "read_vasp_forces", _read_forces)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "velph.toml").write_text(tomli_w.dumps(site_mixture_velph_toml(False)))
    create_phonopy_yaml(
        pathlib.Path("velph.toml"), pathlib.Path("phelel_disp.yaml"), "phelel"
    )

    ph = phonopy.load(tmp_path / "phelel" / "phonopy_params.yaml", produce_fc=False)
    assert ph.site_mixture_scheme == "merge"
    assert ph.unmerged_supercell is not None
    assert len(ph.unmerged_supercell) == n_atoms
    assert ph.forces.shape == (len(phe.supercells_with_displacements), n_atoms, 3)
