"""Test for force constants creation."""

from __future__ import annotations

import copy
import lzma
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from phonopy import Phonopy

from phelel import Phelel
from phelel.interface.phelel_yaml import PhelelYaml
from phelel.interface.vasp.derivatives import (
    create_derivatives,
    read_files,
    read_forces_from_vasprunxmls,
)

cwd = Path(__file__).parent


def test_read_forces_from_vasprunxmls_NaCl111(phelel_empty_NaCl111: Phelel):
    """Test by NaCl conv. unit cell 1x1x1."""
    phe = phelel_empty_NaCl111
    filenames = [
        cwd / f"vasprun_NaCl111_disp{num}.xml.xz" for num in ["000", "001", "002"]
    ]
    forces = read_forces_from_vasprunxmls(
        filenames,
        phe.supercell,
        subtract_rfs=True,
        log_level=0,
    )

    ref = [
        [
            [-0.04119964, -0.0, 0.0],
            [-0.00271915, 0.0, 0.0],
            [0.00518832, 0.0, 0.0],
            [0.00518832, 0.0, 0.0],
            [0.02826969, 0.0, -0.0],
            [-0.00242275, 0.0, -0.0],
            [0.00384761, 0.0, 0.0],
            [0.00384761, 0.0, -0.0],
        ],
        [
            [0.02930525, -0.0, 0.0],
            [-0.00158137, -0.0, 0.0],
            [0.00468871, 0.0, 0.0],
            [0.00468871, 0.0, 0.0],
            [-0.06772416, 0.0, -0.0],
            [-0.00328894, -0.0, 0.0],
            [0.0169559, 0.0, 0.0],
            [0.0169559, 0.0, 0.0],
        ],
    ]
    np.testing.assert_allclose(ref, forces, atol=1e-8)


def test_read_files_C111():
    """Test rading files with C-1x1x1."""
    phelel = _get_phelel_C111("phelel_disp_C111.yaml")
    dir_names = [cwd / "C111_disp-000", cwd / "C111_disp-001"]
    _ = read_files(phelel, dir_names, subtract_rfs=True, log_level=1)


def test_create_derivatives_C111():
    """Test creating derivatives with C-1x1x1.

    Check force constants have the full matrix shape.

    """
    phelel = _get_phelel_C111("phelel_disp_C111.yaml")
    dir_names = [cwd / "C111_disp-000", cwd / "C111_disp-001"]
    create_derivatives(phelel, dir_names, subtract_rfs=True, log_level=1)
    fc = phelel.force_constants
    assert fc.shape[0] == fc.shape[1]


@pytest.mark.parametrize(
    "dataset_key,atol", [("first_atoms", 1e-6), ("displacements", 1e-3)]
)
def test_create_derivatives_C111_phonon_supercell_C222(dataset_key: str, atol: float):
    """Test creating derivatives with C-1x1x1 and the phonon supercell 2x2x2.

    The force constants have to be those of Phonopy with the 2x2x2 supercell
    from the same vasprun.xml, and the phonon dataset must not be replaced by
    the dataset of the supercells.  With "displacements", the displacement of
    C222_disp-001 is given twice, so the number of the phonon supercells
    differs from that of the supercells.  The force constants are then computed
    by symfc, which gives a difference of about 2e-4 from the traditional
    calculator for the largest element of about 47.

    """
    dir_names = [cwd / "C111_disp-000", cwd / "C111_disp-001"]
    calc_dir_names = [cwd / "C222_disp-000", cwd / "C222_disp-001"]
    phelel = _get_phelel_C111(
        "phelel_disp_C111.yaml", phonon_supercell_matrix=np.diag([2, 2, 2])
    )
    ph = _get_phonopy_C222(phelel)
    assert ph.dataset is not None and ph.supercells_with_displacements is not None
    if dataset_key == "first_atoms":
        phelel.phonon_dataset = dict(copy.deepcopy(ph.dataset))
        phonon_dir_names = calc_dir_names
    else:
        displacement = (
            ph.supercells_with_displacements[0].positions - ph.supercell.positions
        )
        phelel.phonon_dataset = {"displacements": np.array([displacement] * 2)}
        phonon_dir_names = calc_dir_names + calc_dir_names[1:]
    _produce_force_constants_C222(ph)

    create_derivatives(phelel, dir_names + phonon_dir_names, subtract_rfs=True)
    assert phelel.phonon_dataset is not None
    assert dataset_key in phelel.phonon_dataset
    if dataset_key == "first_atoms":
        assert phelel.phonon_dataset["natom"] == len(ph.supercell)
    assert len(phelel.forces) == len(phonon_dir_names) - 1
    assert phelel.force_constants is not None and ph.force_constants is not None
    np.testing.assert_allclose(phelel.force_constants, ph.force_constants, atol=atol)


def test_create_derivatives_C111_phonon_supercell_C222_vaspout_h5(tmp_path: Path):
    """Test reading forces of the phonon supercells from vaspout.h5.

    vaspout.h5 is read when it is found in all the directories of the phonon
    supercells.  h5py does not read a compressed file, so vaspout.h5 is
    decompressed into tmp_path, where vasprun.xml is not found.  The force
    constants have to be those from the forces in vasprun.xml, from which they
    differ by about 4e-7.

    """
    phonon_dir_names = []
    for name in ("C222_disp-000", "C222_disp-001"):
        (tmp_path / name).mkdir()
        with lzma.open(cwd / name / "vaspout.h5.xz") as f:
            (tmp_path / name / "vaspout.h5").write_bytes(f.read())
        phonon_dir_names.append(tmp_path / name)
    phelel = _get_phelel_C111(
        "phelel_disp_C111.yaml", phonon_supercell_matrix=np.diag([2, 2, 2])
    )
    ph = _get_phonopy_C222(phelel)
    assert ph.dataset is not None
    phelel.phonon_dataset = dict(copy.deepcopy(ph.dataset))
    _produce_force_constants_C222(ph)

    dir_names = [cwd / "C111_disp-000", cwd / "C111_disp-001"]
    create_derivatives(phelel, dir_names + phonon_dir_names, subtract_rfs=True)
    assert phelel.force_constants is not None and ph.force_constants is not None
    np.testing.assert_allclose(phelel.force_constants, ph.force_constants, atol=1e-5)


def test_create_derivatives_C111_phonon_supercell_without_dataset():
    """A phonon supercell without the phonon dataset is an error."""
    phelel = _get_phelel_C111(
        "phelel_disp_C111.yaml", phonon_supercell_matrix=np.diag([2, 2, 2])
    )
    dir_names = [cwd / "C111_disp-000", cwd / "C111_disp-001"]
    with pytest.raises(RuntimeError, match="phonon dataset"):
        create_derivatives(phelel, dir_names, subtract_rfs=True)


def test_read_files_C111_ncl():
    """Test reading files with non-collinear case of C-1x1x1."""
    phelel = _get_phelel_C111("phelel_disp_C111.yaml")
    dir_names = [cwd / "C111-ncl_disp-000", cwd / "C111-ncl_disp-001"]
    _ = read_files(phelel, dir_names, subtract_rfs=True, log_level=1)


def test_create_derivatives_C111_ncl():
    """Test creating derivatives with non-collinear case of C-1x1x1.

    Check force constants have the full matrix shape.

    """
    phelel = _get_phelel_C111("phelel_disp_C111.yaml")
    dir_names = [cwd / "C111-ncl_disp-000", cwd / "C111-ncl_disp-001"]
    create_derivatives(phelel, dir_names, subtract_rfs=True, log_level=1)
    fc = phelel.force_constants
    assert fc.shape[0] == fc.shape[1]


def _get_phelel_C111(
    phelel_yaml_filename: str, phonon_supercell_matrix: NDArray | None = None
) -> Phelel:
    phe_yml = PhelelYaml().read(cwd / phelel_yaml_filename)
    phelel = Phelel(
        phe_yml.unitcell,
        supercell_matrix=phe_yml.supercell_matrix,
        primitive_matrix=phe_yml.primitive_matrix,
        phonon_supercell_matrix=phonon_supercell_matrix,
        fft_mesh=[18, 18, 18],
        log_level=1,
    )
    phelel.dataset = phe_yml.dataset
    return phelel


def _get_phonopy_C222(phelel: Phelel) -> Phonopy:
    """Return Phonopy with the displacement of C222_disp-001."""
    ph = Phonopy(
        phelel.unitcell,
        supercell_matrix=np.diag([2, 2, 2]),
        primitive_matrix=phelel.primitive_matrix,
    )
    ph.generate_displacements(distance=0.01)
    return ph


def _produce_force_constants_C222(ph: Phonopy) -> None:
    """Produce force constants from vasprun.xml of C222_disp-*."""
    ph.forces = read_forces_from_vasprunxmls(
        [cwd / name / "vasprun.xml.xz" for name in ("C222_disp-000", "C222_disp-001")],
        ph.supercell,
        subtract_rfs=True,
    )
    ph.produce_force_constants()
