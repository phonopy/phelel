"""Test for force constants creation."""

from __future__ import annotations

import copy
import lzma
from pathlib import Path

import h5py
import numpy as np
import pytest
from numpy.typing import NDArray
from phonopy import Phonopy

import phelel
import phelel.interface.vasp.derivatives as derivatives
from phelel import Phelel
from phelel.exception import MagneticMomentsError, MagneticMomentsWarning
from phelel.interface.phelel_yaml import PhelelYaml
from phelel.interface.vasp.derivatives import (
    create_derivatives,
    read_files,
    read_vasp_forces,
)

cwd = Path(__file__).parent


def test_read_vasp_forces_NaCl111(phelel_empty_NaCl111: Phelel):
    """Test by NaCl conv. unit cell 1x1x1.

    With the file of the perfect supercell, its residual forces are subtracted.
    Without it, the forces of all the files are returned as they are.  The
    residual forces of NaCl are zero by symmetry, so the subtraction is checked
    by giving disp001 as the file of the perfect supercell.

    """
    phe = phelel_empty_NaCl111
    filenames = [
        cwd / f"vasprun_NaCl111_disp{num}.xml.xz" for num in ["000", "001", "002"]
    ]
    forces = read_vasp_forces(
        filenames[1:],
        phe.supercell,
        perfect_filename=filenames[0],
        log_level=0,
    )
    raw_forces = read_vasp_forces(filenames, phe.supercell)
    assert len(raw_forces) == 3
    np.testing.assert_allclose(np.array(raw_forces[1:]) - raw_forces[0], forces)
    assert np.abs(raw_forces[1]).max() > 1e-3
    (forces_from_disp001,) = read_vasp_forces(
        filenames[2:], phe.supercell, perfect_filename=filenames[1]
    )
    np.testing.assert_allclose(forces_from_disp001, raw_forces[2] - raw_forces[1])

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


@pytest.mark.parametrize(
    "vaspout_names", [("C222_disp-000", "C222_disp-001"), ("C222_disp-000",)]
)
def test_create_derivatives_C111_phonon_supercell_C222_vaspout_h5(
    tmp_path: Path, vaspout_names: tuple[str, ...]
):
    """Test reading forces of the phonon supercells from vaspout.h5.

    vaspout.h5 is read in each directory where it is found, and vasprun.xml in
    the others.  h5py does not read a compressed file, so vaspout.h5 of the
    directories of vaspout_names is decompressed into tmp_path, where
    vasprun.xml is not found.  The force constants have to be those from the
    forces in vasprun.xml, from which they differ by about 4e-7.

    """
    phonon_dir_names = []
    for name in ("C222_disp-000", "C222_disp-001"):
        if name not in vaspout_names:
            phonon_dir_names.append(cwd / name)
            continue
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


def test_read_files_without_force_files(tmp_path: Path):
    """Without vaspout.h5 and vasprun.xml, the directory is named in the error."""
    phelel = _get_phelel_C111("phelel_disp_C111.yaml")
    (tmp_path / "disp-001").mkdir()
    dir_names = [cwd / "C111_disp-000", cwd / "C111_disp-001"]
    with pytest.raises(FileNotFoundError, match="disp-001"):
        read_files(
            phelel, dir_names, phonon_dir_names=[dir_names[0], tmp_path / "disp-001"]
        )


def _make_dir(
    dir_name: Path, old_format_dir: Path | None = None, has_vaspout_h5: bool = False
) -> Path:
    """Make a directory with the files of the old format and an empty vaspout.h5.

    The files of the old format are links to those in old_format_dir.

    """
    dir_name.mkdir()
    if old_format_dir is not None:
        for path in old_format_dir.iterdir():
            (dir_name / path.name).symlink_to(path)
    if has_vaspout_h5:
        (dir_name / "vaspout.h5").touch()
    return dir_name


def _read(reader: str, dir_names: list[Path]) -> list:
    inwap_per = derivatives.read_inwap_yaml(cwd / "C111_disp-000" / "inwap.yaml")
    if reader == "local_potentials":
        loc_pots = derivatives._read_local_potentials(dir_names, inwap_per)
        assert loc_pots is not None
        return loc_pots
    Dijs, qijs = derivatives._read_PAW_strength_and_overlap(dir_names, inwap_per)
    return Dijs + qijs


@pytest.fixture
def no_vaspout_h5_reader(monkeypatch: pytest.MonkeyPatch):
    """Make reading vaspout.h5 an error."""

    def reader(*args, **kwargs):
        raise AssertionError("vaspout.h5 is read.")

    monkeypatch.setattr(derivatives, "read_local_potential_vaspouth5", reader)
    monkeypatch.setattr(derivatives, "read_PAW_Dij_qij_vaspouth5", reader)


@pytest.mark.parametrize("reader", ["local_potentials", "PAW_strength_and_overlap"])
def test_read_old_format_in_place_of_vaspout_h5(
    reader: str, tmp_path: Path, no_vaspout_h5_reader
):
    """The old format is read when every directory has LOCAL-POTENTIAL.bin.

    Every directory also has vaspout.h5, which is an empty file, and reading it
    is an error.  The data have to be those of the old format.

    """
    old_format_dirs = [cwd / "C111_disp-000", cwd / "C111_disp-001"]
    dir_names = [
        _make_dir(tmp_path / d.name, old_format_dir=d, has_vaspout_h5=True)
        for d in old_format_dirs
    ]
    for data, ref in zip(
        _read(reader, dir_names), _read(reader, old_format_dirs), strict=True
    ):
        np.testing.assert_array_equal(data, ref)


@pytest.mark.parametrize(
    "reader,filename",
    [
        ("local_potentials", "LOCAL-POTENTIAL.bin"),
        ("PAW_strength_and_overlap", "PAW-STRENGTH.bin"),
    ],
)
def test_read_with_old_format_in_some_directories(
    reader: str, filename: str, tmp_path: Path, no_vaspout_h5_reader
):
    """The old format found in only some of the directories is an error.

    Every directory has vaspout.h5, and only the first one has the files of the
    old format.  Reading the old format of the first directory and vaspout.h5
    of the others would mix the two formats, so the file read by the reader and
    the directory without it have to be named in the error.

    """
    dir_names = [
        _make_dir(
            tmp_path / "disp-000",
            old_format_dir=cwd / "C111_disp-000",
            has_vaspout_h5=True,
        ),
        _make_dir(tmp_path / "disp-001", has_vaspout_h5=True),
    ]
    with pytest.raises(RuntimeError, match=f"{filename}.*disp-001"):
        _read(reader, dir_names)


@pytest.mark.parametrize("reader", ["local_potentials", "PAW_strength_and_overlap"])
def test_read_without_files(reader: str, tmp_path: Path):
    """Neither the old format nor vaspout.h5 is an error."""
    dir_names = [_make_dir(tmp_path / "disp-000"), _make_dir(tmp_path / "disp-001")]
    with pytest.raises(RuntimeError, match="Neither.*disp-000"):
        _read(reader, dir_names)


@pytest.mark.parametrize("reader", ["local_potentials", "PAW_strength_and_overlap"])
def test_read_with_vaspout_h5_in_some_directories(
    reader: str, tmp_path: Path, no_vaspout_h5_reader
):
    """vaspout.h5 found in only some of the directories is an error.

    No directory has the files of the old format, and only the first one has
    vaspout.h5.  The directory without vaspout.h5 has to be named in the error.

    """
    dir_names = [
        _make_dir(tmp_path / "disp-000", has_vaspout_h5=True),
        _make_dir(tmp_path / "disp-001"),
    ]
    with pytest.raises(RuntimeError, match="vaspout.h5.*disp-001"):
        _read(reader, dir_names)


@pytest.mark.parametrize("subtract_rfs", [True, False])
def test_read_files_perfect_supercell_forces(
    subtract_rfs: bool, monkeypatch: pytest.MonkeyPatch
):
    """The file of the first directory is the perfect supercell only to subtract.

    The residual forces of C-1x1x1 are nearly zero by symmetry, so whether they
    are subtracted is checked by the arguments given to read_vasp_forces.

    """
    calls = []

    def _read_vasp_forces(filenames, supercell, perfect_filename=None, log_level=0):
        calls.append((filenames, perfect_filename))
        return [np.zeros((len(supercell), 3))] * len(filenames)

    monkeypatch.setattr(derivatives, "read_vasp_forces", _read_vasp_forces)
    phelel = _get_phelel_C111("phelel_disp_C111.yaml")
    dir_names = [cwd / "C111_disp-000", cwd / "C111_disp-001"]
    read_files(phelel, dir_names, subtract_rfs=subtract_rfs)
    ((filenames, perfect_filename),) = calls
    assert [f.parent.name for f in filenames] == ["C111_disp-001"]
    if subtract_rfs:
        assert perfect_filename.parent.name == "C111_disp-000"
    else:
        assert perfect_filename is None


def test_read_vasp_forces_wrong_number_of_atoms():
    """Forces of a supercell of a different number of atoms are an error."""
    phelel = _get_phelel_C111("phelel_disp_C111.yaml")
    with pytest.raises(ValueError, match="Forces of the 8 atoms"):
        read_vasp_forces(
            [cwd / f"C222_disp-{i:03d}" / "vasprun.xml.xz" for i in range(2)],
            phelel.supercell,
        )


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

    Check force constants have the full matrix shape. phelel_disp_C111.yaml
    has no magnetic moments, so the non-collinear calculation gives a warning.

    """
    phelel = _get_phelel_C111("phelel_disp_C111.yaml")
    dir_names = [cwd / "C111-ncl_disp-000", cwd / "C111-ncl_disp-001"]
    with pytest.warns(MagneticMomentsWarning, match="ncdij=4"):
        create_derivatives(phelel, dir_names, subtract_rfs=True, log_level=1)
    fc = phelel.force_constants
    assert fc.shape[0] == fc.shape[1]


def test_create_derivatives_C111_ncl_with_collinear_magmoms():
    """Collinear magnetic moments are rejected for the non-collinear case."""
    phe_yml = PhelelYaml().read(cwd / "phelel_disp_C111.yaml")
    unitcell = phe_yml.unitcell.copy()
    unitcell.magnetic_moments = [1.0] * len(unitcell)
    phelel = Phelel(
        unitcell,
        supercell_matrix=phe_yml.supercell_matrix,
        primitive_matrix=phe_yml.primitive_matrix,
        fft_mesh=[18, 18, 18],
    )
    phelel.dataset = phe_yml.dataset
    dir_names = [cwd / "C111-ncl_disp-000", cwd / "C111-ncl_disp-001"]
    with pytest.raises(MagneticMomentsError, match="ncdij=4"):
        create_derivatives(phelel, dir_names, subtract_rfs=True)


def test_run_derivatives_without_phonon_dataset_in_input():
    """The phonon dataset stored in Phelel is used when the input has none.

    The dataset of the supercells (C-1x1x1) must not be given to the phonon
    supercell (2x2x2), whose number of atoms is different.

    """
    dir_names = [cwd / "C111_disp-000", cwd / "C111_disp-001"]
    phonon_dir_names = [cwd / "C222_disp-000", cwd / "C222_disp-001"]
    phelel = _get_phelel_C111(
        "phelel_disp_C111.yaml", phonon_supercell_matrix=np.diag([2, 2, 2])
    )
    ph = _get_phonopy_C222(phelel)
    assert ph.dataset is not None
    phelel.phonon_dataset = dict(copy.deepcopy(ph.dataset))
    _produce_force_constants_C222(ph)

    phe_input = read_files(
        phelel, dir_names, phonon_dir_names=phonon_dir_names, subtract_rfs=True
    )
    phe_input.phonon_dataset = None
    phelel.run_derivatives(phe_input)
    assert phelel.phonon_dataset is not None
    assert phelel.phonon_dataset["natom"] == len(ph.supercell)
    assert phelel.force_constants is not None and ph.force_constants is not None
    np.testing.assert_allclose(phelel.force_constants, ph.force_constants, atol=1e-6)


def test_run_derivatives_without_any_phonon_dataset():
    """Without the phonon dataset anywhere, the phonon supercell is an error."""
    dir_names = [cwd / "C111_disp-000", cwd / "C111_disp-001"]
    phonon_dir_names = [cwd / "C222_disp-000", cwd / "C222_disp-001"]
    phelel = _get_phelel_C111(
        "phelel_disp_C111.yaml", phonon_supercell_matrix=np.diag([2, 2, 2])
    )
    phelel.phonon_dataset = dict(copy.deepcopy(_get_phonopy_C222(phelel).dataset))
    phe_input = read_files(
        phelel, dir_names, phonon_dir_names=phonon_dir_names, subtract_rfs=True
    )
    phe_input.phonon_dataset = None
    phelel.phonon_dataset = None
    with pytest.raises(RuntimeError, match="Phelel.phonon_dataset"):
        phelel.run_derivatives(phe_input)


@pytest.mark.parametrize("name", ["C", "Mg"])
def test_create_derivatives_vaspout_h5(name: str):
    """Derivatives from vaspout.h5 are equal to the reference values.

    The data are diamond (conventional cell, 8 atoms) and hcp Mg (2 atoms), with
    vaspout.h5 reduced to the data that phelel reads (see
    derivatives_vaspout/README.md).  The reference dVdu and dDijdu were computed
    by phelel from the same files with fft_mesh stored in reference.hdf5.  qij
    of PAW depends only on the species, and qij of all supercells of these data
    are equal, so dqijdu is zero.

    """
    data_dir = cwd / "derivatives_vaspout" / name
    with h5py.File(data_dir / "reference.hdf5") as f:
        fft_mesh = [int(n) for n in f.attrs["fft_mesh"]]
        ref = {key: f[key][()] for key in ("dVdu", "dDijdu")}
    phe = phelel.load(data_dir / "phelel_disp.yaml", fft_mesh=fft_mesh, is_nac=False)
    create_derivatives(phe, sorted(data_dir.glob("disp-*")))
    assert phe.dVdu is not None
    for key, value in (("dVdu", phe.dVdu.dVdu), ("dDijdu", phe.dDijdu.dDijdu)):
        atol = 1e-6 * np.abs(ref[key]).max()
        np.testing.assert_allclose(value, ref[key], rtol=0, atol=atol)
    np.testing.assert_array_equal(phe.dDijdu.dqijdu, 0)


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
    ph.forces = read_vasp_forces(
        [cwd / "C222_disp-001" / "vasprun.xml.xz"],
        ph.supercell,
        perfect_filename=cwd / "C222_disp-000" / "vasprun.xml.xz",
    )
    ph.produce_force_constants()
