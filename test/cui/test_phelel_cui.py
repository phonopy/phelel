"""Tests of phelel command line script."""

from __future__ import annotations

import itertools
import os
import pathlib
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, fields

import h5py
import pytest

from phelel import load
from phelel.cui.phelel_script import main

cwd = pathlib.Path(__file__).parent
cwd_called = pathlib.Path.cwd()


@dataclass
class MockArgs:
    """Mock args of ArgumentParser."""

    filename: Sequence[str] | None = None
    conf_filename: str | None = None
    log_level: int | None = None
    cell_filename: str | None = None
    supercell_dimension: str | None = None
    create_derivatives: Sequence[str] | None = None
    is_displacement: bool = False
    is_plusminus_displacements: bool = False
    fft_mesh_numbers: str | None = None

    def __iter__(self):
        """Make self iterable to support in."""
        return (getattr(self, field.name) for field in fields(self))

    def __contains__(self, item):
        """Implement in operator."""
        return item in (field.name for field in fields(self))


@pytest.mark.parametrize(
    "filename,with_BORN_file",
    [
        (pathlib.Path("..") / "phelel_disp_C111.yaml", False),
        (pathlib.Path("..") / "phelel_disp_C111.yaml", True),
        (pathlib.Path("phelel_disp_C111_nac.yaml"), False),
    ],
)
def test_phelel_script(filename: pathlib.Path, with_BORN_file: bool):
    """Test phelel command."""
    born_str = """4.399652
5.7 0 0 0 5.7 0 0 0 5.7
0 0 0 0 0 0 0 0 0"""

    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        if with_BORN_file:
            born_file = pathlib.Path("BORN")
            with born_file.open("w", encoding="utf-8") as f:
                f.write(born_str)
        else:
            # Test not to read BORN directory.
            born_dir = pathlib.Path("BORN")
            os.mkdir(born_dir)

        try:
            # Check sys.exit(0)
            cell_filename = str(cwd / filename)
            argparse_control = _get_phelel_load_args(cell_filename=cell_filename)
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            if with_BORN_file:
                born_file.unlink()
            else:
                born_dir.rmdir()

            # Clean files created by phonopy-load script.
            for created_filename in ("phelel.yaml",):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


@pytest.mark.parametrize(
    "use_poscar,with_BORN_file", itertools.product([True, False], [True, False])
)
def test_phelel_script_create_derivatives(use_poscar: bool, with_BORN_file: bool):
    """Test phelel command.

    With POSCAR as input structure, ``phelel_disp.yaml`` has to be
    located at the current directory otherwise raise ``RuntimeError``, which
    is verified in this test. This feature is deprecated.

    With ``phelel_disp_C111.yaml`` as input structure,
    the computation will suceeded and ``phelel_params`` is created.

    """
    born_str = """4.399652
5.7 0 0 0 5.7 0 0 0 5.7
0 0 0 0 0 0 0 0 0"""

    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        if with_BORN_file:
            born_file = pathlib.Path("BORN")
            with born_file.open("w", encoding="utf-8") as f:
                f.write(born_str)
        else:
            # Test not to read BORN directory.
            born_dir = pathlib.Path("BORN")
            os.mkdir(born_dir)

        try:
            # Check sys.exit(0)
            dirname = cwd / ".." / "interface" / "vasp"
            if use_poscar:
                cell_filename = str(dirname / "POSCAR-unitcell_C111")
                supercell_dimension = "1 1 1"
            else:
                cell_filename = str(dirname / "phelel_disp_C111.yaml")
                supercell_dimension = None

            fft_mesh_numbers = "1 1 1"

            dispdirs = [str(dirname / "C111_disp-000"), str(dirname / "C111_disp-001")]
            argparse_control = _get_phelel_load_args(
                cell_filename=cell_filename,
                create_derivatives=dispdirs,
                supercell_dimenstion=supercell_dimension,
                fft_mesh_numbers=fft_mesh_numbers,
            )

            if use_poscar:
                with pytest.raises(RuntimeError) as excinfo:
                    main(**argparse_control)
            else:
                with pytest.raises(SystemExit) as excinfo:
                    main(**argparse_control)
                assert excinfo.value.code == 0

                _check_create_params_file(with_BORN_file)

                for created_filename in ("phelel_params.hdf5",):
                    file_path = pathlib.Path(created_filename)
                    assert file_path.exists()
                    file_path.unlink()

            if with_BORN_file:
                born_file.unlink()
            else:
                born_dir.rmdir()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


@pytest.mark.parametrize("is_plusminus_displacements", [True, False])
def test_phelel_script_create_displacements(is_plusminus_displacements: bool):
    """Test phelel command for creating displacements."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            # Check sys.exit(0)
            dirname = cwd / ".." / "interface" / "vasp"
            cell_filename = str(dirname / "POSCAR-unitcell_C111")
            supercell_dimension = "1 1 1"

            argparse_control = _get_phelel_load_args(
                cell_filename=cell_filename,
                supercell_dimenstion=supercell_dimension,
                is_displacement=True,
                is_plusminus_displacements=is_plusminus_displacements,
            )

            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            if is_plusminus_displacements:
                created_filenames = (
                    "phelel_disp.yaml",
                    "SPOSCAR",
                    "POSCAR-001",
                    "POSCAR-002",
                )
                for created_filename in ("POSCAR-003", "POSCAR_PH-003"):
                    file_path = pathlib.Path(created_filename)
                    assert not file_path.exists()
            else:
                created_filenames = (
                    "phelel_disp.yaml",
                    "SPOSCAR",
                    "POSCAR-001",
                )
                for created_filename in ("POSCAR-002",):
                    file_path = pathlib.Path(created_filename)
                    assert not file_path.exists()

            for created_filename in created_filenames:
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()

            _check_no_files()

        finally:
            os.chdir(original_cwd)


@pytest.mark.parametrize("with_BORN_file", [True, False])
def test_phelel_loader_with_reading_BORN(with_BORN_file: bool):
    """Test phelel loader with reading BORN file."""
    born_str = """4.399652
5.7 0 0 0 5.7 0 0 0 5.7
0 0 0 0 0 0 0 0 0"""

    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        if with_BORN_file:
            born_file = pathlib.Path("BORN")
            with born_file.open("w", encoding="utf-8") as f:
                f.write(born_str)

        dirname = cwd / ".." / "interface" / "vasp"
        cell_filename = str(dirname / "phelel_disp_C111.yaml")
        dispdirs = [str(dirname / "C111_disp-000"), str(dirname / "C111_disp-001")]
        phe = load(cell_filename, dir_names=dispdirs, fft_mesh=[1, 1, 1], log_level=2)

        if with_BORN_file:
            assert phe.nac_params is not None
            assert "born" in phe.nac_params
        else:
            assert phe.nac_params is None

        phe.save_hdf5(filename="phelel_params.hdf5")
        _check_create_params_file(with_BORN_file)

        file_path = pathlib.Path("phelel_params.hdf5")
        assert file_path.exists()
        file_path.unlink()

        if with_BORN_file:
            born_file.unlink()

        _check_no_files()

        os.chdir(original_cwd)


def _check_create_params_file(with_BORN_file: bool):
    """Check created phelel_params.hdf5 file."""
    if with_BORN_file:
        # extra keys: 'born_effective_charges' and 'dielectric_constant'
        ref_keys = set(
            [
                "Dij",
                "FFT_mesh",
                "atom_indices_in_derivatives",
                "born_effective_charges",
                "dDijdu",
                "dVdu",
                "dielectric_constant",
                "direct_rotations",
                "displacements_atom_indices",
                "displacements_vectors",
                "dqijdu",
                "force_constants",
                "grid_point",
                "lattice_point",
                "p2s_map",
                "primitive_lattice",
                "primitive_masses",
                "primitive_matrix",
                "primitive_numbers",
                "primitive_positions",
                "qij",
                "s2p_map",
                "shortest_vector_multiplicities",
                "shortest_vectors",
                "spacegroup_number",
                "supercell_lattice",
                "supercell_masses",
                "supercell_matrix",
                "supercell_numbers",
                "supercell_positions",
                "transformation_matrix",
                "unitcell_lattice",
                "unitcell_masses",
                "unitcell_numbers",
                "unitcell_positions",
            ]
        )
    else:
        ref_keys = set(
            [
                "Dij",
                "FFT_mesh",
                "atom_indices_in_derivatives",
                "dDijdu",
                "dVdu",
                "direct_rotations",
                "displacements_atom_indices",
                "displacements_vectors",
                "dqijdu",
                "force_constants",
                "grid_point",
                "lattice_point",
                "p2s_map",
                "primitive_lattice",
                "primitive_masses",
                "primitive_matrix",
                "primitive_numbers",
                "primitive_positions",
                "qij",
                "s2p_map",
                "shortest_vector_multiplicities",
                "shortest_vectors",
                "spacegroup_number",
                "supercell_lattice",
                "supercell_masses",
                "supercell_matrix",
                "supercell_numbers",
                "supercell_positions",
                "transformation_matrix",
                "unitcell_lattice",
                "unitcell_masses",
                "unitcell_numbers",
                "unitcell_positions",
            ]
        )

    with h5py.File("phelel_params.hdf5") as f:
        assert set(f) == ref_keys


def _get_phelel_load_args(
    cell_filename: str | None = None,
    supercell_dimenstion: str | None = None,
    create_derivatives: Sequence[str] | None = None,
    is_displacement: bool = False,
    is_plusminus_displacements: bool = False,
    fft_mesh_numbers: str | None = None,
):
    # Mock of ArgumentParser.args.
    mockargs = MockArgs(
        filename=[],
        log_level=1,
        cell_filename=str(cell_filename),
        supercell_dimension=supercell_dimenstion,
        create_derivatives=create_derivatives,
        is_displacement=is_displacement,
        is_plusminus_displacements=is_plusminus_displacements,
        fft_mesh_numbers=fft_mesh_numbers,
    )

    # See phonopy-load script.
    argparse_control = {"args": mockargs}
    return argparse_control


def _ls():
    current_dir = pathlib.Path(".")
    for file in current_dir.iterdir():
        print(file.name)


def _check_no_files():
    assert not list(pathlib.Path(".").iterdir())
