"""Tests of phelel command line script."""

from __future__ import annotations

import os
import pathlib
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, fields

import pytest

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
    "filename",
    [
        pathlib.Path("..") / "phelel_disp_C111.yaml",
        pathlib.Path("phelel_disp_C111_nac.yaml"),
    ],
)
def test_phelel_script(filename: pathlib.Path):
    """Test phelel command."""
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

        try:
            # Check sys.exit(0)
            cell_filename = str(cwd / filename)
            argparse_control = _get_phelel_load_args(cell_filename=cell_filename)
            with pytest.raises(SystemExit) as excinfo:
                main(**argparse_control)
            assert excinfo.value.code == 0

            # Clean files created by phonopy-load script.
            for created_filename in ("phelel.yaml",):
                file_path = pathlib.Path(created_filename)
                assert file_path.exists()
                file_path.unlink()
        finally:
            os.chdir(original_cwd)


@pytest.mark.parametrize("use_poscar", [True, False])
def test_phelel_script_create_derivatives(use_poscar: bool):
    """Test phelel command.

    With POSCAR as input structure, ``phelel_disp.yaml`` has to be
    located at the current directory otherwise raise ``RuntimeError``, which
    is verified in this test. This feature is deprecated.

    With ``phelel_disp_C111.yaml`` as input structure,
    the computation will suceeded and ``phelel_params`` is created.

    """
    with tempfile.TemporaryDirectory() as temp_dir:
        original_cwd = pathlib.Path.cwd()
        os.chdir(temp_dir)

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

                for created_filename in ("phelel_params.hdf5",):
                    file_path = pathlib.Path(created_filename)
                    assert file_path.exists()
                    file_path.unlink()
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
        finally:
            os.chdir(original_cwd)


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
