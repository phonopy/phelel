"""Test for force constants creation."""

from pathlib import Path

import numpy as np

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


def _get_phelel_C111(phelel_yaml_filename: str) -> Phelel:
    phe_yml = PhelelYaml().read(cwd / phelel_yaml_filename)
    phelel = Phelel(
        phe_yml.unitcell,
        supercell_matrix=phe_yml.supercell_matrix,
        primitive_matrix=phe_yml.primitive_matrix,
        fft_mesh=[18, 18, 18],
        log_level=1,
    )
    phelel.dataset = phe_yml.dataset
    return phelel
