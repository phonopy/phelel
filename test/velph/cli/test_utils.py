"""Tests cli/utils.py."""

import io
import itertools
import pathlib
from collections.abc import Callable

import numpy as np
import pytest
from phonopy.interface.phonopy_yaml import read_cell_yaml
from phonopy.structure.atoms import PhonopyAtoms

from phelel.velph.cli.utils import write_jobs
from phelel.velph.config.schema import KpointsMesh
from phelel.velph.jobs.vasp_job import VaspJob
from phelel.velph.utils.structure import get_reduced_cell


def test_get_reduced_cell_bi2te3(
    helper_methods: Callable, bi2te3_prim_cell: PhonopyAtoms
):
    """Test of get_reduced_cell using Bi2Te3 primitive cell.

    Input cell is primitive rhombohedral.

    """
    ref_cell_str = """lattice:
- [    -2.221502746054457,     3.847755625320100,     0.000000000000000 ] # a
- [    -4.443005492108914,     0.000000000000000,     0.000000000000000 ] # b
- [    -2.221502746054457,     1.282585208440033,    10.479344379716814 ] # c
points:
- symbol: Bi # 1
  coordinates: [  0.398482502929835,  0.398482502929836,  0.804552491210493 ]
  mass: 208.980400
- symbol: Bi # 2
  coordinates: [  0.601517497070165,  0.601517497070165,  0.195447508789506 ]
  mass: 208.980400
- symbol: Te # 3
  coordinates: [  0.213380750346874,  0.213380750346874,  0.359857748959378 ]
  mass: 127.600000
- symbol: Te # 4
  coordinates: [  0.000000000000000,  0.000000000000000,  0.000000000000000 ]
  mass: 127.600000
- symbol: Te # 5
  coordinates: [  0.786619249653126,  0.786619249653126,  0.640142251040622 ]
  mass: 127.600000
"""
    ref_cell = read_cell_yaml(io.StringIO(ref_cell_str))
    reduced_cell = get_reduced_cell(bi2te3_prim_cell)
    ref_lengths = np.linalg.norm(ref_cell.cell, axis=1)
    reduced_lengths = np.linalg.norm(reduced_cell.cell, axis=1)

    # Check lenghts of basis vectors.
    is_found = False
    for ref_perm in itertools.permutations(ref_lengths):
        for reduced_perm in itertools.permutations(reduced_lengths):
            if np.allclose(ref_perm, reduced_perm):
                is_found = True
                break
    assert is_found

    if np.allclose(ref_cell.cell, reduced_cell.cell):
        helper_methods.compare_positions_with_order(
            reduced_cell.scaled_positions, ref_cell.scaled_positions, ref_cell.cell
        )
    else:
        msg = (
            "Reduced cell algorithm may be sensitive to the numerical precision of "
            "computers. Therefore this failure might happen due to it. Please "
            "recondier this test."
        )
        raise AssertionError(msg)


@pytest.mark.parametrize("with_potcar", [True, False])
def test_write_jobs(
    with_potcar: bool,
    nacl_cell: PhonopyAtoms,
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
):
    """Each directory is shown, and a missing POTCAR is said once, first.

    A file copied as POSCAR is shown with the directory.

    """
    contcar = tmp_path / "CONTCAR"
    contcar.write_text("contcar\n")
    copied_files = {"POTCAR": contcar} if with_potcar else {}
    kpoints = {"KPOINTS": KpointsMesh(mesh=(2, 2, 2))}
    jobs = [
        VaspJob(
            directory=tmp_path / name,
            cell=nacl_cell,
            incar={"nsw": 0},
            kpoints=kpoints,
            copied_files=copied_files,
            poscar_source=contcar if name == "b" else None,
        )
        for name in ("a", "b")
    ]
    write_jobs(jobs)
    lines = capsys.readouterr().out.splitlines()
    expected = [
        f'VASP input files were written in "{tmp_path / "a"}".',
        f'"{contcar}" was copied to "{tmp_path / "b" / "POSCAR"}".',
        f'VASP input files were written in "{tmp_path / "b"}".',
    ]
    if not with_potcar:
        expected.insert(
            0,
            '"POTCAR" was not found in the current directory, so the VASP input '
            "files are written without POTCAR.",
        )
    assert lines == expected
    assert (tmp_path / "b" / "POSCAR").read_text() == "contcar\n"
