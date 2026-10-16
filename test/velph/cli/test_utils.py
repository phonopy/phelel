"""Tests cli/utils.py."""

import io
import itertools
import pathlib
from collections.abc import Callable

import click
import numpy as np
import pytest
from phonopy.interface.phonopy_yaml import read_cell_yaml
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import apply_site_mixture

from phelel.velph.cli.utils import (
    echo_vasp_vca_hint,
    make_job_script,
    write_poscar,
)
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


def test_make_job_script():
    """The keys in braces in scheduler_template are replaced by the values."""
    scheduler = {
        "scheduler_template": "#!/bin/bash\n#$ -N {job_name}\n#$ -pe {pe} {cores}\n",
        "job_name": "PbTe-001",
        "pe": "mpi*",
        "cores": 48,
        "walltime": "12:00:00",
    }
    assert make_job_script(scheduler) == "#!/bin/bash\n#$ -N PbTe-001\n#$ -pe mpi* 48\n"


@pytest.mark.parametrize(
    "scheduler_dict,missing",
    [
        ({"job_name": "x"}, "scheduler_template"),
        ({"scheduler_template": "#!/bin/bash\n# {pe}\n", "job_name": "x"}, "pe"),
    ],
)
def test_make_job_script_missing_keys(scheduler_dict: dict, missing: str):
    """Test that a missing scheduler setting is an error naming the key."""
    with pytest.raises(click.ClickException, match=missing):
        make_job_script(scheduler_dict)


@pytest.mark.parametrize("template", ["echo }\n", "echo {\n", "echo {0}\n"])
def test_make_job_script_unreadable_template(template: str):
    """A template that str.format cannot read is an error on braces."""
    with pytest.raises(click.ClickException) as e:
        make_job_script({"scheduler_template": template})
    assert e.value.message.startswith("The job script template cannot be read:")
    assert e.value.message.endswith(
        'To write "{" or "}" in _job.sh, write "{{" or "}}" in scheduler_template.'
    )


def _get_GeSnTe_weighted_cell() -> PhonopyAtoms:
    """Return CsCl-like cell with Ge and Sn (0.5 each) co-located and Te."""
    cell = PhonopyAtoms(
        cell=np.eye(3) * 4.0,
        symbols=["Ge", "Sn", "Te"],
        scaled_positions=[[0, 0, 0], [0, 0, 0], [0.5, 0.5, 0.5]],
    )
    return apply_site_mixture(cell, [0.5, 0.5, 1.0])


def test_write_poscar_of_site_mixture(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
):
    """A cell with weighted species is written for VASP VCA with the hint."""
    cell = _get_GeSnTe_weighted_cell()
    write_poscar(tmp_path, cell)
    lines = (tmp_path / "POSCAR").read_text().splitlines()
    assert lines[5].split() == ["Ge", "Sn", "Te"]
    assert lines[6].split() == ["1", "1", "1"]
    echo_vasp_vca_hint(cell)
    assert "INCAR:               VCA = 0.5 0.5 1" in capsys.readouterr().out


def test_write_poscar_of_ordinary_cell(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
):
    """An ordinary cell is written without the hint of VCA."""
    cell = PhonopyAtoms(
        cell=np.eye(3) * 4.0,
        symbols=["Na", "Cl"],
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    write_poscar(tmp_path, cell)
    assert (tmp_path / "POSCAR").read_text().splitlines()[5].split() == ["Na", "Cl"]
    echo_vasp_vca_hint(cell)
    assert capsys.readouterr().out == ""
