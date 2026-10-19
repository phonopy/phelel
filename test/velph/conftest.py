"""Pytest conftest."""

import pathlib
import tomllib
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pytest
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from phelel import __version__
from phelel.velph.cli.init.build import build_velph_config
from phelel.velph.cli.phonopy.init import run_init as run_phonopy_init
from phelel.velph.config.io import (
    dumps_config,
    parse_init,
    parse_template,
    parse_velph_toml,
)
from phelel.velph.jobs.project import Project
from phelel.velph.jobs.project_directory import ProjectDirectory

cwd = pathlib.Path(__file__).parent
VELPH_TOML_V014_DIR = cwd / "data" / "velph_toml_v014"


@pytest.fixture(scope="session")
def nacl_cell() -> PhonopyAtoms:
    """Return PhonopyAtoms class instance of conventional unit cell of NaCl."""
    symbols = ["Na"] * 4 + ["Cl"] * 4
    lattice = np.eye(3) * 5.69
    points = [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [0.0, 0.0, 0.5],
        [0.0, 0.5, 0.0],
    ]
    return PhonopyAtoms(cell=lattice, scaled_positions=points, symbols=symbols)


@pytest.fixture(scope="session")
def si_prim_cell() -> PhonopyAtoms:
    """Return PhonopyAtoms class instance of primitive cell of Si."""
    symbols = ["Si"] * 2
    lattice = [[0, 2.73, 2.73], [2.73, 0, 2.73], [2.73, 2.73, 0]]
    points = [[0.75, 0.75, 0.75], [0.5, 0.5, 0.5]]
    return PhonopyAtoms(cell=lattice, scaled_positions=points, symbols=symbols)


@pytest.fixture(scope="session")
def aln_cell() -> PhonopyAtoms:
    """Return AlN cell (P6_3mc)."""
    a = 3.111
    c = 4.978
    lattice = [[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]]
    symbols = ["Al", "Al", "N", "N"]
    positions = [
        [1.0 / 3, 2.0 / 3, 0.0009488200000000],
        [2.0 / 3, 1.0 / 3, 0.5009488200000001],
        [1.0 / 3, 2.0 / 3, 0.6190511800000000],
        [2.0 / 3, 1.0 / 3, 0.1190511800000000],
    ]
    cell = PhonopyAtoms(cell=lattice, symbols=symbols, scaled_positions=positions)
    return cell


@pytest.fixture(scope="session")
def agno2_cell() -> PhonopyAtoms:
    """Return AgNO2 unit cell (Imm2)."""
    lattice = [[3.29187833, 0, 0], [0, 6.41646788, 0], [0, 0, 4.99692701]]
    positions = [
        [0.5, 0.5, 0.10954443],
        [0, 0, 0.60954443],
        [0.5, 0.66543538, 0.97306173],
        [0.5, 0.33456462, 0.97306173],
        [0, 0.16543538, 0.47306173],
        [0, 0.83456462, 0.47306173],
        [0.5, 0.5, 0.54433214],
        [0, 0, 0.04433214],
    ]
    symbols = ["N"] * 2 + ["O"] * 4 + ["Ag"] * 2
    cell = PhonopyAtoms(cell=lattice, symbols=symbols, scaled_positions=positions)
    return cell


@pytest.fixture(scope="session")
def tio2_prim_cell() -> PhonopyAtoms:
    """TiO2 anataze primitive cell."""
    lattice = [
        [-1.888070425, 1.888070425, 4.79024315],
        [1.888070425, -1.888070425, 4.79024315],
        [1.888070425, 1.888070425, -4.79024315],
    ]
    positions = [
        [0.45738631, 0.95738631, 0.5],
        [0.70738631, 0.70738631, 0],
        [0.04261369, 0.54261369, 0.5],
        [0.29261369, 0.29261369, 0],
        [0.5, 0.5, 0],
        [0.25, 0.75, 0.5],
    ]
    symbols = ["O"] * 4 + ["Ti"] * 2
    cell = PhonopyAtoms(cell=lattice, symbols=symbols, scaled_positions=positions)
    return cell


@pytest.fixture(scope="session")
def bi2te3_prim_cell() -> PhonopyAtoms:
    """Bi2Te3 primitive cell."""
    lattice = [
        [2.221502746054457, 1.282585208440033, 10.479344379716814],
        [-2.221502746054457, 1.282585208440033, 10.479344379716814],
        [-0.0, -2.565170416880067, 10.479344379716814],
    ]
    positions = [
        [0.601517497070164, 0.601517497070164, 0.601517497070165],
        [0.398482502929835, 0.398482502929836, 0.398482502929835],
        [0.786619249653126, 0.786619249653126, 0.786619249653126],
        [0.0, 0.0, 0.0],
        [0.213380750346874, 0.213380750346874, 0.213380750346874],
    ]
    symbols = ["Bi"] * 2 + ["Te"] * 3
    cell = PhonopyAtoms(cell=lattice, symbols=symbols, scaled_positions=positions)
    return cell


@pytest.fixture(scope="session")
def helper_methods() -> Callable:
    """Return methods to compare cells."""

    class HelperMethods:
        @classmethod
        def compare_cells_with_order(
            cls, cell: PhonopyAtoms, cell_ref: PhonopyAtoms, symprec=1e-5
        ) -> None:
            """Compare two cells with the same orders of positions."""
            np.testing.assert_allclose(cell.cell, cell_ref.cell, atol=symprec)
            cls.compare_positions_with_order(
                cell.scaled_positions, cell_ref.scaled_positions, cell.cell
            )
            np.testing.assert_array_equal(cell.numbers, cell_ref.numbers)
            np.testing.assert_allclose(cell.masses, cell_ref.masses, atol=symprec)
            if cell.magnetic_moments is None:
                assert cell_ref.magnetic_moments is None
            else:
                np.testing.assert_allclose(
                    cell.magnetic_moments, cell_ref.magnetic_moments, atol=symprec
                )

        @classmethod
        def compare_positions_with_order(
            cls, pos, pos_ref, lattice, symprec=1e-5
        ) -> None:
            """Compare two lists of positions and orders.

            lattice :
                Basis vectors in row vectors.

            """
            diff = pos - pos_ref
            diff -= np.rint(diff)
            dist = (np.dot(diff, lattice) ** 2).sum(axis=1)
            assert (dist < symprec).all()

        @classmethod
        def compare_cells(
            cls, cell: PhonopyAtoms, cell_ref: PhonopyAtoms, symprec=1e-5
        ) -> None:
            """Compare two cells where position orders can be different."""
            np.testing.assert_allclose(cell.cell, cell_ref.cell, atol=symprec)

            indices = cls.compare_positions_in_arbitrary_order(
                cell.scaled_positions, cell_ref.scaled_positions, cell.cell
            )
            np.testing.assert_array_equal(cell.numbers, cell_ref.numbers[indices])
            np.testing.assert_allclose(
                cell.masses, cell_ref.masses[indices], atol=symprec
            )
            if cell.magnetic_moments is None:
                assert cell_ref.magnetic_moments is None
            else:
                np.testing.assert_allclose(
                    cell.magnetic_moments,
                    cell_ref.magnetic_moments[indices],
                    atol=symprec,
                )

        @classmethod
        def compare_positions_in_arbitrary_order(
            cls, pos_in, pos_ref, lattice, symprec=1e-5
        ) -> list:
            """Compare two sets of positions irrespective of orders.

            lattice :
                Basis vectors in row vectors.

            """
            indices = []
            for pos in pos_in:
                diff = pos_ref - pos
                diff -= np.rint(diff)
                dist = (np.dot(diff, lattice) ** 2).sum(axis=1)
                matches = np.where(dist < symprec)[0]
                assert len(matches) == 1
                indices.append(matches[0])
            return indices

    return HelperMethods


# [scheduler] that velph init of velph 0.14 wrote by default.
SCHEDULER_TEMPLATE_TOML = '''
[scheduler]
job_name = "vasp-elph"
mpirun_command = "mpirun"
vasp_binary = "vasp_std"
prepend_text = ""
append_text = ""
nodes = 1
walltime = "96:00:00"
scheduler_template = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}
#SBATCH --time={walltime}
#SBATCH --output=ci_%j.log

{prepend_text}
{mpirun_command} {vasp_binary}
{append_text}
"""
'''


def _velph_init_toml(cell: PhonopyAtoms, options: dict[str, Any]) -> str:
    """Return velph.toml text that velph init writes for cell and [init] options.

    options are keys of [init], such as {"supercell_dimension": [2, 2, 2]} for
    "velph init --dim 2 2 2".  The template has SCHEDULER_TEMPLATE_TOML, so
    velph.toml has [scheduler].

    """
    template = parse_template(tomllib.loads(SCHEDULER_TEMPLATE_TOML))
    config = build_velph_config(cell, template=template, options=parse_init(options))
    return dumps_config(config)


@pytest.fixture(scope="session")
def nacl_velph_toml(nacl_cell: PhonopyAtoms) -> Callable[[], dict]:
    """Return a function giving velph.toml dict that velph init writes for NaCl.

    The cell is the conventional unit cell of NaCl, and the supercell is 2x2x2
    of it, as "velph init --dim 2 2 2" gives.  [scheduler] is that of
    SCHEDULER_TEMPLATE_TOML given as the template.  Each call returns a new
    dict, so a test can modify it.

    """
    text = _velph_init_toml(nacl_cell, {"supercell_dimension": [2, 2, 2]})

    def _get_velph_toml() -> dict:
        return tomllib.loads(text)

    return _get_velph_toml


@pytest.fixture(scope="session")
def site_mixture_velph_toml() -> Callable[[bool], dict]:
    """Return a function giving velph.toml dict of a site-mixture cell.

    The dict is that of a velph.toml written by velph init for a CsCl-like cell
    (a = 4 Angstrom) with co-located Ge and Sn (weights 0.5) at (0, 0, 0) and
    Te at (1/2, 1/2, 1/2), with the 2x2x2 supercell.  The cells in velph.toml
    keep Ge and Sn with weights.  The argument is split_site_mixture: False
    gives the merge scheme, with which phonopy merges Ge and Sn into a GeSn
    site, and True gives the split scheme.  Each call returns a new dict.

    """
    cell = PhonopyAtoms(
        cell=np.eye(3) * 4.0,
        symbols=["Ge", "Sn", "Te"],
        scaled_positions=[[0, 0, 0], [0, 0, 0], [0.5, 0.5, 0.5]],
    )
    options = {"supercell_dimension": [2, 2, 2], "site_mixture": "0.5 0.5 1.0"}
    texts = {
        split: _velph_init_toml(cell, {**options, "split_site_mixture": split})
        for split in (False, True)
    }

    def _get_velph_toml(split_site_mixture: bool) -> dict:
        return tomllib.loads(texts[split_site_mixture])

    return _get_velph_toml


@pytest.fixture(scope="session")
def nacl_velph_toml_v014() -> Callable[[], dict]:
    """Return a function giving velph.toml dict that velph 0.14 writes for NaCl.

    The supercell is 2x2x2 of the conventional unit cell, and [scheduler] is
    written.  Each call returns a new dict, so a test can modify it.

    """

    def _get_velph_toml() -> dict:
        return _load_velph_toml_v014("nacl_222.toml")

    return _get_velph_toml


@pytest.fixture(scope="session")
def velph_toml_v014() -> Callable[[str], dict]:
    """Return a function giving velph.toml dict of a file of velph 0.14.

    The argument is the name of a file in data/velph_toml_v014, which is
    written by velph init of velph 0.14 as its header says.

    """
    return _load_velph_toml_v014


def _load_velph_toml_v014(filename: str) -> dict:
    """Return velph.toml dict of a file written by velph init of velph 0.14."""
    with open(VELPH_TOML_V014_DIR / filename, "rb") as f:
        return tomllib.load(f)


@pytest.fixture(scope="session")
def code_init_project() -> Callable[[dict], Project]:
    """Return a function giving Project of a part of velph.toml.

    The argument is a dict of the sections of velph.toml that the init of a
    code reads, such as [phelel], [phonopy], [phono3py], [vasp.nac] and
    [unitcell].  [velph] version is added, and [primitive_cell] is the unit
    cell when it is not given.  The dict is read by parse_velph_toml, so a key
    that velph.toml does not have is an error.

    """

    def _get_project(data: dict) -> Project:
        full = {"velph": {"version": __version__}, "primitive_cell": data["unitcell"]}
        full.update(data)
        return Project.from_config(parse_velph_toml(full))

    return _get_project


_TI_UNITCELL = """
[unitcell]
lattice = [
  [ 2.930720886111760, 0.000000000000000, 0.000000000000000 ],
  [ -1.465360443055880, 2.538078738774425, 0.000000000000000 ],
  [ 0.000000000000000, 0.000000000000000, 4.646120482318025 ],
]
[[unitcell.points]]
symbol = "Ti"
coordinates = [ 0.333333333333336, 0.666666666666664, 0.25 ]
[[unitcell.points]]
symbol = "Ti"
coordinates = [ 0.666666666666664, 0.333333333333336, 0.75 ]
"""


@pytest.fixture(scope="session")
def ti_velph_toml() -> Callable[[Sequence[str]], dict]:
    """Return a function giving velph.toml dict of hcp Ti with a phonon supercell.

    [phelel] and [phono3py] give the 1x1x1 supercell and [phonopy] the 2x2x1
    supercell.  The argument is the lines added to [phonopy], such as
    "amplitude = 0.01".

    """

    def _get_velph_toml(phonopy_lines: Sequence[str]) -> dict:
        toml_str = "\n".join(
            [
                "[phelel]",
                "supercell_dimension = [1, 1, 1]",
                "[phono3py]",
                "supercell_dimension = [1, 1, 1]",
                "[phonopy]",
                "supercell_dimension = [2, 2, 1]",
                *phonopy_lines,
            ]
        )
        return tomllib.loads(toml_str + _TI_UNITCELL)

    return _get_velph_toml


@pytest.fixture(scope="session")
def write_phonopy_disp_yaml() -> Callable[[Project, pathlib.Path], Phonopy]:
    """Return a function writing phonopy/phonopy_disp.yaml as velph phonopy init.

    The arguments are the project and the directory where phonopy/ is made.  The
    function returns Phonopy written in the file.

    """

    def _write(project: Project, directory: pathlib.Path) -> Phonopy:
        project_dir = ProjectDirectory(directory)
        ph = run_phonopy_init(project, project_dir=project_dir)
        filename = project_dir.disp_yaml("phonopy")
        filename.parent.mkdir(parents=True, exist_ok=True)
        ph.save(filename)
        return ph

    return _write
