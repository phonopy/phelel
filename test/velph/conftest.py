"""Pytest conftest."""

import contextlib
import io
import pathlib
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pytest
import tomli
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

from phelel.velph.cli.init.init import _run_init
from phelel.velph.cli.phonopy.init import run_init as run_phonopy_init
from phelel.velph.cli.supercells import PHONOPY_DISP_YAML
from phelel.velph.cli.utils import VelphInitOptions
from phelel.velph.config.io import parse_template
from phelel.velph.config.legacy import parse_legacy_velph_toml
from phelel.velph.config.schema import (
    CODES,
    InputSetSection,
    VaspSection,
    VelphConfig,
)

cwd = pathlib.Path(__file__).parent


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


@pytest.fixture(scope="session")
def site_mixture_velph_toml() -> Callable[[bool], dict]:
    """Return a function giving velph.toml dict of a site-mixture cell.

    The cell is CsCl-like with co-located Ge and Sn (weights 0.5) and Te. The
    cells in velph.toml keep Ge and Sn with weights. The argument is
    split_site_mixture: False writes the merge scheme, with which phonopy
    merges Ge and Sn into a GeSn site, and True writes the split scheme.

    """

    def _get_velph_toml(split_site_mixture: bool) -> dict:
        cell = PhonopyAtoms(
            symbols=["Ge", "Sn", "Te"],
            cell=np.eye(3) * 4.0,
            scaled_positions=[[0, 0, 0], [0, 0, 0], [0.5, 0.5, 0.5]],
        )
        with contextlib.redirect_stdout(io.StringIO()):
            toml_lines = _run_init(
                cell,
                VelphInitOptions(
                    site_mixture="0.5 0.5 1.0",
                    split_site_mixture=split_site_mixture,
                    supercell_dimension=(2, 2, 2),
                ),
            )
        assert toml_lines is not None
        return tomli.loads("\n".join(toml_lines))

    return _get_velph_toml


@pytest.fixture(scope="session")
def nacl_velph_toml_v014(nacl_cell: PhonopyAtoms) -> Callable[[], dict]:
    """Return a function giving velph.toml dict that velph 0.14 writes for NaCl.

    The supercell is 2x2x2 of the conventional unit cell, and [scheduler] is
    written.  Each call returns a new dict, so a test can modify it.

    """
    with contextlib.redirect_stdout(io.StringIO()):
        toml_lines = _run_init(
            nacl_cell, VelphInitOptions(supercell_dimension=(2, 2, 2))
        )
    assert toml_lines is not None
    toml_str = "\n".join(toml_lines)

    def _get_velph_toml() -> dict:
        return tomli.loads(toml_str)

    return _get_velph_toml


@pytest.fixture(scope="session")
def code_init_config() -> Callable[[dict], VelphConfig]:
    """Return a function giving VelphConfig that the init of a code reads.

    The argument is a dict of velph.toml that gives [phelel], [phonopy] or
    [phono3py], [unitcell], and optionally [primitive_cell] and "cell" of
    [vasp.nac].  Without [primitive_cell], the unit cell is used.  Other keys,
    such as fft_mesh and [vasp.supercell] of velph 0.14, are ignored.  A dict
    with [phelel] version, which velph init of velph 0.14 writes, is read by
    the legacy loader.

    """

    def _get_config(data: dict) -> VelphConfig:
        if "version" in data.get("phelel", {}):
            return parse_legacy_velph_toml(data)
        sections: dict[str, Any] = {
            code: {key: value for key, value in data[code].items() if key != "fft_mesh"}
            for code in CODES
            if code in data
        }
        template = parse_template(sections)
        cell = data.get("vasp", {}).get("nac", {}).get("cell")
        return VelphConfig(
            phelel=template.phelel,
            phonopy=template.phonopy,
            phono3py=template.phono3py,
            vasp=VaspSection(input_sets={"nac": InputSetSection(cell=cell)}),
            unitcell=data["unitcell"],
            primitive_cell=data.get("primitive_cell", data["unitcell"]),
        )

    return _get_config


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
        return tomli.loads(toml_str + _TI_UNITCELL)

    return _get_velph_toml


@pytest.fixture(scope="session")
def write_phonopy_disp_yaml() -> Callable[[VelphConfig, pathlib.Path], Phonopy]:
    """Return a function writing phonopy/phonopy_disp.yaml as velph phonopy init.

    The arguments are the config and the directory where phonopy/ is made.  The
    function returns Phonopy written in the file.

    """

    def _write(config: VelphConfig, directory: pathlib.Path) -> Phonopy:
        ph = run_phonopy_init(config, current_directory=directory)
        filename = directory / PHONOPY_DISP_YAML
        filename.parent.mkdir(parents=True, exist_ok=True)
        ph.save(filename)
        return ph

    return _write
