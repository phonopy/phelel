"""Pytest conftest."""

import pathlib
from collections.abc import Callable

import numpy as np
import pytest
from phonopy.structure.atoms import PhonopyAtoms

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
