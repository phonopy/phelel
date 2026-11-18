"""Tests of phelel.base.augmented_fit.

The derivative test data of phelel have q = -q for all q points, so they cannot
detect the sign of q or the change from q to -q of time reversal.  The tests on
synthetic potentials therefore use a supercell with q != -q.  A synthetic
potential is a sum of Gaussians at the atoms, which is a smooth function of the
displacements with a known derivative.

The lattice L_p of the output grid (p2s_matrix) need not be kept by the
rotations S of the operations of the supercell.  The tests on the 2x2x2
supercell of the cubic cell use lattices with S L_p != L_p for some S
(OUTPUT_LATTICES).  The finite differences of DLocalPotential, the reference of
the fit, are also tested with these lattices.

"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import pytest
from numpy.typing import NDArray
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import Primitive, get_primitive, get_supercell
from phonopy.structure.symmetry import Symmetry

import phelel
from phelel.base.augmented_fit import (
    DLocalPotentialAugmentedFit,
    _get_inverse_normal_matrices,
    get_coset_representatives,
)
from phelel.base.fourier_series import FourierSeriesOnGrid
from phelel.base.local_potential import DLocalPotential
from phelel.base.potential_change import (
    add_to_augmented_normal_equation_rhs,
    apply_inverse_augmented_normal_matrices,
)
from phelel.base.symmetrization import select_equivalent_atoms
from phelel.interface.vasp.derivatives import read_files

cwd = Path(__file__).parent
vasp_dir = cwd.parent / "interface" / "vasp"

LATTICE_PARAMETER = 4.0
SIGMA = 0.7
P2S_MATRIX = np.diag([3, 1, 1]).astype("int64")
SUPERCELL_MESH = (36, 12, 12)
FFT_MESH = (10, 10, 10)

# 2x2x2 supercell of the cubic cell and the lattices L_p of the output grid with
# their FFT meshes.  "cubic" is the cubic lattice, kept by all operations.
# "elongated" (2a, a, a) and "sheared" (2a, 0, 0), (a, a, 0), (0, 0, a) are not
# kept by some of the 4-fold rotations.  The grid points of these three
# lattices are the same points with the spacing a / 6 along x, y and z, and
# the lattice of the grid points is kept by all operations.  With the coarser
# meshes of "elongated_coarse" and "sheared_coarse", the lattice of the grid
# points is not kept by some of the 4-fold rotations either.  "bcc" is the
# lattice of the bcc primitive cell, for the structure "bcc" only.  The grid
# points of all are on the mesh with the spacing a / 6.
CUBIC_SUPERCELL_MATRIX = np.diag([2, 2, 2]).astype("int64")
CUBIC_SUPERCELL_MESH = (16, 16, 16)
GRID_POINTS_PER_SUPERCELL_EDGE = 12
OUTPUT_LATTICES = {
    "cubic": (np.diag([2, 2, 2]).astype("int64"), (6, 6, 6)),
    "elongated": (np.diag([1, 2, 2]).astype("int64"), (12, 6, 6)),
    "sheared": (
        np.array([[1, -1, 0], [0, 2, 0], [0, 0, 2]], dtype="int64"),
        (12, 6, 6),
    ),
    "elongated_coarse": (np.diag([1, 2, 2]).astype("int64"), (6, 6, 6)),
    "sheared_coarse": (
        np.array([[1, -1, 0], [0, 2, 0], [0, 0, 2]], dtype="int64"),
        (6, 6, 6),
    ),
    "bcc": (
        np.array([[0, 2, 2], [2, 0, 2], [2, 2, 0]], dtype="int64"),
        (3, 3, 3),
    ),
}

Structure = Literal["CsCl", "magnetic", "screw", "bcc"]


def _get_cells(
    structure: Structure,
    is_symmetry: bool = True,
    supercell_matrix: NDArray = P2S_MATRIX,
    p2s_matrix: NDArray = P2S_MATRIX,
) -> tuple[PhonopyAtoms, Primitive, Symmetry]:
    """Return a supercell of a cubic cell, its primitive cell and symmetry.

    The supercell is supercell_matrix times the cubic cell, and p2s_matrix is
    the supercell matrix relative to the primitive cell.  By default both are
    3x1x1, so the primitive cell is the cubic cell.

    "CsCl" is the CsCl-type cell.  In "magnetic", the two atoms of the CsCl-type
    cell are of one species with the moments +1 and -1, so the body-centering
    translation is a symmetry only with time reversal.  In "screw", two atoms
    of one species are related by the 2_1 screw (x + 1/2, -y, -z) along x, and
    the space group is P2_1/m.  Only "screw" has a fractional translation
    without time reversal, and in "CsCl" every permutation of the coset
    representatives is its own inverse.  "bcc" is the CsCl-type cell of one
    species without moments, whose cubic cell is not the smallest cell.

    """
    if structure == "screw":
        symbols = ["Cs", "Cs"]
        positions = [[0, 0.2, 0.1], [0.5, 0.8, 0.9]]
    else:
        symbols = ["Cs", "Cl"] if structure == "CsCl" else ["Cs", "Cs"]
        positions = [[0, 0, 0], [0.5, 0.5, 0.5]]
    unitcell = PhonopyAtoms(
        symbols=symbols,
        cell=np.eye(3) * LATTICE_PARAMETER,
        scaled_positions=positions,
        magnetic_moments=[1.0, -1.0] if structure == "magnetic" else None,
    )
    supercell = get_supercell(unitcell, supercell_matrix)
    primitive = get_primitive(supercell, np.linalg.inv(p2s_matrix))
    return supercell, primitive, Symmetry(supercell, is_symmetry=is_symmetry)


def _gaussians(
    points: NDArray, cell: PhonopyAtoms, displacement: NDArray | None
) -> tuple[NDArray, NDArray]:
    """Return Gaussians of each atom and their position differences at the points.

    Parameters
    ----------
    points : ndarray
        Cartesian points.
        shape=(n_points, 3)
    cell : PhonopyAtoms
        Supercell.
    displacement : ndarray or None
        Displacements of the atoms.
        shape=(n_atoms, 3)

    Returns
    -------
    gaussians : ndarray
        Sum over the periodic images of the Gaussian of each atom.
        shape=(n_atoms, n_points)
    gradients : ndarray
        Derivative of gaussians with respect to the position of the atom.
        shape=(n_atoms, n_points, 3)

    """
    positions = cell.positions
    if displacement is not None:
        positions = positions + displacement
    images = np.array(list(np.ndindex(3, 3, 3))) - 1
    gaussians = np.zeros((len(positions), len(points)))
    gradients = np.zeros((len(positions), len(points), 3))
    for image in images @ cell.cell:
        differences = points[None, :, :] - positions[:, None, :] - image
        values = np.exp(-(differences**2).sum(axis=2) / (2 * SIGMA**2))
        gaussians += values
        gradients += values[:, :, None] * differences / SIGMA**2
    return gaussians, gradients


def _get_atom_weights(cell: PhonopyAtoms, magnetic: bool) -> NDArray:
    """Return the weight of the Gaussian of each atom for each spin component.

    shape=(ncdij, n_atoms).  Without magnetism, the weights depend on the species.
    With magnetism, the components are V + V_z and V - V_z with V_z given by the
    moment of the atom.

    """
    if magnetic:
        moments = np.array(cell.magnetic_moments)
        return np.array([1.0 + 0.3 * moments, 1.0 - 0.3 * moments])
    return np.array([[1.0 if s == "Cs" else 0.6 for s in cell.symbols]])


def _get_potential(
    cell: PhonopyAtoms,
    magnetic: bool,
    displacement: NDArray | None = None,
    mesh: tuple[int, int, int] = SUPERCELL_MESH,
) -> NDArray:
    """Return the potential on the supercell FFT mesh, shape=(ncdij, nz, ny, nx)."""
    m3, m2, m1 = np.indices(mesh[::-1]).reshape(3, -1)
    fractional = np.c_[m1 / mesh[0], m2 / mesh[1], m3 / mesh[2]]
    gaussians, _ = _gaussians(fractional @ cell.cell, cell, displacement)
    potential = _get_atom_weights(cell, magnetic) @ gaussians
    return potential.reshape(-1, *mesh[::-1]).astype("complex128")


def _get_analytic_dVdu(
    cell: PhonopyAtoms, primitive: Primitive, magnetic: bool, grid_points: NDArray
) -> NDArray:
    """Return dV/du of the atoms of p2s_map, shape=(ncdij, n_p, 3, n_grid_points)."""
    _, gradients = _gaussians(grid_points @ cell.cell, cell, None)
    weights = _get_atom_weights(cell, magnetic)
    atoms = primitive.p2s_map
    return np.einsum("ca,apk->cakp", weights[:, atoms], gradients[atoms])


def _get_type1_displacements(primitive: Primitive, structure: Structure) -> list[dict]:
    """Return displacements along x, y and z of the symmetry-unique atoms."""
    atoms = primitive.p2s_map if structure == "CsCl" else primitive.p2s_map[:1]
    return [
        {"number": atom, "displacement": direction * 0.02}
        for atom in atoms
        for direction in np.eye(3)
    ]


def _to_type2(displacements: list[dict], n_atoms: int) -> NDArray:
    """Return displacements of a type-1 dataset as an array of a type-2 dataset."""
    array = np.zeros((len(displacements), n_atoms, 3))
    for i, displacement in enumerate(displacements):
        array[i, displacement["number"]] = displacement["displacement"]
    return array


def _get_fit(
    supercell: PhonopyAtoms,
    primitive: Primitive,
    symmetry: Symmetry,
    lang: Literal["Rust", "Python"] = "Rust",
) -> DLocalPotentialAugmentedFit:
    return DLocalPotentialAugmentedFit(
        FFT_MESH, P2S_MATRIX, supercell, primitive, symmetry, verbose=False, lang=lang
    )


def _assert_close(values: NDArray, reference: NDArray, tolerance: float) -> None:
    """Check values against reference relative to the largest absolute value."""
    scale = np.abs(reference).max()
    assert scale > 0
    assert np.abs(values - reference).max() / scale < tolerance


@pytest.mark.parametrize(
    "data_dir",
    [
        "derivatives_vaspout/C",
        "derivatives_vaspout/Mg",
        "derivatives_lwap/Cr",
        "derivatives_lwap/Fe-ncl",
    ],
)
def test_type1_data_equal_finite_differences(data_dir: str):
    """The augmented fit of type-1 data agrees with the finite differences.

    Cr has operations with time reversal, and Fe-ncl the rotation in spin space.

    """
    directory = vasp_dir / data_dir
    with h5py.File(directory / "reference.hdf5") as f:
        fft_mesh = [int(n) for n in f.attrs["fft_mesh"]]
    phe = phelel.load(
        directory / "phelel_disp.yaml",
        fft_mesh=fft_mesh,
        is_nac=False,
        dv_evaluation_method="shortest",
    )
    dir_names = sorted(directory.glob("disp-*"))
    phe_input = read_files(phe, dir_names, phonon_dir_names=dir_names)
    potentials = phe_input.local_potentials
    displacements = phe.dataset["first_atoms"]
    finite_differences = DLocalPotential(
        fft_mesh,
        phe.p2s_matrix,
        phe.supercell,
        symmetry=phe.symmetry,
        atom_indices=phe.primitive.p2s_map,
        dv_evaluation_method="shortest",
        verbose=False,
    )
    finite_differences.run(potentials[0], potentials[1:], displacements)

    fit = DLocalPotentialAugmentedFit(
        fft_mesh,
        phe.p2s_matrix,
        phe.supercell,
        phe.primitive,
        phe.symmetry,
        verbose=False,
    )
    fit.run(
        potentials[0],
        potentials[1:],
        _to_type2(displacements, len(phe.supercell)),
    )
    _assert_close(fit.dVdu, finite_differences.dVdu, 1e-12)


@pytest.mark.parametrize("lang", ["Rust", "Python"])
@pytest.mark.parametrize("structure", ["CsCl", "magnetic", "screw"])
def test_type1_synthetic_data_equal_finite_differences(
    structure: Structure, lang: Literal["Rust", "Python"]
):
    """The fit agrees with the finite differences on a supercell with q != -q.

    In "magnetic", half of the operations have time reversal.  The finite
    differences are computed with lang="Rust" for both values of lang of the fit,
    so that the two implementations of the fit are compared with one reference.

    """
    magnetic = structure == "magnetic"
    supercell, primitive, symmetry = _get_cells(structure)
    displacements = _get_type1_displacements(primitive, structure)
    array = _to_type2(displacements, len(supercell))
    perfect = _get_potential(supercell, magnetic)
    potentials = [_get_potential(supercell, magnetic, d) for d in array]

    finite_differences = DLocalPotential(
        FFT_MESH,
        P2S_MATRIX,
        supercell,
        symmetry=symmetry,
        atom_indices=primitive.p2s_map,
        dv_evaluation_method="shortest",
        verbose=False,
    )
    finite_differences.run(perfect, potentials, displacements)

    fit = _get_fit(supercell, primitive, symmetry, lang=lang)
    fit.run(perfect, potentials, array)
    _assert_close(fit.dVdu, finite_differences.dVdu, 1e-12)



def test_type1_grey_group_equal_finite_differences():
    """The fit agrees with the finite differences for zero magnetic moments.

    The symmetry is that of the "magnetic" cell with the moments set to zero,
    a grey group, and the potentials are those of the "magnetic" cell, whose
    spin up and down differ.  If Symmetry lacks the operations with time
    reversal, both methods add them, and spin up and down of dV/du become
    equal.  Without them in the fit, the two methods differ by about 0.3 of
    the largest absolute value of dV/du.

    """
    magnetic_cell, primitive, _ = _get_cells("magnetic")
    grey_cell = magnetic_cell.copy()
    grey_cell.magnetic_moments = np.zeros(len(grey_cell))
    symmetry = Symmetry(grey_cell)
    displacements = _get_type1_displacements(primitive, "magnetic")
    array = _to_type2(displacements, len(grey_cell))
    perfect = _get_potential(magnetic_cell, True)
    potentials = [_get_potential(magnetic_cell, True, d) for d in array]

    finite_differences = DLocalPotential(
        FFT_MESH,
        P2S_MATRIX,
        grey_cell,
        symmetry=symmetry,
        atom_indices=primitive.p2s_map,
        dv_evaluation_method="shortest",
        verbose=False,
    )
    finite_differences.run(perfect, potentials, displacements)

    fit = _get_fit(grey_cell, primitive, symmetry)
    fit.run(perfect, potentials, array)
    _assert_close(fit.dVdu, finite_differences.dVdu, 1e-12)

@pytest.mark.parametrize("structure", ["CsCl", "magnetic", "screw"])
def test_random_displacements_equal_analytic_derivative(structure: Structure):
    """All atoms displaced with +/- pairs give the derivative of the Gaussians."""
    magnetic = structure == "magnetic"
    supercell, primitive, symmetry = _get_cells(structure)
    rng = np.random.default_rng(0)
    directions = rng.standard_normal((3, len(supercell), 3))
    directions /= np.linalg.norm(directions, axis=2, keepdims=True)
    array = np.concatenate([0.01 * directions, -0.01 * directions])
    perfect = _get_potential(supercell, magnetic)
    potentials = [_get_potential(supercell, magnetic, d) for d in array]

    fit = _get_fit(supercell, primitive, symmetry)
    fit.run(perfect, potentials, array)
    assert fit.normal_matrix_eigenvalues.min() > 1e-6
    analytic = _get_analytic_dVdu(supercell, primitive, magnetic, fit.grid_points)
    _assert_close(fit.dVdu, analytic, 1e-3)


def test_random_displacements_without_drift():
    """Without a drift, the fit misses only the uniform translation at q = 0.

    The three eigenvalues of A(0) along the uniform translation of all atoms are
    discarded, so the fit gives the analytic derivative with the sum over the
    atoms of the primitive cell and over the lattice points removed.  The grid
    points are blocks of the mesh of the primitive cell shifted by the lattice
    points, so the sum over the lattice points is that over the blocks.  With
    rcond=0, the eigenvalues of round-off are inverted and the fit fails.

    """
    supercell, primitive, symmetry = _get_cells("screw")
    rng = np.random.default_rng(0)
    directions = rng.standard_normal((3, len(supercell), 3))
    directions -= directions.mean(axis=1, keepdims=True)
    array = np.concatenate([0.01 * directions, -0.01 * directions])
    perfect = _get_potential(supercell, False)
    potentials = [_get_potential(supercell, False, d) for d in array]

    fit = _get_fit(supercell, primitive, symmetry)
    fit.run(perfect, potentials, array)
    eigvals = fit.normal_matrix_eigenvalues
    np.testing.assert_allclose(FourierSeriesOnGrid(FFT_MESH, P2S_MATRIX).qpoints[0], 0)
    np.testing.assert_array_equal(
        (eigvals <= 1e-10 * eigvals.max()).sum(axis=1), [3, 0, 0]
    )

    analytic = _get_analytic_dVdu(supercell, primitive, False, fit.grid_points)
    ncdij, n_p = analytic.shape[:2]
    num_cells = len(fit.lattice_points)
    blocks = analytic.reshape(ncdij, n_p, 3, num_cells, -1)
    uniform = blocks.sum(axis=(1, 3), keepdims=True) / (n_p * num_cells)
    _assert_close(fit.dVdu, (blocks - uniform).reshape(analytic.shape), 1e-3)


def _get_cubic_supercell_cells(
    structure: Structure, output_lattice: str
) -> tuple[PhonopyAtoms, Primitive, Symmetry]:
    """Return the cells of the 2x2x2 supercell with a lattice of OUTPUT_LATTICES."""
    return _get_cells(
        structure,
        supercell_matrix=CUBIC_SUPERCELL_MATRIX,
        p2s_matrix=OUTPUT_LATTICES[output_lattice][0],
    )


def _count_rotations_not_keeping_lattice(
    rotations: NDArray, p2s_matrix: NDArray
) -> int:
    """Return the number of rotations S with S L_p != L_p.

    With the rotation W of S in the supercell basis vectors and M = p2s_matrix,
    S in the basis vectors of L_p is M W M^-1, which is an integer matrix if
    and only if S L_p = L_p.

    """
    count = 0
    for rotation in rotations:
        matrix = p2s_matrix @ rotation @ np.linalg.inv(p2s_matrix)
        if not np.allclose(matrix, np.rint(matrix), atol=1e-8):
            count += 1
    return count


def _get_dVdu_by_atom_and_point(
    fit: DLocalPotentialAugmentedFit | DLocalPotential, primitive: Primitive
) -> dict[tuple[int, tuple[int, ...]], NDArray]:
    """Return dV/du keyed by the atom of the supercell and the grid point.

    The grid point is labeled by its fractional coordinates in the supercell
    basis vectors times GRID_POINTS_PER_SUPERCELL_EDGE, modulo this number.

    """
    labels = np.rint(fit.grid_points * GRID_POINTS_PER_SUPERCELL_EDGE).astype(int)
    labels %= GRID_POINTS_PER_SUPERCELL_EDGE
    return {
        (int(atom), tuple(label)): fit.dVdu[:, i_atom, :, i_point]
        for i_atom, atom in enumerate(primitive.p2s_map)
        for i_point, label in enumerate(labels)
    }


@pytest.mark.parametrize("lang", ["Rust", "Python"])
@pytest.mark.parametrize(
    "output_lattice", ["elongated", "sheared", "elongated_coarse", "sheared_coarse"]
)
@pytest.mark.parametrize("structure", ["CsCl", "magnetic"])
def test_type1_synthetic_data_with_lattice_not_kept(
    structure: Structure, output_lattice: str, lang: Literal["Rust", "Python"]
):
    """The fit agrees with the finite differences when S L_p != L_p for some S.

    Some coset representatives have S L_p != L_p, which is checked.  In
    "magnetic", some of them have time reversal.  The symmetry-independent atoms
    are displaced along x, y and z.

    """
    magnetic = structure == "magnetic"
    p2s_matrix, fft_mesh = OUTPUT_LATTICES[output_lattice]
    supercell, primitive, symmetry = _get_cubic_supercell_cells(
        structure, output_lattice
    )
    cosets = get_coset_representatives(symmetry, supercell, p2s_matrix)
    assert _count_rotations_not_keeping_lattice(cosets.rotations, p2s_matrix) > 0
    assert cosets.time_reversals.any() == magnetic

    displacements = [
        {"number": int(atom), "displacement": direction * 0.02}
        for atom in symmetry.get_independent_atoms()
        for direction in np.eye(3)
    ]
    array = _to_type2(displacements, len(supercell))
    perfect = _get_potential(supercell, magnetic, mesh=CUBIC_SUPERCELL_MESH)
    potentials = [
        _get_potential(supercell, magnetic, d, mesh=CUBIC_SUPERCELL_MESH) for d in array
    ]

    finite_differences = DLocalPotential(
        fft_mesh,
        p2s_matrix,
        supercell,
        symmetry=symmetry,
        atom_indices=primitive.p2s_map,
        dv_evaluation_method="shortest",
        verbose=False,
    )
    finite_differences.run(perfect, potentials, displacements)

    fit = DLocalPotentialAugmentedFit(
        fft_mesh, p2s_matrix, supercell, primitive, symmetry, verbose=False, lang=lang
    )
    fit.run(perfect, potentials, array)
    _assert_close(fit.dVdu, finite_differences.dVdu, 1e-12)


def _count_site_symmetries_not_keeping_lattice(
    symmetry: Symmetry, primitive: Primitive, displaced_atoms: list[int]
) -> int:
    """Return the number of operations with S L_p != L_p of the finite differences.

    The operations are those that send a displaced atom to an atom of the
    primitive cell, which the finite differences use for that atom.  An
    operation is counted once for each atom.

    """
    p2s_matrix = np.rint(np.linalg.inv(primitive.primitive_matrix)).astype("int64")
    rotations = symmetry.symmetry_operations["rotations"]
    return sum(
        _count_rotations_not_keeping_lattice(rotations[sitesyms], p2s_matrix)
        for atom in displaced_atoms
        for sitesyms in select_equivalent_atoms(atom, symmetry, primitive.p2s_map)[1]
    )


@pytest.mark.parametrize(
    "structure,reference,output_lattice",
    [
        (structure, "cubic", output_lattice)
        for structure in ("CsCl", "magnetic")
        for output_lattice in (
            "elongated",
            "sheared",
            "elongated_coarse",
            "sheared_coarse",
        )
    ]
    + [("bcc", "bcc", "cubic"), ("bcc", "bcc", "sheared_coarse")],
)
def test_finite_differences_do_not_depend_on_lattice_of_output_grid(
    structure: Structure, reference: str, output_lattice: str
):
    """dV/du by finite differences does not depend on L_p.

    dV/du of the same atom of the supercell at the same point is compared, at
    the atoms and points that the two output grids have in common.  The
    finite differences use operations with S L_p != L_p, which is checked,
    except for the cubic cell of "bcc", which is kept by all operations but is
    not the smallest cell.  For the coarse meshes, the lattice of the grid
    points, whose basis vectors are those of L_p divided by fft_mesh, is not
    kept by some operations, so g^-1 x of a grid point x is not always a grid
    point.

    """
    magnetic = structure == "magnetic"
    supercell, primitive, symmetry = _get_cubic_supercell_cells(structure, reference)
    displaced_atoms = [int(atom) for atom in symmetry.get_independent_atoms()]
    displacements = [
        {"number": atom, "displacement": direction * 0.02}
        for atom in displaced_atoms
        for direction in np.eye(3)
    ]
    array = _to_type2(displacements, len(supercell))
    perfect = _get_potential(supercell, magnetic, mesh=CUBIC_SUPERCELL_MESH)
    potentials = [
        _get_potential(supercell, magnetic, d, mesh=CUBIC_SUPERCELL_MESH) for d in array
    ]

    tables = []
    for name in (reference, output_lattice):
        p2s_matrix, fft_mesh = OUTPUT_LATTICES[name]
        supercell, primitive, symmetry = _get_cubic_supercell_cells(structure, name)
        finite_differences = DLocalPotential(
            fft_mesh,
            p2s_matrix,
            supercell,
            symmetry=symmetry,
            atom_indices=primitive.p2s_map,
            dv_evaluation_method="shortest",
            verbose=False,
        )
        finite_differences.run(perfect, potentials, displacements)
        np.testing.assert_array_equal(
            finite_differences.atom_indices, primitive.p2s_map
        )
        tables.append(_get_dVdu_by_atom_and_point(finite_differences, primitive))

    num_not_kept = _count_site_symmetries_not_keeping_lattice(
        symmetry, primitive, displaced_atoms
    )
    assert (num_not_kept > 0) == (output_lattice != "cubic")
    if output_lattice.endswith("_coarse"):
        rotations = symmetry.symmetry_operations["rotations"]
        grid_lattice = np.diag(fft_mesh) @ p2s_matrix
        assert _count_rotations_not_keeping_lattice(rotations, grid_lattice) > 0

    reference_table, other = tables
    keys = sorted(set(reference_table) & set(other))
    atoms = {atom for atom, _ in reference_table} & {atom for atom, _ in other}
    points = {point for _, point in reference_table} & {point for _, point in other}
    assert atoms and points
    assert len(keys) == len(atoms) * len(points)
    _assert_close(
        np.array([other[key] for key in keys]),
        np.array([reference_table[key] for key in keys]),
        1e-12,
    )


@pytest.mark.parametrize("output_lattice", ["elongated", "sheared"])
@pytest.mark.parametrize("structure", ["CsCl", "magnetic"])
def test_random_displacements_do_not_depend_on_lattice_of_output_grid(
    structure: Structure, output_lattice: str
):
    """dV/du with S L_p != L_p for some S equals that with the cubic L_p.

    The augmented snapshots of all the coset representatives make the normal
    equations invariant under the operations of the supercell, so the
    least-squares solution does not depend on L_p.  dV/du of the same atom of
    the supercell at the same point is compared.  These are all the atoms of
    the cubic primitive cell at all the grid points.

    """
    magnetic = structure == "magnetic"
    supercell, _, _ = _get_cubic_supercell_cells(structure, "cubic")
    rng = np.random.default_rng(0)
    directions = rng.standard_normal((3, len(supercell), 3))
    directions /= np.linalg.norm(directions, axis=2, keepdims=True)
    array = np.concatenate([0.01 * directions, -0.01 * directions])
    perfect = _get_potential(supercell, magnetic, mesh=CUBIC_SUPERCELL_MESH)
    potentials = [
        _get_potential(supercell, magnetic, d, mesh=CUBIC_SUPERCELL_MESH) for d in array
    ]

    tables = []
    for name in ("cubic", output_lattice):
        p2s_matrix, fft_mesh = OUTPUT_LATTICES[name]
        supercell, primitive, symmetry = _get_cubic_supercell_cells(structure, name)
        fit = DLocalPotentialAugmentedFit(
            fft_mesh, p2s_matrix, supercell, primitive, symmetry, verbose=False
        )
        fit.run(perfect, potentials, array)
        assert fit.normal_matrix_eigenvalues.min() > 1e-6
        tables.append(_get_dVdu_by_atom_and_point(fit, primitive))

    cubic, other = tables
    assert set(cubic) <= set(other)
    keys = list(cubic)
    _assert_close(
        np.array([other[key] for key in keys]),
        np.array([cubic[key] for key in keys]),
        1e-12,
    )


def test_get_coset_representatives_of_supercell():
    """The operations are grouped into cosets of N_c operations."""
    supercell, _, symmetry = _get_cells("CsCl")
    cosets = get_coset_representatives(symmetry, supercell, P2S_MATRIX)
    n_operations = len(symmetry.symmetry_operations["rotations"])
    assert len(cosets.rotations) * 3 == n_operations
    assert not cosets.time_reversals.any()
    positions = supercell.scaled_positions
    for rotation, translation, permutation in zip(
        cosets.rotations, cosets.translations, cosets.permutations, strict=True
    ):
        moved = positions @ rotation.T + translation - positions[permutation]
        np.testing.assert_allclose(moved - np.rint(moved), 0, atol=1e-8)


def test_get_coset_representatives_of_screw():
    """The screw cell has the representatives that the CsCl cell lacks.

    These are a fractional translation without time reversal and a permutation
    that is not its own inverse.  The fit tests on "screw" need them to tell a
    permutation from its inverse and to check the translation.

    """
    supercell, _, symmetry = _get_cells("screw")
    cosets = get_coset_representatives(symmetry, supercell, P2S_MATRIX)
    assert not cosets.time_reversals.any()
    fractional = cosets.translations @ P2S_MATRIX.T
    fractional -= np.rint(fractional)
    assert (np.abs(fractional).max(axis=1) > 1e-8).any()
    assert any(
        (permutation != np.argsort(permutation)).any()
        for permutation in cosets.permutations
    )


def test_get_coset_representatives_with_time_reversal():
    """With magnetic moments, the operations with time reversal are kept."""
    supercell, _, symmetry = _get_cells("magnetic")
    cosets = get_coset_representatives(symmetry, supercell, P2S_MATRIX)
    assert cosets.time_reversals.any()
    assert not cosets.time_reversals.all()


def test_get_coset_representatives_of_identity_only():
    """With is_symmetry=False, the identity is the only representative."""
    supercell, _, symmetry = _get_cells("CsCl", is_symmetry=False)
    cosets = get_coset_representatives(symmetry, supercell, P2S_MATRIX)
    assert len(cosets.rotations) == 1


def test_get_coset_representatives_raises_for_nonmagnetic_primitive_cell():
    """A primitive translation that needs time reversal raises ValueError."""
    supercell = PhonopyAtoms(
        symbols=["Cs", "Cs"],
        cell=np.diag([2 * LATTICE_PARAMETER, LATTICE_PARAMETER, LATTICE_PARAMETER]),
        scaled_positions=[[0, 0, 0], [0.5, 0, 0]],
        magnetic_moments=[1.0, -1.0],
    )
    with pytest.raises(ValueError, match="primitive cell of the magnetic structure"):
        get_coset_representatives(
            Symmetry(supercell), supercell, np.diag([2, 1, 1]).astype("int64")
        )


@pytest.mark.parametrize("with_s2p_map", [False, True])
@pytest.mark.parametrize("moments", [[1.0, -1.0], [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]])
def test_fit_raises_for_primitive_cell_without_moments(
    moments: list, with_s2p_map: bool
):
    """A primitive cell that merges atoms of opposite moments raises ValueError.

    The supercell is 2x2x2 times the cubic cell of the antiferromagnetic CsCl-type
    cell of one species, and the primitive cell is the bcc cell of one atom,
    which phonopy builds without looking at the moments.  With
    is_symmetry=False, get_coset_representatives does not catch it, with or
    without s2p_map.

    """
    unitcell = PhonopyAtoms(
        symbols=["Cs", "Cs"],
        cell=np.eye(3) * LATTICE_PARAMETER,
        scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
        magnetic_moments=moments,
    )
    supercell = get_supercell(unitcell, CUBIC_SUPERCELL_MATRIX)
    p2s_matrix = np.array([[0, 2, 2], [2, 0, 2], [2, 2, 0]], dtype="int64")
    primitive = get_primitive(supercell, np.linalg.inv(p2s_matrix))
    assert len(primitive) == 1
    s2p_map = primitive.s2p_map if with_s2p_map else None
    symmetry = Symmetry(supercell, is_symmetry=False, s2p_map=s2p_map)
    with pytest.raises(ValueError, match="Magnetic moments of the supercell differ"):
        DLocalPotentialAugmentedFit(
            (4, 4, 4), p2s_matrix, supercell, primitive, symmetry, verbose=False
        )


def test_indices_of_minus_qpoints():
    """Q + (-q) is an integer vector, and -q differs from q for diag(3, 1, 1)."""
    fourier_series = FourierSeriesOnGrid(FFT_MESH, P2S_MATRIX)
    qpoints = fourier_series.qpoints
    indices = fourier_series.indices_of_minus_qpoints
    total = qpoints + qpoints[indices]
    np.testing.assert_allclose(total - np.rint(total), 0, atol=1e-12)
    assert (indices != np.arange(3)).sum() == 2


def test_add_to_augmented_normal_equation_rhs():
    """The function adds weights[q, i] * values[c][q, m]."""
    rng = np.random.default_rng(1)
    rhs = rng.standard_normal((2, 6, 3, 5)) + 1j * rng.standard_normal((2, 6, 3, 5))
    weights = rng.standard_normal((3, 6)) + 1j * rng.standard_normal((3, 6))
    values = [
        rng.standard_normal((3, 5)) + 1j * rng.standard_normal((3, 5)) for _ in range(2)
    ]
    expected = rhs + np.einsum("qi,cqm->ciqm", weights, np.array(values))
    add_to_augmented_normal_equation_rhs(rhs, weights, values)
    np.testing.assert_allclose(rhs, expected, atol=1e-14)


def test_apply_inverse_augmented_normal_matrices():
    """The function multiplies by the matrix of each q point."""
    rng = np.random.default_rng(2)
    rhs = rng.standard_normal((2, 6, 3, 5)) + 1j * rng.standard_normal((2, 6, 3, 5))
    matrices = rng.standard_normal((3, 6, 6)) + 1j * rng.standard_normal((3, 6, 6))
    expected = np.einsum("qij,cjqm->ciqm", matrices, rhs)
    apply_inverse_augmented_normal_matrices(rhs, matrices)
    np.testing.assert_allclose(rhs, expected, atol=1e-13)


def test_init_raises_for_primitive_cell_of_another_supercell():
    """A primitive cell of the supercell with the atoms rolled raises ValueError.

    primitive.s2p_map refers to the order of the atoms of its own supercell.
    With the order rolled by one, a Cl atom is compared with a Cs atom, which is
    not at a lattice vector of the primitive cell from it.

    """
    supercell, primitive, _ = _get_cells("CsCl")
    rolled_supercell = PhonopyAtoms(
        symbols=list(np.roll(supercell.symbols, 1)),
        cell=supercell.cell,
        scaled_positions=np.roll(supercell.scaled_positions, 1, axis=0),
    )
    with pytest.raises(ValueError, match="primitive cell has to be that of"):
        _get_fit(rolled_supercell, primitive, Symmetry(rolled_supercell))


def test_run_raises_for_wrong_shape_of_displacements():
    """The number of snapshots has to agree with that of the potentials."""
    supercell, primitive, symmetry = _get_cells("CsCl")
    perfect = _get_potential(supercell, magnetic=False)
    fit = _get_fit(supercell, primitive, symmetry)
    with pytest.raises(ValueError, match="Shape of displacements"):
        fit.run(perfect, [perfect], np.zeros((2, len(supercell), 3)))


def test_get_inverse_normal_matrices():
    """Discard the eigenvalues at or below rcond times the largest over all q.

    Each A(q) has rank 4 of 6, so two eigenvalues of each q are discarded, and
    A pinv(A) A = A.  A(q) of the last q is scaled by 1e-12, below rcond times
    the largest eigenvalue over all q, so all its eigenvalues are discarded
    and pinv(A) is zero there.  A threshold taken for each q separately would
    keep four of them.

    """
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((3, 6, 4)) + 1j * rng.standard_normal((3, 6, 4))
    normal_matrices = vectors @ np.swapaxes(vectors.conj(), 1, 2)
    inverse, eigvals = _get_inverse_normal_matrices(normal_matrices[:2], 1e-10)
    np.testing.assert_array_equal(
        (eigvals <= 1e-10 * eigvals.max()).sum(axis=1), [2, 2]
    )
    np.testing.assert_allclose(
        normal_matrices[:2] @ inverse @ normal_matrices[:2],
        normal_matrices[:2],
        atol=1e-10,
    )

    normal_matrices[2] *= 1e-12
    inverse, eigvals = _get_inverse_normal_matrices(normal_matrices, 1e-10)
    assert eigvals.shape == (3, 6)
    np.testing.assert_array_equal(inverse[2], 0)
    np.testing.assert_allclose(
        normal_matrices[:2] @ inverse[:2] @ normal_matrices[:2],
        normal_matrices[:2],
        atol=1e-10,
    )
