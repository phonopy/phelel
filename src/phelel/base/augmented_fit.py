"""Compute dV/du from supercells with the displacements of all atoms.

In a snapshot j, all atoms of the supercell are displaced.  To first order in
the displacements, the change of the local potential is a convolution over the
lattice points of dV/du and the displacements.  The lattice Fourier sums over
the lattice points l,

    u~_(tau alpha)(q) = sum_l u_(l tau alpha) exp(-2 pi i q . l),
    dV~(r_0, q) = sum_l dV(r_0 + R_l) exp(-2 pi i q . l) = N_c phi_q[dV](m),

therefore split the fit into one small least-squares problem for each q point
commensurate with the supercell.  u_(l tau alpha) is the displacement of the
atom tau in the cell l, r_0 is the point m of the output grid, and
phi_q[dV](m) is that of FourierSeriesOnGrid.get_phi.  N_c = |det(p2s_matrix)| is
the number of the primitive cells in the supercell, and n_p is the number of the
atoms of the primitive cell.

The augmented fit adds, for each snapshot, the snapshots transformed by the
operations of the supercell modulo the translations of the primitive cell
(get_coset_representatives).  The augmented snapshot (j, g) is snapshot j
transformed by the coset representative g.  The normal equations

    A(q) = sum_(j, g) conj(u~^(j,g)(q)) (x) u~^(j,g)(q),
    B(q) = sum_(j, g) conj(u~^(j,g)(q)) (x) phi_q[dV^(j,g)](m),

are accumulated one augmented snapshot at a time, and X(q) = pinv(A(q)) B(q),
where the eigenvalues of A(q) at or below rcond times the largest one over all
q points are discarded in pinv(A(q)).
With phi_q in place of N_c phi_q in B(q), sum_over_qpoints applied to X(q) gives
dV/du with the factor 1 / N_c of the inverse lattice Fourier sum included.

Only g_point_selection="shortest" of FourierSeriesOnGrid is used.

"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import Primitive
from phonopy.structure.symmetry import Symmetry

from phelel.base.fourier_series import FourierSeriesOnGrid, get_fourier_coefficients
from phelel.base.local_potential import get_grid_points
from phelel.base.magnetic_symmetry import (
    check_magnetic_moments,
    check_primitive_translations,
    get_time_reversals,
)
from phelel.base.potential_change import (
    add_to_augmented_normal_equation_rhs,
    apply_inverse_augmented_normal_matrices,
    apply_time_reversal_to_spin_components,
    rotate_spin_components,
)
from phelel.utils.spinor import SpinorRotationMatrices


@dataclass
class CosetRepresentatives:
    """Coset representatives of the operations of the supercell.

    G is the group of the operations (S|v) theta of the supercell, where
    theta = 1 for an operation without time reversal and theta = 1' for one
    with time reversal, and v is taken modulo the lattice of the supercell.

        T = {(E|R_l) 1 | R_l in L}

    is the N_c translations by the lattice points L of the primitive cell in
    the supercell.  T is fixed by the choice of the primitive cell, not by G,
    and the fit needs T to be a subgroup of G.  G/T is the set of the right
    cosets T (S|v) theta = {(S|v + R_l) theta | R_l in L}, and
    N_P = |G/T| = |G| / N_c is the number of the coset representatives.  A pure
    translation of G that is not in T, such as the anti-translation (E|v) 1' of
    a magnetic space group of type IV, is in another coset than T.

    Attributes
    ----------
    rotations : ndarray
        Rotations of (S|v) in the supercell basis vectors.
        dtype='int64', shape=(N_P, 3, 3)
    translations : ndarray
        Translations v in the supercell basis vectors.
        dtype='double', shape=(N_P, 3)
    time_reversals : ndarray
        True for theta = 1' and False for theta = 1.
        dtype='bool', shape=(N_P,)
    permutations : ndarray
        The operation k moves atom i to the site of atom permutations[k, i].
        dtype='int64', shape=(N_P, n_atoms)
    cartesian_rotations : ndarray
        Rotations S in Cartesian coordinates.
        dtype='double', shape=(N_P, 3, 3)

    """

    rotations: NDArray[np.int64]
    translations: NDArray[np.double]
    time_reversals: NDArray[np.bool_]
    permutations: NDArray[np.int64]
    cartesian_rotations: NDArray[np.double]


def get_coset_representatives(
    symmetry: Symmetry, supercell: PhonopyAtoms, p2s_matrix: NDArray[np.int64]
) -> CosetRepresentatives:
    """Return the coset representatives modulo the primitive translations.

    The operations of the supercell are grouped by the rotation, the time
    reversal and the translation modulo the lattice of the primitive cell.  A
    pure translation outside this lattice, as for a primitive cell that is not
    the smallest cell of the crystal, stays a representative of its own.

    Parameters
    ----------
    symmetry : Symmetry
        Symmetry of the supercell.
    supercell : PhonopyAtoms
        Perfect supercell.
    p2s_matrix : ndarray
        Supercell matrix relative to the primitive cell.
        dtype='int64', shape=(3, 3)

    Returns
    -------
    CosetRepresentatives

    Raises
    ------
    ValueError
        When T is not a subgroup of G (see CosetRepresentatives).  This check
        is skipped when the only operation is the identity, as with
        is_symmetry=False and no s2p_map.  With is_symmetry=False and s2p_map,
        the operations are the translations of the primitive cell, which
        phonopy takes from s2p_map without looking at the magnetic moments,
        so the check passes whatever the moments are.
        DLocalPotentialAugmentedFit checks the moments separately.

    """
    rotations = symmetry.symmetry_operations["rotations"]
    translations = symmetry.symmetry_operations["translations"]
    time_reversals = get_time_reversals(symmetry)
    supercell_lattice = supercell.cell.T
    primitive_lattice = supercell_lattice @ np.linalg.inv(p2s_matrix)

    representatives: list[int] = []
    coset_sizes: list[int] = []
    for i_op in range(len(rotations)):
        for i_coset, i_rep in enumerate(representatives):
            if time_reversals[i_op] != time_reversals[i_rep]:
                continue
            if not (rotations[i_op] == rotations[i_rep]).all():
                continue
            difference = p2s_matrix @ (translations[i_op] - translations[i_rep])
            difference -= np.rint(difference)
            if np.linalg.norm(primitive_lattice @ difference) < symmetry.tolerance:
                coset_sizes[i_coset] += 1
                break
        else:
            representatives.append(i_op)
            coset_sizes.append(1)

    num_cells = abs(round(np.linalg.det(p2s_matrix)))
    if len(rotations) > 1 and any(size != num_cells for size in coset_sizes):
        raise ValueError(
            "The translations of the primitive cell are not all symmetry operations "
            "of the supercell without time reversal. With magnetic moments, use a "
            "primitive cell of the magnetic structure."
        )

    selected_rotations = np.array(rotations[representatives], dtype="int64")
    return CosetRepresentatives(
        rotations=selected_rotations,
        translations=np.array(translations[representatives], dtype="double"),
        time_reversals=np.array(time_reversals[representatives], dtype=bool),
        permutations=np.array(
            symmetry.atomic_permutations[representatives], dtype="int64"
        ),
        cartesian_rotations=np.array(
            [
                supercell_lattice @ rotation @ np.linalg.inv(supercell_lattice)
                for rotation in selected_rotations
            ],
            dtype="double",
        ),
    )


class DLocalPotentialAugmentedFit:
    """Compute dV/du from snapshots by the augmented fit.

    See the module docstring.

    Attributes
    ----------
    grid_points : ndarray
        Grid points of get_grid_points.
        dtype='double', shape=(prod(fft_mesh) * det(p2s_matrix), 3)
    lattice_points : ndarray
        Lattice points of get_grid_points.
        dtype='int64', shape=(det(p2s_matrix), 3)
    dVdu : ndarray
        dV/du of the atoms of primitive.p2s_map on the grid points.
        dtype='complex128', shape=(ncdij, n_p, 3, n_grid_points)
    normal_matrix_eigenvalues : ndarray
        Eigenvalues of A(q) in ascending order, for each q point.  Those at or
        below rcond times the largest one are discarded in the inverse of
        A(q).  They mean that the snapshots do not determine dV/du along the
        eigenvectors, as at q = 0 for snapshots without a drift of the center
        of mass.
        dtype='double', shape=(N_c, 3 n_p)

    """

    def __init__(
        self,
        fft_mesh: Sequence[int] | NDArray,
        p2s_matrix: Sequence[Sequence[int]] | NDArray[np.int64],
        supercell: PhonopyAtoms,
        primitive: Primitive,
        symmetry: Symmetry,
        rcond: float = 1e-10,
        verbose: bool = True,
        lang: Literal["Rust", "Python"] = "Rust",
        time_reversal_form: Literal["conjugate", "linear"] = "conjugate",
    ):
        """Init method.

        Parameters
        ----------
        fft_mesh : array_like
            Mesh numbers for primitive cell.
            dtype='int64', shape=(3,)
        p2s_matrix : array_like
            Supercell matrix relative to primitive cell.
            dtype='int64', shape=(3, 3)
        supercell : PhonopyAtoms
            Perfect supercell.
        primitive : Primitive
            Primitive cell of the supercell.
        symmetry : Symmetry
            Symmetry of the supercell.
        rcond : float, optional
            Eigenvalues of A(q) at or below rcond times the largest eigenvalue
            over all q points are discarded in the inverse of A(q).  Default is
            1e-10.
        verbose : bool, optional
            To display log or not.
        lang : str, optional
            Implementation of FourierSeriesOnGrid and of the steps of
            phelel.base.potential_change that have one in Rust, "Rust"
            (default) by phonors or "Python" by NumPy.
        time_reversal_form : str, optional
            How time reversal of a coset representative is applied to phi_q.
            "conjugate" (default) applies
            apply_time_reversal_to_spin_components, which takes the complex
            conjugate, and takes the values at -q.  "linear" exchanges and
            negates the spin components at the same q without the complex
            conjugate (_apply_linear_time_reversal).  The two give the same
            dV/du when the potential changes are Hermitian, and "linear" is
            kept to compare them.

        Raises
        ------
        ValueError
            When time_reversal_form is neither "conjugate" nor "linear".
            When primitive is not the primitive cell of supercell and
            p2s_matrix, when the magnetic moments of supercell are not kept by
            the translations of primitive without time reversal (see
            check_primitive_translations), and as in get_coset_representatives.

        """
        self._supercell = supercell
        self._rcond = rcond
        self._verbose = verbose
        self._lang: Literal["Rust", "Python"] = lang
        if time_reversal_form not in ("conjugate", "linear"):
            raise ValueError(
                f'time_reversal_form must be "conjugate" or "linear", '
                f"not {time_reversal_form!r}."
            )
        self._time_reversal_form = time_reversal_form
        self._p2s_matrix = np.array(p2s_matrix, dtype="int64", order="C")
        self._grid_points, self._lattice_points = get_grid_points(
            fft_mesh, self._p2s_matrix
        )
        self._fourier_series = FourierSeriesOnGrid(
            fft_mesh,
            self._p2s_matrix,
            g_point_selection="shortest",
            supercell_lattice=supercell.cell,
            lang=lang,
        )
        check_primitive_translations(supercell, self._p2s_matrix, symmetry.tolerance)
        self._coset_representatives = get_coset_representatives(
            symmetry, supercell, self._p2s_matrix
        )
        self._set_atoms_and_phases(primitive, symmetry.tolerance)
        self._dVdu: NDArray[np.cdouble] | None = None
        self._normal_matrix_eigenvalues: NDArray[np.double] | None = None

    @property
    def grid_points(self) -> NDArray[np.double]:
        """Return grid points."""
        return self._grid_points

    @property
    def lattice_points(self) -> NDArray[np.int64]:
        """Return lattice points."""
        return self._lattice_points

    @property
    def dVdu(self) -> NDArray[np.cdouble] | None:
        """Return dV/du."""
        return self._dVdu

    @property
    def normal_matrix_eigenvalues(self) -> NDArray[np.double] | None:
        """Return eigenvalues of A(q)."""
        return self._normal_matrix_eigenvalues

    def run(
        self,
        V_loc_per: NDArray[np.cdouble],
        V_loc_disps: Sequence[NDArray[np.cdouble]],
        displacements: NDArray,
    ) -> None:
        """Compute dV/du.

        The results are stored in dVdu and normal_matrix_eigenvalues.

        Parameters
        ----------
        V_loc_per : ndarray
            Local potential of perfect supercell.
            dtype='complex128', shape=(ncdij, nz, ny, nx)
        V_loc_disps : sequence of ndarray
            Local potentials of the snapshots.
            dtype='complex128', shape=(ncdij, nz, ny, nx) each
        displacements : ndarray
            Displacements of all atoms of the snapshots in Cartesian
            coordinates.
            dtype='double', shape=(n_snapshots, n_atoms, 3)

        """
        ncdij = V_loc_per.shape[0]
        displacements = self._get_checked_displacements(
            ncdij, V_loc_disps, displacements
        )

        # BLAS is called only here, before the loops, where phonors runs its
        # kernels in parallel with rayon.
        normal_matrices = self._get_normal_matrices(displacements)
        inverse_normal_matrices, self._normal_matrix_eigenvalues = (
            _get_inverse_normal_matrices(normal_matrices, self._rcond)
        )
        if self._verbose:
            self._print_discarded_eigenvalues()
        spin_rotations = self._get_spin_rotations() if ncdij == 4 else None

        normal_equation_rhs = self._get_normal_equation_rhs(
            V_loc_per, V_loc_disps, displacements, spin_rotations
        )
        apply_inverse_augmented_normal_matrices(
            normal_equation_rhs, inverse_normal_matrices
        )
        self._dVdu = self._sum_over_qpoints(normal_equation_rhs)

    def _get_checked_displacements(
        self,
        ncdij: int,
        V_loc_disps: Sequence[NDArray[np.cdouble]],
        displacements: NDArray,
    ) -> NDArray[np.double]:
        """Return the displacements after checking the inputs of run.

        The magnetic moments of the supercell are checked against ncdij by
        check_magnetic_moments, and the displacements must have one row of all
        atoms for each potential of V_loc_disps.

        Parameters
        ----------
        ncdij : int
            Number of spin components of the potentials.
        V_loc_disps : sequence of ndarray
            See run.
        displacements : array_like
            See run.

        Returns
        -------
        ndarray
            dtype='double', shape=(n_snapshots, n_atoms, 3), C-contiguous

        Raises
        ------
        ValueError
            When the shape of displacements is not (len(V_loc_disps),
            n_atoms, 3), and as in check_magnetic_moments.

        """
        check_magnetic_moments(self._supercell, ncdij)
        displacements = np.array(displacements, dtype="double", order="C")
        expected_shape = (len(V_loc_disps), len(self._supercell), 3)
        if displacements.shape != expected_shape:
            raise ValueError(
                f"Shape of displacements is {displacements.shape}, not "
                f"{expected_shape}."
            )
        return displacements

    def _print_discarded_eigenvalues(self) -> None:
        """Print how many eigenvalues of A(q) are discarded in pinv(A(q)).

        They are those at or below rcond times the largest eigenvalue over all
        q points (_get_inverse_normal_matrices).

        """
        eigvals = self._normal_matrix_eigenvalues
        assert eigvals is not None
        num_discarded = int((eigvals <= self._rcond * eigvals.max()).sum())
        print(
            f"Discarded {num_discarded} of {eigvals.size} eigenvalues of "
            f"A(q) at or below {self._rcond} times the largest one."
        )

    def _get_normal_equation_rhs(
        self,
        V_loc_per: NDArray[np.cdouble],
        V_loc_disps: Sequence[NDArray[np.cdouble]],
        displacements: NDArray[np.double],
        spin_rotations: list[NDArray[np.cdouble]] | None,
    ) -> NDArray[np.cdouble]:
        """Return B(q) = U(q)^H Y(q), accumulated one augmented snapshot at a time.

        Row (j, g) of Y(q) is phi_q of the potential change of the augmented
        snapshot (j, g), and column (j, g) of U(q)^H is conj(u~^(j,g)(q)).
        B(q) is the sum over (j, g) of their outer products, and each pair
        (j, g) adds its outer product to all elements of B(q)
        (add_to_augmented_normal_equation_rhs).  Y(q) is not stored, because
        it would take the memory of all augmented snapshots on the output grid.

        The loop over the snapshots j is outside, so that the Fourier
        coefficients of the potential change of snapshot j are computed by one
        FFT and used for all coset representatives g.  The loops do not call
        BLAS (see the comment in run).

        Parameters
        ----------
        V_loc_per, V_loc_disps : ndarray, sequence of ndarray
            See run.
        displacements : ndarray
            See run.
            dtype='double', shape=(n_snapshots, n_atoms, 3)
        spin_rotations : list of ndarray or None
            D^(1/2) of the coset representatives for ncdij=4, otherwise None.

        Returns
        -------
        ndarray
            B(q) with phi_q in place of N_c phi_q (see the module docstring).
            dtype='complex128',
            shape=(ncdij, 3 n_p, N_c, n_grid_points / N_c)

        """
        ncdij = V_loc_per.shape[0]
        num_cells = len(self._lattice_points)
        normal_equation_rhs = np.zeros(
            (
                ncdij,
                3 * self._num_primitive_atoms,
                num_cells,
                len(self._grid_points) // num_cells,
            ),
            dtype="complex128",
            order="C",
        )
        for displacement, V_loc_disp in zip(displacements, V_loc_disps, strict=True):
            coefficients = [
                get_fourier_coefficients(dV) for dV in V_loc_disp - V_loc_per
            ]
            for i_op in range(len(self._coset_representatives.rotations)):
                phis = self._get_phi_of_augmented_snapshot(
                    coefficients,
                    i_op,
                    None if spin_rotations is None else spin_rotations[i_op],
                )
                weights = self._get_displacement_sums(displacement, i_op).conj()
                add_to_augmented_normal_equation_rhs(
                    normal_equation_rhs,
                    weights,
                    [phi.reshape(num_cells, -1) for phi in phis],
                )
            if self._verbose:
                print("Added a snapshot to the normal equations.")
        return normal_equation_rhs

    def _set_atoms_and_phases(self, primitive: Primitive, tolerance: float) -> None:
        """Set the sublattices and exp(-2 pi i q . l) of each atom of the supercell.

        For atom i of the supercell, the atom of the primitive cell is
        primitive.s2p_map[i], and l is the difference of the fractional
        coordinates of the two atoms in the basis of the primitive cell.  This l
        can differ from one of get_lattice_points by a lattice vector of the
        supercell, which does not change the phase.

        Raises
        ------
        ValueError
            When a difference is farther than tolerance in Cartesian
            coordinates from a lattice vector of the primitive cell, as for a
            primitive cell that is not that of the supercell and p2s_matrix.
            The magnetic moments are checked before by
            check_primitive_translations in __init__.

        """
        p2s_map = np.array(primitive.p2s_map, dtype="int64")
        s2p_map = np.array(primitive.s2p_map, dtype="int64")
        self._num_primitive_atoms = len(p2s_map)
        self._sublattices = [np.where(s2p_map == atom)[0] for atom in p2s_map]
        positions = self._supercell.scaled_positions
        differences = (positions - positions[s2p_map]) @ self._p2s_matrix.T
        lattice_vectors = np.rint(differences).astype("int64")
        primitive_lattice = np.linalg.inv(self._p2s_matrix.T) @ self._supercell.cell
        residuals = (differences - lattice_vectors) @ primitive_lattice
        if (np.linalg.norm(residuals, axis=1) > tolerance).any():
            raise ValueError(
                "Atoms of the supercell are not at lattice vectors of the primitive "
                "cell from their atoms of the primitive cell. The primitive cell "
                "has to be that of the supercell and p2s_matrix."
            )
        self._phases = np.exp(
            -2j * np.pi * (self._fourier_series.qpoints @ lattice_vectors.T)
        )

    def _get_displacement_sums(
        self, displacement: NDArray[np.double], i_op: int
    ) -> NDArray[np.cdouble]:
        """Return the lattice Fourier sums of the displacements of one (j, g).

        The operation moves atom i to the site of atom perm[i] and rotates its
        displacement by S.  np.einsum without optimize does not call BLAS, which
        would compete with the rayon threads of phonors in the loops of run.

        Parameters
        ----------
        displacement : ndarray
            Displacements of all atoms of one snapshot.
            dtype='double', shape=(n_atoms, 3)
        i_op : int
            Index of the coset representative.

        Returns
        -------
        ndarray
            u~_(tau alpha)(q) of the augmented snapshot, with the index
            i = 3 tau + alpha of the second axis.
            dtype='complex128', shape=(N_c, 3 n_p)

        """
        rotated = np.zeros_like(displacement)
        rotated[self._coset_representatives.permutations[i_op]] = np.einsum(
            "ij,nj->ni",
            self._coset_representatives.cartesian_rotations[i_op],
            displacement,
        )
        sums = np.zeros(
            (self._phases.shape[0], self._num_primitive_atoms, 3), dtype="complex128"
        )
        for i_atom, sublattice in enumerate(self._sublattices):
            sums[:, i_atom] = np.einsum(
                "qn,na->qa", self._phases[:, sublattice], rotated[sublattice]
            )
        return sums.reshape(sums.shape[0], -1)

    def _get_normal_matrices(
        self, displacements: NDArray[np.double]
    ) -> NDArray[np.cdouble]:
        """Return A(q) of each q point.

        A(q) = U(q)^H U(q) is the matrix of the normal equations, where U(q)
        has one row u~^(j,g)(q) for each augmented snapshot (j, g)
        (_get_displacement_sums), so that

            A(q) = sum_(j, g) conj(u~^(j,g)(q)) (x) u~^(j,g)(q).

        Parameters
        ----------
        displacements : ndarray
            See run.

        Returns
        -------
        ndarray
            dtype='complex128', shape=(N_c, 3 n_p, 3 n_p)

        """
        n_pairs = 3 * self._num_primitive_atoms
        normal_matrices = np.zeros(
            (self._phases.shape[0], n_pairs, n_pairs), dtype="complex128"
        )
        for displacement in displacements:
            for i_op in range(len(self._coset_representatives.rotations)):
                sums = self._get_displacement_sums(displacement, i_op)
                normal_matrices += np.einsum("qi,qj->qij", sums.conj(), sums)
        return normal_matrices

    def _get_spin_rotations(self) -> list[NDArray[np.cdouble]]:
        """Return the 2x2 matrices D^(1/2) of the coset representatives."""
        lattice = self._supercell.cell.T
        return [
            SpinorRotationMatrices(rotation, lattice).run().Delta
            for rotation in self._coset_representatives.rotations
        ]

    def _get_phi_of_augmented_snapshot(
        self,
        coefficients: list[NDArray[np.cdouble]],
        i_op: int,
        spin_rotation: NDArray[np.cdouble] | None,
    ) -> list[NDArray[np.cdouble]]:
        """Return phi_q(m) of the potential change of one augmented snapshot.

        The rotation in spin space has no complex conjugate, so it is applied
        at the same q.  Time reversal takes the complex conjugate of the values
        on the grid, and the conjugate of the lattice Fourier sum at -q is the
        lattice Fourier sum of the conjugate at q.  So the values at q are taken
        from -q after apply_time_reversal_to_spin_components.  With
        time_reversal_form="linear", _apply_linear_time_reversal is applied at
        the same q instead.

        Parameters
        ----------
        coefficients : list of ndarray
            Fourier coefficients of the potential change of the snapshot for
            each spin component.
        i_op : int
            Index of the coset representative.
        spin_rotation : ndarray or None
            D^(1/2) of the operation for ncdij=4, otherwise None.

        Returns
        -------
        list of ndarray
            dtype='complex128', shape=(N_c, N^p_3, N^p_2, N^p_1) each

        """
        phis = self._fourier_series.get_phi(
            coefficients,
            self._coset_representatives.rotations[i_op],
            self._coset_representatives.translations[i_op],
        )
        if spin_rotation is not None:
            rotate_spin_components(phis, spin_rotation, lang=self._lang)
        if self._coset_representatives.time_reversals[i_op]:
            if self._time_reversal_form == "linear":
                return _apply_linear_time_reversal(phis)
            apply_time_reversal_to_spin_components(phis, lang=self._lang)
            minus_q = self._fourier_series.indices_of_minus_qpoints
            phis = [phi[minus_q] for phi in phis]
        return phis

    def _sum_over_qpoints(self, solutions: NDArray[np.cdouble]) -> NDArray[np.cdouble]:
        """Return dV/du on the grid points from X(q).

        Parameters
        ----------
        solutions : ndarray
            X(q).
            dtype='complex128', shape=(ncdij, 3 n_p, N_c, n_mesh_points)

        Returns
        -------
        ndarray
            dtype='complex128', shape=(ncdij, n_p, 3, n_grid_points)

        """
        ncdij, n_pairs = solutions.shape[:2]
        dVdu = np.zeros(
            (ncdij, n_pairs, len(self._grid_points)), dtype="complex128", order="C"
        )
        for i_component in range(ncdij):
            for i_pair in range(n_pairs):
                dVdu[i_component, i_pair] = self._fourier_series.sum_over_qpoints(
                    solutions[i_component, i_pair]
                )
        return dVdu.reshape(ncdij, self._num_primitive_atoms, 3, -1)


def _apply_linear_time_reversal(
    phis: list[NDArray[np.cdouble]],
) -> list[NDArray[np.cdouble]]:
    """Return phi_q of the time-reversed spin components without conjugate.

    For Hermitian spin components [v_0, v_1, v_2, v_3] in the order
    [11, 12, 21, 22] of VASP, v_0 and v_3 are real and v_2 = conj(v_1), so the
    complex conjugate exchanges v_1 and v_2.  Time reversal of
    apply_time_reversal_to_spin_components then becomes the linear map

        [v_0, v_1, v_2, v_3] -> [v_3, -v_1, -v_2, v_0],

    which has no complex conjugate and is applied to phi_q at the same q.  For
    two components, [v_0, v_1] -> [v_1, v_0], and one component is kept.  For
    data that are not Hermitian, the result differs from that of
    apply_time_reversal_to_spin_components followed by the values at -q.

    Parameters
    ----------
    phis : list of ndarray
        phi_q(m) of one, two or four spin components.  They are not modified.
        dtype='complex128', shape=(N_c, N^p_3, N^p_2, N^p_1) each

    Returns
    -------
    list of ndarray
        phi_q(m) of the time-reversed spin components.

    """
    if len(phis) == 4:
        return [phis[3], -phis[1], -phis[2], phis[0]]
    if len(phis) == 2:
        return [phis[1], phis[0]]
    return phis


def _get_inverse_normal_matrices(
    normal_matrices: NDArray[np.cdouble], rcond: float
) -> tuple[NDArray[np.cdouble], NDArray[np.double]]:
    """Return pinv(A(q)) and the eigenvalues of A(q) of each q point.

    With A(q) = V(q) diag(lambda(q)) V(q)^H by eigh,

        pinv(A(q)) = V(q) diag(1 / lambda(q)) V(q)^H,

    where 1 / lambda is replaced by zero for lambda at or below rcond times
    the largest lambda over all q points.  An eigenvalue that is negative by
    round-off is discarded too, since A(q) is positive semidefinite.

    Parameters
    ----------
    normal_matrices : ndarray
        A(q) of each q point, Hermitian and positive semidefinite.
        dtype='complex128', shape=(N_c, 3 n_p, 3 n_p)
    rcond : float
        Eigenvalues at or below rcond times the largest eigenvalue over all q
        points are discarded.

    Returns
    -------
    inverse_normal_matrices : ndarray
        pinv(A(q)) of each q point.
        dtype='complex128', shape=(N_c, 3 n_p, 3 n_p)
    eigenvalues : ndarray
        Eigenvalues of A(q) in ascending order, for each q point.
        dtype='double', shape=(N_c, 3 n_p)

    """
    eigvals, eigvecs = np.linalg.eigh(normal_matrices)
    is_kept = eigvals > rcond * eigvals.max()
    inverse_eigvals = np.zeros_like(eigvals)
    inverse_eigvals[is_kept] = 1 / eigvals[is_kept]
    inverse_normal_matrices = (eigvecs * inverse_eigvals[:, None, :]) @ np.swapaxes(
        eigvecs.conj(), 1, 2
    )
    return inverse_normal_matrices, np.array(eigvals, dtype="double")
