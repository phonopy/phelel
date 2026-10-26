"""Calculation of DDijQij."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.symmetry import Symmetry

from phelel.base.magnetic_symmetry import check_magnetic_moments, get_time_reversals
from phelel.base.symmetrization import (
    check_displaced_atoms,
    collect_site_symmetry_operations,
    get_displacements_with_rotations,
)
from phelel.exception import MagneticMomentsWarning
from phelel.utils.spherical_harmonics import LxLyLzMatrices, SHRotationMatrices
from phelel.utils.spinor import (
    apply_time_reversal_in_spin_space,
    rotate_delta_vals_in_spin_space,
)


class DeltaDijQij:
    """Container to store delta-Dij and delta-qij.

    Attributes
    ----------
    dDij : ndarray
        Difference of PAW strengths with/without a displacement.
        dtype=complex128
        shape=(ncdij, atoms, lm, lm')
    dqij :
        Difference of <phi_i|phi_j>-<phi_i~|phi_j~>  with/without a
        displacement.
        dtype=complex128
        shape=(ncdij, atoms, lm, lm')
    displacement : dict
        Dispalcement of one atom.
        keys :
            'displacement': Displacement in Cartesian coordinates
            'number': Index of displaced atom
    lm_channels : list of dicts
        lm channels in Dij and qij of atoms
        keys of each distionary:
            'channels' : List of l channels
            'l', 'm' in each channel : l and list of m

    """

    def __init__(
        self,
        Dij_per: NDArray,
        Dij_disp: NDArray,
        qij_per: NDArray,
        qij_disp: NDArray,
        displacement: dict,
        lm_channels: list[dict],
    ):
        """Init method.

        Note
        ----
        The lm index runs over the channels of ``lm_channels`` of the atom in
        their order, and over m = -l, ..., l in each channel.  An l can have
        more than one channel.  For example, with the channels s, s, p, p of
        carbon, the order is
            (l=0,m=0), (l=0,m=0), (l=1,m=-1), (l=1,m=0), (l=1,m=1),
            (l=1,m=-1), (l=1,m=0), (l=1,m=1).
        The two lm indices of Dij and qij are in the same order.

        Parameters
        ----------
        Dij_per : ndarray
            PAW strength of perfect supercell
            dtype=complex128
            shape=(ncdij, atoms, lm, lm')
        Dij_disp : ndarray
            PAW strength of supercell with a displacement
            dtype=complex128
            shape=(ncdij, atoms, lm, lm')
        qij_per : ndarray
            <phi_i|phi_j>-<phi_i~|phi_j~> of perfect supercell
            dtype=complex128
            shape=(ncdij, atoms, lm, lm')
        qij_disp : ndarray
            <phi_i|phi_j>-<phi_i~|phi_j~> of supercell with a displacement
            dtype=complex128
            shape=(ncdij, atoms, lm, lm')
        displacement : dict
            Dispalcement of one atom
            keys :
                'displacement': Displacement in Cartesian coordinates
                'number': Index of displaced atom
        lm_channels : list of dicts
            lm channels in Dij and qij of atoms
            keys of each distionary:
                'channels' : List of l channels
                'l', 'm' in each channel : l and list of m

        """
        self.dDij = Dij_disp - Dij_per
        self.dqij = qij_disp - qij_per
        self.displacement = displacement
        self.lm_channels = lm_channels


class DDijQijFit:
    """Compute dDij/du and dqij/du for one atom.

    Note
    ----
    Currently ncdij=1.

    Attributes
    ----------
    dDijdu : ndarray
        Derivative of Dij with respect to displacement of an atom
        dtype=complex128
        shape=(ncdij, atom_indices, 3, supercell_atoms, lm, lm')
    dqijdu : ndarray
        Derivative of qij (<phi_i|phi_j>-<phi_i~|phi_j~>) with respect to
        displacement of an atom
        dtype=complex128
        shape=(ncdij, atom_indices, 3, supercell_atoms, lm, lm')
    atom_indices : list of int
        Atom indices in supercell where dV will be computed. If those indices
        don't belong to symmetrically equivalent atoms to the dispalced atom.

    """

    def __init__(
        self,
        delta_Dij_qijs: list[DeltaDijQij],
        supercell: PhonopyAtoms,
        symmetry: Symmetry,
        atom_indices: Sequence[int] | NDArray | None = None,
        verbose: bool = True,
    ):
        """Init method.

        Note
        ----

        See array shape of Dij and qij at DeltaDijQij class. The private
        vaiables and algorithm related to site symmetry (or site symmetry like)
        are equivalent to those of LocalPotentialInterpolation. A bit more
        detailed comments are written there.

        Parameters
        ----------
        dDij_qijs : list of DeltaDijQij's
            Changes of Dij's and qij's between displaced and perfect supercells
        supercell : PhonopyAtoms
            Perfect supercell
        symmetry : Symmetry
            Symmetry of supercell
        atom_indices : list of int
            Atom indices in supercell where dDij and dqij will be expected to be
            computed. If None, supposed to be all atoms. Internally only
            symmetrically equivalent atoms to the dispalced atom are selected to
            compute.
        verbose : bool
            To display log or not

        """
        # Inputs
        self._delta_Dij_qijs = delta_Dij_qijs
        self._supercell = supercell
        self._symmetry = symmetry
        self._verbose = verbose
        self._atom_indices_in = atom_indices

        self._dDijdu: NDArray | None = None
        self._dqijdu: NDArray | None = None
        self._atom_indices: NDArray | None = None

        # See comments in LocalPotentialInterpolation for these variables
        self._i_atom: int | None = None
        self._sitesym_sets: NDArray | None = None

        self._setup()

    @property
    def dDijdu(self):
        """Return dDij/du."""
        return self._dDijdu

    @property
    def dqijdu(self):
        """Return dqij/du."""
        return self._dqijdu

    @property
    def atom_indices(self) -> NDArray | None:
        """Return atom indices where dDijdu and dqijdu are stored."""
        return self._atom_indices

    def __iter__(self):
        """Activate iterator."""
        return self

    def __next__(self):
        """Iterate over atom_indices."""
        assert self._atom_indices is not None
        assert self._i_atom is not None
        if len(self._atom_indices) == self._i_atom:
            raise StopIteration

        if self._verbose:
            print(
                "Computing dDij/du and dqij/du of displaced atom %d:"
                % (self._atom_indices[self._i_atom] + 1)
            )

        self._run_at_atom()
        self._i_atom += 1

    def run(self):
        """Calculate by iterating at atom_indices."""
        for _ in self:
            pass

    def __str__(self) -> str:
        """Return Dij and qij in str fashion."""
        assert self._dDijdu is not None
        assert self._dqijdu is not None
        text = ""
        Dij_shape = self._dDijdu.shape[-2:]
        for i_spinor, (Dij_spinor, qij_spinor) in enumerate(
            zip(self._dDijdu, self._dqijdu, strict=True)
        ):
            for i_eatom, (Dij_eatom, qij_eatom) in enumerate(
                zip(Dij_spinor, qij_spinor, strict=True)
            ):
                for i in range(len(self._supercell)):
                    for j in range(3):
                        text += self._str_format(
                            Dij_eatom[j, i].real,
                            i_spinor,
                            i_eatom,
                            i,
                            j,
                            "D",
                            "real",
                            Dij_shape,
                        )
                        text += self._str_format(
                            qij_eatom[j, i].real,
                            i_spinor,
                            i_eatom,
                            i,
                            j,
                            "q",
                            "real",
                            Dij_shape,
                        )
        return text

    def _str_format(
        self,
        xij: NDArray,
        i_spinor: int,
        i_eatom: int,
        i: int,
        j: int,
        str1: str,
        str2: str,
        Dij_shape: tuple,
    ) -> str:
        text = (
            f"Spin {i_spinor + 1}, Disp-Atom {i_eatom + 1}, "
            f"Atom {i + 1} d{str1}ij/d{'xyz'[j]} {str2} part "
            f"({Dij_shape[0]}, {Dij_shape[1]})\n"
        )
        for xij_l in xij:
            text += " ".join(["%8.1e" % v for v in xij_l]) + "\n"
        return text

    def _setup(self):
        check_magnetic_moments(self._supercell, self._delta_Dij_qijs[0].dDij.shape[0])
        self._i_atom = 0
        disp_atom = self._delta_Dij_qijs[0].displacement["number"]
        sitesym_sets, equiv_atoms = collect_site_symmetry_operations(
            disp_atom, self._symmetry
        )

        if self._atom_indices_in is None:
            atoms = np.arange(len(self._supercell))
        else:
            atoms = self._atom_indices_in
        self._atom_indices = np.array(
            [i for i in atoms if i in equiv_atoms], dtype="int64"
        )
        # The rows of sitesym_sets follow equiv_atoms.  They are picked in the
        # order of the returned atoms.
        i_equiv_atom = {int(atom): i for i, atom in enumerate(equiv_atoms)}
        self._sitesym_sets = sitesym_sets[
            [i_equiv_atom[int(atom)] for atom in self._atom_indices]
        ]

        dtype = "c%d" % (np.dtype("double").itemsize * 2)
        natom = len(self._supercell)
        ncdij = self._delta_Dij_qijs[0].dDij.shape[0]
        lmdim = self._delta_Dij_qijs[0].dDij.shape[-2]
        self._dDijdu = np.zeros(
            (ncdij, len(self._atom_indices), 3, natom, lmdim, lmdim),
            dtype=dtype,
            order="C",
        )
        self._dqijdu = np.zeros(
            (ncdij, len(self._atom_indices), 3, natom, lmdim, lmdim),
            dtype=dtype,
            order="C",
        )

    def _run_at_atom(self):
        assert self._dDijdu is not None
        assert self._dqijdu is not None
        assert self._sitesym_sets is not None
        sitesyms = self._sitesym_sets[self._i_atom]
        natom = len(self._supercell)
        lmdim = self._delta_Dij_qijs[0].dDij.shape[-2]
        dtype = f"c{np.dtype('double').itemsize * 2}"
        ncdij = self._dDijdu.shape[0]

        lattice = self._supercell.cell.T
        rotations = self._symmetry.symmetry_operations["rotations"][sitesyms]
        time_reversals = get_time_reversals(self._symmetry)[sitesyms]
        atomic_permutations = self._symmetry.atomic_permutations[sitesyms]

        disps = get_displacements_with_rotations(
            rotations, lattice, self._delta_Dij_qijs
        )
        disps_inv = np.linalg.pinv(disps)
        dDij_rotated_all = np.zeros(
            (ncdij, len(disps), natom * lmdim**2), dtype=dtype, order="C"
        )
        dqij_rotated_all = np.zeros(
            (ncdij, len(disps), natom * lmdim**2), dtype=dtype, order="C"
        )

        count = 0
        for delta_Dij_qij in self._delta_Dij_qijs:
            for r, perm, tr in zip(
                rotations, atomic_permutations, time_reversals, strict=True
            ):
                dDij_rotated, dqij_rotated = self._rotate_Dij_qij(
                    ncdij, delta_Dij_qij, perm, r
                )
                if ncdij == 4:  # Need to rotate in spin space, too.
                    dDij_spinor_rotated = rotate_delta_vals_in_spin_space(
                        dDij_rotated, r, lattice
                    )
                    dqij_spinor_rotated = rotate_delta_vals_in_spin_space(
                        dqij_rotated, r, lattice
                    )
                    dDij_rotated = dDij_spinor_rotated
                    dqij_rotated = dqij_spinor_rotated
                if tr:
                    dDij_rotated = apply_time_reversal_in_spin_space(dDij_rotated)
                    dqij_rotated = apply_time_reversal_in_spin_space(dqij_rotated)
                for i_cdij in range(ncdij):
                    dDij_rotated_all[i_cdij, count] = dDij_rotated[i_cdij]
                    dqij_rotated_all[i_cdij, count] = dqij_rotated[i_cdij]
                count += 1

        # Compute dDij/du and dqij/du
        # shape=(ncdij, len(self._atom_indices), 3, natom, lmdim, lmdim)
        shape = (ncdij, 3, natom, lmdim, lmdim)
        self._dDijdu[:, self._i_atom] = (disps_inv @ dDij_rotated_all).reshape(shape)
        self._dqijdu[:, self._i_atom] = (disps_inv @ dqij_rotated_all).reshape(shape)

    def _rotate_Dij_qij(
        self, ncdij: int, delta_Dij_qij: DeltaDijQij, perm: NDArray, r: NDArray
    ) -> tuple[list[NDArray], list[NDArray]]:
        """Rotate dDij and dqij by the symmetry operation g with rotation r.

        dDij and dqij of each spin component are transformed by g in the space
        of the atomic-like orbitals, g dDij g^-1, as written in
        _rotate_Dij_in_orbital_space. The spin rotation for ncdij=4 and time
        reversal are applied by the caller. The displacement u is rotated to
        R u in get_displacements_with_rotations, so the result is dDij and dqij
        of the displacement R u.

        """
        shr = SHRotationMatrices(r, self._supercell.cell.T, LxLyLzMatrices().run())
        shr.run()
        assert shr.Delta is not None
        lm_channels = delta_Dij_qij.lm_channels
        rot_dDij: list[NDArray] = []
        rot_dqij: list[NDArray] = []
        for i_cdij in range(ncdij):
            rot_dDij.append(
                _rotate_Dij_in_orbital_space(
                    delta_Dij_qij.dDij[i_cdij], lm_channels, shr.Delta, perm
                ).ravel()
            )
            rot_dqij.append(
                _rotate_Dij_in_orbital_space(
                    delta_Dij_qij.dqij[i_cdij], lm_channels, shr.Delta, perm
                ).ravel()
            )
        return rot_dDij, rot_dqij


class DDijQij:
    """Compute dDij/du and dqij/du.

    Note
    ----
    ncdij = 1 without spin
    ncdij = 2 with collinear spin (up and down)
    ncdij = 4 with non-collinear spin (2x2 localpotential)

    Attributes
    ----------
    dDijdu : ndarray
        Derivative of Dij with respect to displacements
        dtype=complex128
        shape=(ncdij, atom_indices, 3, supercell_atoms, lm, lm')
    dqijdu : ndarray
        Derivative of qij (<phi_i|phi_j>-<phi_i~|phi_j~>) with respect to
        displacements
        dtype=complex128
        shape=(ncdij, atom_indices, 3, supercell_atoms, lm, lm')
    lm_channels : list of dicts
        lm channels in Dij and qij of atoms
        keys of each distionary:
            'channels' : List of l channels
            'l', 'm' in each channel : l and list of m
    symmetry : Symmetry
        Symmetry of supercell
    atom_indices : ndarray
        Atom indices in supercell where dV is computed. This is made as
        np.unique(atom_indices given at __init__). If None, all atoms in
        supercell.
        shape=(len(atom_indices),)
        dtype='int64'

    """

    def __init__(
        self,
        supercell: PhonopyAtoms,
        symmetry: Symmetry | None = None,
        atom_indices: Sequence[int] | NDArray | None = None,
        verbose: bool = True,
    ):
        """Init method.

        Parameters
        ----------
        supercell : PhonopyAtoms
            Supercell
        symmetry : Symmetry, optional
            Symmetry of supercell. If None, symmetry is searched in this class
            object.
        atom_indices : list of int, optional
            Atom indices in supercell where dDijdu will be expected to be
            computed. If None, supposed to be all atoms. Internally only
            symmetrically equivalent atoms to the dispalced atom are selected
            to compute.

        """
        self._supercell = supercell
        self._verbose = verbose

        if atom_indices is None:
            self._atom_indices = np.arange(len(self._supercell), dtype="int64")
        else:
            self._atom_indices = np.array(np.unique(atom_indices), dtype="int64")
        if symmetry is None:
            self.symmetry = Symmetry(supercell)
        else:
            self.symmetry = symmetry

        self._Dij: NDArray | None = None
        self._qij: NDArray | None = None
        self._dDijdu: NDArray | None = None
        self._dqijdu: NDArray | None = None

    @property
    def dDijdu(self):
        """Getter and setter of dDijdu."""
        return self._dDijdu

    @dDijdu.setter
    def dDijdu(self, dDijdu):
        natom = len(self._supercell)
        shape = (len(self._atom_indices), 3, natom)
        if dDijdu.shape[1:4] == shape:
            dtype = "c%d" % (np.dtype("double").itemsize * 2)
            self._dDijdu = np.array(dDijdu, dtype=dtype, order="C")
        else:
            raise RuntimeError(
                "Array shape[1:4] disagreement is found, %s!=%s."
                % (shape, dDijdu.shape[1:4])
            )

    @property
    def dqijdu(self):
        """Getter and setter of dqijdu."""
        return self._dqijdu

    @dqijdu.setter
    def dqijdu(self, dqijdu):
        natom = len(self._supercell)
        shape = (len(self._atom_indices), 3, natom)
        if dqijdu.shape[1:4] == shape:
            dtype = "c%d" % (np.dtype("double").itemsize * 2)
            self._dqijdu = np.array(dqijdu, dtype=dtype, order="C")
        else:
            raise RuntimeError(
                "Array shape[1:4] disagreement is found, %s!=%s."
                % (shape, dqijdu.shape[1:4])
            )

    @property
    def Dij(self):
        """Getter and setter of Dij."""
        return self._Dij

    @Dij.setter
    def Dij(self, Dij):
        if Dij.shape[1] == len(self._atom_indices):
            dtype = "c%d" % (np.dtype("double").itemsize * 2)
            self._Dij = np.array(Dij, dtype=dtype, order="C")
        else:
            raise RuntimeError(
                "Number of atoms in supercell disagrees, %s!=%s."
                % (len(self._atom_indices), Dij.shape[1])
            )

    @property
    def qij(self):
        """Getter and setter of qij."""
        return self._qij

    @qij.setter
    def qij(self, qij):
        if qij.shape[1] == len(self._atom_indices):
            dtype = "c%d" % (np.dtype("double").itemsize * 2)
            self._qij = np.array(qij, dtype=dtype, order="C")
        else:
            raise RuntimeError(
                "Number of atoms in supercell disagrees, %s!=%s."
                % (len(self._atom_indices), qij.shape[1])
            )

    @property
    def atom_indices(self) -> NDArray:
        """Return atom indices where dDijdu and dqijdu are stored."""
        return self._atom_indices

    def run(self, Dij_per, Dij_disps, qij_per, qij_disps, displacements, lm_channels):
        """Compute dDij/du and dqij/du.

        Parameters
        ----------
        Dij_per : ndarray
            PAW strength of perfect supercell
            dtype=complex128
            shape=(ncdij, atoms, lm, lm')
        Dij_disp : ndarray
            PAW strength of supercell with a displacement
            dtype=complex128
            shape=(ncdij, atoms, lm, lm')
        qij_per : ndarray
            <phi_i|phi_j>-<phi_i~|phi_j~> of perfect supercell
            dtype=complex128
            shape=(ncdij, atoms, lm, lm')
        qij_disp : ndarray
            <phi_i|phi_j>-<phi_i~|phi_j~> of supercell with a displacement
            dtype=complex128
            shape=(ncdij, atoms, lm, lm')
        displacement : dict
            Dispalcement of one atom
            keys :
                'displacement': Displacement in Cartesian coordinates
                'number': Index of displaced atom
        lm_channels : list of dicts
            lm channels in Dij and qij of atoms
            keys of each distionary:
                'channels' : List of l channels
                'l', 'm' in each channel : l and list of m

        """
        check_displaced_atoms(self._atom_indices, displacements, self.symmetry)
        _check_Dij_symmetry(Dij_per, lm_channels, self._supercell, self.symmetry)
        if self.dDijdu is None:
            self._allocate_arrays(Dij_per.shape[0], Dij_per.shape[2])

        assert self._dDijdu is not None
        assert self._dqijdu is not None

        for disp_atom in np.unique([d["number"] for d in displacements]):
            delta_Dij_qijs = []
            for i, d in enumerate(displacements):
                if d["number"] == disp_atom:
                    delta_Dij_qijs.append(
                        DeltaDijQij(
                            Dij_per,
                            Dij_disps[i],
                            qij_per,
                            qij_disps[i],
                            displacements[i],
                            lm_channels,
                        )
                    )
            ddijqij = DDijQijFit(
                delta_Dij_qijs,
                self._supercell,
                self.symmetry,
                atom_indices=self._atom_indices,
                verbose=self._verbose,
            )
            ddijqij.run()
            assert ddijqij.atom_indices is not None

            indices = []
            for ai in ddijqij.atom_indices:
                indices.append(np.where(self._atom_indices == ai)[0][0])
            self._dDijdu[:, indices] = ddijqij._dDijdu
            self._dqijdu[:, indices] = ddijqij._dqijdu

        self.Dij = Dij_per[:, self._atom_indices, :, :]
        self.qij = qij_per[:, self._atom_indices, :, :]

    def _allocate_arrays(self, ncdij, lmdim):
        dtype = "c%d" % (np.dtype("double").itemsize * 2)
        natom = len(self._supercell)
        shape = (ncdij, len(self._atom_indices), 3, natom, lmdim, lmdim)
        self._dDijdu = np.zeros(shape, dtype=dtype, order="C")
        self._dqijdu = np.zeros(shape, dtype=dtype, order="C")


def _get_orbital_rotation_matrix(
    Delta: list[NDArray], lm_channels: dict, lmdim: int
) -> NDArray:
    """Return the rotation matrix of the atomic-like orbitals of an atom.

    This matrix combines the rotation matrices of the l channels of the atom,
    and is a unitary matrix.

    Parameters
    ----------
    Delta : list[NDArray]
        Rotation matrices of real spherical harmonics for l = 0, 1, ...,
        SHRotationMatrices.Delta.
    lm_channels : dict
        lm channels of the atom.
    lmdim : int
        Size of the matrix.

    """
    bigDelta = np.zeros((lmdim, lmdim), dtype="complex128")
    row = 0
    if lm_channels["channels"] is None:
        return bigDelta
    for ll in lm_channels["channels"]:
        n = ll["l"] * 2 + 1
        bigDelta[row : (row + n), row : (row + n)] = Delta[ll["l"]]
        row += n
    return bigDelta


def _rotate_Dij_in_orbital_space(
    Dij: NDArray, lm_channels: list[dict], Delta: list[NDArray], perm: NDArray
) -> NDArray:
    """Rotate Dij of all atoms of one spin component by a symmetry operation.

    For the operation g = {R|t}, which moves atom i to atom perm[i], the result
    is

        Dij'[perm[i]] = Delta(R) Dij[i] Delta(R)^+,

    where Delta(R) = <lm'|R|lm> is the matrix of the rotation R in the real
    spherical harmonics of the atomic-like orbitals. This is Dij transformed by
    g, g Dij g^-1, so Dij of a crystal that is invariant under g is unchanged.

    Parameters
    ----------
    Dij : NDArray
        Dij (or qij) of one spin component, shape=(natom, lmdim, lmdim).
    lm_channels : list[dict]
        lm channels of the atoms.
    Delta : list[NDArray]
        SHRotationMatrices.Delta of the rotation R of the operation, Delta(R)
        for each l.
    perm : NDArray
        Atomic permutation of the operation. Atom i is moved to atom perm[i].

    Returns
    -------
    NDArray
        Rotated Dij, shape=(natom, lmdim, lmdim).

    """
    lmdim = Dij.shape[-1]
    rot_Dij = np.zeros_like(Dij)
    for i, (Dij_atom, lm_atom) in enumerate(zip(Dij, lm_channels, strict=True)):
        bigDelta = _get_orbital_rotation_matrix(Delta, lm_atom, lmdim)
        rot_Dij[i] = bigDelta @ Dij_atom @ bigDelta.T.conj()
    perm_inv = np.zeros_like(perm)
    perm_inv[perm] = np.arange(len(perm))
    return rot_Dij[perm_inv]


def _check_Dij_symmetry(
    Dij: NDArray,
    lm_channels: list[dict],
    supercell: PhonopyAtoms,
    symmetry: Symmetry,
    tolerance: float = 1e-2,
) -> None:
    """Warn when a symmetry operation changes Dij of the perfect supercell.

    Every symmetry operation, with its spin rotation for ncdij=4 and with time
    reversal where the operation has it, should leave Dij of the perfect
    supercell unchanged. When an operation changes Dij by more than tolerance
    (in eV), the magnetic moments included in the symmetry operations do not
    describe the magnetic state of the VASP calculation, and
    MagneticMomentsWarning is issued. Nothing is checked for ncdij=1.

    Parameters
    ----------
    Dij : NDArray
        Dij of the perfect supercell, shape=(ncdij, natom, lmdim, lmdim).
    lm_channels : list[dict]
        lm channels of the atoms.
    supercell : PhonopyAtoms
        Perfect supercell.
    symmetry : Symmetry
        Symmetry of the supercell.
    tolerance : float, optional
        Largest change of Dij in eV that is accepted. Default is 1e-2.

    """
    ncdij = Dij.shape[0]
    if ncdij == 1:
        return
    rotations = symmetry.symmetry_operations["rotations"]
    lattice = np.array(supercell.cell.T, dtype="double", order="C")
    perms = symmetry.atomic_permutations
    lxlylz = LxLyLzMatrices().run()
    Deltas: dict[bytes, list[NDArray]] = {}
    deviations = []
    for r, perm, tr in zip(rotations, perms, get_time_reversals(symmetry), strict=True):
        key = r.tobytes()
        if key not in Deltas:
            shr = SHRotationMatrices(r, lattice, lxlylz)
            shr.run()
            assert shr.Delta is not None
            Deltas[key] = shr.Delta
        vals = [
            _rotate_Dij_in_orbital_space(Dij[i], lm_channels, Deltas[key], perm)
            for i in range(ncdij)
        ]
        if ncdij == 4:
            vals = rotate_delta_vals_in_spin_space(vals, r, lattice)
        if tr:
            vals = apply_time_reversal_in_spin_space(vals)
        deviations.append(
            max(np.abs(v - Dij_s).max() for v, Dij_s in zip(vals, Dij, strict=True))
        )
    deviations_array = np.array(deviations)
    n_changed = int((deviations_array > tolerance).sum())
    if n_changed == 0:
        return
    warnings.warn(
        "Every symmetry operation should leave the PAW strengths Dij of the "
        f"perfect supercell unchanged, but {n_changed} of the {len(rotations)} "
        f"operations change them by up to {deviations_array.max():.2g} eV "
        f"(tolerance {tolerance:g} eV). The magnetic moments included in the "
        "symmetry operations do not describe the magnetic state of the VASP "
        f"calculation (ncdij={ncdij}), and the derivatives obtained with these "
        "operations are wrong. The magnetic moments given by phelel-init "
        "--magmom have to describe the magnetic state of the VASP calculation, "
        "and the supercells have to be generated again with them.",
        MagneticMomentsWarning,
        stacklevel=1,
    )
