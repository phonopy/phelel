"""Calculation of DDijQij."""

from __future__ import annotations

from typing import Optional

import numpy as np
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import compute_all_sg_permutations
from phonopy.structure.symmetry import Symmetry

from phelel.base.local_potential import (
    collect_site_symmetry_operations,
    get_displacements_with_rotations,
    rotate_delta_vals_in_spin_space,
)
from phelel.utils.spherical_harmonics import LxLyLzMatrices, SHRotationMatrices


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
        Dij_per: np.ndarray,
        Dij_disp: np.ndarray,
        qij_per: np.ndarray,
        qij_disp: np.ndarray,
        displacement: dict,
        lm_channels: list[dict],
    ):
        """Init method.

        Note
        ----
        The order of lm is like
            (l=0,m=0), (l=1,m=-1), (l=1,m=0), (l=1,m=1), (l=2,m=-2),
            (l=2,m=-1), (l=2,m=0), ...

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
        atom_indices: Optional[list[int]] = None,
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

        self._dDijdu: Optional[np.ndarray] = None
        self._dqijdu: Optional[np.ndarray] = None
        self._atom_indices: Optional[np.ndarray] = None

        # See comments in LocalPotentialInterpolation for these variables
        self._i_atom: Optional[int] = None
        self._sitesym_sets: Optional[np.ndarray] = None

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
    def atom_indices(self):
        """Return atom indices where dDijdu and dqijdu are stored."""
        return self._atom_indices

    def __iter__(self):
        """Activate iterator."""
        return self

    def __next__(self):
        """Iterate over atom_indices."""
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

    def __str__(self):
        """Return Dij and qij in str fashion."""
        text = ""
        Dij_shape = self._dDijdu.shape[-2:]
        for i_spinor, (Dij_spinor, qij_spinor) in enumerate(
            zip(self._dDijdu, self._dqijdu)
        ):
            for i_eatom, (Dij_eatom, qij_eatom) in enumerate(
                zip(Dij_spinor, qij_spinor)
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
        xij: np.ndarray,
        i_spinor: int,
        i_eatom: int,
        i: int,
        j: int,
        str1: str,
        str2: str,
        Dij_shape: tuple,
    ):
        text = (
            f"Spin {i_spinor + 1}, Disp-Atom {i_eatom + 1}, "
            f'Atom {i + 1} d{str1}ij/d{"xyz"[j]} {str2} part '
            f"({Dij_shape[0]}, {Dij_shape[1]})\n"
        )
        for xij_l in xij:
            text += " ".join(["%8.1e" % v for v in xij_l]) + "\n"
        return text

    def _setup(self):
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
            [i for i in atoms if i in equiv_atoms], dtype="int_"
        )
        sitesym_selected_indices = [
            i for i, eq_atom in enumerate(equiv_atoms) if eq_atom in self._atom_indices
        ]
        self._sitesym_sets = sitesym_sets[sitesym_selected_indices]

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
        sitesyms = self._sitesym_sets[self._i_atom]
        natom = len(self._supercell)
        lmdim = self._delta_Dij_qijs[0].dDij.shape[-2]
        dtype = f'c{np.dtype("double").itemsize * 2}'
        ncdij = self._dDijdu.shape[0]

        lattice = self._supercell.cell.T
        rotations = self._symmetry.symmetry_operations["rotations"][sitesyms]
        translations = self._symmetry.symmetry_operations["translations"][sitesyms]
        atomic_permutations = compute_all_sg_permutations(
            self._supercell.scaled_positions,
            rotations,
            translations,
            np.array(lattice, dtype="double", order="C"),
            self._symmetry.tolerance,
        )

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
            for r, perm in zip(rotations, atomic_permutations):
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
        self, ncdij: int, delta_Dij_qij: DeltaDijQij, perm: np.ndarray, r: np.ndarray
    ) -> list[np.ndarray, np.ndarray]:
        """Rotate Dij and qij.

        This rotation is the direct product of rotations of atomic permutation
        and atomic-like orbitals on atomic points. The displacements are rotated
        actively (R), and atomic permutation and atomic-like orbitals are
        rotated passively (R^-1).

        """
        rot_dDij = []
        rot_dqij = []
        for i_cdij in range(ncdij):
            perm_inv = np.zeros_like(perm)
            for j, k in enumerate(perm):
                perm_inv[k] = j
            _rot_dDij, _rot_dqij = self._get_inv_rotated_dDij_qij(
                delta_Dij_qij, r, i_cdij
            )
            rot_dDij.append(_rot_dDij[perm_inv].ravel())
            rot_dqij.append(_rot_dqij[perm_inv].ravel())
        return rot_dDij, rot_dqij

    def _get_inv_rotated_dDij_qij(self, delta_Dij_qij: DeltaDijQij, r, i_spinor):
        """Inverse-rotate dDij and dqij."""
        lattice = self._supercell.cell.T
        shr = SHRotationMatrices(r, lattice, LxLyLzMatrices().run())
        shr.run()

        rot_dDij = np.zeros_like(delta_Dij_qij.dDij[0])
        rot_dqij = np.zeros_like(delta_Dij_qij.dqij[0])

        # loop over atoms
        for i, (dDij, dqij, lm_channels) in enumerate(
            zip(
                delta_Dij_qij.dDij[i_spinor],
                delta_Dij_qij.dqij[i_spinor],
                delta_Dij_qij.lm_channels,
            )
        ):
            bigDelta = self._get_big_Delta(shr.Delta, lm_channels, dDij)
            rot_dDij[i] = bigDelta @ dDij @ bigDelta.T.conj()
            rot_dqij[i] = bigDelta @ dqij @ bigDelta.T.conj()

        return rot_dDij, rot_dqij

    def _get_big_Delta(self, Delta, lm_channels, dDij):
        """Return orbital rotation matrix.

        This matrix combines rotation matrices of different orbitals, and
        is a unitary matrix.

        """
        bigDelta = np.zeros_like(dDij)
        row = 0
        if lm_channels["channels"] is None:
            return bigDelta
        for ll in lm_channels["channels"]:
            n = ll["l"] * 2 + 1
            bigDelta[row : (row + n), row : (row + n)] = Delta[ll["l"]]
            row += n
        return bigDelta


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
        dtype='int_'

    """

    def __init__(
        self,
        supercell: PhonopyAtoms,
        symmetry: Optional[Symmetry] = None,
        atom_indices=None,
        verbose=True,
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
            self.atom_indices = np.arange(len(self._supercell), dtype="int_")
        else:
            self.atom_indices = np.array(np.unique(atom_indices), dtype="int_")
        if symmetry is None:
            self.symmetry = Symmetry(supercell)
        else:
            self.symmetry = symmetry

        self._Dij = None
        self._qij = None
        self._dDijdu = None
        self._dqijdu = None

    @property
    def dDijdu(self):
        """Getter and setter of dDijdu."""
        return self._dDijdu

    @dDijdu.setter
    def dDijdu(self, dDijdu):
        natom = len(self._supercell)
        shape = (len(self.atom_indices), 3, natom)
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
        shape = (len(self.atom_indices), 3, natom)
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
        if Dij.shape[1] == len(self.atom_indices):
            dtype = "c%d" % (np.dtype("double").itemsize * 2)
            self._Dij = np.array(Dij, dtype=dtype, order="C")
        else:
            raise RuntimeError(
                "Number of atoms in supercell disagrees, %s!=%s."
                % (len(self.atom_indices), Dij.shape[1])
            )

    @property
    def qij(self):
        """Getter and setter of qij."""
        return self._qij

    @qij.setter
    def qij(self, qij):
        if qij.shape[1] == len(self.atom_indices):
            dtype = "c%d" % (np.dtype("double").itemsize * 2)
            self._qij = np.array(qij, dtype=dtype, order="C")
        else:
            raise RuntimeError(
                "Number of atoms in supercell disagrees, %s!=%s."
                % (len(self.atom_indices), qij.shape[1])
            )

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
        if self.dDijdu is None:
            self._allocate_arrays(Dij_per.shape[0], Dij_per.shape[2])

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
                atom_indices=self.atom_indices,
                verbose=self._verbose,
            )
            ddijqij.run()

            indices = []
            for ai in ddijqij._atom_indices:
                indices.append(np.where(self.atom_indices == ai)[0][0])
            self._dDijdu[:, indices] = ddijqij._dDijdu
            self._dqijdu[:, indices] = ddijqij._dqijdu

        self.Dij = Dij_per[:, self.atom_indices, :, :]
        self.qij = qij_per[:, self.atom_indices, :, :]

    def _allocate_arrays(self, ncdij, lmdim):
        dtype = "c%d" % (np.dtype("double").itemsize * 2)
        natom = len(self._supercell)
        shape = (ncdij, len(self.atom_indices), 3, natom, lmdim, lmdim)
        self._dDijdu = np.zeros(shape, dtype=dtype, order="C")
        self._dqijdu = np.zeros(shape, dtype=dtype, order="C")
