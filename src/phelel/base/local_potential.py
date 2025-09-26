"""Calculate derivative of local potential."""

from __future__ import annotations

import os
import textwrap
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phelel.base.Dij_qij import DeltaDijQij

import numpy as np
from numpy.typing import NDArray
from phonopy.interface.vasp import get_vasp_structure_lines
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import SNF3x3, determinant, get_supercell
from phonopy.structure.symmetry import Symmetry
from phonopy.utils import similarity_transformation

from phelel.interface.vasp.file_IO import get_CHGCAR
from phelel.utils.data import real2cmplx
from phelel.utils.lattice_points import get_lattice_points
from phelel.utils.spinor import SpinorRotationMatrices


class DeltaLocalPotential:
    """Container to store change in local potential by atomic displacement.

    Note
    ----
    ncdij = 1 without spin
    ncdij = 2 with collinear spin (up and down)
    ncdij = 4 with non-collinear spin (2x2 localpotential)

    Attributes
    ----------
    dV : ndarray
        Difference of local potentials with and without an atom displacement
        dtype='complex128'
        shape=(ncdij, nz, ny, nx)
    displacements : dict
        See docstring of __init__.

    """

    def __init__(self, V_loc_per: NDArray, V_loc_disp: NDArray, displacement: dict):
        """Init method.

        Parameters
        ----------
        V_loc_per : ndarray
            Local potential of perfect supercell
            dtype='complex128'
            shape=(ncdij, nz, ny, nx)
        V_loc_disp : ndarray
            Local potential of sueprcell with a displacement
            dtype=''complex128''
            shape=(ncdij, nz, ny, nx)
        displacement : dict
            Displacement of one atom
            keys :
                'displacement' : Displacement in Cartesian coordinates
                'number' : Index of displaced atom

        """
        self.dV = V_loc_disp - V_loc_per
        self.displacement = displacement

    def write(
        self,
        supercell: PhonopyAtoms,
        filename: str | bytes | os.PathLike | None = None,
        verbose: bool = False,
    ):
        """Write local potential to a text file."""
        if filename:
            _filename = filename
        else:
            _filename = "locpot.dat"

        if verbose:
            print(f'dV_loc is written in "{_filename}" in LOCPOT like format.')

        header = "\n".join(get_vasp_structure_lines(supercell))
        locpot = get_CHGCAR(self.dV[0].real, header)  # only cdij=0

        with open(_filename, "w") as w:
            w.write(locpot)


class LocalPotentialInterpolationNUFFT:
    """Interpolation of local potential by non-uniform FFT.

    For one displaced atom and maybe also symmetrically equivalent atoms to the
    dispalced atom.

    There are two loops:
    1) Atomic diplacement directions (outer)
       iFFT of dV on supercell FFT mesh
    2) Symmetry operations (inner)
       NUFFT of iFFT(dV) onto primitive cell FFT mesh grid where
       the grid is rotated by symmetry operations passively.

    Attributes
    ----------
    p2s_matrix : ndarray
        Transformation matrix to supercell from primitive cell
        dtype='int64'
        shape=(3,3)
    grid_points : ndarray
        Points to be used to compute <psi|dV_loc|psi'> given as
        coordinates of supercell. Coordinates are given as
        [[x1, y1, z1],
         [x2, y2, z2],
         ...         ]
        dtype='double'
        shape=(prod(fft_mesh)*det(p2s_matrix), 3)
    lattice_points : ndarray
        Lattice points in supercell in primitive cell coordinates
        dtype='int64'
        shape=(det(supercell_matrxi), 3)
    dVdu : ndarray
        Displacement derivative of local potential in supercell interpolated on
        mesh grid of primitve cell
        dtype='complex128'
        shape=(ncdij, atom_indices, 3, grid_points)
    atom_indices_returned : ndarray
        Atom indices in supercell where dV will be computed. If those indices
        don't belong to symmetrically equivalent atoms to the dispalced atom.
        dtype='int64', shape=(atom_indices,)
    dVs : list of DeltaLocalPotential
        This is given from DLocalPotential.run().

    """

    def __init__(
        self,
        fft_mesh: int | float | Sequence | NDArray,
        p2s_matrix: NDArray,
        supercell: PhonopyAtoms,
        symmetry: Symmetry,
        atom_indices: NDArray | None = None,
        nufft: str | None = None,
        finufft_eps: float | None = None,
        verbose: bool = True,
    ):
        """Init method.

        Parameters
        ----------
        fft_mesh : array_like
            Mesh numbers for primitive cell
            dtype='int64'
            shape=(3,)
        p2s_matrix : array_like
            Supercell matrix relative to primitve cell
            dtype='int64'
            shape=(3,3)
        supercell : PhonopyAtoms
            Perfect supercell
        symmetry : Symmetry
            Symmetry of supercell
        atom_indices : list of int, optional
            Atom indices in supercell where dV will be expected to be computed.
            If None, supposed to be all atoms. Internally only symmetrically
            equivalent atoms to the dispalced atom are selected to compute.
        nufft : str or None, optional
            'finufft' only. Default is None, which corresponds to 'finufft'.
        finufft_eps : float or None, optional
            Accuracy of finufft interpolation. Default is None, which
            corresponds to 1e-6.
        verbose : bool, optional
            To display log or not

        """
        ##########
        # Inputs #
        ##########
        self._fft_mesh = np.array(fft_mesh, dtype="int64")
        self._verbose = verbose
        self._supercell = supercell
        self._symmetry = symmetry
        self._atom_indices_in = atom_indices
        if nufft is None:
            self._nufft = "finufft"
        else:
            self._nufft = nufft
        if finufft_eps is None:
            self._finufft_eps = 1e-6
        else:
            self._finufft_eps = finufft_eps

        ##########
        # Public #
        ##########
        self._p2s_matrix = np.array(p2s_matrix, dtype="int64", order="C")
        (self._grid_points, self._lattice_points) = get_grid_points(
            fft_mesh, self._p2s_matrix
        )
        assert 3 == self._grid_points.shape[1]

        self._dVdu: NDArray | None = None
        self._atom_indices_returned: NDArray | None = None

        ###########
        # Private #
        ###########
        # Delta local potentials before interpolation
        self._delta_Vs: list[DeltaLocalPotential] | None = None

        # counter of equivalent atoms calculations
        self._i_atom = 0
        # Sets of symmetry operations to send
        # displaced atom to ones of
        # symmetrically equivalent atoms.
        # The set is the same as
        # site-symmetry when displaced atom
        # and the one is identical.
        self._sitesym_sets = None

        # Used in the first loop for each one of symmetrically equivalent atoms
        self._dV_itpl = None
        # Interpolated dV for different displacements
        # [[dV(0,0,0), dV(0,0,1), dV(0,0,2), ..., dV(0,0,N_g)],
        #  [dV(0,1,0), dV(0,1,1), dV(0,1,2), ..., dV(0,1,N_g)],
        #  ...
        #  [dV(1,0,0), dV(1,0,1), dV(1,0,2), ..., dV(1,0,N_g)],
        #  [dV(1,1,0), dV(1,1,1), dV(1,1,2), ..., dV(1,1,N_g)],
        #  ...
        #  [dV(n,N,0), dV(n,N,1), dV(n,N,2), ..., dV(n,N,N_g)]]
        # n:   number of displacements,
        # N:   number of symmetry operations
        # N_g: number of interpolated grid points

        # Inverse Fourier transformed original dV
        self._dV_iFT = None

        # [[d_x(0,0), d_y(0,0), d_z(0,0)],
        #  [d_x(0,1), d_y(0,1), d_z(0,1)],
        #  ...
        #  [d_x(1,0), d_y(1,0), d_z(1,0)],
        #  [d_x(1,1), d_y(1,1), d_z(1,1)],
        #  ...
        #  [d_x(n,N), d_y(n,N), d_z(n,N)]]
        # n: number of displacements,
        # N: number of symmetry operations
        self._disps = None

        self._finufft_plan = None

    def __iter__(self):
        """Enable iterator."""
        return self

    def __next__(self):
        """Enable iterator."""
        if len(self._atom_indices_returned) == self._i_atom:
            self._dV_itpl = None
            self._disps = None
            raise StopIteration

        self._run_at_atom()
        self._i_atom += 1

    def next(self):
        """Enable iterator."""
        return self.__next__()

    @property
    def p2s_matrix(self):
        """Return supercell matrix."""
        return self._p2s_matrix

    @property
    def grid_points(self):
        """Return grid points."""
        return self._grid_points

    @property
    def lattice_points(self):
        """Return lattice points."""
        return self._lattice_points

    @property
    def dVdu(self) -> NDArray | None:
        """Return dVdu."""
        return self._dVdu

    @property
    def atom_indices_returned(self):
        """Return atom indices."""
        return self._atom_indices_returned

    @property
    def delta_Vs(self) -> list[DeltaLocalPotential] | None:
        """Return dVs."""
        return self._delta_Vs

    @delta_Vs.setter
    def delta_Vs(self, delta_Vs: list[DeltaLocalPotential]):
        """Prepare dV interpolation calculation.

        Note
        ----
        Atom index list (atom_indices_returned) is created as a union of
        atom_indices_in and symmetrically equivalent atoms of the displaced atom.

        Parameters
        ----------
        dVs : list of DeltaLocalPotentials
            Changes in local potentials by atomic displacements. This
            can be set later.

        """
        self._delta_Vs = delta_Vs
        self._i_atom = 0
        disp_atom = self._delta_Vs[0].displacement["number"]
        sitesym_sets, equiv_atoms = collect_site_symmetry_operations(
            disp_atom, self._symmetry
        )

        if self._atom_indices_in is None:
            atoms = np.arange(len(self._supercell))
        else:
            atoms = self._atom_indices_in
        self._atom_indices_returned = np.array(
            [i for i in atoms if i in equiv_atoms], dtype="int64"
        )
        sitesym_selected_indices = [
            i
            for i, eq_atom in enumerate(equiv_atoms)
            if eq_atom in self._atom_indices_returned
        ]
        self._sitesym_sets = sitesym_sets[sitesym_selected_indices]

        dtype = f"c{np.dtype('double').itemsize * 2}"
        ncdij = self._delta_Vs[0].dV.shape[0]
        self._dVdu = np.zeros(
            (ncdij, len(self._atom_indices_returned), 3, len(self._grid_points)),
            dtype=dtype,
            order="C",
        )

    def delete_dVdu(self):
        """Delete large object dVdu."""
        self._dVdu = None

    def run(self):
        """Possibly iterate over selected symmetrically equivalent atoms."""
        for _ in self:
            pass

    def _run_at_atom(self):
        """Calculate at an atom in selected symmetrically equivalent atoms.

        Called by iterator method __next__.

        np.dot(disps_inv, full_dV_itpl) is stored in self.dVdu. But full_dV_tpl
        requires large memory space, it is divided into those at site-symmetry
        operations and multiplied with disps_inv piece by piece.

        disps:
            Stacked atomic displacements. Site-symmetry runs fastest.
            shape = (n_sitesyms * n_disps, 3)
        disps_inv:
            shape = (3, n_sitesyms * n_disps)
        dVs_*:
            shape = (ncdij, n_grid_points)
        self._dVdu:
            shape = (ncdij, natom, 3, n_grid_points)

        ``dVs_iFT``, ``dVs_rotated``, and ``self._finufft_plan`` are the list of
        size ``ncdij``, where ``ncdij`` is 1 (non-magnetic), 2 (collinear
        magnetic), or 4 (non-collinear).

        Spinor rotation
        ---------------
        When non-collinear case, dV is considered as 2x2 matrix of scalar fields
        of delta V in real space with respect to each atomic displacement. This
        dV is rotated to expand symemtrized information to non-symmetrized
        information by leveraging site symmetry information for computing the
        derivative.

        """
        sitesyms = self._sitesym_sets[self._i_atom]
        rotations = self._symmetry.symmetry_operations["rotations"][sitesyms]
        translations = self._symmetry.symmetry_operations["translations"][sitesyms]
        lattice = self._supercell.cell.T
        disps = get_displacements_with_rotations(rotations, lattice, self._delta_Vs)
        disps_inv = np.linalg.pinv(disps)

        if self._verbose:
            print("Running finufft (eps=%.3e)..." % self._finufft_eps)

        ncdij = self._dVdu.shape[0]
        self._init_finufft(ncdij)

        count = 0
        for delta_V in self._delta_Vs:
            dVs_iFT = [self._get_iFFT_of_dV(dV) for dV in delta_V.dV]
            for r, t in zip(rotations, translations):
                dVs_rotated = self._rotate_dV(dVs_iFT, r, t)
                if ncdij == 4:  # Need to rotate in spin space, too.
                    dVs_spinor_rotated = rotate_delta_vals_in_spin_space(
                        dVs_rotated, r, lattice
                    )
                    dVs_rotated.clear()
                    dVs_rotated = dVs_spinor_rotated
                for i_cdij in range(ncdij):
                    self._dVdu[i_cdij, self._i_atom] += np.outer(
                        disps_inv[:, count], dVs_rotated[i_cdij]
                    )
                count += 1
                dVs_rotated.clear()
            dVs_iFT.clear()

        self._finalize_finufft()

    def _rotate_dV(
        self, dV_iFTs: list[NDArray], r: NDArray, t: NDArray
    ) -> list[NDArray]:
        """Rotate dV by rotating coordinates of delta potential passively.

        Instead of rotating delta potential, grid points are rotated.

        """
        r_inv = np.linalg.inv(r)
        t_inv = -r_inv @ t
        grid_points = self._grid_points @ r_inv.T + t_inv
        grid_points -= np.rint(grid_points)
        return self._run_finufft(grid_points, dV_iFTs)

    def _get_iFFT_of_dV(self, dV: NDArray) -> NDArray:
        """Inverse FFT."""
        dims = dV.shape
        assert 3 == len(dims)
        dV_iFT = np.fft.fftshift(np.fft.ifftn(dV))
        return np.array(dV_iFT, dtype=dV_iFT.dtype, order="C")

    def _run_finufft(
        self, grid_points: NDArray, dV_iFTs: list[NDArray]
    ) -> list[NDArray]:
        """Transform from uniform to non-uniform points.

        3D Type-2 transform.

            finufft.nufft3d2(z, y, x, dV_iFT, eps=self._finufft_eps)

        dV_iFT is FFT of dV whwere dV values are stored in Fortran order.
        So x, y, z are alined as (z, y, x) in nufft3d2.

        """
        retval = []
        for i, dV_iFT in enumerate(dV_iFTs):
            x, y, z = [
                np.array(v, dtype=grid_points.dtype, order="C")
                for v in (grid_points * (np.pi * 2)).T
            ]
            self._finufft_plan[i].setpts(z, y, x)
            retval.append(self._finufft_plan[i].execute(dV_iFT))
        return retval

    def _init_finufft(self, ncdij: int):
        import finufft

        dtype = f"c{np.dtype('double').itemsize * 2}"
        self._finufft_plan = [
            finufft.Plan(
                2, self._delta_Vs[0].dV[i].shape, eps=self._finufft_eps, dtype=dtype
            )
            for i in range(ncdij)
        ]

    def _finalize_finufft(self):
        self._finufft_plan.clear()


class DLocalPotential:
    """Compute derivative of local potential wrt. atomic displacements.

    dV/du.

    Note
    ----
    ncdij = 1 without spin
    ncdij = 2 with collinear spin (up and down)
    ncdij = 4 with non-collinear spin (2x2 localpotential)

    Attributes
    ----------
    p2s_matrix : ndarray
        Transformation matrix to supercell from primitive cell.
        dtype='int64', shape=(3,3)
    grid_points : ndarray
        Points to be used to compute <psi|dV_loc|psi'> given as
        coordinates of supercell. Coordinates are given as
        [[x1, y1, z1],
         [x2, y2, z2],
         ...         ]
        dtype='double'
        shape=(prod(fft_mesh)*det(p2s_matrix), 3)
    lattice_points : ndarray
        Lattice points in supercell in primitive cell coordinates.
        dtype='int64', shape=(det(supercell_matrxi), 3)
    supercell : PhonopyAtoms
        Supercell.
    symmetry : Symmetry
        Symmetry of supercell.
    dVdu : ndarray
        Displacement derivative of local potential in supercell interpolated on
        mesh grid of primitve cell.
        dtype='complex128', shape=(ncdij, atom_indices, 3, grid_points)
    atom_indices : ndarray, optional
        Atom indices in supercell where dV is computed. This is made as
        np.unique(atom_indices given at __init__). If None, all atoms in
        supercell.
        shape=(len(atom_indices),), dtype='int64'
    fft_mesh : array_like
        Mesh numbers for primitive cell.
        dtype='int64', shape=(3,)

    """

    def __init__(
        self,
        fft_mesh: int | float | Sequence | NDArray,
        p2s_matrix: NDArray,
        supercell: PhonopyAtoms,
        symmetry: Symmetry | None = None,
        atom_indices: NDArray | None = None,
        nufft: str | None = None,
        finufft_eps: float | None = None,
        verbose: bool = True,
    ):
        """Init method.

        Parameters
        ----------
        fft_mesh : array_like
            Mesh numbers for primitive cell.
            dtype='int64', shape=(3,)
        p2s_matrix : ndarray or 2d list
            Supercell matrix relative to primitve cell.
            dtype='int64', shape=(3,3)
        supercell : PhonopyAtoms
            Supercell.
        symmetry : Symmetry, optional
            Symmetry of supercell. If None, symmetry is searched in this class
            object.
        atom_indices : list of int, optional
            Atom indices in supercell where dV will be expected to be computed.
            If None, supposed to be all atoms. Internally only symmetrically
            equivalent atoms to the dispalced atom are selected to compute.
        nufft : str or None
            'finufft' only. Default is None, which corresponds to 'finufft'.
        finufft_eps : float or None, optional
            Accuracy of finufft interpolation. Default is None, which
            corresponds to 1e-6.
        verbose : bool
            To display log or not

        """
        self._verbose = verbose

        self._supercell = supercell
        self._fft_mesh = fft_mesh
        self._p2s_matrix = p2s_matrix
        if atom_indices is None:
            self._atom_indices = np.arange(len(self._supercell), dtype="int64")
        else:
            self._atom_indices = np.array(np.unique(atom_indices), dtype="int64")
        if symmetry is None:
            self._symmetry = Symmetry(self._supercell)
        else:
            self._symmetry = symmetry
        self._nufft = nufft
        self._finufft_eps = finufft_eps
        self._lattice_points = None
        self._grid_points = None
        self._dVdu = None  # self.dVdu is provided by @property.

    @property
    def p2s_matrix(self):
        """Return supercell matrix."""
        return self._p2s_matrix

    @property
    def supercell(self):
        """Return supercell."""
        return self._supercell

    @property
    def symmetry(self):
        """Return symmetry of supercell."""
        return self._symmetry

    @property
    def atom_indices(self):
        """Return atom indices."""
        return self._atom_indices

    @property
    def fft_mesh(self):
        """Return FFT mesh."""
        return self._fft_mesh

    @property
    def dVdu(self):
        """Return dVdu.

        See detail at attribute section of this class's docstring.

        """
        return self._dVdu

    @dVdu.setter
    def dVdu(self, dVdu: NDArray):
        if dVdu.dtype == "double":
            _dVdu = real2cmplx(dVdu)
        else:
            _dVdu = dVdu
        if _dVdu.shape[1:] == self._get_dVdu_shape():
            self._allocate_arrays(_dVdu.shape[0])
            self._dVdu[:] = _dVdu
        else:
            raise RuntimeError(
                "Array shape[1:] disagreement is found, %s!=%s."
                % (self._get_dVdu_shape(), _dVdu.shape[1:])
            )

    @property
    def lattice_points(self):
        """Return lattice points.

        See detail at attribute section of this class's docstring.

        """
        return self._lattice_points

    @lattice_points.setter
    def lattice_points(self, lattice_points: NDArray):
        multi = determinant(self.p2s_matrix)
        if lattice_points.shape == (multi, 3):
            self._lattice_points = np.array(lattice_points, dtype="int64", order="C")
        else:
            raise RuntimeError(
                "Array shape disagreement is found, %s!=%s."
                % ((multi, 3), lattice_points.shape)
            )

    @property
    def grid_points(self):
        """Return grid_points.

        See detail at attribute section of this class's docstring.

        """
        return self._grid_points

    @grid_points.setter
    def grid_points(self, grid_points: NDArray):
        N = np.prod(self.fft_mesh) * determinant(self.p2s_matrix)
        if grid_points.shape == (N, 3):
            self._grid_points = np.array(grid_points, dtype="double", order="C")
        else:
            raise RuntimeError(
                "Array shape disagreement is found, %s!=%s."
                % ((N, 3), grid_points.shape)
            )

    def run(
        self,
        V_loc_per: NDArray,
        V_loc_disps: list[NDArray],
        displacements: list[dict],
    ):
        """Calculate dV/du.

        Calculation results are stored in self._dVdu.

        Parameters
        ----------
        V_loc_per : ndarray
            Local potential of perfect supercell
            dtype='complex128'
            shape=(ncdij, nz, ny, nx)
        V_loc_disps : list of ndarrays
            Local potentials of sueprcells with respective displacements
            dtype='complex128'
            shape=(ndisp, ncdij, nz, ny, nx)
        displacements : list of dicts
            Displacements of displaced atoms
            keys of each dict:
                'displacement' : Displacement in Cartesian coordinates
                'number' : Index of displaced atom

        """
        lpi = LocalPotentialInterpolationNUFFT(
            self._fft_mesh,
            self._p2s_matrix,
            self._supercell,
            self._symmetry,
            atom_indices=self._atom_indices,
            nufft=self._nufft,
            finufft_eps=self._finufft_eps,
        )
        self._lattice_points = lpi.lattice_points.copy(order="C")
        self._grid_points = lpi.grid_points.copy(order="C")

        if self._dVdu is None:
            self._allocate_arrays(V_loc_per.shape[0])

        for disp_atom in np.unique([d["number"] for d in displacements]):
            dVs = []
            for i, d in enumerate(displacements):
                if d["number"] == disp_atom:
                    dVs.append(
                        DeltaLocalPotential(V_loc_per, V_loc_disps[i], displacements[i])
                    )
            lpi.delta_Vs = dVs
            for i_atom, _ in enumerate(lpi):  # Run lpi by iterator.next()
                if self._verbose:
                    print(
                        "Computed dV/du by displaced atom %d"
                        % (lpi.atom_indices_returned[i_atom] + 1)
                    )

            indices = []
            for ai in lpi.atom_indices_returned:
                indices.append(np.where(self.atom_indices == ai)[0][0])
            self._dVdu[:, indices, :, :] = lpi.dVdu

    def visualize(self, pcell: PhonopyAtoms, i_atom: int):
        """Visualize dV/du in x, y, z."""
        for i_dir in range(3):
            xyz = "xyz"[i_dir]
            # spin 1 only
            multi = visualize_distribution(
                pcell,
                self._p2s_matrix,
                self._fft_mesh,
                self._grid_points,
                self._dVdu[0, i_atom, i_dir],
                filename="locpot_viz-%03d-%s.dat"
                % (self._atom_indices[i_atom] + 1, xyz),
            )

        if self._verbose:
            msg = (
                "[%d, %d, %d] supercell of primitive cell is "
                "used. dV/du is written in "
                '"locpot_viz-%d-{x,y,z}.dat", in LOCPOT-like '
                "format." % (tuple(multi) + (i_atom + 1,))
            )
            print(
                textwrap.fill(
                    msg, width=70, initial_indent="    ", subsequent_indent="    "
                )
            )

    def _get_dVdu_shape(self):
        num_gp = np.prod(self._fft_mesh) * determinant(self._p2s_matrix)
        return (len(self._atom_indices), 3, num_gp)

    def _allocate_arrays(self, ncdij: int):
        dtype = "c%d" % (np.dtype("double").itemsize * 2)
        shape = (ncdij,) + self._get_dVdu_shape()
        self._dVdu = np.zeros(shape, dtype=dtype, order="C")


def visualize_distribution(
    pcell: PhonopyAtoms,
    p2s_matrix: NDArray,
    fft_mesh: int | float | Sequence | NDArray,
    grid_points: NDArray,
    data: NDArray,
    filename: str | bytes | os.PathLike | None = None,
) -> NDArray:
    """Visualize scalar distribution in space.

    Note
    ----
    multiplicity is the three integers that simply extend primitive cell. This
    extended cell includes the supercell. See more detail in
    _get_multipliticy_for_visualization. This is because the grid is defined
    along primitive cell basis vectors.

    """
    plat = pcell.cell.T
    multiplicity = _get_multipliticy_for_visualization(p2s_matrix, plat)

    M = np.dot(np.linalg.inv(p2s_matrix), np.diag(multiplicity))
    M = np.rint(M).astype(int)
    n = int(np.rint(np.linalg.det(M)))
    d = grid_points.shape[0]

    # Copy grid point coordinates to fill the visualized cell with respect
    # to the basis vectors of local potential cell.
    gp_viz = np.zeros((d * n, 3), dtype="double", order="C")
    snf = SNF3x3(M)
    snf.run()
    P_inv = np.linalg.inv(snf.P)
    diag_A = tuple(np.diagonal(snf.D))

    # Supercell lattice points inside visuzalization cell.
    # Grid points of supercell + the lattice point is concatenated.
    for i, lp in enumerate(np.dot(list(np.ndindex(diag_A)), P_inv.T)):
        gp_viz[(i * d) : ((i + 1) * d), :] = grid_points + lp

    # Grid point coordinates are transformed to those in the visualized cell
    # basis.
    M_inv = np.linalg.inv(M)
    gp_viz = np.dot(gp_viz, M_inv.T)

    # Coordinates are converted to integers.
    multi_to_int = fft_mesh * multiplicity
    gp_viz *= multi_to_int
    gp_viz = np.rint(gp_viz).astype(int) % multi_to_int
    shape = tuple(fft_mesh * multiplicity)[::-1]
    dV_viz = np.zeros(shape, dtype=data.dtype, order="C")
    done = np.zeros(shape, dtype=int, order="C")

    n_ip = len(data)
    for i, index in enumerate(gp_viz):
        dV_viz[tuple(index[::-1])] = data[i % n_ip]
        done[tuple(index[::-1])] = 1

    assert (done == 1).all(), "%d in %d (%d x %d x %d) are done." % (
        done.sum(),
        np.prod(done.shape),
        done.shape[0],
        done.shape[1],
        done.shape[2],
    )

    scell = get_supercell(pcell, np.diag(multiplicity))
    header = "\n".join(get_vasp_structure_lines(scell))
    locpot = get_CHGCAR(dV_viz.real, header)
    with open(filename, "w") as w:
        w.write(locpot)

    return multiplicity


def collect_site_symmetry_operations(
    disp_atom: int, symmetry: Symmetry
) -> tuple[NDArray, NDArray]:
    """Collect site symmetry operations.

    Collect symmetry operations from a diplaced atom to symmetrically
    equivalent atoms. When the symmetrically equivalent atom is the
    displaced atom, those operations are the site-symmetry
    operations. In the other cases, those operations are bound for
    each symmetrically equivalent atom.

    Returns
    -------
    sitesym_sets : ndarray
        Sets of indices of symmetry operations. Each set contains symmetry
        operations that send the displaced atom to specific one of equivalent
        atoms.
        dtype='int64'
        shape=(equiv_atoms, site_syms)
    equiv_atoms : ndarray
        Indices of symmetrically equivalent atoms
        dtype='int64'
        shape=(equiv_atoms,)

    """
    perms = symmetry.atomic_permutations[:, disp_atom]

    # This gives something like [0, 0, 0, 0, 4, 4, 4, 4, ...].
    map_to_indep_atoms = symmetry.get_map_atoms()
    # This gives something like [0, 1, 2, 3] for disp_atom=0.
    equiv_atoms = np.array(
        [
            i
            for i, x in enumerate(map_to_indep_atoms)
            if map_to_indep_atoms[disp_atom] == x
        ],
        dtype="int64",
    )
    ops_sets = []
    for i_atom in equiv_atoms:
        isyms = np.where(perms == i_atom)[0]
        if len(isyms) > 0:
            ops_sets.append(isyms)

    return np.array(ops_sets, dtype="int64", order="C"), equiv_atoms


def get_displacements_with_rotations(
    rotations: NDArray,
    lattice: NDArray,
    delta_vals: list[DeltaLocalPotential] | list[DeltaDijQij],
) -> NDArray:
    """Rotate displacements by site-symmetry actively.

    Displacements are stored in the following order:

        disps = []
        for calc_disp_dir in disps_of_one_atoms:
            for r in sitesym:
                disps.append(dot(r, calc_disp_dir))

    Returns
    -------
    disps : ndarray
        shape=(calc_disp_dir * sitesyms, 3)

    """
    disps = np.zeros((len(rotations) * len(delta_vals), 3), dtype="double", order="C")
    count = 0
    for delta_V in delta_vals:
        disp = delta_V.displacement["displacement"]
        for r in rotations:
            r_cart = similarity_transformation(lattice, r)
            disps[count] = np.dot(r_cart, disp)
            count += 1
    return disps


def rotate_delta_vals_in_spin_space(
    delta_vals_rotated: list[NDArray], r: NDArray, lattice: NDArray
) -> list[NDArray]:
    r"""Take linear combination of delta vals with spin rotation matrix.

    In this method A B A^+ is computed, where A is the 2x2 rotation matrix of
    spin space and B is ``delta_vals_rotated``. What is computed is more
    precisely,

    \sum_{i_s1, i_s2} A_{i_sp, i_s1} B_{i_s1, i_s2} (A^+)_{i_s2, i_s}

    The data order of ``delta_vals_rotated`` is as follows. This method is
    performed on when ncdij=4. This 4 corresponds to 2x2 matrix in spinor
    representation. In VASP covention, the data order is defined as

    cdij=[1, 2, 3, 4] -> [11, 12, 21, 22].

    SpinorRotationMatrices.Delta is a 2x2 numpy array.

    Parameters
    ----------
    delta_vals_rotated : list[NDArray]
        Rotated delta local potential or delta-Dijqij.
    lattice : ndarray.
        Basis vectors of supercell in column vectors.
        shape=(3, 3)
    r : ndarray
        Rotation matrix wrt basis vectors (not Cartesian).
        shape=(3, 3).

    """
    srm = SpinorRotationMatrices(r, lattice).run()
    assert srm.Delta.shape == (2, 2)

    delta_vals_spinor_rotated = [np.zeros_like(delta_vals_rotated[0]) for _ in range(4)]
    for i_sp in (0, 1):
        for i_s in (0, 1):
            for i_s1 in (0, 1):
                for i_s2 in (0, 1):
                    d_d = srm.Delta[i_sp, i_s1] * srm.Delta.T.conj()[i_s2, i_s]
                    delta_vals_spinor_rotated[i_sp * 2 + i_s] += (
                        d_d * delta_vals_rotated[i_s1 * 2 + i_s2]
                    )
    return delta_vals_spinor_rotated


def get_grid_points(
    fft_mesh: int | float | Sequence | NDArray, p2s_matrix: NDArray
) -> tuple[NDArray, NDArray]:
    """Return grid points with respect to supercell basis vectors.

    Parameters
    ----------
    fft_mesh: Number of grid points along a, b, c basis vectors of primitve
        cell
    p2s_matrix: An integer matrix that is used to create basis vectors
        of supercell from those of primitive cell

    Returns
    -------
    grid_points : ndarray
        Points to be used to compute <psi|dV_loc|psi'> given as
        coordinates of supercell. Coordinates are given as
        [[x1, y1, z1],
         [x2, y2, z2],
         ...         ]
        dtype='double'
        shape=(prod(fft_mesh)*det(p2s_matrix), 3)
    lattice_points : ndarray
        Lattice points in supercell in primitive cell coordinates
        dtype='int64'
        shape=(det(p2s_matrix), 3)

    """
    # x runs fastest, then y, z. An example is shown belw:
    #
    # In [3]: b, c, a = np.meshgrid(range(3), range(4), range(2))
    # In [4]: a.ravel()
    # Out[4]:
    # array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
    #        0, 1])
    # In [5]: b.ravel()
    # Out[5]:
    # array([0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 0, 0, 1, 1,
    #        2, 2])
    # In [6]: c.ravel()
    # Out[6]:
    # array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,
    #        3, 3])
    b, c, a = np.meshgrid(range(fft_mesh[1]), range(fft_mesh[2]), range(fft_mesh[0]))
    grid_points = np.multiply(
        np.c_[a.ravel(), b.ravel(), c.ravel()], 1.0 / np.array(fft_mesh)
    )

    # lattice points
    lattice_points, multi = get_lattice_points(p2s_matrix)

    # copy at lattice points
    n = np.prod(fft_mesh)
    gp_super = np.zeros((n * np.prod(multi), 3), dtype="double", order="C")
    for i, lp in enumerate(lattice_points):
        gp_super[(i * n) : ((i + 1) * n), :] = grid_points + lp

    # Transform grid points coordinates from those of primitive cell to those
    # of supercell.
    gp_super[:] = np.dot(gp_super, np.linalg.inv(p2s_matrix).T)

    return gp_super, lattice_points


def _get_multipliticy_for_visualization(
    p2s_matrix: NDArray, prim_lattice: NDArray
) -> NDArray:
    """Find minimum multiplication of primitive cell that can contain supercell.

    Parameters
    ----------
    p2s_matrix : ndarray
        Transformation matrix from primitive cell to supercell.
        shape=(3, 3), dtype='double'
    prim_lattice : ndarray
        Basis vectors of primitive cell given as column vectors.
        shape=(3, 3), dtype='double'

    Returns
    -------
    Multiplicity with respect to primitive cell.

    Note:
        Ls = Lp Ms
        Lv = Ls Mv
        Lv = Lp N
        Ls Mv = Lp N
        Ms Mv = N
        Mv = Ms^-1 N

    """
    plat = prim_lattice  # column vectors
    smat = p2s_matrix
    abc = np.sqrt(np.diagonal(np.dot(plat.T, plat)))
    n = np.ones(3, dtype=int)

    for _ in range(100):
        vmat = np.dot(np.linalg.inv(smat), np.diag(n))
        if (abs(vmat - np.rint(vmat)) < 1e-5).all():
            break
        i_min = np.argmin(abc * n)
        n[i_min] += 1

    return n
