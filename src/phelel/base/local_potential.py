"""Calculate derivative of local potential."""

from __future__ import annotations

import os
import pathlib
import textwrap
from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from phonopy.harmonic.displacement import FirstAtomDisplacement
from phonopy.interface.vasp import get_vasp_structure_lines
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import determinant, get_supercell
from phonopy.structure.symmetry import Symmetry
from phonors import snf3x3

from phelel.base.fourier_series import FourierSeriesOnGrid, get_fourier_coefficients
from phelel.base.magnetic_symmetry import check_magnetic_moments, get_time_reversals
from phelel.base.symmetrization import (
    check_displaced_atoms,
    get_displacements_with_rotations,
    select_equivalent_atoms,
)
from phelel.interface.vasp.file_IO import get_CHGCAR
from phelel.utils.lattice_points import get_lattice_points
from phelel.utils.spinor import (
    apply_time_reversal_in_spin_space,
    rotate_delta_vals_in_spin_space,
)


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

    def __init__(
        self,
        V_loc_per: NDArray,
        V_loc_disp: NDArray,
        displacement: FirstAtomDisplacement,
    ):
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
        Lattice points of the primitive cell, one from each class modulo the
        lattice of the supercell, in the basis vectors of the primitive cell.
        They are not reduced into the supercell.  See get_lattice_points.
        dtype='int64'
        shape=(det(p2s_matrix), 3)
    dVdu : ndarray
        Displacement derivative of local potential in supercell interpolated on
        mesh grid of primitve cell
        dtype='complex128'
        shape=(ncdij, atom_indices, 3, grid_points)
    atom_indices_returned : ndarray
        Atom indices in supercell where dV will be computed. If those indices
        don't belong to symmetrically equivalent atoms to the dispalced atom.
        dtype='int64', shape=(atom_indices,)

    """

    def __init__(
        self,
        fft_mesh: Sequence[int] | NDArray,
        p2s_matrix: NDArray,
        supercell: PhonopyAtoms,
        symmetry: Symmetry,
        atom_indices: NDArray | None = None,
        finufft_eps: float | None = None,
        nufft: Literal["finufft", "native", "native-box"] = "finufft",
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
        finufft_eps : float or None, optional
            Accuracy of finufft interpolation. Default is None, which
            corresponds to 1e-6.
        nufft : {"finufft", "native", "native-box"}, optional
            Method to evaluate the Fourier series of dV at the grid points.
            "finufft" uses finufft with finufft_eps.  "native" and "native-box"
            evaluate the series by FourierSeriesOnGrid without the approximation
            of NUFFT, and finufft_eps is not used.  They differ in the g chosen
            from each g + N^s n (see phelel.base.fourier_series), which changes
            the values between the points of the FFT mesh of the supercell.
            "native" takes the shortest reciprocal lattice vectors, which keeps
            the symmetry and the reality of dV there.  "native-box" takes the
            box used with finufft.  Default is "finufft".
        verbose : bool, optional
            To display log or not

        """
        ##########
        # Inputs #
        ##########
        self._verbose = verbose
        self._supercell = supercell
        self._symmetry = symmetry
        self._atom_indices_in = atom_indices
        if finufft_eps is None:
            self._finufft_eps = 1e-6
        else:
            self._finufft_eps = finufft_eps
        self._nufft = nufft

        ##########
        # Public #
        ##########
        self._p2s_matrix = np.array(p2s_matrix, dtype="int64", order="C")
        (self._grid_points, self._lattice_points) = get_grid_points(
            fft_mesh, self._p2s_matrix
        )
        assert 3 == self._grid_points.shape[1]
        self._fourier_series: FourierSeriesOnGrid | None = None
        if nufft == "native":
            self._fourier_series = FourierSeriesOnGrid(
                fft_mesh,
                self._p2s_matrix,
                representatives="shortest",
                supercell_lattice=supercell.cell,
            )
        elif nufft == "native-box":
            self._fourier_series = FourierSeriesOnGrid(fft_mesh, self._p2s_matrix)

        self._dVdu: NDArray | None = None
        self._atom_indices_returned: NDArray | None = None

        ###########
        # Private #
        ###########
        self._finufft_plan = None

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

    def run(self, delta_Vs: list[DeltaLocalPotential]):
        """Compute dV/du of the atoms equivalent to the displaced atom.

        The atoms are those of atom_indices given at __init__ that are
        equivalent to the displaced atom by symmetry.  They are stored in
        atom_indices_returned, and dV/du of them in dVdu.

        Parameters
        ----------
        delta_Vs : list of DeltaLocalPotential
            Changes in local potentials by the displacements of one atom.

        """
        check_magnetic_moments(self._supercell, delta_Vs[0].dV.shape[0])
        disp_atom = delta_Vs[0].displacement["number"]
        atoms, sitesym_sets = select_equivalent_atoms(
            disp_atom, self._symmetry, self._atom_indices_in
        )

        ncdij = delta_Vs[0].dV.shape[0]
        dVdu = np.zeros(
            (ncdij, len(atoms), 3, len(self._grid_points)),
            dtype="complex128",
            order="C",
        )
        for i_atom, sitesyms in enumerate(sitesym_sets):
            dVdu[:, i_atom] = self._run_at_atom(delta_Vs, sitesyms)
            if self._verbose:
                print("Computed dV/du by displaced atom %d" % (atoms[i_atom] + 1))
        self._atom_indices_returned = atoms
        self._dVdu = dVdu

    def _run_at_atom(
        self, delta_Vs: list[DeltaLocalPotential], sitesyms: NDArray
    ) -> NDArray:
        """Return dV/du of one of the atoms equivalent to the displaced atom.

        Let a be the displaced atom and b this atom.  Each operation g = {R|t}
        of sitesyms sends a to b.  Applied to the change dV_k by the k-th
        displacement u_k of a, g gives the change by the displacement R u_k of
        b,

            dV_kg(r) = dV_k(g^-1 r),

        which is evaluated at the grid points by _rotate_dV.  For
        ncdij=4, dV_kg is also rotated in spin space by R, and for the
        operations combined with time reversal, time reversal is applied in
        spin space.

        dV/du of b is the least-squares solution of

            dV_kg = (R u_k) . dV/du  for all k and g,

        that is, dV/du = pinv(U) dV, where the rows of U are R u_k with g
        running fastest (get_displacements_with_rotations).  The products of
        the columns of pinv(U) and dV_kg are summed one (k, g) at a time, so
        that dV_kg of all (k, g) are not stored at once.

        Parameters
        ----------
        delta_Vs : list of DeltaLocalPotential
            Changes in local potentials by the displacements u_k of the
            displaced atom a.
        sitesyms : ndarray
            Indices of the symmetry operations that send a to b.
            dtype='int64', shape=(n_site_syms,)

        Returns
        -------
        ndarray
            dV/du of b.
            dtype='complex128', shape=(ncdij, 3, n_grid_points)

        """
        rotations = self._symmetry.symmetry_operations["rotations"][sitesyms]
        translations = self._symmetry.symmetry_operations["translations"][sitesyms]
        time_reversals = get_time_reversals(self._symmetry)[sitesyms]
        lattice = self._supercell.cell.T
        disps = get_displacements_with_rotations(rotations, lattice, delta_Vs)
        disps_inv = np.linalg.pinv(disps)

        if self._verbose:
            if self._fourier_series is None:
                print("Running finufft (eps=%.3e)..." % self._finufft_eps)
            else:
                print(
                    f"Evaluating Fourier series of dV at grid points ({self._nufft})..."
                )

        ncdij = delta_Vs[0].dV.shape[0]
        dVdu = np.zeros(
            (ncdij, 3, len(self._grid_points)), dtype="complex128", order="C"
        )
        if self._fourier_series is None:
            self._init_finufft([dV.shape for dV in delta_Vs[0].dV])

        count = 0
        for delta_V in delta_Vs:
            dVs_rec = [self._get_Fourier_coefficients(dV) for dV in delta_V.dV]
            for r, t, tr in zip(rotations, translations, time_reversals, strict=True):
                dVs_rotated = self._rotate_dV(dVs_rec, r, t)
                if ncdij == 4:  # Need to rotate in spin space, too.
                    dVs_spinor_rotated = rotate_delta_vals_in_spin_space(
                        dVs_rotated, r, lattice
                    )
                    dVs_rotated.clear()
                    dVs_rotated = dVs_spinor_rotated
                if tr:
                    dVs_rotated = apply_time_reversal_in_spin_space(dVs_rotated)
                for i_cdij in range(ncdij):
                    dVdu[i_cdij] += np.outer(disps_inv[:, count], dVs_rotated[i_cdij])
                count += 1
                dVs_rotated.clear()
            dVs_rec.clear()

        if self._fourier_series is None:
            self._finalize_finufft()
        return dVdu

    def _rotate_dV(
        self, dVs_rec: list[NDArray], r: NDArray, t: NDArray
    ) -> list[NDArray]:
        """Rotate dV by rotating coordinates of delta potential passively.

        With finufft, the grid points are rotated instead of delta potential.
        With nufft="native" or "native-box", the Fourier coefficients are
        relabeled by FourierSeriesOnGrid.

        Parameters
        ----------
        dVs_rec : list of ndarray
            Results of _get_Fourier_coefficients for each spin component.
            dtype='complex128', shape=(nz, ny, nx) each
        r : ndarray
            Rotation of the operation g in the supercell basis vectors.
            dtype='int64', shape=(3, 3)
        t : ndarray
            Translation of g in the supercell basis vectors.
            dtype='double', shape=(3,)

        Returns
        -------
        list of ndarray
            dV(g^-1 x) at the grid points x for g = {r|t}, for each spin
            component.
            dtype='complex128', shape=(n_grid_points,) each

        """
        if self._fourier_series is not None:
            return self._fourier_series.evaluate(dVs_rec, rotation=r, translation=t)
        r_inv = np.linalg.inv(r)
        t_inv = -r_inv @ t
        grid_points = self._grid_points @ r_inv.T + t_inv
        grid_points -= np.rint(grid_points)
        return self._run_finufft(grid_points, dVs_rec)

    def _get_Fourier_coefficients(self, dV: NDArray) -> NDArray:
        """Return the Fourier coefficients of dV in the layout of the method.

        Parameters
        ----------
        dV : ndarray
            Change in the local potential of one spin component on the mesh of
            VASP.
            dtype='complex128', shape=(nz, ny, nx)

        Returns
        -------
        ndarray
            get_fourier_coefficients(dV) with nufft="native" or "native-box",
            otherwise _get_iFFT_of_dV(dV).
            dtype='complex128', shape=dV.shape

        """
        if self._fourier_series is None:
            return self._get_iFFT_of_dV(dV)
        return get_fourier_coefficients(dV)

    def _get_iFFT_of_dV(self, dV: NDArray) -> NDArray:
        """Inverse FFT.

        Parameters
        ----------
        dV : ndarray
            Change in the local potential of one spin component on the mesh of
            VASP.
            dtype='complex128', shape=(nz, ny, nx)

        Returns
        -------
        ndarray
            Inverse FFT of dV, shifted by fftshift so that the zero frequency is
            at the center, as finufft takes the Fourier coefficients.
            dtype='complex128', shape=dV.shape

        """
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

        Parameters
        ----------
        grid_points : ndarray
            Points in the supercell basis vectors at which the Fourier series
            are evaluated.
            dtype='double', shape=(n_points, 3)
        dV_iFTs : list of ndarray
            Fourier coefficients from _get_iFFT_of_dV for each spin
            component, with a plan of _init_finufft for each.
            dtype='complex128', shape=(nz, ny, nx) each

        Returns
        -------
        list of ndarray
            Values at grid_points of the Fourier series of each of dV_iFTs.
            dtype='complex128', shape=(len(grid_points),) each

        """
        assert self._finufft_plan is not None
        retval = []
        for i, dV_iFT in enumerate(dV_iFTs):
            x, y, z = [
                np.array(v, dtype=grid_points.dtype, order="C")
                for v in (grid_points * (np.pi * 2)).T
            ]
            self._finufft_plan[i].setpts(z, y, x)
            retval.append(self._finufft_plan[i].execute(dV_iFT))
        return retval

    def _init_finufft(self, mesh_shapes: list[tuple[int, ...]]):
        """Make a type-2 plan of finufft for each spin component.

        Parameters
        ----------
        mesh_shapes : list of tuple of int
            Shapes (nz, ny, nx) of the Fourier coefficients of the spin
            components.

        """
        import finufft

        self._finufft_plan = [
            finufft.Plan(2, shape, eps=self._finufft_eps, dtype="complex128")
            for shape in mesh_shapes
        ]

    def _finalize_finufft(self):
        assert self._finufft_plan is not None
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
        Lattice points of the primitive cell, one from each class modulo the
        lattice of the supercell, in the basis vectors of the primitive cell.
        They are not reduced into the supercell.  See get_lattice_points.
        dtype='int64', shape=(det(p2s_matrix), 3)
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
        fft_mesh: Sequence[int] | NDArray,
        p2s_matrix: NDArray,
        supercell: PhonopyAtoms,
        symmetry: Symmetry | None = None,
        atom_indices: NDArray | None = None,
        finufft_eps: float | None = None,
        nufft: Literal["finufft", "native", "native-box"] = "finufft",
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
        finufft_eps : float or None, optional
            Accuracy of finufft interpolation. Default is None, which
            corresponds to 1e-6.
        nufft : {"finufft", "native", "native-box"}, optional
            Method to evaluate the Fourier series of dV at the grid points.  See
            LocalPotentialInterpolationNUFFT.  Default is "finufft".
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
        self._finufft_eps: float | None = finufft_eps
        self._nufft: Literal["finufft", "native", "native-box"] = nufft
        self._lattice_points: NDArray | None = None
        self._grid_points: NDArray | None = None
        self._dVdu: NDArray | None = None  # self.dVdu is provided by @property.

    @classmethod
    def from_arrays(
        cls,
        fft_mesh: Sequence[int] | NDArray,
        p2s_matrix: NDArray,
        supercell: PhonopyAtoms,
        dVdu: NDArray,
        grid_points: NDArray,
        lattice_points: NDArray,
        symmetry: Symmetry | None = None,
        atom_indices: NDArray | None = None,
        verbose: bool = True,
    ) -> DLocalPotential:
        """Return DLocalPotential that holds dV/du computed before.

        This is used to read phelel_params.hdf5.  The parameters other than
        dVdu, grid_points and lattice_points are those of __init__.

        Parameters
        ----------
        dVdu : ndarray
            dV/du on the grid points.
            dtype='complex128', shape=(ncdij, atom_indices, 3, grid_points)
        grid_points : ndarray
            Grid points in the supercell basis vectors, as get_grid_points
            returns.
            dtype='double', shape=(prod(fft_mesh) * det(p2s_matrix), 3)
        lattice_points : ndarray
            Lattice points of the primitive cell in the basis vectors of the
            primitive cell, as get_grid_points returns.
            dtype='int64', shape=(det(p2s_matrix), 3)

        Returns
        -------
        DLocalPotential

        """
        dlp = cls(
            fft_mesh,
            p2s_matrix,
            supercell,
            symmetry=symmetry,
            atom_indices=atom_indices,
            verbose=verbose,
        )
        n_lattice_points = determinant(dlp.p2s_matrix)
        n_grid_points = int(np.prod(dlp.fft_mesh)) * n_lattice_points
        for name, value, shape in (
            ("dVdu", dVdu, (dVdu.shape[0], *dlp._get_dVdu_shape())),
            ("grid_points", grid_points, (n_grid_points, 3)),
            ("lattice_points", lattice_points, (n_lattice_points, 3)),
        ):
            if value.shape != shape:
                raise ValueError(f"Shape of {name} is {value.shape}, not {shape}.")
        dlp._dVdu = np.array(dVdu, dtype="complex128", order="C")
        dlp._grid_points = np.array(grid_points, dtype="double", order="C")
        dlp._lattice_points = np.array(lattice_points, dtype="int64", order="C")
        return dlp

    @property
    def p2s_matrix(self) -> NDArray:
        """Return supercell matrix."""
        return self._p2s_matrix

    @property
    def supercell(self) -> PhonopyAtoms:
        """Return supercell."""
        return self._supercell

    @property
    def symmetry(self) -> Symmetry:
        """Return symmetry of supercell."""
        return self._symmetry

    @property
    def atom_indices(self) -> NDArray:
        """Return atom indices."""
        return self._atom_indices

    @property
    def fft_mesh(self) -> Sequence[int] | NDArray:
        """Return FFT mesh."""
        return self._fft_mesh

    @property
    def dVdu(self) -> NDArray | None:
        """Return dVdu.

        See detail at attribute section of this class's docstring.

        """
        return self._dVdu

    @property
    def lattice_points(self) -> NDArray | None:
        """Return lattice points.

        See detail at attribute section of this class's docstring.

        """
        return self._lattice_points

    @property
    def grid_points(self):
        """Return grid_points.

        See detail at attribute section of this class's docstring.

        """
        return self._grid_points

    def run(
        self,
        V_loc_per: NDArray,
        V_loc_disps: list[NDArray],
        displacements: Sequence[FirstAtomDisplacement],
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
        check_displaced_atoms(self._atom_indices, displacements, self._symmetry)
        lpi = LocalPotentialInterpolationNUFFT(
            self._fft_mesh,
            self._p2s_matrix,
            self._supercell,
            self._symmetry,
            atom_indices=self._atom_indices,
            finufft_eps=self._finufft_eps,
            nufft=self._nufft,
            verbose=self._verbose,
        )
        self._lattice_points = lpi.lattice_points.copy(order="C")
        self._grid_points = lpi.grid_points.copy(order="C")

        if self._dVdu is None:
            self._allocate_arrays(V_loc_per.shape[0])

        assert self._dVdu is not None

        for disp_atom in np.unique([d["number"] for d in displacements]):
            dVs = []
            for i, d in enumerate(displacements):
                if d["number"] == disp_atom:
                    dVs.append(
                        DeltaLocalPotential(V_loc_per, V_loc_disps[i], displacements[i])
                    )
            lpi.run(dVs)
            assert lpi.atom_indices_returned is not None
            indices = []
            for ai in lpi.atom_indices_returned:
                indices.append(np.where(self.atom_indices == ai)[0][0])
            self._dVdu[:, indices, :, :] = lpi.dVdu

    def visualize(
        self,
        pcell: PhonopyAtoms,
        i_atom: int,
        directory: str | os.PathLike = ".",
    ) -> list[pathlib.Path]:
        """Write dV/du of an atom in x, y, z in LOCPOT-like files.

        The parts v, v_x, v_y, v_z of get_potential_and_magnetic_parts are
        written in

            locpot_viz-{atom number}-{x,y,z}.dat for v,
            locpot_viz-{atom number}-{x,y,z}-{vx,vy,vz}.dat for the others,

        where the atom number counts the atoms of the supercell from 1.

        Parameters
        ----------
        pcell : PhonopyAtoms
            Primitive cell, whose basis vectors are p2s_matrix^-1 times those
            of the supercell.
        i_atom : int
            Index of the atom in atom_indices.
        directory : str or os.PathLike, optional
            Directory where the files are written.  Default is the current
            directory.

        Returns
        -------
        list of pathlib.Path
            Written files.

        """
        assert self._grid_points is not None
        assert self._dVdu is not None
        atom_number = self._atom_indices[i_atom] + 1
        filenames = []
        for i_dir, xyz in enumerate("xyz"):
            parts = get_potential_and_magnetic_parts(self._dVdu[:, i_atom, i_dir])
            for name, values in parts.items():
                suffix = "" if name == "v" else f"-{name}"
                filename = (
                    pathlib.Path(directory)
                    / f"locpot_viz-{atom_number:03d}-{xyz}{suffix}.dat"
                )
                multi = visualize_distribution(
                    filename,
                    pcell,
                    self._p2s_matrix,
                    self._fft_mesh,
                    self._grid_points,
                    values,
                )
                filenames.append(filename)

        if self._verbose:
            names = ", ".join(
                f'"locpot_viz-{atom_number:03d}-{{x,y,z}}'
                + ("" if name == "v" else f"-{name}")
                + '.dat"'
                for name in parts
            )
            msg = (
                "[%d, %d, %d] supercell of primitive cell is "
                "used. dV/du is written in %s, in LOCPOT-like "
                "format." % (*multi, names)
            )
            print(
                textwrap.fill(
                    msg, width=70, initial_indent="    ", subsequent_indent="    "
                )
            )
        return filenames

    def _get_dVdu_shape(self) -> tuple[int, int, int]:
        """Return the shape of dVdu without the spin index.

        Returns
        -------
        tuple of int
            (len(atom_indices), 3, number of grid points)

        """
        num_gp = int(
            self._fft_mesh[0]
            * self._fft_mesh[1]
            * self._fft_mesh[2]
            * determinant(self._p2s_matrix)
        )
        return (len(self._atom_indices), 3, num_gp)

    def _allocate_arrays(self, ncdij: int):
        shape = (ncdij, *self._get_dVdu_shape())
        self._dVdu = np.zeros(shape, dtype="complex128", order="C")


def get_potential_and_magnetic_parts(
    values: NDArray[np.cdouble],
) -> dict[str, NDArray[np.double]]:
    """Return the potential part and the magnetic parts of spin components.

    The spin components c are those of the local potential read from VASP and
    of dVdu of DLocalPotential: (v) without spin, (v + v_z, v - v_z) with
    collinear spin, and the 2x2 matrix (v + v_z, v_x - i v_y, v_x + i v_y,
    v - v_z) with non-collinear spin.  They are converted back by

        ncdij = 2: v = (c_0 + c_1) / 2, v_z = (c_0 - c_1) / 2,
        ncdij = 4: v = (c_0 + c_3) / 2, v_x = (c_1 + c_2) / 2,
                   v_y = i (c_1 - c_2) / 2, v_z = (c_0 - c_3) / 2.

    v, v_x, v_y, v_z are real, and their real parts are returned.

    Parameters
    ----------
    values : ndarray
        Spin components.
        dtype='complex128', shape=(ncdij, ...)

    Returns
    -------
    dict
        "v", and "vz" for ncdij = 2 or "vx", "vy", "vz" for ncdij = 4, in this
        order.
        dtype='double', shape=values.shape[1:] each

    """
    c = values
    if len(c) == 1:
        parts = {"v": c[0]}
    elif len(c) == 2:
        parts = {"v": (c[0] + c[1]) / 2, "vz": (c[0] - c[1]) / 2}
    elif len(c) == 4:
        parts = {
            "v": (c[0] + c[3]) / 2,
            "vx": (c[1] + c[2]) / 2,
            "vy": 1j * (c[1] - c[2]) / 2,
            "vz": (c[0] - c[3]) / 2,
        }
    else:
        raise ValueError(
            f"Number of spin components has to be 1, 2 or 4, not {len(c)}."
        )
    return {name: np.array(part.real, dtype="double") for name, part in parts.items()}


def visualize_distribution(
    filename: str | os.PathLike,
    pcell: PhonopyAtoms,
    p2s_matrix: NDArray,
    fft_mesh: int | float | Sequence | NDArray,
    grid_points: NDArray,
    data: NDArray,
) -> NDArray:
    """Visualize scalar distribution in space.

    Note
    ----
    multiplicity is the three integers that simply extend primitive cell. This
    extended cell includes the supercell. See more detail in
    _get_multipliticy_for_visualization. This is because the grid is defined
    along primitive cell basis vectors.

    Returns
    -------
    ndarray
        Multiplicity of the primitive cell along its basis vectors, which gives
        the cell written in the file.
        dtype=int, shape=(3,)

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
    d_diag, P, _ = snf3x3(np.array(M, dtype="int64"))
    P_inv = np.linalg.inv(P)
    diag_A = tuple(d_diag)

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
        *done.shape,
    )

    scell = get_supercell(pcell, np.diag(multiplicity))
    header = "\n".join(get_vasp_structure_lines(scell))
    locpot = get_CHGCAR(dV_viz.real, header)  # type: ignore

    with open(filename, "w") as w:
        w.write(locpot)

    return multiplicity


def get_grid_points(
    fft_mesh: Sequence[int] | NDArray, p2s_matrix: NDArray
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
        Lattice points of the primitive cell, one from each class modulo the
        lattice of the supercell, in the basis vectors of the primitive cell.
        They are not reduced into the supercell.  See get_lattice_points.
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
