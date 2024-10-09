"""Phelel API."""

from __future__ import annotations

import io
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import Primitive, Supercell, isclose
from phonopy.structure.symmetry import Symmetry

from phelel.base.Dij_qij import DDijQij
from phelel.base.local_potential import DLocalPotential
from phelel.file_IO import write_phelel_params_hdf5
from phelel.version import __version__


@dataclass
class PhelelDataset:
    """Data structure of input data to run derivatives."""

    local_potentials: list[np.ndarray]
    Dijs: list[np.ndarray]
    qijs: list[np.ndarray]
    lm_channels: list[dict]
    dataset: Optional[dict] = None
    phonon_dataset: Optional[dict] = None
    forces: Optional[np.ndarray] = None


class Phelel:
    """Phelel class.

    Attributes
    ----------
    unitcell : PhonopyAtoms
        Unit cell.
    supercell : Supercell
        Spercell.
    primitive : Primitive
        Primitive cell.
    symmetry : Symmetry
        Symmetry of supercell.
    phonon : Phonopy
        Phonopy class instance.
    supercell_matrix : ndarray
        Supercell matrix relative to unit cell.
        dtype='int_', shape=(3,3)
    primitive_matrix : ndarray
        Primitive matrix relative to unit cell. Default is None.
        dtype='double', shape=(3,3)
    atom_indices_in_derivatives : ndarray
        Atom indices in supercell used for computing derivatives.
        dtype='int_', shape=(atom_indices_in_derivatives,)
    fft_mesh : ndarray
        FFT mesh numbers for primitive cell used for generating local potential
        derivative interpolation grid.
        dtype='int_', shape=(3,)
    dVdu : DLocalPotential
        A class instance to calculate and store dV/du.
    dDijdu : DDijQij
        A class instance to calculate and store dDij/du and dQij/du.

    """

    def __init__(
        self,
        unitcell: PhonopyAtoms,
        supercell_matrix: Optional[Union[Sequence, np.ndarray]] = None,
        primitive_matrix: Optional[Union[str, Sequence, np.ndarray]] = None,
        phonon_supercell_matrix: Optional[
            Union[int, float, Sequence, np.ndarray]
        ] = None,
        fft_mesh: Optional[Union[Sequence, np.ndarray]] = None,
        symprec: float = 1e-5,
        is_symmetry: bool = True,
        calculator: Optional[str] = None,
        nufft: Optional[str] = None,
        finufft_eps: Optional[float] = None,
        log_level: int = 0,
    ):
        """Init method.

        Parameters
        ----------
        unitcell : PhonopyAtoms
            Unit cell.
        supercell_matrix : array_like, optional
            Supercell matrix multiplied to input cell basis vectors.
            shape=(3, ) or (3, 3), where the former is considered a diagonal
            matrix. Default is the unit matrix.
            dtype=int
        primitive_matrix : array_like or str, optional
            Primitive matrix multiplied to input cell basis vectors. Default is
            None, which is equivalent to 'auto'.
            For array_like, shape=(3, 3), dtype=float.
            When 'F', 'I', 'A', 'C', or 'R' is given instead of a 3x3 matrix,
            the primitive matrix for the character found at
            https://spglib.github.io/spglib/definition.html
            is used.
        phonon_supercell_matrix : array_like, optional
            Supercell matrix used for phonon calculation. Supercell matrix for
            derivatives of potentials and phonon (derivative of forces) can be
            different. Unless setting this, supercell_matrix is used as
            phonon_supercell_matrix. Default is None.
        fft_mesh : array_like, optional
            FFT mesh numbers for primitive cell. Default is None.
            dtype='int_', shape=(3,)
        symprec : float, optional
            Symmetry tolerance used to search crystal symmetry. Default
            is 1e-5.
        is_symmetry : bool, optional
            Use crystal symmetry or not. Default is True.
        calculator :
            A dummy parameter.
        nufft : str or None, optional
            'finufft' only. Default is None, which corresponds to 'finufft'.
        finufft_eps : float or None, optional
            Accuracy of finufft interpolation. Default is None, which
            corresponds to 1e-6.
        log_level : int, optional
            Log level. 0 is most quiet. Default is 0.

        """
        self._unitcell = unitcell
        self._symprec = symprec
        self._is_symmetry = is_symmetry
        self._calculator = calculator
        self._nufft = nufft
        self._finufft_eps = finufft_eps
        self._log_level = log_level

        ph = self._get_phonopy(supercell_matrix, primitive_matrix)
        self._primitive_matrix = ph.primitive_matrix
        self._supercell_matrix = ph.supercell_matrix
        self._primitive = ph.primitive
        self._supercell = ph.supercell
        self._symmetry = ph.symmetry
        self._dataset = ph.dataset
        self._atom_indices_in_derivatives = self._primitive.p2s_map

        if phonon_supercell_matrix is None:
            self._phonon = ph
        else:
            self._phonon = self._get_phonopy(phonon_supercell_matrix, primitive_matrix)
            assert isclose(self._primitive, self._phonon.primitive)

        p2s_mat_float = np.linalg.inv(self.primitive.primitive_matrix)
        self._p2s_matrix = np.rint(p2s_mat_float).astype("int_")
        assert (abs(self._p2s_matrix - p2s_mat_float) < 1e-5).all()

        if phonon_supercell_matrix is None:
            self._phonon_supercell_matrix = None
        else:
            self._phonon_supercell_matrix = self._phonon.supercell_matrix

        self._dVdu = None
        self._dDijdu = None

        if fft_mesh is None:
            self._fft_mesh = None
        else:
            self.fft_mesh = fft_mesh

        self._dDijdu = DDijQij(
            self._supercell,
            symmetry=self._symmetry,
            atom_indices=self._atom_indices_in_derivatives,
            verbose=self._log_level > 0,
        )

    @property
    def version(self) -> str:
        """Return phelel release version number.

        str
            Phelel release version number

        """
        return __version__

    @property
    def calculator(self) -> str:
        """Return calculator interface name.

        str
            Calculator name such as 'vasp', 'qe', etc.

        """
        return self._calculator

    @property
    def unitcell(self) -> PhonopyAtoms:
        """Return unitcell."""
        return self._unitcell

    @property
    def supercell(self) -> Supercell:
        """Return supercell."""
        return self._supercell

    @property
    def primitive(self) -> Primitive:
        """Return primitive cell."""
        return self._primitive

    @property
    def p2s_matrix(self):
        """Return supercell matrix."""
        return self._p2s_matrix

    @property
    def dataset(self) -> dict:
        """Setter and getter of potential dataset."""
        return self._dataset

    @dataset.setter
    def dataset(self, dataset: dict):
        self._dataset = dataset

    @property
    def atom_indices_in_derivatives(self) -> np.ndarray:
        """Return atom indices used for calculation of derivatives."""
        return self._atom_indices_in_derivatives

    @property
    def symmetry(self) -> Symmetry:
        """Return symmetry of supercell."""
        return self._symmetry

    @property
    def supercell_matrix(self) -> np.ndarray:
        """Return supercell matrix."""
        return self._supercell_matrix

    @property
    def primitive_matrix(self) -> Optional[np.ndarray]:
        """Return primitive matrix."""
        return self._primitive_matrix

    @property
    def phonon_supercell_matrix(self) -> Optional[np.ndarray]:
        """Return supercell matrix used for phonon calculation.

        If ``phonon_supercell_matrix`` is not specified at instantiation of
        ``Phelel`` class, this returns None.

        """
        return self._phonon_supercell_matrix

    @property
    def phonon_supercell(self) -> Supercell:
        """Return phonon supercell."""
        return self._phonon.supercell

    @property
    def phonon_primitive(self) -> Primitive:
        """Return phonon primitive cell."""
        return self._phonon.primitive

    @property
    def phonon_dataset(self) -> dict:
        """Setter and getter of phonon dataset."""
        return self._phonon.dataset

    @phonon_dataset.setter
    def phonon_dataset(self, phonon_dataset):
        self._phonon.dataset = phonon_dataset

    @property
    def nac_params(self) -> Optional[dict]:
        """Setter and getter of parameters for non-analytical term correction.

        dict
            Parameters used for non-analytical term correction
            'born': ndarray
                Born effective charges
                shape=(primitive cell atoms, 3, 3), dtype='double', order='C'
            'factor': float
                Unit conversion factor
            'dielectric': ndarray
                Dielectric constant tensor
                shape=(3, 3), dtype='double', order='C'

        """
        return self._phonon._nac_params

    @nac_params.setter
    def nac_params(self, nac_params: dict):
        self._phonon.nac_params = nac_params

    @property
    def force_constants(self) -> Optional[np.ndarray]:
        """Return force constants."""
        return self._phonon.force_constants

    @force_constants.setter
    def force_constants(self, force_constants: np.ndarray):
        self._phonon.force_constants = force_constants

    @property
    def unit_conversion_factor(self) -> float:
        """Return phonon frequency unit conversion factor.

        float
            Phonon frequency unit conversion factor. This factor
            converts sqrt(<force>/<distance>/<AMU>)/2pi/1e12 to THz
            (ordinary frequency).

        """
        return self._phonon.unit_conversion_factor

    @property
    def phonon(self) -> Phonopy:
        """Return Phonopy class instance."""
        return self._phonon

    @property
    def fft_mesh(self) -> np.ndarray:
        """Setter and getter of FFT mesh numbers."""
        return self._fft_mesh

    @fft_mesh.setter
    def fft_mesh(self, fft_mesh: Union[Sequence, np.ndarray]):
        self._fft_mesh = np.array(fft_mesh, dtype="int_")
        self._dVdu = DLocalPotential(
            self._fft_mesh,
            self._p2s_matrix,
            self._supercell,
            symmetry=self._symmetry,
            atom_indices=self._atom_indices_in_derivatives,
            nufft=self._nufft,
            finufft_eps=self._finufft_eps,
            verbose=self._log_level > 0,
        )

    @property
    def dVdu(self) -> Optional[DLocalPotential]:
        """Return DLocalPotential class instance."""
        return self._dVdu

    @dVdu.setter
    def dVdu(self, dVdu: DLocalPotential):
        self._dVdu = dVdu

    @property
    def dDijdu(self) -> DDijQij:
        """Return DDijQij class instance."""
        return self._dDijdu

    @dDijdu.setter
    def dDijdu(self, dDijdu: DDijQij):
        self._dDijdu = dDijdu

    @property
    def supercells_with_displacements(self) -> Optional[list[PhonopyAtoms]]:
        """Return supercells with displacements.

        list of PhonopyAtoms
            Supercells with displacements generated by
            Phonopy.generate_displacements.

        """
        ph = self._get_phonopy(self._supercell_matrix, self._primitive_matrix)
        ph.dataset = self._dataset
        return ph.supercells_with_displacements

    @property
    def phonon_supercells_with_displacements(self) -> Optional[list[PhonopyAtoms]]:
        """Return supercells with displacements for phonons.

        list of PhonopyAtoms
            Supercells with displacements generated by
            Phonopy.generate_displacements.

        """
        return self._phonon.supercells_with_displacements

    def generate_displacements(
        self,
        distance=0.01,
        is_plusminus="auto",
        is_diagonal=True,
    ):
        """Generate displacement dataset."""
        ph = self._get_phonopy(self._supercell_matrix, self._primitive_matrix)
        ph.generate_displacements(
            distance=distance, is_plusminus=is_plusminus, is_diagonal=is_diagonal
        )
        self._dataset = ph.dataset

    def run_derivatives(self, phe_input: PhelelDataset):
        """Run displacement derivatives calculations from temporary raw data.

        Note
        ----
        After calculation, temporary raw data may be deleted.
        Force constants are created to have full matrix shape.

        """
        if self._fft_mesh is None:
            msg = (
                "fft_mesh for dV/du interpolation has to be set before running this "
                "method."
            )
            raise RuntimeError(msg)

        if phe_input.dataset is not None:
            self._dataset = phe_input.dataset
        loc_pots = phe_input.local_potentials
        Dijs = phe_input.Dijs
        qijs = phe_input.qijs

        if phe_input.phonon_dataset is not None:
            self._prepare_phonon(
                dataset=phe_input.phonon_dataset,
                forces=phe_input.forces,
                calculate_full_force_constants=True,
            )
        else:
            self._prepare_phonon(
                dataset=self._dataset,
                forces=phe_input.forces,
                calculate_full_force_constants=True,
            )
        self._dVdu.run(loc_pots[0], loc_pots[1:], self._dataset["first_atoms"])
        self._dDijdu.run(
            Dijs[0],
            Dijs[1:],
            qijs[0],
            qijs[1:],
            self._dataset["first_atoms"],
            phe_input.lm_channels,
        )

    def save_hdf5(
        self, filename: Union[str, bytes, os.PathLike, io.IOBase] = "phelel_params.hdf5"
    ):
        """Write phelel_params.hdf5."""
        write_phelel_params_hdf5(
            dVdu=self._dVdu,
            dDijdu=self._dDijdu,
            supercell_matrix=self._supercell_matrix,
            primitive_matrix=self._primitive_matrix,
            primitive=self._primitive,
            unitcell=self._unitcell,
            supercell=self._supercell,
            atom_indices_in_derivatives=self._atom_indices_in_derivatives,
            disp_dataset=self._dataset,
            force_constants=self._phonon.force_constants,
            phonon_supercell_matrix=self._phonon.supercell_matrix,
            phonon_primitive=self._phonon.primitive,
            phonon_supercell=self._phonon.supercell,
            nac_params=self._phonon.nac_params,
            symmetry_dataset=self._phonon.primitive_symmetry.dataset,
            filename=filename,
        )

    def _prepare_phonon(
        self,
        dataset: Optional[dict] = None,
        forces: Optional[
            Union[
                Sequence,
                np.ndarray,
            ]
        ] = None,
        force_constants: Optional[np.ndarray] = None,
        calculate_full_force_constants: bool = True,
        fc_calculator: Optional[str] = None,
        fc_calculator_options: Optional[str] = None,
        show_drift: bool = True,
    ):
        """Initialize phonon calculation."""
        if dataset is not None:
            self._phonon.dataset = dataset
        if forces is not None:
            self._phonon.forces = forces
            self._phonon.produce_force_constants(
                calculate_full_force_constants=calculate_full_force_constants,
                fc_calculator=fc_calculator,
                fc_calculator_options=fc_calculator_options,
                show_drift=show_drift,
            )
        if force_constants is not None:
            self._phonon.force_constants = force_constants

    def _get_phonopy(
        self,
        supercell_matrix: Optional[Union[int, float, Sequence, np.ndarray]],
        primitive_matrix: Optional[Union[str, Sequence, np.ndarray]],
    ) -> Phonopy:
        """Return Phonopy instance.

        Note
        ----
        store_dense_svecs=False is necessary to be compatible with code in VASP.

        """
        return Phonopy(
            self._unitcell,
            supercell_matrix=supercell_matrix,
            primitive_matrix=primitive_matrix,
            symprec=self._symprec,
            is_symmetry=self._is_symmetry,
            store_dense_svecs=False,
            calculator=self._calculator,
            log_level=self._log_level,
        )
