"""Phelel API."""

from __future__ import annotations

import io
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from phonopy import Phonopy
from phonopy.exception import ForcesetsNotFoundError
from phonopy.harmonic.dynamical_matrix import DynamicalMatrix
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

    local_potentials: list[NDArray]
    Dijs: list[NDArray]
    qijs: list[NDArray]
    lm_channels: list[dict]
    dataset: dict | None = None
    phonon_dataset: dict | None = None
    forces: NDArray | None = None


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
        dtype='int64', shape=(3,3)
    primitive_matrix : ndarray
        Primitive matrix relative to unit cell. Default is None.
        dtype='double', shape=(3,3)
    atom_indices_in_derivatives : ndarray
        Atom indices in supercell used for computing derivatives.
        dtype='int64', shape=(atom_indices_in_derivatives,)
    fft_mesh : ndarray
        FFT mesh numbers for primitive cell used for generating local potential
        derivative interpolation grid.
        dtype='int64', shape=(3,)
    dVdu : DLocalPotential
        A class instance to calculate and store dV/du.
    dDijdu : DDijQij
        A class instance to calculate and store dDij/du and dQij/du.

    """

    def __init__(
        self,
        unitcell: PhonopyAtoms,
        supercell_matrix: ArrayLike | None = None,
        primitive_matrix: str | ArrayLike | None = None,
        phonon_supercell_matrix: ArrayLike | None = None,
        fft_mesh: ArrayLike | None = None,
        symprec: float = 1e-5,
        is_symmetry: bool = True,
        calculator: str | None = None,
        nufft: str | None = None,
        finufft_eps: float | None = None,
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
            dtype='int64', shape=(3,)
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

        self._phelel_phonon = self._get_phonopy(supercell_matrix, primitive_matrix)
        self._atom_indices_in_derivatives = self._phelel_phonon.primitive.p2s_map

        self._phonon: Optional[Phonopy] = None
        if phonon_supercell_matrix is not None:
            self._phonon = self._get_phonopy(phonon_supercell_matrix, primitive_matrix)
            assert isclose(self._phelel_phonon.primitive, self._phonon.primitive)

        p2s_mat_float = np.linalg.inv(self.primitive.primitive_matrix)
        self._p2s_matrix = np.rint(p2s_mat_float).astype("int64")
        assert (abs(self._p2s_matrix - p2s_mat_float) < 1e-5).all()

        self._dVdu = None

        if fft_mesh is None:
            self._fft_mesh = None
        else:
            self.fft_mesh = fft_mesh

        self._dDijdu = DDijQij(
            self._phelel_phonon.supercell,
            symmetry=self._phelel_phonon.symmetry,
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
    def calculator(self) -> str | None:
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
        return self._phelel_phonon.supercell

    @property
    def primitive(self) -> Primitive:
        """Return primitive cell."""
        return self._phelel_phonon.primitive

    @property
    def p2s_matrix(self):
        """Return supercell matrix."""
        return self._p2s_matrix

    @property
    def dataset(self) -> dict | None:
        """Setter and getter of potential dataset."""
        return self._phelel_phonon.dataset

    @dataset.setter
    def dataset(self, dataset: dict):
        self._phelel_phonon.dataset = dataset

    @property
    def atom_indices_in_derivatives(self) -> np.ndarray:
        """Return atom indices used for calculation of derivatives."""
        return self._atom_indices_in_derivatives

    @property
    def symmetry(self) -> Symmetry:
        """Return symmetry of supercell."""
        return self._phelel_phonon.symmetry

    @property
    def primitive_symmetry(self) -> Symmetry:
        """Return symmetry of primitive cell."""
        return self._phelel_phonon.primitive_symmetry

    @property
    def supercell_matrix(self) -> np.ndarray:
        """Return supercell matrix."""
        return self._phelel_phonon.supercell_matrix

    @property
    def primitive_matrix(self) -> Optional[np.ndarray]:
        """Return primitive matrix."""
        return self._phelel_phonon.primitive_matrix

    @property
    def phonon_supercell_matrix(self) -> Optional[np.ndarray]:
        """Return supercell matrix used for phonon calculation.

        If ``phonon_supercell_matrix`` is not specified at instantiation of
        ``Phelel`` class, this returns None.

        """
        if self._phonon is None:
            return None
        else:
            return self._phonon.supercell_matrix

    @property
    def phonon_supercell(self) -> Optional[Supercell]:
        """Return phonon supercell."""
        if self._phonon is None:
            return None
        else:
            return self._phonon.supercell

    @property
    def phonon_primitive(self) -> Optional[Primitive]:
        """Return phonon primitive cell."""
        if self._phonon is None:
            return None
        else:
            return self._phonon.primitive

    @property
    def phonon_dataset(self) -> Optional[dict]:
        """Setter and getter of phonon dataset."""
        if self._phonon is None:
            return None
        else:
            return self._phonon.dataset

    @phonon_dataset.setter
    def phonon_dataset(self, phonon_dataset):
        if self._phonon is None:
            raise RuntimeError("Phonon instance is not initialized.")
        self._phonon.dataset = phonon_dataset

    @property
    def nac_params(self) -> dict | None:
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
        return self._phelel_phonon.nac_params

    @nac_params.setter
    def nac_params(self, nac_params: dict):
        self._phelel_phonon.nac_params = nac_params
        if self._phonon is not None:
            self._phonon.nac_params = nac_params

    @property
    def force_constants(self) -> Optional[np.ndarray]:
        """Return force constants."""
        if self._phonon is None:
            return self._phelel_phonon.force_constants
        else:
            return self._phonon.force_constants

    @force_constants.setter
    def force_constants(self, force_constants: np.ndarray):
        if self._phonon is None:
            self._phelel_phonon.force_constants = force_constants
        else:
            self._phonon.force_constants = force_constants

    @property
    def forces(self) -> np.ndarray:
        """Setter and getter of forces of supercells."""
        if self._phonon is None:
            return self._phelel_phonon.forces
        else:
            return self._phonon.forces

    @forces.setter
    def forces(self, forces: Union[Sequence, np.ndarray]):
        _forces = np.array(forces, dtype="double", order="C")
        if self._phonon is None:
            self._phelel_phonon.forces = _forces
        else:
            self._phonon.forces = _forces

    @property
    def unit_conversion_factor(self) -> float | None:
        """Return phonon frequency unit conversion factor.

        float
            Phonon frequency unit conversion factor. This factor
            converts sqrt(<force>/<distance>/<AMU>)/2pi/1e12 to THz
            (ordinary frequency).

        """
        if self._phonon is None:
            return None
        else:
            return self._phonon.unit_conversion_factor

    @property
    def fft_mesh(self) -> NDArray | None:
        """Setter and getter of FFT mesh numbers."""
        return self._fft_mesh

    @fft_mesh.setter
    def fft_mesh(self, fft_mesh: ArrayLike):
        self._fft_mesh = np.array(fft_mesh, dtype="int64")
        self._dVdu = DLocalPotential(
            self._fft_mesh,
            self._p2s_matrix,
            self._phelel_phonon.supercell,
            symmetry=self.symmetry,
            atom_indices=self.atom_indices_in_derivatives,
            nufft=self._nufft,
            finufft_eps=self._finufft_eps,
            verbose=self._log_level > 0,
        )

    @property
    def dVdu(self) -> DLocalPotential | None:
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
    def supercells_with_displacements(self) -> list[PhonopyAtoms] | None:
        """Return supercells with displacements.

        list of PhonopyAtoms
            Supercells with displacements generated by
            Phonopy.generate_displacements.

        """
        return self._phelel_phonon.supercells_with_displacements

    @property
    def phonon_supercells_with_displacements(self) -> list[PhonopyAtoms] | None:
        """Return supercells with displacements for phonons.

        list of PhonopyAtoms
            Supercells with displacements generated by
            Phonopy.generate_displacements.

        """
        if self._phonon is None:
            return None
        else:
            return self._phonon.supercells_with_displacements

    @property
    def dynamical_matrix(self) -> DynamicalMatrix | None:
        """Return DynamicalMatrix class instance."""
        return self._phelel_phonon.dynamical_matrix

    def generate_displacements(
        self,
        distance: float | None = 0.01,
        is_plusminus: Literal["auto"] | bool = "auto",
        is_diagonal: bool = True,
    ):
        """Generate displacement dataset."""
        self._phelel_phonon.generate_displacements(
            distance=distance, is_plusminus=is_plusminus, is_diagonal=is_diagonal
        )

    def generate_phonon_displacements(
        self,
        distance: float | None = 0.01,
        is_plusminus: Literal["auto"] | bool = "auto",
        is_diagonal: bool = True,
    ):
        """Generate displacement dataset."""
        if self._phonon is None:
            msg = "Phonon instance has to be initialized."
            raise RuntimeError(msg)
        self._phonon.generate_displacements(
            distance=distance, is_plusminus=is_plusminus, is_diagonal=is_diagonal
        )

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
            self._phelel_phonon.dataset = phe_input.dataset
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
                dataset=self._phelel_phonon.dataset,
                forces=phe_input.forces,
                calculate_full_force_constants=True,
            )
        assert self._phelel_phonon.dataset is not None
        assert self._dVdu is not None
        self._dVdu.run(
            loc_pots[0], loc_pots[1:], self._phelel_phonon.dataset["first_atoms"]
        )
        self._dDijdu.run(
            Dijs[0],
            Dijs[1:],
            qijs[0],
            qijs[1:],
            self._phelel_phonon.dataset["first_atoms"],
            phe_input.lm_channels,
        )

    def save_hdf5(
        self, filename: Union[str, bytes, os.PathLike, io.IOBase] = "phelel_params.hdf5"
    ):
        """Write phelel_params.hdf5."""
        params = {
            "dVdu": self.dVdu,
            "dDijdu": self.dDijdu,
            "supercell_matrix": self.supercell_matrix,
            "primitive_matrix": self.primitive_matrix,
            "primitive": self.primitive,
            "unitcell": self.unitcell,
            "supercell": self.supercell,
            "atom_indices_in_derivatives": self._atom_indices_in_derivatives,
            "disp_dataset": self.dataset,
            "nac_params": self.nac_params,
            "force_constants": self.force_constants,
            "symmetry_dataset": self.primitive_symmetry.dataset,
            "filename": filename,
        }
        if self._phonon is not None:
            params.update(
                {
                    "phonon_supercell_matrix": self.phonon_supercell_matrix,
                    "phonon_primitive": self.phonon_primitive,
                    "phonon_supercell": self.phonon_supercell,
                }
            )
        write_phelel_params_hdf5(**params)

    def save_phonon(
        self,
        filename: str | os.PathLike = "phonopy_params.yaml",
        settings: dict | None = None,
        hdf5_settings: dict | None = None,
        compression: str | bool = False,
    ):
        """Save phonon in yaml format."""
        if self._phonon is None:
            ph = self._phelel_phonon
        else:
            ph = self._phonon
        ph.save(
            filename=filename,
            settings=settings,
            hdf5_settings=hdf5_settings,
            compression=compression,
        )

    def _prepare_phonon(
        self,
        dataset: dict | None = None,
        forces: ArrayLike | None = None,
        force_constants: ArrayLike | None = None,
        calculate_full_force_constants: bool = True,
        fc_calculator: Literal["traditional", "symfc", "alm"] | None = None,
        fc_calculator_options: str | None = None,
        show_drift: bool = True,
    ):
        """Initialize phonon calculation."""
        if self._phonon is None:
            ph = self._phelel_phonon
        else:
            ph = self._phonon
        if dataset is not None:
            ph.dataset = dataset
        if forces is not None:
            ph.forces = forces
        try:
            ph.produce_force_constants(
                calculate_full_force_constants=calculate_full_force_constants,
                fc_calculator=fc_calculator,
                fc_calculator_options=fc_calculator_options,
                show_drift=show_drift,
            )
        except ForcesetsNotFoundError:
            if force_constants is not None:
                ph.force_constants = force_constants

    def _get_phonopy(
        self,
        supercell_matrix: ArrayLike | None = None,
        primitive_matrix: ArrayLike | None = None,
    ) -> Phonopy:
        """Return Phonopy instance."""
        return Phonopy(
            self._unitcell,
            supercell_matrix=supercell_matrix,
            primitive_matrix=primitive_matrix,
            symprec=self._symprec,
            is_symmetry=self._is_symmetry,
            calculator=self._calculator,
            log_level=self._log_level,
        )
