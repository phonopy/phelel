"""Calculation of Eliashberg function."""

from collections.abc import Sequence
from typing import Optional, Union

import h5py
import numpy as np
import spglib
from phono3py.file_IO import get_filename_suffix
from phono3py.other.kaccum import KappaDOSTHM
from phono3py.phonon.grid import BZGrid, get_grid_point_from_address, get_ir_grid_points
from phonopy import Phonopy
from phonopy.phonon.tetrahedron_mesh import TetrahedronMesh
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.grid_points import GridPoints
from phonopy.units import THzToEv


def get_Eliashberg_function(h5_filename: str):
    """Get Eliashberg function."""
    with h5py.File(h5_filename) as f:
        # [itemp]
        temps = f["results/electron_phonon/phonons/self_energy_1/temps"][:]
        # [ikpt, 3]
        ir_kpoints = f["results/electron_phonon/phonons/self_energy_1/kpoint_coords"][:]
        lat_scale = f["results/positions/scale"][()]
        lattice = f["results/positions/lattice_vectors"][:] * lat_scale
        # ion_types = f["results/positions/ion_types"][:]
        number_ion_types = f["results/positions/number_ion_types"][:]
        numbers = []
        for i, nums in enumerate(number_ion_types):
            numbers += [i + 1] * nums

        positions = f["results/positions/position_ions"][:]
        # row vectors in python (and KPOINTS file, too)
        k_gen_vecs = f[
            "results/electron_phonon/phonons/self_energy_1/kpoint_generating_vectors"
        ][:]
        # [ispin, itemp] in 1/eV
        dos_at_ef = f["results/electron_phonon/phonons/self_energy_1/dos_at_ef"][:, :]
        # [ispin, ib, ikpt , 1, temp, (re,im)] in eV
        gamma = -f["results/electron_phonon/phonons/self_energy_1/selfen_ph"][
            :, :, :, 0, :, 1
        ]
        # [ikpt, ib] in 2piTHz
        freqs = f["results/electron_phonon/phonons/self_energy_1/phonon_freqs_ibz"][:]
        # [ispin, itemp, freq_points]
        a2f_vasp = f["results/electron_phonon/phonons/self_energy_1/a2F"][:]
        # in 2piTHz
        freq_points_vasp = f[
            "results/electron_phonon/phonons/self_energy_1/frequency_grid"
        ][:]

    sym_dataset = spglib.get_symmetry_dataset((lattice, positions, numbers))
    mesh = np.linalg.inv(lattice.T @ k_gen_vecs).T
    mesh = np.rint(mesh).astype(int)
    bz_grid = BZGrid(mesh, lattice=lattice, symmetry_dataset=sym_dataset)
    ir_grid_points, ir_grid_weights, ir_grid_map = get_ir_grid_points(bz_grid)

    # ir_gps indices in phonopy are mapped to those in vasp.
    ir_addresss = np.rint(ir_kpoints @ mesh).astype(int)
    gps = get_grid_point_from_address(ir_addresss, bz_grid.D_diag)
    irgp = ir_grid_map[gps]
    id_map = [np.where(irgp == gp)[0][0] for gp in ir_grid_points]

    a2f_at_qj = np.zeros(freqs.shape + (len(temps),), dtype="double")
    for itemp, _ in enumerate(temps):
        a2f_at_qj[:, :, itemp] = (
            (
                2
                * gamma[0, :, :, itemp].T
                / (freqs * THzToEv / (2 * np.pi))
                / (dos_at_ef[0, itemp] * 2 * np.pi)
            )
            / THzToEv
            * 2
            * np.pi
        )
    kappados = KappaDOSTHM(
        a2f_at_qj[None, id_map, :, :],
        freqs[id_map],
        bz_grid,
        ir_grid_points=ir_grid_points,
        ir_grid_weights=ir_grid_weights,
        ir_grid_map=ir_grid_map,
        num_sampling_points=201,
    )
    freq_points, a2f = kappados.get_kdos()

    import matplotlib.pyplot as plt

    plt.figure()
    for itemp, temp in enumerate(temps):
        plt.plot(freq_points_vasp, a2f_vasp[0, itemp, :], label=f"{temp} K")
        plt.plot(freq_points, a2f[0, :, 1, itemp], label=f"{temp} K")
    plt.xlabel("Frequency (2piTHz)")
    plt.ylabel("a2F")
    plt.title("a2F vs Frequency")
    plt.legend()
    plt.show()


class EliashbergFunction:
    """Eliashberg function class."""

    def __init__(
        self,
        mesh: Union[Sequence, np.ndarray],
        primitive: PhonopyAtoms,
        rotations: np.ndarray,
        gamma: np.ndarray,
        frequencies: np.ndarray,
        eigenvalues: np.ndarray,
        fermi_level: Optional[float] = None,
    ):
        """Init method."""
        self._mesh = mesh
        self._primitive = primitive
        self._frequencies = frequencies
        self._eigenvalues = eigenvalues
        self._gamma = gamma
        self._fermi_level = fermi_level

        self._gp = GridPoints(
            self._mesh,
            np.linalg.inv(self._primitive.cell),
            is_gamma_center=True,
            rotations=rotations,
        )
        assert len(self._gp.ir_grid_points) == len(frequencies)
        self._eliashberg = None

        # 2pi cancels to convert unit of delta function
        self._unit_conversion = 1.0 / THzToEv / 2 / np.prod(self._mesh)

        if self._fermi_level is not None:
            self.run_electronic_DOS(np.array([self._fermi_level], dtype="double"))
            self._unit_conversion /= self._eldos[0]

    @property
    def eliashberg(self):
        """Return Eliashberg function."""
        return self._eliashberg

    @property
    def gp(self):
        """Return GridPoints instance."""
        return self._gp

    @property
    def electronic_dos(self):
        """Return electronic DOS."""
        return self._eldos

    def run_tetrahedron_method(self, frequency_points: Union[Sequence, np.ndarray]):
        """Calculate Eliashberg function spectrum using tetrahedron method."""
        thm = TetrahedronMesh(
            self._primitive,
            self._frequencies,
            self._mesh,
            self._gp.grid_address,
            self._gp.grid_mapping_table,
            self._gp.ir_grid_points,
        )
        self._eliashberg = np.zeros(len(frequency_points), dtype="double")
        thm.set(value="I", frequency_points=frequency_points)
        coef = self._gamma / self._frequencies
        for i, iw in enumerate(thm):
            # iw (fpoints, bands)
            if i == 0:
                continue
            w = self._gp.weights[i]
            self._eliashberg += np.dot(iw, coef[i]) * w
        self._eliashberg *= self._unit_conversion

    def run_electronic_DOS(self, frequency_points: Union[Sequence, np.ndarray]):
        """Calculate electronic DOS using tetrahedron method."""
        thm = TetrahedronMesh(
            self._primitive,
            self._eigenvalues,
            self._mesh,
            self._gp.grid_address,
            self._gp.grid_mapping_table,
            self._gp.ir_grid_points,
        )
        self._eldos = np.zeros(len(frequency_points), dtype="double")
        thm.set(value="I", frequency_points=frequency_points)
        for i, iw in enumerate(thm):
            # iw (fpoints, bands)
            if i == 0:
                continue
            w = self._gp.weights[i]
            self._eldos += iw.sum(axis=1) * w


def _collect_frequency_and_gamma(phonon: Phonopy):
    frequencies = phonon.get_mesh_dict()["frequencies"]
    ir_grid_points = phonon.mesh.ir_grid_points
    assert len(ir_grid_points) == len(frequencies)
    gamma = np.zeros_like(frequencies)
    for i, gp in enumerate(ir_grid_points):
        suffix = get_filename_suffix(phonon.mesh_numbers, grid_point=gp)
        filename = "gamma%s.hdf5" % suffix
        with h5py.File(filename, "r") as f:
            gamma[i] = f["gamma"][:]

    return frequencies, gamma


def _get_eigenvalues(mesh_numbers: np.ndarray, ir_grid_points: np.ndarray):
    suffix = get_filename_suffix(mesh_numbers)
    filename = "electron%s.hdf5" % suffix
    with h5py.File(filename, "r") as f:
        eigenvalues = f["eigenvalues"][:]
    return np.array(eigenvalues[ir_grid_points, :, 0], order="C", dtype="double")


def show_Eliashberg_function(
    phonon: Phonopy,
    num_frequency_points: int = 4001,
    fermi_level: Optional[float] = None,
):
    """Show Eliashberg functions at frequency points."""
    frequencies, gamma = _collect_frequency_and_gamma(phonon)
    ir_grid_points = phonon.mesh.ir_grid_points
    eigenvalues = _get_eigenvalues(phonon.mesh_numbers, ir_grid_points)
    ef = EliashbergFunction(
        phonon.mesh_numbers,
        phonon.primitive,
        phonon.primitive_symmetry.get_pointgroup_operations(),
        gamma,
        frequencies,
        eigenvalues,
        fermi_level=fermi_level,
    )
    print("# DOS at fermi level: %f" % ef.electronic_dos[0])
    fpts = np.linspace(
        frequencies.min() - 0.1, frequencies.max() + 0.1, num_frequency_points
    )
    ef.run_tetrahedron_method(fpts)
    for fp, dos in zip(fpts, ef.eliashberg):
        print(fp, dos)

    # To plot electronic DOS
    # fpts = np.linspace(eigenvalues.min() - 0.1, eigenvalues.max() + 0.1,
    #                    num_frequency_points)
    # ef.run_electronic_DOS(fpts)
    # print("")
    # print("")
    # for fp, dos in zip(fpts, ef.electronic_dos * 2):
    #     print(fp, dos)


def show_frequency_and_gamma(phonon: Phonopy):
    """Show frequecnies and Eliashberg functions."""
    frequencies, gamma = _collect_frequency_and_gamma(phonon)
    for f_band, g_band in zip(frequencies, gamma):
        for f, g in zip(f_band, g_band):
            print(f, g)
