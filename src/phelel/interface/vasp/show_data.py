"""Convenient routines to show detailed data."""

import io
import os

import h5py
import numpy as np
from phonopy.interface.vasp import VasprunxmlExpat, get_vasp_structure_lines, read_vasp

from phelel.interface.vasp.file_IO import (
    get_CHGCAR,
    read_dprojectors,
    read_eigenvalues,
    read_inwap_yaml,
    read_local_potential,
    read_PAW_Dij_qij,
    read_qtot,
    read_waves,
)
from phelel.interface.vasp.procar import QTOT, CoreCharge, Procar


class DijQij:
    """Container of Dij and qij.

    Attributes
    ----------
    Dij : ndarray
        PAW strength
        dtype=complex128
        shape=(spin, atom, lm, l'm')
    qij : ndarray
        <phi_i|phi_j>-<phi_i~|phi_j~>
        dtype=complex128
        shape=(spin, atom, lm, l'm')

    """

    def __init__(self, Dij, qij):
        """Init method.

        Note
        ----
        The order of lm is:
            (l=0,m=0), (l=1,m=-1), (l=1,m=0), (l=1,m=1), (l=2,m=-2),
            (l=2,m=-1), (l=2,m=0), ...

        Parameters
        ----------
        Dij : ndarray
            PAW strength
            dtype=complex128
            shape=(spin, atom, lm, l'm')
        qij : ndarray
            <phi_i|phi_j>-<phi_i~|phi_j~>
            dtype=complex128
            shape=(spin, atom, lm, l'm')

        """
        self.Dij = Dij
        self.qij = qij

    def __str__(self):
        """Generate qij and Dij as text."""
        # qij are real and Dij are expected to be real.
        text = self._str_format(self.Dij.real, "Dij real part")
        # text += self._str_format(self.Dij.imag, "Dij imaginary part")
        text += self._str_format(self.qij.real, "qij real part")
        # text += self._str_format(self.qij.imag, "qij imaginary part")
        return text

    def _str_format(self, val, comment=""):
        text = ""
        for i, dij_spin in enumerate(val):
            for j, dij_atom in enumerate(dij_spin):
                text += "Spin %d, Atom %d, %s (%d, %d)\n" % (
                    (i + 1, j + 1) + (comment,) + dij_atom.shape
                )
                for dij_l in dij_atom:
                    # if (abs(dij_l) > 1e-4).any():
                    #     text += " ".join(["%8.4f" % x for x in dij_l]) + "\n"
                    # else:
                    #     text += " ".join(["%8.1e" % x for x in dij_l]) + "\n"
                    text += " ".join(["%8.1e" % x for x in dij_l]) + "\n"
        return text


class LocalPotential:
    """Container of Local potential."""

    def __init__(self, V_loc):
        """Init method.

        Parameters
        ----------
        V_loc : ndarray
            V_loc stores local potential.
            dtype=complex128
            shape=(ncdij, nz, ny, nx)

            ncdij=1 without spin
            ncdij=2 with collinear spin (up and down)
            ncdij=4 with non-collinear spin (2x2 localpotential)

        """
        self._V_loc = V_loc

    @property
    def V(self):
        """Return local potential."""
        return self._V_loc

    def write_locpot(self, header, spin_polarized=False, filename="locpot.dat"):
        """Write local potential to locpot.dat."""
        text = get_CHGCAR(self._V_loc[0].real, header)
        with open(filename, "w") as w:
            w.write(text)

    def __str__(self):
        """Return array shape of local potential as text."""
        text = "Local potential (ncdij, nz, ny, nx) = "
        text += str(self._V_loc.shape) + "\n"
        # text += str(self._V_loc)
        return text


class WaveFunction:
    """Container of pseudo wave functions."""

    def __init__(self, waves, weights=None, occupancy=None):
        """Init method.

        Parameters
        ----------
        waves : ndarray
            wavefunctions
            shape=(nkpts, nbtot, ispin, nrspinors, nz, ny, nx)
            dtype=complex128
        weights : list of float values
            weights of k-points. This is not given by wap.
            shape=(nkpts,)
        occupancy : list of lists of float values
            Electron occupations at bands and k-points. This is not given by
            wap.
            shape=(nkpts, nbtot)

        """
        self._waves = waves
        self._weights = weights
        self._occupancy = occupancy

    def write_charge(self, header, spin_polarized=False, filename="charge.dat"):
        """Write sum of squared orbitals to charge.dat."""
        charge = self._sum_charge(spin_polarized)
        text = get_CHGCAR(charge, header)
        with open(filename, "w") as w:
            w.write(text)
        return charge.sum() / np.prod(charge.shape)

    def _square_sum(self, ikpt=0, iband=0, ispin=0, ispinor=0):
        wave = self._waves[ikpt, iband, ispin, ispinor]
        return (wave.real**2 + wave.imag**2).sum()

    def _sum_charge(self, spin_polarized):
        """Calculate pseudo charge density.

        Note that wave functions are not normalized by themselves due to overlap
        matrix. Charge density out of PAW spheres is expected to agree with
        CHGCAR except for mesh density.

        """
        charge = self._waves.real**2 + self._waves.imag**2
        sum_charge = charge.sum(axis=2).sum(axis=2)
        if spin_polarized:
            occupancy = self._occupancy
        else:
            occupancy = np.array(self._occupancy) * 2

        for i, w in enumerate(self._weights):
            for j, occ in enumerate(occupancy[i]):
                sum_charge[i, j] *= w * occ
        sum_charge = sum_charge.sum(axis=0).sum(axis=0)
        return sum_charge

    def __str__(self):
        """Return total sum of squared orbitals on grid as text."""
        text = "Waves (nkpts, nbtot, ispin, nrspinors, nz, ny, nx) = "
        text += str(self._waves.shape)
        text += "\n"
        # text += str(self._waves)
        for ikpt, iband in list(np.ndindex(self._waves.shape[:2])):
            text += "sum_i|pseudo-wave_i|^2 at k-point %d, band %d: %f\n" % (
                ikpt + 1,
                iband + 1,
                self._square_sum(ikpt=ikpt, iband=iband),
            )
        return text


class VaspShowData:
    """Show el-ph related data generated by VASP."""

    def __init__(self):
        """Init method."""
        self.parse()

    def parse(self):
        """Parse files."""
        self._inwap = read_inwap_yaml(filename="inwap.yaml")
        self._Dij = read_PAW_Dij_qij(self._inwap, "PAW-STRENGTH.bin")
        self._qij = read_PAW_Dij_qij(self._inwap, "PAW-OVERLAP.bin")
        self._qtot = read_qtot(self._inwap, filename="QTOT.bin")
        self._V_loc = read_local_potential(self._inwap, filename="LOCAL-POTENTIAL.bin")

        if os.path.exists("EIGENVALUE.bin"):
            self._eigvals = read_eigenvalues(self._inwap, filename="EIGENVALUE.bin")
        else:
            self._eigvals = None
        if os.path.exists("WAVES.bin"):
            self._waves = read_waves(self._inwap, filename="WAVES.bin")
        else:
            self._waves = None
        if os.path.exists("DPROJECTORS.bin"):
            self._proj = read_dprojectors(self._inwap, filename="DPROJECTORS.bin")
            filename = "dprojects.hdf5"
            w = h5py.File(filename, "w")
            w.create_dataset("cproj", data=self._proj)
            w.close()
        else:
            self._proj = None

        self._cell_el = read_vasp("POSCAR")
        self._parse_vasprun_xml()

    def show(self):
        """Show data."""
        if self._waves is not None:
            print(self.waves)
        print(self.dij)
        print(self.V_loc)
        print(self.Q_tot)
        if self._proj is not None:
            print(self.procar)
            print(self.core_charge)
        print(self.eigenvalues)

    def chgcar(self):
        """Return total charge and write orbital squared to a file."""
        header = "\n".join(get_vasp_structure_lines(self._cell_el))
        n_charge = self.waves.write_charge(header)

        msg = (
            "VASP CHGCAR like data is generated from wave function. "
            "k-point symmetry is not expanded in phelel, therefore "
            "to obtain very similar results to CHGCAR, ISYM=0 has to "
            "be specified."
        )
        import textwrap

        text = textwrap.fill(msg, width=70) + "\n\n"
        text += 'The result is written into "charge.dat".\n'
        text += "Total number of electrons = %f" % n_charge
        return text

    def locpot(self):
        """Write local potential to locpot.dat."""
        header = "\n".join(get_vasp_structure_lines(self._cell_el))
        self.V_loc.write_locpot(header)

        msg = (
            'The local potential is written in "locpot.dat" in VASP LOCPOT-like format.'
        )
        import textwrap

        text = textwrap.fill(msg, width=70)
        return text

    @property
    def dij(self):
        """Return DijQij class instance."""
        dij = DijQij(self._Dij, self._qij)
        return dij

    @property
    def V_loc(self):
        """Return LocalPotential class instance."""
        V_loc = LocalPotential(self._V_loc)
        return V_loc

    @property
    def eigenvalues(self):
        """Return Eigenvalue class instance."""
        e = Eigenvalue(self._eigvals)
        return e

    @property
    def waves(self):
        """Return WaveFunction class instance."""
        occupancy = []
        for eig_k in self._xml_eigvals[0]:  # spin 1
            occupancy.append([eig_b[1] for eig_b in eig_k])
        waves = WaveFunction(self._waves, weights=self._k_weights, occupancy=occupancy)
        return waves

    @property
    def Q_tot(self):
        """Return QTOT class instance."""
        qtot = QTOT(self._qtot)
        return qtot

    @property
    def procar(self):
        """Return Procar class instance."""
        proj = Procar(self._proj, self._inwap["lm_orbitals"], self._qtot)
        return proj

    @property
    def core_charge(self):
        """Return CoreCharge class instance."""
        rho = CoreCharge(self._proj, self._qij)
        return rho

    def _parse_vasprun_xml(self, filename="vasprun.xml"):
        with io.open(filename, "rb") as f:
            vxml = VasprunxmlExpat(f)
            if vxml.parse():
                self._xml_eigvals = vxml.get_eigenvalues()
                self._k_weights = vxml.get_k_weights()
            else:
                print("Parsing vasprun.xml failed.")


class Eigenvalue:
    """Container of eigenvalues."""

    def __init__(self, e):
        """Init method.

        Parameters
        ----------
        e : ndarray
            Eigenvalues
            dtype='double'
            shape=(nbtot, nkpts, ispin)

        """
        self._e = e

    def __str__(self):
        """Generate eigenvalues information as text."""
        text = "Eigenvalues (nbtot, nkpts, ispin) = "
        text += str(self._e.shape) + "\n"
        for i in range(self._e.shape[2]):
            for j, e_k in enumerate(self._e[:, :, i].T):
                text += "ISPIN: %d/%d, k-point: %d/%d\n" % (
                    i + 1,
                    self._e.shape[2],
                    j + 1,
                    self._e.shape[1],
                )
                text += "%5s %16s\n" % ("Band", "Eigenvalue")
                for k, e_b in enumerate(e_k):
                    text += "%5d %16.6f\n" % (k + 1, e_b)
                text += "\n"
        return text
