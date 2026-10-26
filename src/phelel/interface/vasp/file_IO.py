"""VASP file IO functions."""

from __future__ import annotations

import os
from typing import Literal

import h5py
import numpy as np
import yaml
from numpy.typing import NDArray
from phonopy.file_IO import get_io_module_to_decompress


###########
# Readers #
###########
def read_bin_stream(
    filename: str | os.PathLike, dtype: np.dtype | None = None
) -> NDArray:
    """Read binary stream."""
    if dtype is None:
        dtype = np.dtype("double")
    myio = get_io_module_to_decompress(filename)
    with myio.open(filename, "rb") as fp:
        data = np.frombuffer(fp.read(), dtype=dtype)
    return data


def read_inwap_vaspouth5(
    filename: str | os.PathLike = "vaspout.h5",
) -> dict:
    """Read inwap-like information in vaspout.h5."""
    inwap = {}
    with h5py.File(filename) as h5:
        # dimensions of the potential
        pot_shape: tuple = h5["results/potential/total"].shape  # type: ignore
        ncdij = pot_shape[0]
        fft_fine = pot_shape[1:]
        # electronic structure
        ispin = h5["/results/electron_eigenvalues/ispin"][()]  # type: ignore
        nbtot = h5["/results/electron_eigenvalues/nb_tot"][()]  # type: ignore
        nkpts = h5["/results/electron_eigenvalues/kpoints"][()]  # type: ignore
        # number of ions
        nions: int = h5["/results/electron_eigenvalues/nions"][()]  # type: ignore
        # kpoint coordinates
        kpoints = h5["/results/electron_eigenvalues/kpoint_coords"][:]  # type: ignore
        inwap["ncdij"] = ncdij
        inwap["nions"] = nions
        inwap["fft_fine"] = fft_fine
        inwap["ispin"] = ispin
        inwap["nkpts"] = nkpts
        inwap["nbtot"] = nbtot
        inwap["nrspinors"] = 2 if ncdij == 4 else 1
        inwap["kpoints"] = kpoints
        lmdim = h5["results/paw/lmdim"][()]  # type: ignore
        ldim = h5["results/paw/ldim"][0]  # type: ignore
        inwap["lmdim"] = lmdim  # data["PAW"]["LMDIM"]
        inwap["ldim"] = ldim  # data["PAW"]["LDIM"]

        # read required information to generate lm_orbitals
        nitypes: NDArray = h5["/input/poscar/number_ion_types"][:]  # type: ignore
        lmax: NDArray = h5["/results/paw/lmax"][:]  # type: ignore
        lps: NDArray = h5["/results/paw/lps"][:]  # type: ignore

        # construct lm_orbitals dictionary here
        itypes = sum([[i] * nityp for i, nityp in enumerate(nitypes)], [])
        lm_orbitals = []
        for i in range(nions):
            ityp = itypes[i]
            channels = []
            for il in range(lmax[ityp]):
                ll = lps[ityp, il]
                channels.append({"l": ll, "m": list(range(-ll, ll + 1))})
            atom_dict = {
                "atom_index": i + 1,
                "num_l_channels": lmax[ityp],
                "channels": channels,
            }
            lm_orbitals.append(atom_dict)
        inwap["lm_orbitals"] = lm_orbitals
        # TODO
        inwap["fft_coarse"] = None  # data["fft_grid"]["coarse"]

    return inwap


def read_inwap_yaml(filename: str | os.PathLike = "inwap.yaml") -> dict:
    """Read inwap.yaml."""
    inwap = {}
    with open(filename) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
        inwap["ncdij"] = data["cell"]["NCDIJ"]
        inwap["nions"] = data["cell"]["NIONS"]
        inwap["fft_coarse"] = data["fft_grid"]["coarse"]
        inwap["fft_fine"] = data["fft_grid"]["fine"]
        inwap["ispin"] = data["cell"]["ISPIN"]
        inwap["nkpts"] = data["cell"]["NKPTS"]
        inwap["nbtot"] = data["cell"]["NBTOT"]
        inwap["nrspinors"] = data["cell"]["NRSPINORS"]
        inwap["lmdim"] = data["PAW"]["LMDIM"]
        inwap["ldim"] = data["PAW"]["LDIM"]
        inwap["lm_orbitals"] = data["PAW"]["lm_orbitals"]
        inwap["kpoints"] = data["KPOINTS"]

    return inwap


def read_PAW_Dij_qij_vaspouth5(
    filename: str | os.PathLike = "vaspout.h5",
) -> tuple[NDArray, NDArray]:
    """Read Dij and qij in vaspout.h5.

    Dij(ncdij, nions, lmdim, lmdim')
    qij(ncdij, nions, lmdim, lmdim')

    Used such as <psi|p><lm|A|lm'><p|psi'>

    """
    with h5py.File(filename) as h5:
        dij_real = h5["/results/paw/dij"][:]  # type: ignore
        qij_real = h5["/results/paw/qij"][:]  # type: ignore
    dij = dij_real[:, :, :, :, 0] + 1j * dij_real[:, :, :, :, 1]  # type: ignore
    qij = qij_real[:, :, :, :, 0] + 1j * qij_real[:, :, :, :, 1]  # type: ignore
    return dij, qij


def read_PAW_Dij_qij(inwap: dict, filename: str | os.PathLike) -> NDArray:
    """Read Dij or qij.

    Dij(ncdij, nions, lmdim, lmdim')
    qij(ncdij, nions, lmdim, lmdim')

    Used such as <psi|p><lm|A|lm'><p|psi'>

    """
    ncdij = inwap["ncdij"]
    nions = inwap["nions"]
    lmdim = inwap["lmdim"]
    shape = (ncdij, nions, lmdim, lmdim)
    data = read_bin_stream(filename)
    data_complex = data.view("complex128")
    if len(data_complex) != np.prod(shape):
        raise ValueError(
            f"{filename}: data size is inconsistent with values in inwap.yaml."
        )
    return data_complex.reshape(shape)


def read_local_potential_vaspouth5(
    filename: str | os.PathLike = "vaspout.h5", key: Literal["total", "xcmu"] = "total"
) -> NDArray:
    """Read local potentials in vaspout.h5.

    For spin unpolarized calculations the potential is real but we convert it to
    complex.
    For collinear calculations the potential is written as (v, v_z), and it is
    returned as (v + v_z, v - v_z), i.e., spin up and spin down.
    For non-collinear calculations the potential is written as (v, v_x, v_y,
    v_z), and it is returned as the 2x2 matrix (v + v_z, v_x - i v_y,
    v_x + i v_y, v - v_z).

    VASP 6.6.1 and older write part of the points of a collinear or
    non-collinear potential converted to spin up and spin down or to the 2x2
    matrix, where v_y is lost.  For these, RuntimeError is raised, and the
    potential has to be read from LOCAL-POTENTIAL.bin written with
    LWAP = .TRUE.

    Kinetic energy density (xcmu) is also read by this function in the same way.

    """
    with h5py.File(filename) as h5:
        # dimensions of the potential
        pot_real: NDArray = h5[f"/results/potential/{key}"][:]  # type: ignore
        ncdij = pot_real.shape[0]
        if ncdij > 1:
            version = _get_vasp_version_vaspouth5(h5)
            if version is None or version <= (6, 6, 1):
                written_by = (
                    "VASP of unknown version"
                    if version is None
                    else "VASP " + ".".join(str(v) for v in version)
                )
                raise RuntimeError(
                    f'The local potential in "{filename}" was written by '
                    f"{written_by}. VASP 6.6.1 and older do not write it "
                    "correctly for ISPIN = 2 or LNONCOLLINEAR = .TRUE. Run "
                    "VASP with LWAP = .TRUE., then LOCAL-POTENTIAL.bin is read "
                    "in place of vaspout.h5."
                )
        pot = np.zeros_like(pot_real, dtype=complex)

        if ncdij == 1:
            v = pot_real[0, :, :, :]
            pot[:, :, :, :] = v
        if ncdij == 2:
            v = pot_real[0, :, :, :]
            vz = pot_real[1, :, :, :]
            pot[0, :, :, :] = v + vz
            pot[1, :, :, :] = v - vz

        if ncdij == 4:
            v = pot_real[0, :, :, :]
            vx = pot_real[1, :, :, :]
            vy = pot_real[2, :, :, :]
            vz = pot_real[3, :, :, :]
            pot[0, :, :, :] = v + vz
            pot[1, :, :, :] = vx - 1j * vy
            pot[2, :, :, :] = vx + 1j * vy
            pot[3, :, :, :] = v - vz
        return pot


def _get_vasp_version_vaspouth5(h5: h5py.File) -> tuple[int, int, int] | None:
    """Return the VASP version (major, minor, patch) in vaspout.h5, or None."""
    if "version" not in h5:
        return None
    group = h5["version"]
    major, minor, patch = (int(group[key][()]) for key in ("major", "minor", "patch"))  # type: ignore
    return major, minor, patch


def read_local_potential(
    inwap: dict, filename: str | os.PathLike = "LOCAL-POTENTIAL.bin"
) -> NDArray:
    """Read LOCAL-POTENTIAL.bin.

    Normally local potential (SV) is real in VASP. But non-collinear version, it
    turns to be complex. So we force local potential to be complex in either
    case.

    """
    ncdij = inwap["ncdij"]
    nx, ny, nz = inwap["fft_fine"]
    shape = (ncdij, nz, ny, nx)
    data = read_bin_stream(filename)
    data_complex = data.view("complex128")
    if len(data_complex) != np.prod(shape):
        raise ValueError(
            f"{filename}: data size is inconsistent with values in inwap.yaml."
        )
    return data_complex.reshape(shape)


def get_CHGCAR(charge: NDArray, header: str) -> str:
    """Return CHGCAR style text from ndarray.

    Each value is preceded by a space, so that a negative value is separated
    from the value before it.

    """
    text = header + "\n%5d%5d%5d\n" % charge.shape[::-1]
    count = 0
    for z in range(charge.shape[0]):
        for y in range(charge.shape[1]):
            for x in range(charge.shape[2]):
                text += " %17.11e" % charge[z, y, x]
                count += 1
                if (count % 5) == 0:
                    count = 0
                    text += "\n"
    return text
