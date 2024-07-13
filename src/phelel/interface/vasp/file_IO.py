"""VASP file IO functions."""

import os
from typing import Union

import h5py
import numpy as np
import yaml
from phono3py.file_IO import get_filename_suffix
from phonopy.file_IO import get_io_module_to_decompress


###########
# Readers #
###########
def read_bin_stream(filename, dtype=None):
    """Read binary stream."""
    if dtype is None:
        dtype = np.dtype("double")
    myio = get_io_module_to_decompress(filename)
    with myio.open(filename, "rb") as fp:
        data = np.frombuffer(fp.read(), dtype=dtype)
    return data


def read_inwap_vaspouth5(
    filename: Union[str, bytes, os.PathLike] = "vaspout.h5",
) -> dict:
    """Read inwap-like information in vaspout.h5."""
    inwap = {}
    with h5py.File(filename) as h5:
        # dimensions of the potential
        pot = h5["results/potential/total"]
        ncdij = pot.shape[0]
        fft_fine = pot.shape[1:]
        # electronic structure
        ispin = h5["/results/electron_eigenvalues/ispin"][()]
        nbtot = h5["/results/electron_eigenvalues/nb_tot"][()]
        nkpts = h5["/results/electron_eigenvalues/kpoints"][()]
        # number of ions
        nions = h5["/results/electron_eigenvalues/nions"][()]
        # kpoint coordinates
        kpoints = h5["/results/electron_eigenvalues/kpoint_coords"][:]
        inwap["ncdij"] = ncdij
        inwap["nions"] = nions
        inwap["fft_fine"] = fft_fine
        inwap["ispin"] = ispin
        inwap["nkpts"] = nkpts
        inwap["nbtot"] = nbtot
        inwap["nrspinors"] = 2 if ncdij == 4 else 1
        inwap["kpoints"] = kpoints
        lmdim = h5["results/paw/lmdim"][()]
        ldim = h5["results/paw/ldim"][0]
        inwap["lmdim"] = lmdim  # data["PAW"]["LMDIM"]
        inwap["ldim"] = ldim  # data["PAW"]["LDIM"]

        # read required information to generate lm_orbitals
        nitypes = h5["/input/poscar/number_ion_types"][:]
        lmax = h5["/results/paw/lmax"][:]
        lps = h5["/results/paw/lps"][:]

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


def read_inwap_yaml(filename: Union[str, bytes, os.PathLike] = "inwap.yaml") -> dict:
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


def read_PAW_Dij_qij_vaspouth5(filename):
    """Read Dij and qij in vaspout.h5.

    Dij(ncdij, nions, lmdim, lmdim')
    qij(ncdij, nions, lmdim, lmdim')

    Used such as <psi|p><lm|A|lm'><p|psi'>

    """
    with h5py.File(filename) as h5:
        dij_real = h5["/results/paw/dij"][:]
        qij_real = h5["/results/paw/qij"][:]
    dij = dij_real[:, :, :, :, 0] + 1j * dij_real[:, :, :, :, 1]
    qij = qij_real[:, :, :, :, 0] + 1j * qij_real[:, :, :, :, 1]
    return dij, qij


def read_PAW_Dij_qij(inwap, filename, is_Rij=False):
    """Read Dij, qij, and Rij.

    Dij(ncdij, nions, lmdim, lmdim')
    qij(ncdij, nions, lmdim, lmdim')
    Rij(2, ncdij, nions, lmdim, lmdim')

    Used such as <psi|p><lm|A|lm'><p|psi'>

    """
    ncdij = inwap["ncdij"]
    nions = inwap["nions"]
    lmdim = inwap["lmdim"]
    if is_Rij:
        shape = (2, ncdij, nions, lmdim, lmdim)
    else:
        shape = (ncdij, nions, lmdim, lmdim)
    data = read_bin_stream(filename)
    dt = "c%d" % (data.itemsize * 2)
    data_complex = data.view(dtype=dt)
    if len(data_complex) != np.prod(shape):
        print("%s: data size is inconsistent with values in inwap.yaml." % filename)
        return None
    return data_complex.reshape(shape)


def read_local_potential_vaspouth5(inwap, filename="vaspout.h5"):
    """Read local potentials in vaspout.h5.

    For spin unpolarized calculations the potential is real but we convert it to
    complex.
    For collinear calculations the potential is written as up and down.
    For non-collinear calculations the potential is written as scalar potential
    + magnetic field potential

    """
    with h5py.File(filename) as h5:
        # dimensions of the potential
        pot_real = h5["/results/potential/total"][:]
        ncdij = pot_real.shape[0]
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


def read_local_potential(inwap, filename="LOCAL-POTENTIAL.bin"):
    """Read LOCAL-POTENTIAL.bin.

    Normally local potential (SV) is real in VASP. But non-collinear version, it
    turns to be complex. So we force local potential to be complex in either
    case.

    """
    ncdij = inwap["ncdij"]
    nx, ny, nz = inwap["fft_fine"]
    shape = (ncdij, nz, ny, nx)
    data = read_bin_stream(filename)
    dt = "c%d" % (data.itemsize * 2)
    data_complex = data.view(dtype=dt)
    if len(data_complex) != np.prod(shape):
        print("%s: data size is inconsistent with values in inwap.yaml." % filename)
        return None
    return data_complex.reshape(shape)


def read_dprojectors(inwap, filename="DPROJECTORS.bin"):
    """Read DPROJECTORS.bin."""
    nbtot = inwap["nbtot"]
    nrspinors = inwap["nrspinors"]
    ispin = inwap["ispin"]
    nkpts = inwap["nkpts"]
    lmdim = inwap["lmdim"]
    nions = inwap["nions"]
    # The last index corresponds to usual projector and its derivatives along
    # x, y, z.
    shape = (nkpts, nbtot, ispin, nrspinors, nions, lmdim, 4)
    data = read_bin_stream(filename)
    dt = "c%d" % (data.itemsize * 2)
    data_complex = data.view(dtype=dt)
    if len(data_complex) != np.prod(shape):
        print("%s: data size is inconsistent with values in inwap.yaml." % filename)
        return None
    return data_complex.reshape(shape)


def read_waves(inwap, filename="WAVES.bin"):
    """Read WAVES.bin."""
    ispin = inwap["ispin"]
    nkpts = inwap["nkpts"]
    nbtot = inwap["nbtot"]
    nrspinors = inwap["nrspinors"]
    nx, ny, nz = inwap["fft_coarse"]
    shape = (nkpts, nbtot, ispin, nrspinors, nz, ny, nx)
    data = read_bin_stream(filename)
    dt = "c%d" % (data.itemsize * 2)
    data_complex = data.view(dtype=dt)
    if len(data_complex) != np.prod(shape):
        print("%s: data size is inconsistent with values in inwap.yaml." % filename)
        return None
    return data_complex.reshape(shape)


def read_eigenvalues(inwap, filename="EIGENVALUE.bin"):
    """Read EIGENVALUE.bin."""
    ispin = inwap["ispin"]
    nkpts = inwap["nkpts"]
    nbtot = inwap["nbtot"]
    shape = (nbtot, nkpts, ispin)
    data = read_bin_stream(filename)
    if len(data) != np.prod(shape):
        print("%s: data size is inconsistent with values in inwap.yaml." % filename)
        return None

    # change shape (nbtot, nkpts, ispin) --> (nkpts, nbtot, ispin)
    ret_data = np.array(data.reshape(shape).swapaxes(0, 1), dtype=data.dtype, order="C")
    return ret_data


def read_qtot(inwap, filename="QTOT.bin"):
    """Read QTOT.bin."""
    ldim = inwap["ldim"]
    nions = inwap["nions"]
    shape = (nions, ldim, ldim)
    data = read_bin_stream(filename)
    if len(data) != np.prod(shape):
        print("%s: data size is inconsistent with values in inwap.yaml." % filename)
        return None
    return data.reshape(shape)


def get_CHGCAR(charge, header):
    """Return CHGCAR style text from ndarray."""
    text = header + "\n%5d%5d%5d\n" % charge.shape[::-1]
    count = 0
    for z in range(charge.shape[0]):
        for y in range(charge.shape[1]):
            for x in range(charge.shape[2]):
                text += "%18.11e" % charge[z, y, x]
                count += 1
                if (count % 5) == 0:
                    count = 0
                    text += "\n"
    return text


###########
# Writers #
###########
def write_mesh_electron_hdf5(dirnames, mesh):
    """Read .bin files and write to hdf5 file.

    Note
    ----
    No spin and spinor are considered.

    """
    inwaps = []
    for dname in dirnames:
        inwaps.append(read_inwap_yaml("%s/%s" % (dname, "inwap.yaml")))

    # check all have the same data sizes except for nkpts.
    assert len(np.unique([iwp["nbtot"] for iwp in inwaps])) == 1
    assert len(np.unique([iwp["nions"] for iwp in inwaps])) == 1
    assert len(np.unique([iwp["lmdim"] for iwp in inwaps])) == 1
    fft_meshes = np.array([iwp["fft_coarse"] for iwp in inwaps])
    assert (fft_meshes[1:] - fft_meshes[0] == 0).all()

    nkpts = np.array([iwp["nkpts"] for iwp in inwaps], dtype=int)
    nbtot = inwaps[0]["nbtot"]
    lmdim = inwaps[0]["lmdim"]
    nions = inwaps[0]["nions"]
    fft_mesh = tuple(fft_meshes[0])
    dtype = "c%d" % (np.dtype("double").itemsize * 2)
    # waves = np.empty((np.sum(nkpts), nbtot, 1, 1) + fft_mesh,
    #                  dtype=dtype)
    waves_shape = (np.sum(nkpts), nbtot, 1, 1) + fft_mesh
    dprojs = np.empty((np.sum(nkpts), nbtot, 1, 1, nions, lmdim, 4), dtype=dtype)
    eigenvalues = np.empty((np.sum(nkpts), nbtot, 1), dtype="double")
    idx = 0
    for dname, iwp, nk in zip(dirnames, inwaps, nkpts):
        # waves[idx:(idx + nk)] = read_waves(
        #     iwp, "%s/%s" % (dname, "WAVES.bin"))
        dprojs[idx : (idx + nk)] = read_dprojectors(
            iwp, "%s/%s" % (dname, "DPROJECTORS.bin")
        )
        eigvals = read_eigenvalues(iwp, "%s/%s" % (dname, "EIGENVALUE.bin"))
        eigenvalues[idx : (idx + nk)] = eigvals
        idx += nk

    suffix = get_filename_suffix(mesh)
    with h5py.File("electron%s.hdf5" % suffix, "w") as w:
        # compression doesn't help much.
        w.create_dataset("mesh", data=mesh)
        # w.create_dataset('waves', data=waves, compression=None)
        w.create_dataset("dprojectors", data=dprojs, compression=None)
        w.create_dataset("eigenvalues", data=eigenvalues, compression=None)

    # This is to avoid using memory space by directly writing to the file
    # object piece by piece.
    with h5py.File("electron%s.hdf5" % suffix, "a") as w:
        waves = w.create_dataset("waves", waves_shape, dtype=dtype)
        idx = 0
        for dname, iwp, nk in zip(dirnames, inwaps, nkpts):
            waves[idx : (idx + nk)] = read_waves(iwp, "%s/%s" % (dname, "WAVES.bin"))
            idx += nk


def write_mesh_KPOINTS(filename, num_per_batch=100):
    """Read grid_address-xxx.hdf5 and write KPOINTS style files.

    The set of k-points is divided into kpoint batches and
    each batch has num_per_batch kpoints.

    """
    with h5py.File(filename, "r") as f:
        mesh = np.array(f["mesh"][:], dtype=float)
        grid_address = f["grid_address"][:]

    num = num_per_batch
    w = None
    for i, kpt in enumerate(grid_address / mesh):
        if i % num == 0:
            if i != 0:
                w.close()
            w = open("KPOINTS-%03d" % (i // num + 1), "w")
            if len(grid_address) - i > num:
                w.write("batch %d-%d\n" % (i + 1, i + num))
                w.write("%d\n" % num)
            else:
                w.write("batch %d-%d\n" % (i + 1, len(grid_address)))
                w.write("%d\n" % (len(grid_address) - i))
            w.write("Reciprocal\n")
        w.write("%12.8f%12.8f%12.8f 1\n" % tuple(kpt))
    w.close()
