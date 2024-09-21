"""Phelel file_IO functions."""

from __future__ import annotations

import io
import os
import pathlib
from typing import Optional, Union

import h5py
import numpy as np
from phonopy.structure.atoms import PhonopyAtoms, atom_data
from phonopy.structure.cells import Primitive
from phonopy.structure.symmetry import Symmetry

from phelel.base.Dij_qij import DDijQij
from phelel.base.local_potential import DLocalPotential
from phelel.utils.data import cmplx2real
from phelel.utils.lattice_points import get_lattice_points


def write_phelel_params_hdf5(
    dVdu: Optional["DLocalPotential"] = None,
    dDijdu: Optional["DDijQij"] = None,
    # Rij=None,
    supercell_matrix=None,
    primitive_matrix=None,
    primitive: Optional[Primitive] = None,
    unitcell: Optional[PhonopyAtoms] = None,
    supercell: Optional[PhonopyAtoms] = None,
    atom_indices_in_derivatives=None,
    disp_dataset=None,
    force_constants=None,
    phonon_supercell_matrix=None,
    phonon_primitive: Optional[Primitive] = None,
    phonon_supercell: Optional[PhonopyAtoms] = None,
    nac_params=None,
    symmetry_dataset=None,
    filename="phelel_params.hdf5",
):
    """Write phelel_params.hdf5."""
    with h5py.File(filename, "w") as w:
        _add_datasets(
            w,
            dVdu=dVdu,
            dDijdu=dDijdu,
            supercell_matrix=supercell_matrix,
            primitive_matrix=primitive_matrix,
            primitive=primitive,
            unitcell=unitcell,
            supercell=supercell,
            atom_indices_in_derivatives=atom_indices_in_derivatives,
            disp_dataset=disp_dataset,
            force_constants=force_constants,
            phonon_supercell_matrix=phonon_supercell_matrix,
            phonon_primitive=phonon_primitive,
            phonon_supercell=phonon_supercell,
            nac_params=nac_params,
            symmetry_dataset=symmetry_dataset,
        )


def write_dVdu_hdf5(
    dVdu,
    supercell_matrix,
    primitive_matrix,
    primitive,
    unitcell,
    supercell,
    filename="dVdu.hdf5",
):
    """Write dVdu.hdf5."""
    with h5py.File(filename, "w") as w:
        _add_datasets(
            w,
            dVdu=dVdu,
            supercell_matrix=supercell_matrix,
            primitive_matrix=primitive_matrix,
            primitive=primitive,
            unitcell=unitcell,
            supercell=supercell,
        )


def read_phelel_params_hdf5(
    filename: Union[str, bytes, os.PathLike, io.IOBase] = "phelel_params.hdf5",
    log_level: int = 0,
) -> tuple[DLocalPotential, DDijQij, np.ndarray, np.ndarray]:
    """Read dV/du and dDij/du from phelel_params.hdf5.

    Parameteters
    ------------
    filename :
        File name of phelel_params.hdf5.
    log_level : int
        Log level.

    Returns
    -------
    tuple :
        dDVdu_obj : DLocalPotential
        dDijdu_obj : DDijQij
        fft_mesh : np.ndarray
        fc : np.ndarray, optional

    """
    if pathlib.Path(filename).exists():
        with h5py.File(filename, "r") as f:
            fft_mesh, dVdu, grid_points, lattice_points = read_dVdu_hdf5(f)
            dDijdu, dqijdu, Dij, qij = read_dDijdu_hdf5(f)
            fc = read_force_constants_hdf5(f)
            supercell = PhonopyAtoms(
                cell=f["supercell_lattice"][:].T,
                scaled_positions=f["supercell_positions"][:],
                symbols=[atom_data[n][1] for n in f["supercell_numbers"][:]],
                masses=f["supercell_masses"][:],
            )
            symmetry = Symmetry(supercell)
            if "atom_indices_in_derivatives" in f:
                atom_indices = f["atom_indices_in_derivatives"][:]
            else:
                atom_indices = f["p2s_map"][:]
            if "primitive_matrix" in f:
                pmat = f["primitive_matrix"][:]
            else:
                pmat = (
                    np.linalg.inv(f["supercell_lattice"][:]) @ f["primitive_lattice"][:]
                )
            p2s_mat_float = np.linalg.inv(pmat)
            p2s_matrix = np.rint(p2s_mat_float).astype("int_")
            assert (abs(p2s_matrix - p2s_mat_float) < 1e-5).all()

        if log_level:
            print(f'dV/du was read from "{filename}".')
            print(f'dDij/du was read from "{filename}".')
    else:
        raise FileNotFoundError(f'"{filename}" was not found.')

    dDVdu_obj = DLocalPotential(
        fft_mesh,
        p2s_matrix,
        supercell,
        symmetry=symmetry,
        atom_indices=atom_indices,
    )
    dDVdu_obj.dVdu = dVdu
    dDVdu_obj.grid_points = grid_points
    dDVdu_obj.lattice_points = lattice_points

    dDijdu_obj = DDijQij(
        supercell,
        symmetry=symmetry,
        atom_indices=atom_indices,
    )
    dDijdu_obj.dDijdu = dDijdu
    dDijdu_obj.dqijdu = dqijdu
    dDijdu_obj.Dij = Dij
    dDijdu_obj.qij = qij

    return dDVdu_obj, dDijdu_obj, fft_mesh, fc


def write_dDijdu_hdf5(dDijdu, filename="dDijdu.hdf5"):
    """Write dDijdu.hdf5."""
    with h5py.File(filename, "w") as w:
        _add_datasets(w, dDijdu=dDijdu)


def read_force_constants_hdf5(f):
    """Read force_constants from hdf5 file object."""
    return f["force_constants"][:]


def read_dVdu_hdf5(f):
    """Read dVdu from hdf5 file object."""
    fft_mesh = f["FFT_mesh"][:]
    dVdu = f["dVdu"][:]
    grid_points = f["grid_point"][:]
    lattice_points = f["lattice_point"][:]
    return fft_mesh, dVdu, grid_points, lattice_points


def read_dDijdu_hdf5(f):
    """Read dDijdu from hdf5 file object."""
    dDijdu = f["dDijdu"][:]
    dqijdu = f["dqijdu"][:]
    Dij = f["Dij"][:]
    qij = f["qij"][:]
    return dDijdu, dqijdu, Dij, qij


def _add_datasets(
    w,
    dVdu: Optional["DLocalPotential"] = None,
    dDijdu: Optional["DDijQij"] = None,
    Rij=None,
    supercell_matrix=None,
    primitive_matrix=None,
    primitive: Optional[Primitive] = None,
    unitcell: Optional[PhonopyAtoms] = None,
    supercell: Optional[PhonopyAtoms] = None,
    atom_indices_in_derivatives=None,
    disp_dataset=None,
    force_constants=None,
    phonon_supercell_matrix=None,
    phonon_primitive: Optional[Primitive] = None,
    phonon_supercell: Optional[PhonopyAtoms] = None,
    nac_params=None,
    symmetry_dataset=None,
):
    if dVdu is not None:
        w.create_dataset("dVdu", data=cmplx2real(dVdu.dVdu))
        w.create_dataset("grid_point", data=dVdu.grid_points)
        w.create_dataset("lattice_point", data=dVdu.lattice_points)
        w.create_dataset("FFT_mesh", data=dVdu.fft_mesh)
    if dDijdu is not None:
        w.create_dataset("dDijdu", data=cmplx2real(dDijdu.dDijdu))
        w.create_dataset("dqijdu", data=cmplx2real(dDijdu.dqijdu))
        w.create_dataset("Dij", data=cmplx2real(dDijdu.Dij))
        w.create_dataset("qij", data=cmplx2real(dDijdu.qij))
    if Rij is not None:
        w.create_dataset("Rij", data=cmplx2real(Rij))
    if supercell_matrix is not None:
        w.create_dataset(
            "supercell_matrix", data=np.array(supercell_matrix, dtype="int_", order="C")
        )
    if primitive_matrix is not None:
        w.create_dataset(
            "primitive_matrix",
            data=np.array(primitive_matrix, dtype="double", order="C"),
        )
    if primitive is not None:
        w.create_dataset(
            "primitive_lattice",
            data=np.array(primitive.cell.T, dtype="double", order="C"),
        )
        w.create_dataset(
            "primitive_positions",
            data=np.array(primitive.scaled_positions, dtype="double", order="C"),
        )
        w.create_dataset(
            "primitive_numbers", data=np.array(primitive.numbers, dtype="int_")
        )
        w.create_dataset(
            "primitive_masses", data=np.array(primitive.masses, dtype="double")
        )
        p2s_vectors, p2s_multiplicities = primitive.get_smallest_vectors()
        w.create_dataset("p2s_map", data=np.array(primitive.p2s_map, dtype="int_"))
        w.create_dataset("s2p_map", data=np.array(primitive.s2p_map, dtype="int_"))
        w.create_dataset("shortest_vectors", data=np.array(p2s_vectors, dtype="double"))
        w.create_dataset(
            "shortest_vector_multiplicities",
            data=np.array(p2s_multiplicities, dtype="int_"),
        )
        if atom_indices_in_derivatives is not None:
            if not np.array_equal(atom_indices_in_derivatives, primitive.p2s_map):
                w.create_dataset(
                    "atom_indices_in_derivatives",
                    data=np.array(atom_indices_in_derivatives, dtype="int_"),
                )
    if unitcell is not None:
        w.create_dataset(
            "unitcell_lattice",
            data=np.array(unitcell.cell.T, dtype="double", order="C"),
        )
        w.create_dataset(
            "unitcell_positions",
            data=np.array(unitcell.scaled_positions, dtype="double", order="C"),
        )
        w.create_dataset(
            "unitcell_numbers", data=np.array(unitcell.numbers, dtype="int_")
        )
        w.create_dataset(
            "unitcell_masses", data=np.array(unitcell.masses, dtype="double")
        )
    if supercell is not None:
        w.create_dataset(
            "supercell_lattice",
            data=np.array(supercell.cell.T, dtype="double", order="C"),
        )
        w.create_dataset(
            "supercell_positions",
            data=np.array(supercell.scaled_positions, dtype="double", order="C"),
        )
        w.create_dataset(
            "supercell_numbers", data=np.array(supercell.numbers, dtype="int_")
        )
        w.create_dataset(
            "supercell_masses", data=np.array(supercell.masses, dtype="double")
        )
    if disp_dataset is not None:
        if "first_atoms" in disp_dataset:
            atom_indices = [d["number"] for d in disp_dataset["first_atoms"]]
            w.create_dataset(
                "displacements_atom_indices", data=np.array(atom_indices, dtype="int_")
            )
            disps = [d["displacement"] for d in disp_dataset["first_atoms"]]
            w.create_dataset(
                "displacements_vectors", data=np.array(disps, dtype="double", order="C")
            )
    if force_constants is not None:
        w.create_dataset(
            "force_constants", data=np.array(force_constants, dtype="double", order="C")
        )
    if phonon_supercell_matrix is not None:
        w.create_dataset(
            "phonon_supercell_matrix",
            data=np.array(phonon_supercell_matrix, dtype="int_", order="C"),
        )
    if phonon_primitive is not None:
        p2s_vectors, p2s_multiplicities = phonon_primitive.get_smallest_vectors()
        w.create_dataset(
            "phonon_p2s_map", data=np.array(phonon_primitive.p2s_map, dtype="int_")
        )
        w.create_dataset(
            "phonon_s2p_map", data=np.array(phonon_primitive.s2p_map, dtype="int_")
        )
        w.create_dataset(
            "phonon_shortest_vectors", data=np.array(p2s_vectors, dtype="double")
        )
        w.create_dataset(
            "phonon_shortest_vector_multiplicities",
            data=np.array(p2s_multiplicities, dtype="int_"),
        )
        p2s_mat_float = np.linalg.inv(phonon_primitive.primitive_matrix)
        p2s_matrix = np.rint(p2s_mat_float).astype("int_")
        assert (abs(p2s_matrix - p2s_mat_float) < 1e-5).all()
        lattice_points, _ = get_lattice_points(p2s_matrix)
        w.create_dataset("phonon_lattice_point", data=lattice_points)
    if phonon_supercell is not None:
        w.create_dataset(
            "phonon_supercell_lattice",
            data=np.array(phonon_supercell.cell.T, dtype="double", order="C"),
        )
        w.create_dataset(
            "phonon_supercell_positions",
            data=np.array(phonon_supercell.scaled_positions, dtype="double", order="C"),
        )
        w.create_dataset(
            "phonon_supercell_numbers",
            data=np.array(phonon_supercell.numbers, dtype="int_"),
        )
        w.create_dataset(
            "phonon_supercell_masses",
            data=np.array(phonon_supercell.masses, dtype="double"),
        )
    if nac_params is not None:
        w.create_dataset(
            "born_effective_charges",
            data=np.array(nac_params["born"], dtype="double", order="C"),
        )
        w.create_dataset(
            "dielectric_constant",
            data=np.array(nac_params["dielectric"], dtype="double", order="C"),
        )
    if symmetry_dataset is not None:
        w.create_dataset(
            "transformation_matrix",
            data=np.array(
                symmetry_dataset.transformation_matrix, dtype="double", order="C"
            ),
        )
        w.create_dataset(
            "direct_rotations",
            data=np.array(symmetry_dataset.rotations, dtype="int_", order="C"),
        )
        sym_dataset_dict = vars(symmetry_dataset)
        if "number" in sym_dataset_dict:  # For non-magnetic case
            w.create_dataset(
                "spacegroup_number", data=int(symmetry_dataset.number), dtype="int_"
            )
        elif "uni_number" in sym_dataset_dict:  # For magnetic case
            w.create_dataset(
                "magnetic_spacegroup_uni_number",
                data=int(symmetry_dataset.uni_number),
                dtype="int_",
            )
