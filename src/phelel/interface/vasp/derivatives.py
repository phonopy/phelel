"""Wrapper functions to calculate derivatives."""

from __future__ import annotations

import os
import pathlib
from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from phonopy.file_IO import get_born_parameters
from phonopy.interface.vasp import parse_set_of_forces
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import Primitive
from phonopy.structure.symmetry import Symmetry

from phelel import Phelel
from phelel.api_phelel import PhelelDataset
from phelel.interface.vasp.file_IO import (
    read_forces_vaspouth5,
    read_inwap_vaspouth5,
    read_inwap_yaml,
    read_local_potential,
    read_local_potential_vaspouth5,
    read_PAW_Dij_qij,
    read_PAW_Dij_qij_vaspouth5,
)


def read_files(
    phelel: Phelel,
    dir_names: Sequence[str | os.PathLike],
    phonon_dir_names: Sequence[str | os.PathLike] | None = None,
    subtract_rfs: bool = False,
    log_level: int = 0,
) -> PhelelDataset:
    """Load files needed to create derivatives."""
    inwap_path = pathlib.Path(dir_names[0]) / "inwap.yaml"
    if inwap_path.exists():
        inwap_per = read_inwap_yaml(inwap_path)
    else:
        # try reading from vaspout.h5
        inwap_path = pathlib.Path(dir_names[0]) / "vaspout.h5"
        inwap_per = read_inwap_vaspouth5(inwap_path)

    if inwap_per["nions"] != len(phelel.supercell):
        raise ValueError(
            "Number of ions in the supercell is different from the number of atoms "
            "in the inwap.yaml or vaspout.h5 file."
        )

    if log_level:
        print(f'Parameters were collected from "{inwap_path}".')

    dataset, _ = _get_datasets(phelel)
    loc_pots = _read_local_potentials(dir_names, inwap_per, log_level=log_level)
    if loc_pots is None:
        raise ValueError(
            "Failed to read required local potentials from the given directories. "
        )
    kin_pots = _read_local_potentials(
        dir_names, inwap_per, key="xcmu", log_level=log_level
    )
    Dijs, qijs = _read_PAW_strength_and_overlap(
        dir_names, inwap_per, log_level=log_level
    )
    if phonon_dir_names is None:
        _dir_names = dir_names
    else:
        _dir_names = phonon_dir_names

    if phelel.phonon_supercell_matrix:
        supercell = phelel.phonon_supercell
    else:
        supercell = phelel.supercell
    assert supercell is not None

    vaspout5_exists = True
    vaspout_filenames = []
    for dir_name in _dir_names:
        try:
            vaspout_filenames.append(next(pathlib.Path(dir_name).glob("vaspout.h5*")))
        except StopIteration:
            vaspout5_exists = False
            break
    if vaspout5_exists:
        forces = _read_forces_from_vaspout_h5(
            vaspout_filenames, subtract_rfs=subtract_rfs, log_level=log_level
        )
    else:
        vasprun_filenames = _get_vasprun_filenames(_dir_names)
        forces = read_forces_from_vasprunxmls(
            vasprun_filenames,
            supercell,
            subtract_rfs=subtract_rfs,
            log_level=log_level,
        )

    if forces[0].shape[0] != len(supercell):
        raise ValueError(
            "Number of ions in the phonon supercell is different from the number of "
            "atoms in the vasprun.xml file."
        )

    phelel.forces = forces

    if phelel.nac_params is None:
        # This situation is possible when this function is called from cui/load.
        nac_params = _read_born(
            phelel.primitive, phelel.primitive_symmetry, log_level=log_level
        )
        if nac_params:
            phelel.nac_params = nac_params

    return PhelelDataset(
        local_potentials=loc_pots,
        Dijs=Dijs,
        qijs=qijs,
        lm_channels=inwap_per["lm_orbitals"],
        kinetic_potentials=kin_pots,
        dataset=dataset,
        forces=np.array(forces, dtype="double", order="C"),
    )


def create_derivatives(
    phelel: Phelel,
    dir_names: Sequence,
    subtract_rfs: bool = False,
    log_level: int = 0,
):
    """Calculate derivatives.

    Input files are read and derivatives are computed. The results are stored in
    Phelel instance.

    When the number of dir_names is equivalent to the number of displacements
    for the el-ph calculation, the same directories are used for calculating
    force constants. To calculate force constants from another directories,
    those directories have to be appdended after the directory names for
    el-ph.

    % phelel -d --dim 2 2 2 --pa auto
    % phelel --fft-mesh 18 18 18 --cd perfect disp-001

    """
    dataset, phonon_dataset = _get_datasets(phelel)
    num_disp = len(dataset["first_atoms"]) + 1
    num_disp_ph = len(phonon_dataset["first_atoms"]) + 1
    if len(dir_names) == num_disp:
        phonon_dir_names = dir_names
    elif len(dir_names) == num_disp + num_disp_ph:
        phonon_dir_names = dir_names[num_disp:]
    else:
        raise RuntimeError("Number of dir_names is wrong.")

    phe_input = read_files(
        phelel,
        dir_names,
        phonon_dir_names=phonon_dir_names,
        subtract_rfs=subtract_rfs,
        log_level=log_level,
    )
    if phelel.fft_mesh is not None:
        phelel.run_derivatives(phe_input)

    # phelel.Rij = read_Rij(dir_names[0], inwap_per)


def read_Rij(dir_name: str | os.PathLike, inwap_per: dict) -> NDArray:
    """Read Rij."""
    return read_PAW_Dij_qij(inwap_per, "%s/PAW-Rnij.bin" % dir_name, is_Rij=True)


def read_forces_from_vasprunxmls(
    vasprun_filenames: list | tuple,
    supercell: PhonopyAtoms,
    subtract_rfs: bool = False,
    log_level: int = 0,
) -> list[NDArray]:
    """Read forces from vasprun.xml's."""
    if log_level:
        for filename in vasprun_filenames:
            print(f'Forces were read from "{filename}".')

    calc_dataset = parse_set_of_forces(len(supercell), vasprun_filenames, verbose=False)
    forces = calc_dataset["forces"]

    return _subtract_residual_forces(forces, subtract_rfs, log_level)


def _subtract_residual_forces(
    forces: list[NDArray], subtract_rfs: bool, log_level: int
) -> list[NDArray]:
    if subtract_rfs:
        _forces = [fset - forces[0] for fset in forces[1:]]
        if log_level:
            print(
                "Residual forces of perfect supercell were subtracted from "
                "supercell forces."
            )
    else:
        _forces = forces[1:]

    return _forces


def _get_vasprun_filenames(dir_names):
    vasprun_filenames = []
    for dir_name in dir_names:
        filename = next(pathlib.Path(dir_name).glob("vasprun.xml*"))
        vasprun_filenames.append(filename)
    return vasprun_filenames


def _read_forces_from_vaspout_h5(
    vaspout_filenames: list | tuple,
    subtract_rfs: bool = False,
    log_level: int = 0,
) -> list[NDArray]:
    """Read forces from vaspout.h5's."""
    forces = []
    for filename in vaspout_filenames:
        if log_level:
            print(f'Forces were read from "{filename}".')
        forces.append(read_forces_vaspouth5(filename))

    return _subtract_residual_forces(forces, subtract_rfs, log_level)


def _read_born(
    primitive: Primitive, primitive_symmetry: Symmetry, log_level: int = 0
) -> dict | None:
    if pathlib.Path("BORN").is_file():
        with open("BORN", "r") as f:
            nac_params = get_born_parameters(f, primitive, primitive_symmetry)
            if log_level:
                print('"BORN" was read.')
        return nac_params
    else:
        return None


def _get_datasets(phelel: Phelel) -> tuple:
    """Return inwap dataset and phonopy dataset."""
    assert phelel.dataset is not None
    if "first_atoms" in phelel.dataset:
        dataset = phelel.dataset
        if (
            phelel.phonon_supercell_matrix
            and phelel.phonon_dataset is not None
            and "first_atoms" in phelel.phonon_dataset
        ):
            phonon_dataset = phelel.phonon_dataset
        else:
            phonon_dataset = dataset
    else:
        raise RuntimeError("Displacement dataset has to be stored in Phelel instance.")

    return dataset, phonon_dataset


def _read_local_potentials(
    dir_names: Sequence[str | os.PathLike],
    inwap_per: dict,
    key: Literal["total", "xcmu"] = "total",
    log_level: int = 0,
) -> list[NDArray] | None:
    loc_pots = []
    vaspout5_exists = True
    for dir_name in dir_names:
        try:
            locpot_path = next(pathlib.Path(dir_name).glob("vaspout.h5*"))
        except StopIteration:
            vaspout5_exists = False
            break

        try:
            loc_pots.append(
                read_local_potential_vaspouth5(filename=locpot_path, key=key)
            )
        except KeyError:
            return None

        if log_level:
            print(f'Local potential was read from "{locpot_path}".')

    if vaspout5_exists:
        return loc_pots

    # Old way for LOCAL-POTENTIAL.bin.
    # Meta-GGA kinetic potential is not available.
    if key == "xcmu":
        return None

    for dir_name in dir_names:
        try:
            locpot_path = next(pathlib.Path(dir_name).glob("LOCAL-POTENTIAL.bin*"))
            loc_pots.append(read_local_potential(inwap_per, filename=locpot_path))
        except StopIteration as e:
            raise RuntimeError(
                f'"LOCAL-POTENTIAL.bin" not found in "{dir_name}".'
            ) from e
    return loc_pots


def _read_PAW_strength_and_overlap(
    dir_names, inwap_per, log_level=0
) -> tuple[list[NDArray], list[NDArray]]:
    Dijs = []
    qijs = []
    vaspout5_exists = True
    for dir_name in dir_names:
        try:
            Dij_qij_path = next(pathlib.Path(dir_name).glob("vaspout.h5*"))
        except StopIteration:
            vaspout5_exists = False
            break

        dij, qij = read_PAW_Dij_qij_vaspouth5(Dij_qij_path)
        Dijs.append(dij)
        qijs.append(qij)
        if log_level:
            print(f'Dijs and qjis were read from "{Dij_qij_path}".')

    if vaspout5_exists:
        return Dijs, qijs

    # Old way for PAW-*.bin.
    for dir_name in dir_names:
        possible_Dij_path = list(pathlib.Path(dir_name).glob("PAW-STRENGTH.bin*"))
        possible_qij_path = list(pathlib.Path(dir_name).glob("PAW-OVERLAP.bin*"))
        if possible_Dij_path and possible_qij_path:
            Dij_path = possible_Dij_path[0]
            qij_path = possible_qij_path[0]
            Dijs.append(read_PAW_Dij_qij(inwap_per, Dij_path))
            qijs.append(read_PAW_Dij_qij(inwap_per, qij_path))
            if log_level:
                print(f'"{Dij_path}" and "{qij_path}" were read.')

    return Dijs, qijs
