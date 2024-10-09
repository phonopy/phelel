"""Wrapper functions to calculate derivatives."""

from __future__ import annotations

import os
import pathlib
from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
from phonopy import Phonopy
from phonopy.file_IO import get_born_parameters
from phonopy.interface.vasp import parse_set_of_forces

from phelel import Phelel
from phelel.api_phelel import PhelelDataset
from phelel.interface.vasp.file_IO import (
    read_inwap_vaspouth5,
    read_inwap_yaml,
    read_local_potential,
    read_local_potential_vaspouth5,
    read_PAW_Dij_qij,
    read_PAW_Dij_qij_vaspouth5,
)


def read_files(
    phelel: Phelel,
    dir_names: Sequence[str],
    phonon_dir_names: Optional[Sequence[str]] = None,
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
        print(f'"{inwap_path}" was read.')
    dataset, _ = _get_datasets(phelel)
    loc_pots = _read_local_potentials(dir_names, inwap_per, log_level=log_level)
    Dijs, qijs = _read_PAW_strength_and_overlap(
        dir_names, inwap_per, log_level=log_level
    )
    if phonon_dir_names is None:
        _dir_names = dir_names
    else:
        _dir_names = phonon_dir_names
    vasprun_filenames = _get_vasprun_filenames(_dir_names, log_level=log_level)
    forces = read_forces_from_vasprunxmls(
        vasprun_filenames,
        phelel.phonon,
        subtract_rfs=subtract_rfs,
        log_level=log_level,
    )

    if forces[0].shape[0] != len(phelel.phonon_supercell):
        raise ValueError(
            "Number of ions in the phonon supercell is different from the number of "
            "atoms in the vasprun.xml file."
        )

    nac_params = _read_born(phelel.phonon)
    phelel.phonon.forces = forces
    if nac_params:
        phelel.phonon.nac_params = nac_params
    return PhelelDataset(
        local_potentials=loc_pots,
        Dijs=Dijs,
        qijs=qijs,
        lm_channels=inwap_per["lm_orbitals"],
        dataset=dataset,
        forces=np.array(forces, dtype="double", order="C"),
    )


def create_derivatives(
    phelel: Phelel,
    dir_names: Sequence,
    subtract_rfs=False,
    log_level=0,
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
    if log_level > 0:
        print("Calculation of dV/du, dDij/du, and force constants")
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


def read_Rij(dir_name, inwap_per):
    """Read Rij."""
    return read_PAW_Dij_qij(inwap_per, "%s/PAW-Rnij.bin" % dir_name, is_Rij=True)


def read_forces_from_vasprunxmls(
    vasprun_filenames: Union[list, tuple],
    phonon: Phonopy,
    subtract_rfs=False,
    log_level=0,
):
    """Read forces from vasprun.xml's and read NAC params from BORN."""
    calc_dataset = parse_set_of_forces(
        len(phonon.supercell), vasprun_filenames, verbose=False
    )
    forces = calc_dataset["forces"]

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


def _get_vasprun_filenames(dir_names, log_level=0):
    vasprun_filenames = []
    for dir_name in dir_names:
        filename = next(pathlib.Path(dir_name).glob("vasprun.xml*"))
        vasprun_filenames.append(filename)
        if log_level:
            print('"%s" was read.' % filename)
    return vasprun_filenames


def _read_born(phonon: Phonopy, log_level=0):
    if pathlib.Path("BORN").exists():
        with open("BORN", "r") as f:
            nac_params = get_born_parameters(
                f, phonon.primitive, phonon.primitive_symmetry
            )
            if log_level:
                print('"BORN" was read.')
        return nac_params
    else:
        return None


def _get_datasets(phelel: Phelel) -> tuple:
    """Return inwap dataset and phonopy dataset."""
    if "first_atoms" in phelel.dataset:
        dataset = phelel.dataset
        if "first_atoms" in phelel.phonon_dataset:
            phonon_dataset = phelel.phonon_dataset
        else:
            phonon_dataset = dataset
    else:
        raise RuntimeError("Displacement dataset has to be stored in Phelel instance.")

    return dataset, phonon_dataset


def _read_local_potentials(
    dir_names: Union[str, bytes, os.PathLike], inwap_per: dict, log_level: int = 0
) -> list[np.ndarray]:
    loc_pots = []
    for dir_name in dir_names:
        # Note glob returns a generator.
        possible_locpot_paths = list(
            pathlib.Path(dir_name).glob("LOCAL-POTENTIAL.bin*")
        )
        if possible_locpot_paths:
            locpot_path = possible_locpot_paths[0]
            loc_pots.append(read_local_potential(inwap_per, filename=locpot_path))
        else:
            locpot_path = pathlib.Path(dir_name) / "vaspout.h5"
            loc_pots.append(
                read_local_potential_vaspouth5(inwap_per, filename=locpot_path)
            )
        if log_level:
            print(f'"{locpot_path}" was read.')
    return loc_pots


def _read_PAW_strength_and_overlap(
    dir_names, inwap_per, log_level=0
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    Dijs = []
    qijs = []
    for dir_name in dir_names:
        # Note glob returns a generator.
        possible_Dij_path = list(pathlib.Path(dir_name).glob("PAW-STRENGTH.bin*"))
        possible_qij_path = list(pathlib.Path(dir_name).glob("PAW-OVERLAP.bin*"))
        if possible_Dij_path and possible_qij_path:
            Dij_path = possible_Dij_path[0]
            qij_path = possible_qij_path[0]
            Dijs.append(read_PAW_Dij_qij(inwap_per, Dij_path))
            qijs.append(read_PAW_Dij_qij(inwap_per, qij_path))
            if log_level:
                print(f'"{Dij_path}" and "{qij_path}" were read.')
        else:
            Dij_qij_path = pathlib.Path(dir_name) / "vaspout.h5"
            dij, qij = read_PAW_Dij_qij_vaspouth5(Dij_qij_path)
            Dijs.append(dij)
            qijs.append(qij)
            if log_level:
                print(f'"{Dij_qij_path}" was read.')
    return Dijs, qijs
