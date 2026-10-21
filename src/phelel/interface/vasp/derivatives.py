"""Wrapper functions to calculate derivatives."""

from __future__ import annotations

import os
import pathlib
from collections.abc import Sequence
from typing import Literal, cast

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
    is_nac: bool = True,
    log_level: int = 0,
) -> PhelelDataset:
    """Load files needed to create derivatives.

    When ``is_nac`` is True and the NAC parameters are not yet stored in
    ``phelel``, they are read from BORN in the current directory if it exists.

    """
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

    dataset, phonon_dataset = _get_datasets(phelel)
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

    if phelel.phonon_supercell_matrix is not None:
        supercell = phelel.phonon_supercell
    else:
        supercell = phelel.supercell
    assert supercell is not None

    force_filenames = get_vasp_force_filenames(_dir_names)
    forces = read_vasp_forces(
        force_filenames[1:],
        supercell,
        perfect_filename=force_filenames[0] if subtract_rfs else None,
        log_level=log_level,
    )
    phelel.forces = forces

    if is_nac and phelel.nac_params is None:
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
        phonon_dataset=phonon_dataset,
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
    num_disp_ph = _get_number_of_displacements(phonon_dataset) + 1
    if len(dir_names) == num_disp:
        phonon_dir_names = dir_names
    elif len(dir_names) == num_disp + num_disp_ph:
        phonon_dir_names = dir_names[num_disp:]
    else:
        raise RuntimeError("Number of dir_names is wrong.")

    phe_input = read_files(
        phelel,
        dir_names[:num_disp],
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


def read_vasp_forces(
    filenames: Sequence[str | os.PathLike],
    supercell: PhonopyAtoms,
    perfect_filename: str | os.PathLike | None = None,
    log_level: int = 0,
) -> list[NDArray]:
    """Read forces of the supercells with displacements from VASP output files.

    filenames are those of the supercells with displacements.  With
    perfect_filename, the residual forces of the perfect supercell read from it
    are subtracted from the forces.  A file whose name ends with ".h5" is read
    as vaspout.h5, which can not be compressed.  vasprun.xml can be compressed.

    """
    all_filenames = list(filenames)
    if perfect_filename is not None:
        all_filenames.insert(0, perfect_filename)
    if log_level:
        for filename in all_filenames:
            print(f'Forces were read from "{filename}".')

    calc_dataset = parse_set_of_forces(len(supercell), all_filenames, verbose=False)
    if not calc_dataset:
        raise ValueError(
            f"Forces of the {len(supercell)} atoms of the supercell could not be "
            "read from the VASP output files."
        )
    forces = calc_dataset["forces"]
    if perfect_filename is None:
        return forces
    if log_level:
        print(
            "Residual forces of perfect supercell were subtracted from supercell "
            "forces."
        )
    return [fset - forces[0] for fset in forces[1:]]


def get_vasp_force_filenames(
    dir_names: Sequence[str | os.PathLike],
) -> list[pathlib.Path]:
    """Return the VASP output file to read forces from, of each directory.

    vaspout.h5 is chosen when it is found, otherwise vasprun.xml, which can be
    compressed (vasprun.xml.xz etc.).  FileNotFoundError is raised for a
    directory with neither.

    """
    filenames = []
    for dir_name in dir_names:
        vaspout_filename = pathlib.Path(dir_name) / "vaspout.h5"
        if vaspout_filename.is_file():
            filenames.append(vaspout_filename)
            continue
        vasprun_filenames = sorted(pathlib.Path(dir_name).glob("vasprun.xml*"))
        if not vasprun_filenames:
            raise FileNotFoundError(
                f'Neither "vaspout.h5" nor "vasprun.xml" is found in "{dir_name}".'
            )
        filenames.append(vasprun_filenames[0])
    return filenames


def _read_born(
    primitive: Primitive, primitive_symmetry: Symmetry, log_level: int = 0
) -> dict | None:
    if pathlib.Path("BORN").is_file():
        with open("BORN") as f:
            nac_params = get_born_parameters(f, primitive, primitive_symmetry)
            if log_level:
                print('"BORN" was read.')
        return nac_params
    else:
        return None


def _get_datasets(phelel: Phelel) -> tuple[dict, dict]:
    """Return inwap dataset and phonopy dataset.

    Without the phonon supercell, the phonopy dataset is the inwap dataset.

    """
    assert phelel.dataset is not None
    if "first_atoms" not in phelel.dataset:
        raise RuntimeError("Displacement dataset has to be stored in Phelel instance.")
    if phelel.phonon_supercell_matrix is None:
        return phelel.dataset, phelel.dataset
    if phelel.phonon_dataset is None:
        raise RuntimeError(
            "The phonon dataset has to be stored in Phelel instance when the phonon "
            "supercell matrix is given."
        )
    return phelel.dataset, phelel.phonon_dataset


def _get_number_of_displacements(dataset: dict) -> int:
    """Return the number of supercells with displacements of a dataset.

    The dataset has "first_atoms" when one atom is displaced in each supercell,
    or "displacements" when all atoms are displaced.

    """
    if "first_atoms" in dataset:
        return len(dataset["first_atoms"])
    return len(dataset["displacements"])


def _find_vaspout_h5(
    dir_names: Sequence[str | os.PathLike],
) -> list[pathlib.Path] | None:
    """Return the vaspout.h5 file of every directory.

    None is returned when no directory has vaspout.h5, in which case the
    files of the old format are read.  When only some of the directories have
    vaspout.h5, the two formats are not mixed, because the data of a
    directory would then be paired with the displacement of another one.

    """
    paths = [next(pathlib.Path(d).glob("vaspout.h5*"), None) for d in dir_names]
    if all(path is not None for path in paths):
        return cast(list[pathlib.Path], paths)
    if all(path is None for path in paths):
        return None
    missing = ", ".join(
        f'"{d}"' for d, path in zip(dir_names, paths, strict=True) if path is None
    )
    raise RuntimeError(
        "vaspout.h5 was found in some of the directories but not in "
        f"{missing}.  All directories have to contain the results in the same "
        "format: vaspout.h5, or LOCAL-POTENTIAL.bin, PAW-STRENGTH.bin and "
        "PAW-OVERLAP.bin."
    )


def _read_local_potentials(
    dir_names: Sequence[str | os.PathLike],
    inwap_per: dict,
    key: Literal["total", "xcmu"] = "total",
    log_level: int = 0,
) -> list[NDArray] | None:
    loc_pots = []
    vaspout_paths = _find_vaspout_h5(dir_names)
    if vaspout_paths is not None:
        for locpot_path in vaspout_paths:
            try:
                loc_pots.append(
                    read_local_potential_vaspouth5(filename=locpot_path, key=key)
                )
            except KeyError:
                return None
            if log_level:
                print(f'Local potential was read from "{locpot_path}".')
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
    vaspout_paths = _find_vaspout_h5(dir_names)
    if vaspout_paths is not None:
        for Dij_qij_path in vaspout_paths:
            dij, qij = read_PAW_Dij_qij_vaspouth5(Dij_qij_path)
            Dijs.append(dij)
            qijs.append(qij)
            if log_level:
                print(f'Dijs and qjis were read from "{Dij_qij_path}".')
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
        else:
            raise RuntimeError(
                f'"PAW-STRENGTH.bin" or "PAW-OVERLAP.bin" not found in "{dir_name}".'
            )

    return Dijs, qijs
