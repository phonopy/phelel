"""Utilities of main CUI script."""

from __future__ import annotations

import dataclasses
import os
import pathlib
from typing import Literal, cast

from numpy.typing import ArrayLike
from phono3py.interface.calculator import (
    get_additional_info_to_write_fc2_supercells,
    get_additional_info_to_write_supercells,
    get_default_displacement_distance,
)
from phonopy import Phonopy
from phonopy.cui.collect_cell_info import CellInfoResult
from phonopy.cui.collect_cell_info import get_cell_info as phonopy_get_cell_info
from phonopy.cui.phonopy_script import store_nac_params
from phonopy.interface.calculator import write_supercells_with_displacements

from phelel.api_phelel import Phelel
from phelel.cui.settings import PhelelSettings
from phelel.interface.phelel_yaml import PhelelYaml


@dataclasses.dataclass
class PhelelCellInfoResult(CellInfoResult):
    """Phelel cell info result.

    This is a subclass of CellInfoResult.

    """

    phelel_yaml: PhelelYaml | None = None
    phonon_supercell_matrix: ArrayLike | None = None


def get_cell_info(
    settings: PhelelSettings,
    cell_filename: str | os.PathLike | None,
    log_level: int,
    load_phonopy_yaml: bool = True,
) -> PhelelCellInfoResult:
    """Return calculator interface and crystal structure information."""
    cell_info = phonopy_get_cell_info(
        settings,
        cell_filename,
        log_level=log_level,
        load_phonopy_yaml=load_phonopy_yaml,
        phonopy_yaml_cls=PhelelYaml,
    )

    cell_info_dict = dataclasses.asdict(cell_info)
    cell_info_dict["phelel_yaml"] = cell_info_dict.pop("phonopy_yaml")
    cell_info = PhelelCellInfoResult(
        **cell_info_dict,
        phonon_supercell_matrix=settings.phonon_supercell_matrix,
    )

    phelel_yaml = cell_info.phelel_yaml
    if cell_info.phonon_supercell_matrix is None and phelel_yaml:
        cell_info.phonon_supercell_matrix = phelel_yaml.phonon_supercell_matrix

    return cell_info


def create_phelel_supercells(
    cell_info: PhelelCellInfoResult,
    settings: PhelelSettings,
    symprec: float,
    interface_mode: str | None = "vasp",
    log_level: int = 1,
):
    """Create displacements and supercells.

    Distance unit used is that for the calculator interface.
    The default unit is Angstron.

    """
    optional_structure_info = cell_info.optional_structure_info
    unitcell_filename = optional_structure_info[0]
    phe_yml = cell_info.phelel_yaml

    phelel = Phelel(
        cell_info.unitcell,
        supercell_matrix=cell_info.supercell_matrix,
        primitive_matrix=cell_info.primitive_matrix,
        phonon_supercell_matrix=cell_info.phonon_supercell_matrix,
        symprec=symprec,
        is_symmetry=settings.is_symmetry,
        calculator=interface_mode,
    )

    if log_level:
        print("")
        print(f'Unit cell was read from "{unitcell_filename}".')

    generate_phelel_supercells(
        phelel,
        interface_mode=interface_mode,
        distance=settings.displacement_distance,
        is_plusminus=settings.is_plusminus_displacement,
        is_diagonal=settings.is_diagonal_displacement,
        log_level=log_level,
    )

    if pathlib.Path("BORN").exists() or (phe_yml and phe_yml.nac_params):
        store_nac_params(
            cast(Phonopy, phelel),
            settings,
            phe_yml,
            unitcell_filename,
            log_level,
        )

    additional_info = get_additional_info_to_write_supercells(
        interface_mode, phelel.supercell_matrix
    )
    assert phelel.supercells_with_displacements is not None
    write_supercells_with_displacements(
        interface_mode,
        phelel.supercell,
        phelel.supercells_with_displacements,
        optional_structure_info=optional_structure_info,
        additional_info=additional_info,
    )

    if phelel.phonon_supercell_matrix is not None:
        additional_info = get_additional_info_to_write_fc2_supercells(
            interface_mode, phelel.phonon_supercell_matrix, suffix="PH"
        )
        assert phelel.phonon_supercell is not None
        assert phelel.phonon_supercells_with_displacements is not None
        write_supercells_with_displacements(
            phelel.calculator,
            phelel.phonon_supercell,
            phelel.phonon_supercells_with_displacements,
            optional_structure_info=optional_structure_info,
            additional_info=additional_info,
        )

    return phelel


def generate_phelel_supercells(
    phelel: Phelel,
    interface_mode: str | None = "vasp",
    distance: float | None = None,
    is_plusminus: Literal["auto"] | bool = "auto",
    is_diagonal: bool = True,
    log_level: int = 0,
):
    """Generate phelel supercells."""
    if distance is None:
        _distance = get_default_displacement_distance(interface_mode)
    else:
        _distance = distance

    phelel.generate_displacements(
        distance=_distance, is_plusminus=is_plusminus, is_diagonal=is_diagonal
    )

    if log_level:
        assert phelel.supercells_with_displacements is not None
        print("Displacement distance: %s" % _distance)
        print("Number of displacements: %d" % len(phelel.supercells_with_displacements))

    if phelel.phonon_supercell_matrix is not None:
        phelel.generate_phonon_displacements(
            distance=distance, is_plusminus=is_plusminus, is_diagonal=is_diagonal
        )
        if log_level:
            assert phelel.phonon_supercells_with_displacements is not None
            print(
                "Number of displacements for phonon: %d"
                % len(phelel.phonon_supercells_with_displacements)
            )
