"""Utilities of main CUI script."""

from typing import Optional, Union

from phono3py.interface.calculator import (
    get_additional_info_to_write_fc2_supercells,
    get_additional_info_to_write_supercells,
    get_default_displacement_distance,
)
from phonopy.interface.calculator import write_supercells_with_displacements

from phelel.api_phelel import Phelel


def create_phelel_supercells(
    cell_info,
    settings,
    symprec,
    interface_mode="vasp",
    log_level=1,
):
    """Create displacements and supercells.

    Distance unit used is that for the calculator interface.
    The default unit is Angstron.

    """
    optional_structure_info = cell_info["optional_structure_info"]

    phelel = Phelel(
        cell_info["unitcell"],
        supercell_matrix=cell_info["supercell_matrix"],
        primitive_matrix=cell_info["primitive_matrix"],
        phonon_supercell_matrix=cell_info["phonon_supercell_matrix"],
        symprec=symprec,
        is_symmetry=settings.is_symmetry,
        calculator=interface_mode,
    )

    if log_level:
        print("")
        print('Unit cell was read from "%s".' % optional_structure_info[0])

    generate_phelel_supercells(
        phelel,
        interface_mode=interface_mode,
        distance=settings.displacement_distance,
        is_plusminus=settings.is_plusminus_displacement,
        is_diagonal=settings.is_diagonal_displacement,
        log_level=log_level,
    )

    additional_info = get_additional_info_to_write_supercells(
        interface_mode, phelel.supercell_matrix
    )
    write_supercells_with_displacements(
        interface_mode,
        phelel.supercell,
        phelel.supercells_with_displacements,
        optional_structure_info=optional_structure_info,
        additional_info=additional_info,
    )

    if phelel.phonon is not None:
        additional_info = get_additional_info_to_write_fc2_supercells(
            interface_mode, phelel.phonon_supercell_matrix, suffix="PH"
        )
        write_supercells_with_displacements(
            phelel.phonon.calculator,
            phelel.phonon.supercell,
            phelel.phonon.supercells_with_displacements,
            optional_structure_info=optional_structure_info,
            additional_info=additional_info,
        )

    return phelel


def generate_phelel_supercells(
    phelel: Phelel,
    interface_mode: str = "vasp",
    distance: Optional[float] = None,
    is_plusminus: Union[str, bool] = "auto",
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
        print("Displacement distance: %s" % _distance)
        print("Number of displacements: %d" % len(phelel.supercells_with_displacements))

    if phelel.phonon is not None:
        phelel.phonon.generate_displacements(
            distance=distance, is_plusminus=is_plusminus, is_diagonal=is_diagonal
        )
        if log_level:
            print(
                "Number of displacements for phonon: %d"
                % len(phelel.phonon.supercells_with_displacements)
            )
