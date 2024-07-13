"""Implementation of velph-supercell-init."""

import pathlib
import xml.parsers.expat
from typing import Optional

import click
import numpy as np
from phelel import Phelel
from phelel.cui.create_supercells import generate_phelel_supercells
from phonopy.interface.vasp import VasprunxmlExpat
from phonopy.structure.atoms import PhonopyAtoms, parse_cell_dict
from phonopy.structure.symmetry import symmetrize_borns_and_epsilon
from phonopy.units import Bohr, Hartree


def run_init(
    toml_dict: dict,
    current_directory: pathlib.Path = pathlib.Path(""),
) -> Optional[Phelel]:
    """Generate displacements and write phelel_disp.yaml.

    current_directory : Path
        Used for test.

    """
    convcell = parse_cell_dict(toml_dict["unitcell"])
    supercell_matrix = toml_dict["phelel"].get("supercell_dimension", None)
    phonon_supercell_matrix = toml_dict["phelel"].get(
        "phonon_supercell_dimension", None
    )
    if "primitive_cell" in toml_dict:
        primitive = parse_cell_dict(toml_dict["primitive_cell"])
        primitive_matrix = np.dot(np.linalg.inv(convcell.cell.T), primitive.cell.T)
    else:
        primitive = convcell
        primitive_matrix = None

    is_symmetry = True
    try:
        if toml_dict["phelel"]["nosym"] is True:
            is_symmetry = False
    except KeyError:
        pass

    phe = Phelel(
        convcell,
        supercell_matrix=supercell_matrix,
        phonon_supercell_matrix=phonon_supercell_matrix,
        primitive_matrix=primitive_matrix,
        is_symmetry=is_symmetry,
        calculator="vasp",
    )

    is_diagonal = toml_dict["phelel"].get("diagonal", True)
    is_plusminus = toml_dict["phelel"].get("plusminus", "auto")
    amplitude = toml_dict["phelel"].get("amplitude", None)

    generate_phelel_supercells(
        phe,
        interface_mode="vasp",
        distance=amplitude,
        is_plusminus=is_plusminus,
        is_diagonal=is_diagonal,
    )

    nac_directory = current_directory / "nac"
    if nac_directory.exists():
        click.echo('Found "nac" directory. Read NAC params.')
        vasprun_path = nac_directory / "vasprun.xml"
        if vasprun_path.exists():
            nac_params = _get_nac_params(
                toml_dict,
                vasprun_path,
                primitive,
                convcell,
                is_symmetry,
            )
            if nac_params is not None:
                phe.nac_params = nac_params
        else:
            click.echo('Not found "nac/vasprun.xml". NAC params were not included.')
            return None

    return phe


def _get_nac_params(
    toml_dict: dict,
    vasprun_path: pathlib.Path,
    primitive: Optional[PhonopyAtoms],
    convcell: PhonopyAtoms,
    is_symmetry: bool,
    symprec: float = 1e-5,
) -> Optional[dict]:
    with open(vasprun_path, "rb") as f:
        try:
            vasprun = VasprunxmlExpat(f)
            vasprun.parse()
        except xml.parsers.expat.ExpatError:
            click.echo(f'Parsing "{vasprun_path}" failed.')
            return None

    nac_cell = convcell
    try:
        if "primitive" in toml_dict["vasp"]["nac"]["cell"]:
            nac_cell = primitive
    except KeyError:
        pass

    borns_, epsilon_ = symmetrize_borns_and_epsilon(
        vasprun.born,
        vasprun.epsilon,
        nac_cell,
        primitive=primitive,
        symprec=symprec,
        is_symmetry=is_symmetry,
    )

    nac_params = {
        "born": borns_,
        "factor": Hartree * Bohr,
        "dielectric": epsilon_,
    }
    return nac_params
