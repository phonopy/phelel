"""phelel command line interface."""

from __future__ import annotations

import pathlib
import sys
from typing import Optional, Union

import numpy as np
from phonopy.cui.phonopy_script import (
    print_end,
    print_error,
    print_error_message,
    print_version,
    store_nac_params,
)
from phonopy.exception import CellNotFoundError
from phonopy.interface.calculator import get_calculator_physical_units
from phonopy.structure.cells import print_cell

from phelel import Phelel
from phelel.cui.create_supercells import create_phelel_supercells, get_cell_info
from phelel.cui.phelel_argparse import get_parser
from phelel.cui.settings import PhelelConfParser
from phelel.interface.phelel_yaml import PhelelYaml
from phelel.version import __version__


# AA is created at http://www.network-science.de/ascii/.
def print_phelel():
    """Show phelel logo."""
    print(
        r"""       _          _      _
 _ __ | |__   ___| | ___| |
| '_ \| '_ \ / _ \ |/ _ \ |
| |_) | | | |  __/ |  __/ |
| .__/|_| |_|\___|_|\___|_|
|_|"""
    )
    print_version(__version__, package_name="phelel", rjust_length=25)


def finalize_phelel(
    phelel: Phelel,
    confs: Optional[dict] = None,
    log_level: Union[int, bool] = 0,
    displacements_mode: bool = False,
    filename: Union[str, pathlib.Path] = "phelel.yaml",
    sys_exit_after_finalize: bool = True,
) -> None:
    """Write phelel.yaml and then exit.

    Parameters
    ----------
    phelel : Phelel
        Phelel instance.
    confs : dict
        This contains the settings and command options that the user set.
    log_level : int
        Log level. 0 means quiet.
    displacements_mode : Bool
        When True, crystal structure is written in the length unit of
        calculator interface in phelel_disp.yaml. Otherwise, the
        default unit (angstrom) is used.
    filename : str, optional
        phelel.yaml is written in this filename.

    """
    if displacements_mode:
        _calculator = phelel.calculator
    else:
        _calculator = None
    _physical_units = get_calculator_physical_units(_calculator)

    yaml_settings = {"force_sets": False, "displacements": displacements_mode}

    phe_yml = PhelelYaml(
        configuration=confs,
        calculator=_calculator,
        physical_units=_physical_units,
        settings=yaml_settings,
    )
    phe_yml.set_phelel_info(phelel)
    with open(filename, "w") as w:
        w.write(str(phe_yml))

    if log_level > 0:
        print("")
        if displacements_mode:
            print(f'Displacement dataset was written in "{filename}".')
        else:
            print(f'Summary of calculation was written in "{filename}".')
        print_end()

    if sys_exit_after_finalize:
        sys.exit(0)


def main(**argparse_control):
    """Phelel CUI main function."""
    #################
    # Option parser #
    #################
    load_phelel_yaml = argparse_control.get("load_phelel_yaml", False)

    if "args" in argparse_control:  # For pytest
        args = argparse_control["args"]
        log_level = args.log_level
    else:
        parser = get_parser()
        args = parser.parse_args()
        # Set log level
        log_level = 1
        if args.verbose:
            log_level = 2
        if args.quiet:
            log_level = 0
        if args.log_level is not None:
            log_level = args.log_level

    if log_level > 0:
        print_phelel()

    if len(args.filename) > 0:
        phelel_conf = PhelelConfParser(filename=args.filename[0], args=args)
        settings = phelel_conf.settings
    else:
        phelel_conf = PhelelConfParser(args=args)
        settings = phelel_conf.settings

    if settings.symmetry_tolerance is None:
        symprec = 1e-5
    else:
        symprec = settings.symmetry_tolerance

    #####################
    # Initialize phelel #
    #####################
    try:
        cell_info = get_cell_info(
            settings=settings,
            cell_filename=settings.cell_filename,
            log_level=log_level,
            load_phonopy_yaml=load_phelel_yaml,
        )
    except CellNotFoundError as e:
        print_error_message(str(e))
        if log_level > 0:
            print_error()
        sys.exit(1)

    unitcell = cell_info.unitcell
    supercell_matrix = cell_info.supercell_matrix
    primitive_matrix = cell_info.primitive_matrix
    unitcell_filename = cell_info.optional_structure_info[0]
    cell_info.phonon_supercell_matrix = settings.phonon_supercell_matrix
    phe_yml = cell_info.phelel_yaml
    if cell_info.phonon_supercell_matrix is None and phe_yml:
        ph_smat = phe_yml.phonon_supercell_matrix
        cell_info.phonon_supercell_matrix = ph_smat
    phonon_supercell_matrix = cell_info.phonon_supercell_matrix

    if settings.create_displacements:
        phelel = create_phelel_supercells(
            cell_info,
            settings,
            symprec,
            log_level=log_level,
        )
        finalize_phelel(
            phelel,
            confs=phelel_conf.confs,
            log_level=log_level,
            displacements_mode=True,
            filename="phelel_disp.yaml",
        )

    fft_mesh = settings.fft_mesh_numbers
    phelel = Phelel(
        unitcell,
        supercell_matrix,
        primitive_matrix=primitive_matrix,
        phonon_supercell_matrix=phonon_supercell_matrix,
        fft_mesh=fft_mesh,
        symprec=symprec,
        is_symmetry=settings.is_symmetry,
        finufft_eps=settings.finufft_eps,
    )

    if log_level > 0:
        print("")
        print(f'Crystal structure was read from "{unitcell_filename}".')
        print("Settings:")
        if (np.diag(np.diag(supercell_matrix)) - supercell_matrix).any():
            print("  Supercell matrix:")
            for v in supercell_matrix:
                print("    %s" % v)
        else:
            print("  Supercell: %s" % np.diag(supercell_matrix))
        if isinstance(primitive_matrix, str) and primitive_matrix == "auto":
            print("  Primitive matrix (Auto):")
        elif primitive_matrix is not None:
            print("  Primitive matrix:")
        if primitive_matrix is not None:
            for v in phelel.primitive_matrix:
                print("    %s" % v)
        if phonon_supercell_matrix is not None:
            print("  Supercell matrix for phonon:")
            for v in phelel.phonon_supercell_matrix:
                print("    %s" % v)
        if fft_mesh is not None:
            print("  FFT mesh: [%d %d %d]" % tuple(fft_mesh))
        print("Space group type: %s" % phelel.symmetry.get_international_table())
        print("-" * 30 + " primitive cell " + "-" * 30)
        print_cell(phelel.primitive)
        print("-" * 32 + " super cell " + "-" * 32)
        print_cell(phelel.supercell)
        if phelel.phonon_supercell_matrix is not None:
            print("-" * 28 + " phonon super cell " + "-" * 29)
            print_cell(phelel.phonon_supercell)
        print("-" * 76)

    ##################################
    # Create dV/du, dDij/du, dqij/du #
    ##################################
    if True:
        from phelel.interface.vasp.derivatives import create_derivatives

        if phelel.dataset is None or "first_atoms" not in phelel.dataset:
            if cell_info.phelel_yaml is not None:
                phe_yml = cell_info.phelel_yaml
            else:
                if log_level:
                    print(
                        "************************************************************"
                    )
                    print("  Use of POSCAR like crystal structure input is deprected.")
                    print("  Please use phelel-yaml like input.")
                    print(
                        "************************************************************"
                    )
                filename = "phelel_disp.yaml"
                if not pathlib.Path(filename).exists():
                    raise RuntimeError(f'"{filename}" not found.')
                if log_level:
                    print(f'Read displacement datasets from "{filename}".')
                phe_yml = PhelelYaml()
                phe_yml.read(filename)

            phelel.dataset = phe_yml.dataset
            if phe_yml.phonon_dataset is not None:
                phelel.phonon_dataset = phe_yml.phonon_dataset

            if pathlib.Path("BORN").exists() or phe_yml.nac_params:
                store_nac_params(
                    phelel.phonon,
                    settings,
                    cell_info.phelel_yaml,
                    unitcell_filename,
                    log_level,
                    load_phonopy_yaml=load_phelel_yaml,
                )

        if settings.create_derivatives:
            create_derivatives(
                phelel,
                settings.create_derivatives,
                subtract_rfs=settings.subtract_rfs,
                log_level=log_level,
            )
            if phelel.fft_mesh is not None:
                phelel.save_hdf5(filename="phelel_params.hdf5")
                if log_level > 0:
                    print('"phelel_params.hdf5" has been created.')
            print_end()
            sys.exit(0)

    if log_level > 0:
        finalize_phelel(
            phelel,
            confs=phelel_conf.confs,
            log_level=log_level,
            filename="phelel.yaml",
        )
