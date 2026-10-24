"""phelel command line interface."""

from __future__ import annotations

import os
import pathlib
import sys
import textwrap
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray
from phonopy import Phonopy
from phonopy.cui.phonopy_script import (
    file_exists,
    print_end,
    print_error,
    print_error_message,
    print_version,
    store_nac_params,
)
from phonopy.exception import (
    CellNotFoundError,
    MagmomValueError,
    NotPhonopyYamlError,
)
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import print_cell

from phelel import Phelel, __version__
from phelel.cui.create_supercells import create_phelel_supercells, get_cell_info
from phelel.cui.phelel_argparse import get_parser
from phelel.cui.settings import PhelelConfParser, PhelelSettings
from phelel.interface.vasp.derivatives import create_derivatives


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
    confs: dict | None = None,
    log_level: int | bool = 0,
    displacements_mode: bool = False,
    filename: str | pathlib.Path = "phelel.yaml",
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
    yaml_settings = {"force_sets": False, "displacements": displacements_mode}
    phe_yml = phelel.to_phelel_yaml(configuration=confs, settings=yaml_settings)
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
    """Phelel CUI main function.

    The commands give argparse_control as follows.

    phelel-init (generation of supercells with displacements):
        {"mode": "init"}
    phelel (calculation of derivatives from phelel_disp.yaml):
        {"mode": "run"}
    phelel-load (deprecated alias of phelel):
        {"mode": "run", "deprecated_command": "phelel-load"}

    For pytest, argparse_control["args"] is given in place of the parsed
    command options.

    """
    mode: Literal["init", "run"] = argparse_control["mode"]
    deprecated_command = argparse_control.get("deprecated_command")
    if deprecated_command is not None:
        print("")
        print(f"WARNING: '{deprecated_command}' is deprecated. Use 'phelel' instead.")
        print("")

    #################
    # Option parser #
    #################
    if "args" in argparse_control:  # For pytest
        args = argparse_control["args"]
        log_level = args.log_level
    else:
        parser = get_parser(mode)
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

    # With phelel-init, the argument is a configuration file.  With phelel, it
    # is a phelel_disp.yaml-like file, and a configuration file is given by
    # --config.
    if mode == "init" and len(args.filename) > 0:
        conf_filename = args.filename[0]
    elif mode == "run":
        conf_filename = args.conf_filename
    else:
        conf_filename = None
    if conf_filename is not None:
        file_exists(conf_filename, log_level=log_level)
    phelel_conf = PhelelConfParser(filename=conf_filename, args=args)
    if log_level > 0 and conf_filename is not None:
        print(f'"{conf_filename}" was read as phelel configuration file.')
    settings = phelel_conf.settings
    if mode == "run" and len(args.filename) > 0:
        cell_filename = args.filename[0]
    else:
        cell_filename = settings.cell_filename

    _check_operation(mode, settings, log_level)

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
            cell_filename=cell_filename,
            log_level=log_level,
            load_phonopy_yaml=(mode == "run"),
        )
    except CellNotFoundError as e:
        print_error_message(str(e))
        if isinstance(e, NotPhonopyYamlError) and mode == "run" and args.filename:
            print(
                f'If "{args.filename[0]}" is a configuration file, give it by --config.'
            )
        if log_level > 0:
            print_error()
        sys.exit(1)
    except MagmomValueError as e:
        print_error_message(str(e))
        if log_level > 0:
            print_error()
        sys.exit(1)
    assert cell_info.supercell_matrix is not None

    unitcell = cell_info.unitcell
    supercell_matrix = cell_info.supercell_matrix
    primitive_matrix = cell_info.primitive_matrix
    unitcell_filename = cell_info.optional_structure_info.unitcell_filename
    phe_yml = cell_info.phelel_yaml
    phonon_supercell_matrix = cell_info.phonon_supercell_matrix

    if mode == "init":
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

    assert phe_yml is not None
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
    if phonon_supercell_matrix is not None:
        assert phelel.phonon_supercell_matrix is not None
        assert phelel.phonon_supercell is not None

    if log_level > 0:
        _show_settings(
            phelel,
            unitcell_filename,
            supercell_matrix,
            primitive_matrix,
            phonon_supercell_matrix,
            fft_mesh,
        )

    if phe_yml.dataset is not None:
        phelel.dataset = phe_yml.dataset
    if phe_yml.phonon_dataset is not None:
        phelel.phonon_dataset = phe_yml.phonon_dataset
    if pathlib.Path("BORN").is_file() or phe_yml.nac_params:
        store_nac_params(
            cast(Phonopy, phelel),
            settings,
            phe_yml,
            unitcell_filename,
            log_level,
        )

    ##################################
    # Create dV/du, dDij/du, dqij/du #
    ##################################
    if settings.create_derivatives:
        if phelel.dataset is None:
            print_error_message(
                _wrap(
                    f'"{unitcell_filename}" has no displacement dataset.',
                    "Generate the supercells with displacements by 'phelel-init -d', "
                    "and give the phelel_disp.yaml that it writes.",
                )
            )
            if log_level:
                print_error()
            sys.exit(1)
        create_derivatives(
            phelel,
            settings.create_derivatives,
            subtract_rfs=settings.subtract_rfs,
            log_level=log_level,
        )
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


def _check_operation(
    mode: Literal["init", "run"], settings: PhelelSettings, log_level: int
) -> None:
    """Stop with an error when the operation does not belong to the command."""
    message = None
    if mode == "run" and settings.create_displacements:
        message = (
            "'-d' generates supercells with displacements, which is done by "
            "'phelel-init'.\n"
            "Replace 'phelel' with 'phelel-init' in your command."
        )
    elif mode == "init" and settings.create_derivatives:
        message = (
            "'--cd' calculates the derivatives, which is done by 'phelel'.\n"
            "Replace 'phelel-init' with 'phelel' in your command."
        )
    elif mode == "init" and not settings.create_displacements:
        message = (
            "'phelel-init' generates supercells with displacements, and needs "
            "'-d' for it.\n"
            "To calculate the derivatives, use 'phelel --cd'."
        )
    elif mode == "run" and settings.create_derivatives:
        if settings.fft_mesh_numbers is None:
            message = _wrap(
                "'--cd' needs '--fft-mesh', the numbers of the FFT mesh of the "
                "primitive cell on which the derivatives are written, e.g., "
                "'--fft-mesh 18 18 18'."
            )
    if message is not None:
        print_error_message(message)
        if log_level:
            print_error()
        sys.exit(1)


def _wrap(*paragraphs: str) -> str:
    """Return the paragraphs wrapped at 79 characters, one after another.

    Each paragraph starts on a new line.  Lines are not broken at hyphens, so
    that command names and options such as 'phelel-init' are kept whole.

    """
    return "\n".join(
        textwrap.fill(
            paragraph, width=79, break_on_hyphens=False, break_long_words=False
        )
        for paragraph in paragraphs
    )


def _show_settings(
    phelel: Phelel,
    unitcell_filename: str | os.PathLike | None,
    supercell_matrix: NDArray,
    primitive_matrix: str | NDArray | None,
    phonon_supercell_matrix: NDArray | None,
    fft_mesh: NDArray | None,
) -> None:
    """Show the crystal structure and the settings."""
    print("")
    print(f'Crystal structure was read from "{unitcell_filename}".')
    print("Settings:")
    if (np.diag(np.diag(supercell_matrix)) - supercell_matrix).any():
        print("  Supercell matrix:")
        for v in supercell_matrix:
            print("    %s" % v)
    else:
        print("  Supercell: %s" % np.diag(supercell_matrix))
    if primitive_matrix is not None:
        if isinstance(primitive_matrix, str) and primitive_matrix == "auto":
            print("  Primitive matrix (Auto):")
        else:
            print("  Primitive matrix:")
        assert phelel.primitive_matrix is not None
        for v in phelel.primitive_matrix:
            print("    %s" % v)
    if phonon_supercell_matrix is not None:
        assert phelel.phonon_supercell_matrix is not None
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
        print_cell(cast(PhonopyAtoms, phelel.phonon_supercell))
    print("-" * 76)
