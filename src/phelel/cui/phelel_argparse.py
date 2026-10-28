"""Option parser."""

from __future__ import annotations

import argparse
from typing import Literal


def get_parser(mode: Literal["init", "run"]) -> argparse.ArgumentParser:
    """Return ArgumentParser instance.

    Parameters
    ----------
    mode : "init" or "run"
        "init" is for phelel-init, which generates supercells with
        displacements.  "run" is for phelel, which calculates the derivatives
        from phelel_disp.yaml and the results of the supercell calculations.

    """
    if mode == "init":
        description = (
            "phelel-init: generate supercells with displacements for "
            "electron-phonon calculation"
        )
    else:
        description = (
            "phelel: calculate derivatives of local potential and PAW strengths "
            "with respect to atomic displacements"
        )
    parser = argparse.ArgumentParser(description=description)
    _add_common_options(parser)
    if mode == "init":
        _add_init_options(parser)
        parser.add_argument("filename", nargs="*", help="Phelel configuration file")
    else:
        _add_run_options(parser)
        _reject_init_options(parser)
        parser.add_argument(
            "filename",
            nargs="*",
            help="phelel_disp.yaml-like file (default: phelel_disp.yaml)",
        )
    return parser


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    """Add options accepted by both phelel-init and phelel."""
    parser.add_argument(
        "--loglevel",
        dest="log_level",
        type=int,
        metavar="INT",
        default=None,
        help="Log level",
    )
    parser.add_argument(
        "--nosym",
        dest="is_nosym",
        action="store_true",
        default=None,
        help="Symmetry is not imposed.",
    )
    parser.add_argument(
        "--pa",
        "--primitive-axis",
        "--primitive-axes",
        nargs="+",
        dest="primitive_axes",
        default=None,
        help="Same as PRIMITIVE_AXES tags",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        dest="quiet",
        action="store_true",
        default=False,
        help="Print out smallest information",
    )
    parser.add_argument(
        "--tolerance",
        dest="symmetry_tolerance",
        type=float,
        default=None,
        help="Symmetry tolerance to search",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Detailed run-time information is displayed.",
    )


def _add_init_options(parser: argparse.ArgumentParser) -> None:
    """Add options that only apply to phelel-init."""
    parser.add_argument(
        "--amplitude",
        dest="displacement_distance",
        type=float,
        default=None,
        help="Distance of displacements",
    )
    parser.add_argument(
        "-c",
        "--cell",
        dest="cell_filename",
        metavar="FILE",
        default=None,
        help="Read unit cell",
    )
    parser.add_argument(
        "-d",
        "--disp",
        dest="is_displacement",
        action="store_true",
        default=False,
        help="Generate supercells with displacements",
    )
    parser.add_argument(
        "--dim",
        nargs="+",
        dest="supercell_dimension",
        metavar="INT",
        default=None,
        help="Supercell dimensions with three integers or matrix with 9 integers",
    )
    parser.add_argument(
        "--dim-phonon",
        nargs="+",
        dest="phonon_supercell_dimension",
        default=None,
        help=(
            "Supercell dimensions for phonon with three integers or "
            "matrix with 9 integers"
        ),
    )
    parser.add_argument(
        "--magmom",
        nargs="+",
        dest="magmoms",
        default=None,
        help=(
            "Magnetic moments of atoms in unit cell, N values (collinear) or "
            "3N values (non-collinear). Same as MAGMOM tag."
        ),
    )
    parser.add_argument(
        "--nodiag",
        dest="is_nodiag",
        action="store_true",
        default=False,
        help="Set displacements parallel to axes",
    )
    parser.add_argument(
        "--pm",
        dest="is_plusminus_displacements",
        action="store_true",
        default=False,
        help="Set plus minus displacements",
    )


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    """Add options that only apply to phelel."""
    parser.add_argument(
        "--config",
        dest="conf_filename",
        metavar="FILE",
        default=None,
        help="Phelel configuration file",
    )
    parser.add_argument(
        "--cd",
        "--create-derivatives",
        nargs="+",
        dest="create_derivatives",
        metavar="DIR",
        default=None,
        help=(
            "Calculate the derivatives and write them in phelel_params.hdf5. "
            "Directory names of perfect and displacement supercells are given "
            "as arguments. --fft-mesh is required."
        ),
    )
    parser.add_argument(
        "--fft-mesh",
        nargs="+",
        dest="fft_mesh_numbers",
        default=None,
        help="FFT mesh numbers used in primitive cell",
    )
    parser.add_argument(
        "--finufft-eps",
        dest="finufft_eps",
        type=float,
        default=None,
        help="Accuracy of finufft interpolation (default=1e-6)",
    )
    parser.add_argument(
        "--nufft",
        dest="nufft",
        choices=["finufft", "native"],
        default=None,
        help=(
            "Method to evaluate the change of the local potential at the grid "
            'points: "finufft" (default) or "native" (FFT without the '
            "approximation of NUFFT)"
        ),
    )
    parser.add_argument(
        "--srfs",
        "--subtract-residual-forces",
        dest="subtract_rfs",
        action="store_true",
        default=False,
        help="Subtract residual forces for force constants calculation.",
    )


class _MovedToInitAction(argparse.Action):
    """Reject an option of phelel-init given to phelel."""

    def __call__(self, parser, namespace, values, option_string=None):
        parser.exit(
            2,
            f"{parser.prog}: error: '{option_string}' is an option of "
            "'phelel-init', which generates supercells with displacements.\n"
            f"Replace '{parser.prog}' with 'phelel-init' in your command.\n"
            "See https://phonopy.github.io/phelel/phelel-command.html\n",
        )


def _reject_init_options(parser: argparse.ArgumentParser) -> None:
    """Make phelel reject the options of phelel-init with a message that points there.

    Without this, argparse rejects them with a generic 'unrecognized
    arguments' message.

    """
    specs: list[tuple[tuple[str, ...], int | str]] = [
        (("--amplitude",), 1),
        (("-c", "--cell"), 1),
        (("-d", "--disp"), 0),
        (("--dim",), "+"),
        (("--dim-phonon",), "+"),
        (("--magmom",), "+"),
        (("--nodiag",), 0),
        (("--pm",), 0),
    ]
    for option_strings, nargs in specs:
        kwargs: dict = {
            "action": _MovedToInitAction,
            "nargs": nargs,
            "help": argparse.SUPPRESS,
            "default": argparse.SUPPRESS,
        }
        if nargs == 0:
            kwargs["const"] = True
        parser.add_argument(*option_strings, **kwargs)
