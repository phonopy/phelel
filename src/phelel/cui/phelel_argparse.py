"""Option parser."""


def get_parser(load_phelel_yaml=False):
    """Return ArgumentParser instance."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Electron-phonon interaction calculator"
    )
    parser.add_argument(
        "--amplitude",
        dest="displacement_distance",
        type=float,
        default=None,
        help="Distance of displacements",
    )
    if not load_phelel_yaml:
        parser.add_argument(
            "-c",
            "--cell",
            dest="cell_filename",
            metavar="FILE",
            default=None,
            help="Read unit cell",
        )
    parser.add_argument(
        "--cd",
        "--create-derivatives",
        nargs="+",
        dest="create_derivatives",
        metavar="DIR",
        default=None,
        help=(
            "Calculate dVdu and dDijdu and create phelel_disp.hdf5. "
            "Directory names of perfect "
            "and displacement supercells are given as arguments."
        ),
    )
    if load_phelel_yaml:
        parser.add_argument(
            "--config",
            dest="conf_filename",
            metavar="FILE",
            default=None,
            help="Read phelel configuration file",
        )
    if not load_phelel_yaml:
        parser.add_argument(
            "-d",
            "--disp",
            dest="is_displacement",
            action="store_true",
            default=False,
            help="As first stage, get least displacements",
        )
        parser.add_argument(
            "--dim",
            nargs="+",
            dest="supercell_dimension",
            metavar="INT",
            default=None,
            help=(
                "Supercell dimensions with three integers or matrix with " "9 integers"
            ),
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
        "--nodiag",
        dest="is_nodiag",
        action="store_true",
        default=False,
        help="Set displacements parallel to axes",
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
        "--pm",
        dest="is_plusminus_displacements",
        action="store_true",
        default=False,
        help="Set plus minus displacements",
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
        "--qpoints",
        nargs="+",
        dest="qpoints",
        metavar="NUM",
        default=None,
        help="Phonon q-points for specific sampling to calculate |g|",
    )
    parser.add_argument(
        "--read-qpoints",
        dest="read_qpoints",
        action="store_true",
        default=False,
        help="Read QPOITNS file for specific phonon sampling to calculate |g|",
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
    parser.add_argument(
        "--srfs",
        "--subtract-residual-forces",
        dest="subtract_rfs",
        action="store_true",
        default=False,
        help="Subtract residual forces for force constants calculation.",
    )
    if load_phelel_yaml:
        parser.add_argument("filename", nargs="*", help="phelel.yaml like file")
    else:
        parser.add_argument("filename", nargs="*", help="Phelel configure file")

    return parser
