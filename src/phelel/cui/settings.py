"""Command line option handler."""

from __future__ import annotations

import argparse
import os

import numpy as np
from phonopy.cui.settings import ConfParser, Settings


class PhelelSettings(Settings):
    """Setting parameter container."""

    def __init__(self):
        """Init method."""
        super().__init__(load_phonopy_yaml=False)
        self.create_derivatives = None
        self.fft_mesh_numbers = None
        self.finufft_eps = None
        self.grid_points = None
        self.phonon_supercell_matrix = None
        self.subtract_rfs = False


class PhelelConfParser(ConfParser):
    """Phelel setting parameter parser."""

    def __init__(
        self,
        filename: str | os.PathLike | None = None,
        args: argparse.Namespace | None = None,
    ):
        """Init method."""
        super().__init__()
        if filename is not None:
            self._read_file(filename)
        if args is not None:
            self._read_options(args)
        self._parse_conf()
        self.settings = PhelelSettings()
        self._set_settings(self.settings)

    def _read_options(self, args: argparse.Namespace):
        super()._read_options(args)  # store data in self._confs
        if "create_derivatives" in args:
            if args.create_derivatives:
                dir_names = args.create_derivatives
                self._confs["create_derivatives"] = " ".join(dir_names)
        if "fft_mesh_numbers" in args:
            if args.fft_mesh_numbers:
                self._confs["fft_mesh"] = " ".join(args.fft_mesh_numbers)
        if "finufft_eps" in args:
            if args.finufft_eps is not None:
                self._confs["finufft_eps"] = args.finufft_eps
        if "phonon_supercell_dimension" in args:
            dim_phonon = args.phonon_supercell_dimension
            if dim_phonon is not None:
                self._confs["dim_phonon"] = " ".join(dim_phonon)
        if "subtract_rfs" in args:
            if args.subtract_rfs:
                self._confs["subtract_rfs"] = ".true."

    def _parse_conf(self):
        super()._parse_conf()
        confs = self._confs

        for conf_key in confs.keys():
            if conf_key == "create_derivatives":
                self._set_parameter(
                    "create_derivatives", confs["create_derivatives"].split()
                )

            if conf_key == "dim_phonon":
                matrix = [int(x) for x in confs["dim_phonon"].split()]
                if len(matrix) == 9:
                    matrix = np.array(matrix).reshape(3, 3)
                elif len(matrix) == 3:
                    matrix = np.diag(matrix)
                else:
                    self.setting_error(
                        "Number of elements of dim-phonon has to be 3 or 9."
                    )

                if matrix.shape == (3, 3):
                    if np.linalg.det(matrix) < 1:
                        self.setting_error(
                            "Determinant of supercell matrix has " + "to be positive."
                        )
                    else:
                        self._set_parameter("dim_phonon", matrix)

            if conf_key == "fft_mesh":
                fft_mesh_nums = [int(x) for x in confs["fft_mesh"].split()]
                if len(fft_mesh_nums) == 3:
                    self._set_parameter("fft_mesh_numbers", fft_mesh_nums)
                else:
                    self.setting_error(
                        "Number of elements of fft_mesh tag has to be 3."
                    )

            if conf_key == "finufft_eps":
                self._set_parameter("finufft_eps", confs["finufft_eps"])

            if conf_key == "subtract_rfs":
                if confs["subtract_rfs"] == ".true.":
                    self._set_parameter("subtract_rfs", True)

    def _set_settings(self, settings: PhelelSettings):
        super()._set_settings(settings)
        params = self._parameters

        if "create_derivatives" in params:
            if params["create_derivatives"]:
                settings.create_derivatives = params["create_derivatives"]

        if "dim_phonon" in params:
            settings.phonon_supercell_matrix = params["dim_phonon"]

        if "fft_mesh_numbers" in params:
            if params["fft_mesh_numbers"]:
                settings.fft_mesh_numbers = params["fft_mesh_numbers"]

        if "finufft_eps" in params:
            if params["finufft_eps"]:
                settings.finufft_eps = params["finufft_eps"]

        if "subtract_rfs" in params:
            if params["subtract_rfs"]:
                settings.subtract_rfs = params["subtract_rfs"]
