"""Command line option handler."""

import numpy as np
from phonopy.cui.settings import ConfParser, Settings


class PhelelSettings(Settings):
    """Setting parameter container."""

    _default = {
        "create_derivatives": None,
        "fft_mesh_numbers": None,
        "finufft_eps": None,
        "grid_points": None,
        "phonon_supercell_matrix": None,
        "subtract_rfs": False,
    }

    def __init__(self, default=None):
        """Init method."""
        super().__init__(default=default)
        self._v.update(PhelelSettings._default.copy())
        if default is not None:
            self._v.update(default)

    def set_create_derivatives(self, val):
        """Setter of create_derivatives."""
        self._v["create_derivatives"] = val

    def set_fft_mesh_numbers(self, val):
        """Setter of fft_mesh_numbers."""
        self._v["fft_mesh_numbers"] = val

    def set_finufft_eps(self, val):
        """Setter of finufft_eps."""
        self._v["finufft_eps"] = val

    def set_grid_points(self, val):
        """Setter of grid_points."""
        self._v["grid_points"] = val

    def set_phonon_supercell_matrix(self, val):
        """Set phonon_supercell_matrix."""
        self._v["phonon_supercell_matrix"] = val

    def set_subtract_rfs(self, val):
        """Setter of subtract_rfs."""
        self._v["subtract_rfs"] = val


class PhelelConfParser(ConfParser):
    """Phelel setting parameter parser."""

    def __init__(self, filename=None, args=None, default_settings=None):
        """Init method."""
        self._settings = PhelelSettings(default=default_settings)
        confs = {}
        if filename is not None:
            super().__init__(filename=filename)
            self.read_file()  # store .conf file setting in self._confs
            self._parse_conf()
            self._set_settings()
            confs.update(self._confs)
        if args is not None:
            super().__init__(args=args)
            self._read_options()
            self._parse_conf()
            self._set_settings()
            confs.update(self._confs)
        self._confs = confs

    def _read_options(self):
        self.read_options()  # store data in self._confs
        if "create_derivatives" in self._args:
            if self._args.create_derivatives:
                dir_names = self._args.create_derivatives
                self._confs["create_derivatives"] = " ".join(dir_names)
        if "fft_mesh_numbers" in self._args:
            if self._args.fft_mesh_numbers:
                self._confs["fft_mesh"] = " ".join(self._args.fft_mesh_numbers)
        if "finufft_eps" in self._args:
            if self._args.finufft_eps is not None:
                self._confs["finufft_eps"] = self._args.finufft_eps
        if "phonon_supercell_dimension" in self._args:
            dim_phonon = self._args.phonon_supercell_dimension
            if dim_phonon is not None:
                self._confs["dim_phonon"] = " ".join(dim_phonon)
        if "subtract_rfs" in self._args:
            if self._args.subtract_rfs:
                self._confs["subtract_rfs"] = ".true."

    def _parse_conf(self):
        self.parse_conf()
        confs = self._confs

        for conf_key in confs.keys():
            if conf_key == "create_derivatives":
                self.set_parameter(
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
                        self.set_parameter("dim_phonon", matrix)

            if conf_key == "fft_mesh":
                fft_mesh_nums = [int(x) for x in confs["fft_mesh"].split()]
                if len(fft_mesh_nums) == 3:
                    self.set_parameter("fft_mesh_numbers", fft_mesh_nums)
                else:
                    self.setting_error(
                        "Number of elements of fft_mesh tag has to be 3."
                    )

            if conf_key == "finufft_eps":
                self.set_parameter("finufft_eps", confs["finufft_eps"])

            if conf_key == "subtract_rfs":
                if confs["subtract_rfs"] == ".true.":
                    self.set_parameter("subtract_rfs", True)

    def _set_settings(self):
        ConfParser.set_settings(self)
        params = self._parameters

        if "create_derivatives" in params:
            if params["create_derivatives"]:
                self._settings.set_create_derivatives(params["create_derivatives"])

        if "dim_phonon" in params:
            self._settings.set_phonon_supercell_matrix(params["dim_phonon"])

        if "fft_mesh_numbers" in params:
            if params["fft_mesh_numbers"]:
                self._settings.set_fft_mesh_numbers(params["fft_mesh_numbers"])

        if "finufft_eps" in params:
            if params["finufft_eps"]:
                self._settings.set_finufft_eps(params["finufft_eps"])

        if "subtract_rfs" in params:
            if params["subtract_rfs"]:
                self._settings.set_subtract_rfs(params["subtract_rfs"])
